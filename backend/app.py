import hashlib
import hmac
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import init_db, get_db, check_db_health, SupportTicket
from rag_engine import TwinRepRAG, MANUALS_DIR

load_dotenv()

LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8001/v1")
LLM_SERVER_ROOT = os.getenv("LLM_SERVER_ROOT", "http://localhost:8001")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instruct")
MANAGER_USERNAME = os.getenv("MANAGER_USERNAME", "admin")
MANAGER_PASSWORD = os.getenv("MANAGER_PASSWORD", "twinrep-admin")

rag = TwinRepRAG()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="TwinRep AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- Authentication Helpers ---
def _make_manager_token(username: str) -> str:
    sig = hmac.new(MANAGER_PASSWORD.encode("utf-8"), username.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{username}:{sig}"


def _verify_manager_token(token: str) -> bool:
    if ":" not in token:
        return False
    username, _ = token.split(":", 1)
    if not hmac.compare_digest(username, MANAGER_USERNAME):
        return False
    expected = _make_manager_token(username)
    return hmac.compare_digest(token, expected)


async def require_manager(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Manager authentication required.")
    token = authorization.removeprefix("Bearer ").strip()
    if not _verify_manager_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired manager session token.")
    return MANAGER_USERNAME


# --- Request Schemas ---
class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    customer_name: str = "Valued Partner"
    question: str


class EditTicketRequest(BaseModel):
    edited_response: str


class PersonaScaleRequest(BaseModel):
    scale: float  # 0.0 = Generic Base LLM, 1.0 = Full Cloned Manager Persona


# --- Auth & Health Endpoints ---
@app.post("/api/auth/login")
async def manager_login(payload: LoginRequest):
    user_ok = hmac.compare_digest(payload.username.strip(), MANAGER_USERNAME)
    pass_ok = hmac.compare_digest(payload.password, MANAGER_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Invalid manager username or password.")
    return {"access_token": _make_manager_token(MANAGER_USERNAME), "username": MANAGER_USERNAME}


@app.get("/api/auth/verify")
async def verify_session(manager: str = Depends(require_manager)):
    return {"authenticated": True, "username": manager}


@app.get("/api/health")
async def system_health():
    db_ok = await check_db_health()
    rag_stats = rag.get_stats()

    llm_status = "offline"
    lora_scale = None
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            h = await client.get(f"{LLM_SERVER_ROOT}/health")
            llm_status = "online" if h.status_code == 200 else f"status_{h.status_code}"
        except Exception:
            llm_status = "offline"

        if llm_status == "online":
            try:
                la = await client.get(f"{LLM_SERVER_ROOT}/lora-adapters")
                if la.status_code == 200:
                    adapters = la.json()
                    if isinstance(adapters, list) and adapters:
                        lora_scale = adapters[0].get("scale")
            except Exception:
                pass

    return {
        "mysql": "connected" if db_ok else "disconnected",
        "faiss": rag_stats,
        "llm": {"status": llm_status, "lora_scale": lora_scale},
    }


# --- Customer-Facing Endpoints ---
@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Retrieve threshold-filtered technical manual & verified ticket chunks
    matches = rag.retrieve(clean_question, k=4)
    context_block = (
        "\n\n---\n\n".join(
            f"[Internal Note - {m['source']}]\n{m['content']}"
            for m in matches
        )
        if matches
        else "NO_MATCHING_NOTES"
    )

    system_prompt = (
        "You are a Senior After-Sales Service Manager replying directly to a customer's support ticket. "
        "Write like an experienced, helpful human engineer—warm, direct, and practical.\n\n"
        "STRICT STYLE & ACCURACY RULES:\n"
        "1. NEVER say 'Based on our knowledge base', 'According to our records', 'historical verified', "
        "'provided documents', or 'As an AI'. Speak naturally from first-hand engineering experience.\n"
        "2. Open like a real person (e.g., 'Thanks for flagging this—let's get your KX-400 back online.') "
        "and go straight into the fix.\n"
        "3. Base all technical steps, torque/pressure specs, and part SKUs strictly on the INTERNAL SERVICE NOTES below, "
        "keeping safety warnings and numbered steps in their exact order.\n"
        "4. If INTERNAL SERVICE NOTES is 'NO_MATCHING_NOTES' or does not cover their specific issue, "
        "do NOT guess—naturally let them know you are escalating the ticket to our Level-2 Engineering team.\n\n"
        f"INTERNAL SERVICE NOTES:\n{context_block}"
    )

    # 2. Query local llama.cpp server (with active LoRA scale)
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            resp = await client.post(
                f"{LLM_API_BASE}/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean_question},
                    ],
                    "temperature": 0.35,
                    "max_tokens": 512,
                },
            )
            resp.raise_for_status()
            ai_draft = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM Server error: {e}")

    # 3. Log ticket in MySQL for Manager Review Loop
    ticket = SupportTicket(
        customer_name=(payload.customer_name or "Valued Partner").strip(),
        question=clean_question,
        retrieved_sources=json.dumps(matches),
        ai_draft=ai_draft,
        status="pending_review",
    )
    db.add(ticket)
    await db.commit()
    await db.refresh(ticket)

    return {
        "ticket_id": ticket.id,
        "ai_draft": ai_draft,
        "sources": matches,
        "status": ticket.status,
    }


@app.get("/api/tickets/{ticket_id}/status")
async def customer_ticket_status(ticket_id: int, db: AsyncSession = Depends(get_db)):
    ticket = await db.get(SupportTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket #{ticket_id} not found.")
    return {
        "id": ticket.id,
        "customer_name": ticket.customer_name,
        "question": ticket.question,
        "status": ticket.status,
        "ai_draft": ticket.ai_draft,
        "final_response": ticket.final_response,
        "created_at": ticket.created_at.isoformat(),
    }


# --- Protected Manager & Knowledge Base Endpoints ---
@app.post("/api/upload")
async def upload_manual(file: UploadFile = File(...), _: str = Depends(require_manager)):
    safe_name = Path(file.filename or "").name
    ext = Path(safe_name).suffix.lower()
    if ext not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")
    dest = MANUALS_DIR / safe_name
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = rag.ingest_file(dest)
    return {"filename": safe_name, **result}


@app.post("/api/ingest-folder")
async def ingest_folder(_: str = Depends(require_manager)):
    result = rag.ingest_existing_manuals()
    return result


@app.get("/api/tickets")
async def list_tickets(db: AsyncSession = Depends(get_db), _: str = Depends(require_manager)):
    result = await db.execute(select(SupportTicket).order_by(desc(SupportTicket.id)))
    tickets = result.scalars().all()
    return [
        {
            "id": t.id,
            "customer_name": t.customer_name,
            "question": t.question,
            "sources": json.loads(t.retrieved_sources or "[]"),
            "ai_draft": t.ai_draft,
            "final_response": t.final_response,
            "status": t.status,
            "created_at": t.created_at.isoformat(),
        }
        for t in tickets
    ]


@app.post("/api/tickets/{ticket_id}/approve")
async def approve_ticket(ticket_id: int, db: AsyncSession = Depends(get_db), _: str = Depends(require_manager)):
    ticket = await db.get(SupportTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    if ticket.status in {"approved", "edited"}:
        raise HTTPException(
            status_code=409,
            detail=f"Ticket #{ticket_id} is already locked ({ticket.status}) and indexed.",
        )

    ticket.status = "approved"
    ticket.final_response = ticket.ai_draft
    await db.commit()

    indexed = rag.add_verified_resolution(ticket.id, ticket.question, ticket.final_response)
    return {"status": "approved", "ticket_id": ticket.id, "newly_indexed": indexed}


@app.post("/api/tickets/{ticket_id}/edit")
async def edit_and_teach_ticket(
    ticket_id: int,
    payload: EditTicketRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_manager),
):
    ticket = await db.get(SupportTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    if ticket.status in {"approved", "edited"}:
        raise HTTPException(
            status_code=409,
            detail=f"Ticket #{ticket_id} is already locked ({ticket.status}) and indexed.",
        )

    clean_response = payload.edited_response.strip()
    if not clean_response:
        raise HTTPException(status_code=400, detail="Edited response cannot be empty.")

    ticket.status = "edited"
    ticket.final_response = clean_response
    await db.commit()

    indexed = rag.add_verified_resolution(ticket.id, ticket.question, ticket.final_response)
    return {"status": "edited_and_indexed", "ticket_id": ticket.id, "newly_indexed": indexed}


@app.post("/api/persona/scale")
async def set_persona_scale(payload: PersonaScaleRequest, _: str = Depends(require_manager)):
    """Hot-swaps LoRA adapter weight on llama-server between 0.0 (Generic) and 1.0 (Cloned Manager)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(f"{LLM_SERVER_ROOT}/lora-adapters", json=[{"id": 0, "scale": payload.scale}])
            r.raise_for_status()
            return {"scale": payload.scale, "llama_server_status": r.status_code}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Could not update LoRA scale on llama-server: {e}")


# Serve Frontend HTML/CSS/JS at root URL
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
