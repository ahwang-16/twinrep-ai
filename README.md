# TwinRep AI — Persona-Adaptive After-Sales Support Agent

EE 542 final project (Fall 2026). Author: Alex Hwang.

TwinRep AI is a local-first, full-stack B2B after-sales support platform built for specialized hardware and heavy machinery manufacturers. Instead of relying on rigid decision trees or generic LLMs that produce robotic answers and hallucinate technical specifications, the system acts as a digital twin of an experienced after-sales service manager.

It separates **conversational style** from **verified domain facts** across three core mechanisms:

1. **Behavioral Cloning (QLoRA + `llama.cpp`):** Fine-tunes `Meta-Llama-3.1-8B-Instruct` on historical manager replies so the model adopts the senior engineer's authentic tone, diagnostic phrasing, and industry terminology—with live runtime adapter scaling (`0.0` Generic Base LLM vs. `1.0` Cloned Manager).
2. **Dynamic Knowledge Base (RAG + FAISS):** Chunks and embeds technical manuals and verified support resolutions locally using `all-MiniLM-L6-v2`, enforcing SHA-256 file deduplication and L2 similarity thresholds to ground answers strictly in verified documentation.
3. **Self-Updating Continuous Learning Loop:** Logs every customer inquiry in MySQL (`pending_review`). When a manager clicks **Approve & Index** or **Edit & Teach AI**, the ticket is locked and the verified resolution is immediately vectorized into the live FAISS index without retraining the LLM.

## Layout

```text
twinrep-ai/
├── backend/
│   ├── app.py               # FastAPI server: RAG chat, LoRA scaling, manager auth, HITL review API
│   ├── database.py          # Async SQLAlchemy engine, health check, and SupportTicket model
│   ├── rag_engine.py        # FAISS vector store, SHA-256 manifest deduplication, threshold retrieval
│   ├── requirements.txt     # Python backend and embedding dependencies
│   └── .env.example         # Environment configuration template (copy to .env)
├── frontend/
│   ├── index.html           # 3-tab web portal (Customer Support, Knowledge Base, Manager Loop)
│   ├── style.css            # Responsive dark-mode B2B dashboard styling and status indicators
│   └── app.js               # Client-side state, ticket tracker, manager auth, and live health bar
├── training/
│   ├── prepare_data.py      # Splits raw support CSV into LoRA JSONL + RAG Q&A corpus
│   └── persona_train.jsonl  # Formatted instruction-response pairs for QLoRA fine-tuning
├── data/
│   ├── manuals/             # Product manuals (.pdf, .txt, .md) and historical_verified_qa.txt
│   ├── raw_logs/            # Historical B2B customer support ticket logs (support_logs.csv)
│   └── faiss_index/         # Persistent FAISS index (index.faiss, index.pkl, indexed_manifest.json)
├── llama.cpp/               # Local C++ inference server compiled with Metal/CUDA (gitignored)
└── models/                  # Base Llama-3.1-8B GGUF + trained twinrep_lora-F16.gguf (gitignored)
```

## Architecture

```text
Customer / Manager Browser (frontend/)
        │
        ▼  REST (JSON + Bearer Auth)
FastAPI Backend (backend/app.py :8000)
   ├──► MySQL 8+ (database.py) ───────────► support_tickets (pending_review | approved | edited)
   ├──► TwinRepRAG (rag_engine.py) ───────► FAISS Index + all-MiniLM-L6-v2 (L2 <= 1.35)
   │         ▲                                    │
   │         └── Hot-Index on Approve/Edit ◄──────┘ (Manager Review Loop)
   └──► llama-server (llama.cpp :8001) ───► Meta-Llama-3.1-8B-Instruct + --lora twinrep_lora-F16.gguf
```

## Data Sources & Dual-Stream Preprocessing (`training/`)

To prevent the fine-tuned model from memorizing outdated specifications while keeping technical facts instantly updatable, `training/prepare_data.py` splits `data/raw_logs/support_logs.csv` (`customer_question`, `manager_reply`, `category`) into two separate datasets:

- **Style & Persona Dataset (`training/persona_train.jsonl`):** Instruction-input-output JSONL pairs used exclusively to train the LoRA adapter on how the senior manager greets clients, structures troubleshooting steps, and uses domain jargon.
- **Factual Knowledge Corpus (`data/manuals/historical_verified_qa.txt`):** Structured historical ticket resolutions stored alongside OEM product manuals and spec sheets in `data/manuals/` for vector retrieval.

```sh
cd training
python3 prepare_data.py
```

## Setup

### 1. MySQL 8+ Database

Create the `twinrep` database and service user:

```sql
CREATE DATABASE twinrep CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'twinrep_user'@'localhost' IDENTIFIED BY 'twinrep_pass';
GRANT ALL PRIVILEGES ON twinrep.* TO 'twinrep_user'@'localhost';
FLUSH PRIVILEGES;
```

### 2. Environment Configuration

Copy `backend/.env.example` to `backend/.env` and update credentials if needed (`.env` is ignored by git):

```sh
cp backend/.env.example backend/.env
```

```env
DATABASE_URL=mysql+asyncmy://twinrep_user:twinrep_pass@localhost:3306/twinrep
LLM_API_BASE=http://localhost:8001/v1
LLM_SERVER_ROOT=http://localhost:8001
LLM_MODEL=llama-3.1-8b-instruct
APP_HOST=0.0.0.0
APP_PORT=8000
MANAGER_USERNAME=admin
MANAGER_PASSWORD=twinrep-admin
```

## Behavioral Cloning (`llama.cpp` + QLoRA)

The behavioral cloning adapter is trained via 4-bit QLoRA using **Unsloth** on Google Colab (`unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, rank `r = 16`, `lora_alpha = 16`, target modules `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, 3 epochs, learning rate `2e-4`) and exported as a standalone GGUF adapter (`twinrep_lora-F16.gguf`, ~160 MB).

Compile `llama.cpp` locally (automatically enables Apple Metal GPU acceleration on macOS) and start `llama-server` on port `8001` with both the quantized base model and the trained LoRA adapter:

```sh
git clone https://github.com/ggerganov/llama.cpp
cmake -B llama.cpp/build -S llama.cpp
cmake --build llama.cpp/build --config Release -j

./llama.cpp/build/bin/llama-server \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --lora ./models/twinrep_lora-F16.gguf \
  --host 0.0.0.0 \
  --port 8001 \
  --ctx-size 4096
```

At runtime, the backend dynamically adjusts the LoRA adapter weight via `POST http://localhost:8001/lora-adapters` (`scale: 1.0` for Cloned Manager vs. `scale: 0.0` for Generic Base LLM) without restarting the inference server.

## Dynamic Knowledge Base (`backend/rag_engine.py`)

`TwinRepRAG` manages local text extraction, chunking, embedding, deduplication, and retrieval:

- **Embeddings & Chunking:** Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional local embeddings, zero external API calls) and `RecursiveCharacterTextSplitter` with `chunk_size=850`, `chunk_overlap=100`, and section-aware separators (`["\n===", "\n\n", "\n", ". ", " ", ""]`) so multi-step repair procedures stay intact within a single chunk.
- **SHA-256 Manifest Deduplication:** Tracks file hashes and verified ticket IDs in `data/faiss_index/indexed_manifest.json`. Re-syncing unchanged files skips duplicate insertions, while uploading an updated version of an existing filename deletes its stale chunks before re-indexing.
- **Startup Auto-Ingest:** Automatically indexes all `.pdf`, `.txt`, and `.md` files in `data/manuals/` on server boot if the FAISS index contains zero real document chunks.
- **Similarity Guardrail:** Queries `similarity_search_with_score` and discards chunks whose L2 distance exceeds `max_l2_distance = 1.35`, converting passing distances into normalized similarity scores (`1 - d/2`). When no chunks meet the threshold, the system prompt instructs the model to escalate to Level-2 Engineering rather than guessing.

## Web Portal (`frontend/`)

Plain HTML5, CSS3, and JavaScript served directly by FastAPI at `http://localhost:8000` with role separation between public customers and authenticated managers:

- **Top Health & Persona Bar:** Polls `GET /api/health` on load and after mutations to display live status pills for `MySQL`, `FAISS` (indexed chunk count), and `LLM` (online status and active `LoRA` scale).
- **Tab 1 — Customer Support Portal (Public):** Clients submit equipment issues, view AI-drafted responses with cited RAG sources and similarity percentages, and use the **Track Ticket Status** bar (`Ticket #` lookup) to view official manager-verified responses.
- **Tab 2 — Knowledge Base (Manager Protected):** Upload `.pdf`, `.txt`, or `.md` technical manuals or sync `data/manuals/` into FAISS with live duplicate detection.
- **Tab 3 — Manager Review Loop (Manager Protected):** Displays all customer tickets from MySQL with retrieved source metadata and an editable response box:
  - **Approve & Index:** Locks the ticket (`approved`) and indexes the verified Q&A pair into FAISS.
  - **Edit & Teach AI:** Saves the manager's revised technical fix (`edited`), locks the ticket, and immediately vectorizes the resolution into the live FAISS index so future inquiries retrieve it automatically.

## REST API & Database (`backend/`)

FastAPI with async SQLAlchemy (`asyncmy`) and stateless HMAC-SHA256 Bearer authentication. Tables are auto-created on startup via `init_db()`.

```sh
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

| Endpoint | Role | Purpose |
|---|---|---|
| `GET /api/health` | public | Live status for MySQL, FAISS chunk/file counts, and `llama-server` LoRA scale |
| `POST /api/chat` | public | Retrieve RAG chunks (`L2 <= 1.35`), generate LoRA-styled draft, log `pending_review` ticket |
| `GET /api/tickets/{ticket_id}/status` | public | Check whether a ticket is `pending_review`, `approved`, or `edited` and fetch `final_response` |
| `POST /api/auth/login` | public | Authenticate manager credentials (`200` with Bearer token, `401` invalid credentials) |
| `GET /api/auth/verify` | manager | Validate active manager Bearer session token |
| `POST /api/upload` | manager | Upload `.pdf`, `.txt`, or `.md` manual, deduplicate by SHA-256, and index into FAISS |
| `POST /api/ingest-folder` | manager | Batch-scan `data/manuals/` and index any new or modified documents |
| `GET /api/tickets` | manager | List all support tickets ordered newest-first with retrieved source metadata |
| `POST /api/tickets/{ticket_id}/approve` | manager | Approve AI draft, lock ticket (`409` if already locked), and index into FAISS |
| `POST /api/tickets/{ticket_id}/edit` | manager | Save edited manager response, lock ticket (`409` if already locked), and index into FAISS |
| `POST /api/persona/scale` | manager | Dynamically set `llama-server` LoRA adapter weight (`0.0` to `1.0`) |

Errors: `400` empty input or unsupported file extension, `401` missing or invalid manager token, `404` unknown ticket ID, `409` attempt to re-approve or re-edit a locked ticket, `502` local `llama-server` unreachable.

### MySQL Schema (`support_tickets`)

| Column | Type | Description |
|---|---|---|
| `id` | `INT` (PK) | Auto-incrementing ticket identifier |
| `customer_name` | `VARCHAR(120)` | Submitting client or company name |
| `question` | `TEXT` | Raw customer technical inquiry |
| `retrieved_sources` | `TEXT` | JSON array of matched RAG filenames, chunks, and similarity scores |
| `ai_draft` | `TEXT` | Initial response generated by the RAG + LoRA pipeline |
| `final_response` | `TEXT` | Manager-approved or edited official resolution (`NULL` while pending) |
| `status` | `VARCHAR(32)` | `pending_review`, `approved`, or `edited` |
| `created_at` | `DATETIME` | UTC creation timestamp |

## End-to-End Verification Workflow

1. **Start Services:** Launch `llama-server` on port `8001` (with `--lora` loaded) and `uvicorn app:app --host 0.0.0.0 --port 8000` in `backend/`.
2. **Verify System Health:** Open `http://localhost:8000` and confirm green header pills for `MySQL: connected`, `FAISS: X chunks`, and `LLM: online (LoRA=1.0)`.
3. **Compare Persona Cloning (Feature 1):** In Tab 1, ask *"Our KX-400 CNC machine just threw an E-04 spindle encoder error on startup and locked the Z-axis. What should our technician check first?"* Log in as Manager (`admin` / `twinrep-admin`) via Tab 2 or Tab 3, switch the header **Persona** selector to `Generic Base LLM (LoRA = 0.0)`, and re-ask the same question to compare generic vs. cloned manager phrasing.
4. **Verify Threshold Guardrail (Feature 2):** Ask an unrelated off-topic question in Tab 1 and confirm `RAG Sources Referenced: None (below similarity threshold)` appears and the agent escalates rather than hallucinating.
5. **Test Continuous Learning Loop (Feature 3):** Ask a novel equipment edge case not present in `data/manuals/`, open Tab 3, enter the engineering fix, and click **Edit & Teach AI**. Re-asking that exact question in Tab 1 immediately retrieves `verified_ticket_<id>` and returns the newly learned solution.