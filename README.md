# TwinRep AI — Persona-Adaptive After-Sales Support Agent

**TwinRep AI** is a local-first, full-stack B2B after-sales support platform that combines **QLoRA behavioral cloning** with **Retrieval-Augmented Generation (RAG)** and a **Human-in-the-Loop (HITL)** continuous learning pipeline.

Instead of relying on generic chatbots that produce robotic answers or hallucinate technical specifications, TwinRep AI separates **conversational style** from **verified domain facts**:

1. **Behavioral Cloning (QLoRA + `llama.cpp`):** Fine-tunes `Meta-Llama-3.1-8B-Instruct` on historical manager replies so the agent adopts the senior engineer's authentic tone, greeting style, and industry terminology—with live runtime adapter scaling (`0.0` Generic vs. `1.0` Cloned Manager).
2. **Dynamic Knowledge Base (RAG + FAISS):** Chunks and embeds product manuals and verified support logs locally using `all-MiniLM-L6-v2`, enforcing SHA-256 file deduplication and L2 similarity thresholds to ground responses strictly in verified documentation.
3. **Self-Updating Continuous Learning Loop:** Logs customer inquiries to MySQL (`pending_review`). When a manager clicks **Approve & Index** or **Edit & Teach AI**, the verified resolution is locked and immediately vectorized into the live FAISS index without retraining the LLM.

---

## Project Structure

```text
twinrep-ai/
├── backend/
│   ├── app.py               # FastAPI server (RAG chat, LoRA scaling, auth, HITL endpoints)
│   ├── database.py          # Async SQLAlchemy + MySQL SupportTicket model
│   ├── rag_engine.py        # FAISS vector store, SHA-256 deduplication, threshold retrieval
│   ├── requirements.txt     # Python backend dependencies
│   └── .env.example         # Environment configuration template
├── frontend/
│   ├── index.html           # 3-tab web portal (Customer Support, Knowledge Base, Manager Loop)
│   ├── style.css            # Responsive B2B dark-mode dashboard styling
│   └── app.js               # Client-side state, manager session auth, and live health checks
├── training/
│   ├── prepare_data.py      # Splits raw support CSV into LoRA JSONL + RAG Q&A corpus
│   └── persona_train.jsonl  # Formatted instruction-tuning dataset
└── data/
    ├── manuals/             # Technical manuals & verified historical Q&A documents
    ├── raw_logs/            # Raw historical customer support ticket logs
    └── faiss_index/         # Persistent FAISS vector index (auto-generated on startup)
