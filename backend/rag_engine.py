import hashlib
import json
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = Path("../data/faiss_index")
MANUALS_DIR = Path("../data/manuals")
MANIFEST_PATH = INDEX_DIR / "indexed_manifest.json"


class TwinRepRAG:
    def __init__(self, max_l2_distance: float = 1.35):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        MANUALS_DIR.mkdir(parents=True, exist_ok=True)
        self.max_l2_distance = max_l2_distance
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=850,
            chunk_overlap=100,
            separators=["\n===", "\n\n", "\n", ". ", " ", ""],
        )
        self.vectorstore = self._load_or_create_index()
        self.manifest = self._load_or_bootstrap_manifest()

        # Auto-ingest manuals folder on startup if no real documents are indexed yet
        if self.get_real_chunk_count() == 0:
            self.ingest_existing_manuals()

    def _load_or_create_index(self) -> FAISS:
        if (INDEX_DIR / "index.faiss").exists():
            return FAISS.load_local(str(INDEX_DIR), self.embeddings, allow_dangerous_deserialization=True)
        vs = FAISS.from_texts(
            texts=["TwinRep AI Technical After-Sales Knowledge Base initialized."],
            embedding=self.embeddings,
            metadatas=[{"source": "system_init"}],
        )
        vs.save_local(str(INDEX_DIR))
        return vs

    def _compute_file_hash(self, file_path: Path) -> str:
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def _load_or_bootstrap_manifest(self) -> dict:
        if MANIFEST_PATH.exists():
            try:
                return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Bootstrap manifest from any sources already present in an existing FAISS index
        existing_sources: dict[str, int] = {}
        existing_tickets: list[int] = []
        for doc in self.vectorstore.docstore._dict.values():
            src = doc.metadata.get("source", "")
            if not src or src == "system_init":
                continue
            if src.startswith("verified_ticket_"):
                try:
                    tid = int(src.replace("verified_ticket_", ""))
                    if tid not in existing_tickets:
                        existing_tickets.append(tid)
                except ValueError:
                    pass
            else:
                existing_sources[src] = existing_sources.get(src, 0) + 1

        files_meta = {}
        for filename, count in existing_sources.items():
            candidate = MANUALS_DIR / filename
            file_hash = self._compute_file_hash(candidate) if candidate.exists() else "legacy_indexed"
            files_meta[filename] = {"sha256": file_hash, "chunks": count}

        manifest = {"files": files_meta, "verified_tickets": sorted(existing_tickets)}
        self._save_manifest(manifest)
        return manifest

    def _save_manifest(self, manifest: dict | None = None):
        target = manifest if manifest is not None else self.manifest
        MANIFEST_PATH.write_text(json.dumps(target, indent=2), encoding="utf-8")

    def _delete_by_source(self, source_name: str):
        ids_to_delete = [
            doc_id
            for doc_id, doc in self.vectorstore.docstore._dict.items()
            if doc.metadata.get("source") == source_name
        ]
        if ids_to_delete:
            self.vectorstore.delete(ids_to_delete)

    def _extract_text(self, file_path: Path) -> str:
        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def ingest_file(self, file_path: Path) -> dict:
        if not file_path.exists():
            return {"status": "missing", "chunks_added": 0}

        file_hash = self._compute_file_hash(file_path)
        prev = self.manifest["files"].get(file_path.name)

        # Deduplication check: skip if identical file content was already indexed
        if prev and prev.get("sha256") == file_hash:
            return {
                "status": "skipped_duplicate",
                "chunks_added": 0,
                "message": f'"{file_path.name}" is unchanged and already indexed ({prev.get("chunks", 0)} chunks).',
            }

        raw_text = self._extract_text(file_path).strip()
        if not raw_text:
            return {"status": "empty", "chunks_added": 0, "message": f'"{file_path.name}" contained no readable text.'}

        # If updating an existing filename with new content, remove stale chunks first
        if prev:
            self._delete_by_source(file_path.name)

        chunks = self.splitter.split_text(raw_text)
        metadatas = [{"source": file_path.name} for _ in chunks]
        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        self.vectorstore.save_local(str(INDEX_DIR))

        self.manifest["files"][file_path.name] = {"sha256": file_hash, "chunks": len(chunks)}
        self._save_manifest()

        action = "Updated" if prev else "Indexed"
        return {
            "status": "indexed",
            "chunks_added": len(chunks),
            "message": f'{action} "{file_path.name}" ({len(chunks)} chunks).',
        }

    def ingest_existing_manuals(self) -> dict:
        added = 0
        skipped = 0
        for file_path in sorted(MANUALS_DIR.glob("*")):
            if file_path.suffix.lower() in {".pdf", ".txt", ".md"}:
                res = self.ingest_file(file_path)
                if res["status"] == "indexed":
                    added += res["chunks_added"]
                elif res["status"] == "skipped_duplicate":
                    skipped += 1
        return {"total_chunks_added": added, "files_skipped": skipped}

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        # Fetch extra candidates so filtering system_init and below-threshold docs still yields up to k matches
        results = self.vectorstore.similarity_search_with_score(query, k=k + 2)
        filtered = []
        for doc, l2_dist in results:
            src = doc.metadata.get("source", "manual")
            if src == "system_init":
                continue
            dist = float(l2_dist)
            if dist > self.max_l2_distance:
                continue
            similarity = round(max(0.0, 1.0 - (dist / 2.0)), 3)
            filtered.append({
                "source": src,
                "content": doc.page_content,
                "l2_distance": round(dist, 3),
                "similarity": similarity,
            })
            if len(filtered) >= k:
                break
        return filtered

    def add_verified_resolution(self, ticket_id: int, question: str, verified_answer: str) -> bool:
        """Feature 3 (Idempotent): Hot-updates FAISS once per ticket_id."""
        if ticket_id in self.manifest["verified_tickets"]:
            return False

        entry = (
            f"Verified Support Resolution (Ticket #{ticket_id})\n"
            f"Customer Question: {question}\n"
            f"Approved Technical Answer: {verified_answer}"
        )
        chunks = self.splitter.split_text(entry)
        metadatas = [{"source": f"verified_ticket_{ticket_id}"} for _ in chunks]
        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        self.vectorstore.save_local(str(INDEX_DIR))

        self.manifest["verified_tickets"].append(ticket_id)
        self._save_manifest()
        return True

    def get_real_chunk_count(self) -> int:
        return sum(
            1
            for doc in self.vectorstore.docstore._dict.values()
            if doc.metadata.get("source") != "system_init"
        )

    def get_stats(self) -> dict:
        return {
            "total_chunks": self.get_real_chunk_count(),
            "indexed_files": list(self.manifest["files"].keys()),
            "verified_tickets_count": len(self.manifest["verified_tickets"]),
            "max_l2_distance": self.max_l2_distance,
        }
