import csv
import json
from pathlib import Path

RAW_CSV = Path("../data/raw_logs/support_logs.csv")
LORA_JSONL = Path("../training/persona_train.jsonl")
RAG_QA_TXT = Path("../data/manuals/historical_verified_qa.txt")

def main():
    if not RAW_CSV.exists():
        print(f"Place your teammate's CSV at {RAW_CSV.resolve()} first.")
        return

    lora_records = []
    rag_entries = []

    with RAW_CSV.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("customer_question") or "").strip()
            a = (row.get("manager_reply") or "").strip()
            cat = (row.get("category") or "General Support").strip()
            if not q or not a:
                continue

            # 1. Format for LoRA Behavioral Cloning (teaches tone, greeting, jargon)
            lora_records.append({
                "instruction": "You are the Senior After-Sales Service Manager. Reply to the customer inquiry in your authentic, professional, and domain-accurate tone.",
                "input": q,
                "output": a
            })

            # 2. Format for RAG Knowledge Base (stores factual resolution)
            rag_entries.append(
                f"=== Historical Verified Resolution ({cat}) ===\n"
                f"Customer Issue: {q}\n"
                f"Manager Resolution: {a}\n"
            )

    with LORA_JSONL.open("w", encoding="utf-8") as out_jsonl:
        for rec in lora_records:
            out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    RAG_QA_TXT.write_text("\n".join(rag_entries), encoding="utf-8")
    print(f"Exported {len(lora_records)} LoRA training pairs -> {LORA_JSONL}")
    print(f"Exported {len(rag_entries)} RAG knowledge entries -> {RAG_QA_TXT}")

if __name__ == "__main__":
    main()
