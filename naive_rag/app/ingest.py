from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from loaders import load_documents, naive_chunk
from rag import RAG


def main():
    docs_dir = Path("docs")
    if not docs_dir.exists():
        raise SystemExit("docs/ not found. Create it and add .txt/.md/.pdf files.")

    rag = RAG()

    # Build chunk list
    all_chunks = []
    for source, text in load_documents(docs_dir):
        chunks = naive_chunk(text)
        for i, c in enumerate(chunks):
            all_chunks.append((source, i, c))

    if not all_chunks:
        raise SystemExit("No supported documents found in docs/.")

    # Batch upserts
    added_total = 0
    batch = []
    for item in tqdm(all_chunks, desc="Ingesting"):
        batch.append(item)
        if len(batch) >= 256:
            added_total += rag.upsert_chunks(batch)
            batch = []
    if batch:
        added_total += rag.upsert_chunks(batch)

    print(f"Done. Added {added_total} new chunks.")
    print("Persisted: data/index.bin, data/meta.sqlite, data/index_config.json")


if __name__ == "__main__":
    main()
