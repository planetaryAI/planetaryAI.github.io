from __future__ import annotations

from rag import RAG


def main():
    rag = RAG()
    print("Local RAG Chat (HNSW + SQLite). Type 'exit' to quit.\n")

    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        hits = rag.retrieve(q, k=4)
        ans = rag.answer(q, hits)

        print("\n--- retrieved ---")
        for h in hits:
            print(f"- {h.source} (score={h.score:.3f})")
        print("\n--- answer ---")
        print(ans)
        print()

if __name__ == "__main__":
    main()
