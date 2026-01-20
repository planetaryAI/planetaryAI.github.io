from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from vectorstore import HNSWSQLiteStore, Retrieved


class RAG:
    def __init__(
        self,
        data_dir: str = "data",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3:4b",
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # create/load store
        self.store = HNSWSQLiteStore(data_dir=data_dir, dim=self.dim)

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model

    @staticmethod
    def stable_chunk_id(source: str, chunk_idx: int, text: str) -> str:
        h = hashlib.sha1((source + "|" + str(chunk_idx) + "|" + text[:200]).encode("utf-8")).hexdigest()
        return h

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # normalize embeddings for cosine space
        embs = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)

    def upsert_chunks(self, chunks: List[Tuple[str, int, str]]):
        """chunks: List[(source_path, chunk_idx, chunk_text)]"""
        if not chunks:
            return 0
        texts = [c[2] for c in chunks]
        embs = self.embed_texts(texts)

        items = []
        for (source, idx, text), emb in zip(chunks, embs):
            cid = self.stable_chunk_id(source, idx, text)
            items.append((cid, source, text, emb))
        return self.store.add_embeddings(items)

    def retrieve(self, query: str, k: int = 4) -> List[Retrieved]:
        q = self.embed_texts([query])
        return self.store.query(q, k=k)

    def answer(self, query: str, retrieved: List[Retrieved]) -> str:
        if retrieved:
            context = "\n\n---\n\n".join(
                f"[source: {r.source} | score: {r.score:.3f}]\n{r.text}" for r in retrieved
            )
        else:
            context = "(no context found)"

        prompt = f"""You are a helpful assistant. Use the CONTEXT to answer the QUESTION.
If the context is insufficient, say so and suggest what to add.

CONTEXT:
{context}

QUESTION:
{query}
"""

        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={"model": self.ollama_model, "prompt": prompt, "stream": False},
            timeout=300,
        )

        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 8192,
                    "temperature": 0.1,
                    "max_new_tokens": 512
                }
            },
            timeout=300,
        )

        resp.raise_for_status()
        return resp.json().get("response", "").strip()
