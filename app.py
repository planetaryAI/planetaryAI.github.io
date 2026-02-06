import os, json
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

APP_TITLE = "Planetary AI – Lunar Mineralogy Demo"

INDEX_PATH = "data/index.faiss"
DOCS_PATH = "data/docs.jsonl"
MODEL_PATH = "models/llama-3.1-3b-instruct-q4_k_m.gguf"

index = faiss.read_index(INDEX_PATH)

docs = []
with open(DOCS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=int(os.environ.get("LLAMA_THREADS", "2")),
)

SYSTEM_PROMPT = (
    "You are a lunar mineralogy tutor focused on petrography and geochemistry. "
    "Answer only using the provided context. "
    "If the answer is not in the context, say you do not know. "
    "Be concise and factual."
)

def retrieve(query, k=4):
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    _, I = index.search(q, k)
    return [docs[i] for i in I[0] if i < len(docs)]

def generate(query, contexts):
    context_text = "\n\n".join(
        f"[{i+1}] {c.get('text','')}" for i, c in enumerate(contexts)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return out["choices"][0]["message"]["content"]

app = FastAPI(title=APP_TITLE)

class Ask(BaseModel):
    query: str

@app.get("/")
def root():
    return {"ok": True, "title": APP_TITLE}

@app.post("/ask")
def ask(req: Ask):
    ctx = retrieve(req.query)
    answer = generate(req.query, ctx)
    citations = [
        {"n": i+1, "source": c.get("source",""), "text": c.get("text","")[:200]}
        for i, c in enumerate(ctx)
    ]
    return {"answer": answer, "citations": citations}