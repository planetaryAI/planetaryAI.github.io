from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from pypdf import PdfReader


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def load_documents(docs_dir: Path) -> Iterable[Tuple[str, str]]:
    """Yields (source_path, full_text)"""
    for p in sorted(docs_dir.rglob("*")):
        if p.is_dir():
            continue
        suf = p.suffix.lower()
        if suf in {".txt", ".md"}:
            yield str(p), read_text_file(p)
        elif suf == ".pdf":
            yield str(p), read_pdf(p)


def naive_chunk(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Naive character chunking."""
    text = " ".join(text.split())
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks
