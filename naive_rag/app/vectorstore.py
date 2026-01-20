from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import hnswlib
import numpy as np


@dataclass
class Retrieved:
    chunk_id: str
    source: str
    text: str
    score: float  # higher is better (rough cosine similarity)


class HNSWSQLiteStore:
    """HNSW index persisted to disk + SQLite for chunk text/metadata.

    - HNSW labels are integers [0..N-1]
    - SQLite table stores chunk_id (string), label (int), source (string), text (string)
    """

    def __init__(
        self,
        data_dir: str = "data",
        index_file: str = "index.bin",
        db_file: str = "meta.sqlite",
        config_file: str = "index_config.json",
        dim: Optional[int] = None,
        space: str = "cosine",
        ef_construction: int = 200,
        M: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.data_dir / index_file
        self.db_path = self.data_dir / db_file
        self.cfg_path = self.data_dir / config_file

        self.space = space
        self.ef_construction = ef_construction
        self.M = M

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                label INTEGER UNIQUE,
                source TEXT,
                text TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_label ON chunks(label)")
        self._conn.commit()

        self.index = hnswlib.Index(space=self.space, dim=dim or 384)

        if self.index_path.exists() and self.cfg_path.exists():
            self._load_index()
        else:
            if dim is None:
                raise ValueError("dim must be provided when creating a new index (no existing index found).")
            self._init_new(dim)

    def _read_cfg(self) -> dict:
        if self.cfg_path.exists():
            return json.loads(self.cfg_path.read_text())
        return {}

    def _write_cfg(self, cfg: dict) -> None:
        self.cfg_path.write_text(json.dumps(cfg, indent=2))

    def _init_new(self, dim: int) -> None:
        cfg = {
            "dim": dim,
            "space": self.space,
            "count": 0,
            "max_elements": 10000,
            "ef_construction": self.ef_construction,
            "M": self.M,
            "ef": 64,
        }
        self.index.init_index(
            max_elements=cfg["max_elements"],
            ef_construction=cfg["ef_construction"],
            M=cfg["M"],
        )
        self.index.set_ef(cfg["ef"])
        self._write_cfg(cfg)

    def _load_index(self) -> None:
        cfg = self._read_cfg()
        dim = int(cfg["dim"])
        self.index = hnswlib.Index(space=cfg.get("space", self.space), dim=dim)
        self.index.load_index(str(self.index_path))
        self.index.set_ef(int(cfg.get("ef", 64)))

    def _ensure_capacity(self, needed_total: int) -> None:
        cfg = self._read_cfg()
        max_elements = int(cfg.get("max_elements", 10000))
        if needed_total <= max_elements:
            return
        # grow ~2x until enough
        new_max = max_elements
        while new_max < needed_total:
            new_max *= 2
        self.index.resize_index(new_max)
        cfg["max_elements"] = new_max
        self._write_cfg(cfg)

    def close(self) -> None:
        self._conn.close()

    def _next_label(self) -> int:
        cfg = self._read_cfg()
        return int(cfg.get("count", 0))

    def has_chunk_id(self, chunk_id: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM chunks WHERE chunk_id=? LIMIT 1", (chunk_id,))
        return cur.fetchone() is not None

    def add_embeddings(
        self,
        items: List[Tuple[str, str, str, np.ndarray]],
    ) -> int:
        """Add items: (chunk_id, source, text, embedding[dim]). Returns number added.

        Assumes embeddings are float32 and *already normalized* if using cosine.
        """
        if not items:
            return 0

        # Filter out already-ingested chunk_ids
        new_items = []
        for chunk_id, source, text, emb in items:
            if not self.has_chunk_id(chunk_id):
                new_items.append((chunk_id, source, text, emb))

        if not new_items:
            return 0

        start_label = self._next_label()
        needed_total = start_label + len(new_items)
        self._ensure_capacity(needed_total)

        labels = np.arange(start_label, start_label + len(new_items), dtype=np.int64)
        vecs = np.vstack([it[3] for it in new_items]).astype(np.float32)

        self.index.add_items(vecs, labels)

        # Insert into SQLite
        self._conn.executemany(
            "INSERT INTO chunks(chunk_id, label, source, text) VALUES (?,?,?,?)",
            [(cid, int(lbl), src, txt) for (cid, src, txt, _), lbl in zip(new_items, labels)],
        )
        self._conn.commit()

        # Update count + persist index
        cfg = self._read_cfg()
        cfg["count"] = needed_total
        self._write_cfg(cfg)
        self.index.save_index(str(self.index_path))

        return len(new_items)

    def query(self, query_vec: np.ndarray, k: int = 4) -> List[Retrieved]:
        """Return top-k results."""
        cfg = self._read_cfg()
        count = int(cfg.get("count", 0))
        if count == 0:
            return []

        k = min(k, count)
        labels, distances = self.index.knn_query(query_vec.astype(np.float32), k=k)
        labels = labels[0].tolist()
        distances = distances[0].tolist()

        out: List[Retrieved] = []
        for lbl, dist in zip(labels, distances):
            cur = self._conn.execute(
                "SELECT chunk_id, source, text FROM chunks WHERE label=? LIMIT 1",
                (int(lbl),),
            )
            row = cur.fetchone()
            if not row:
                continue
            chunk_id, source, text = row
            # cosine distance -> similarity (rough)
            score = float(1.0 - dist)
            out.append(Retrieved(chunk_id=chunk_id, source=source, text=text, score=score))
        return out
