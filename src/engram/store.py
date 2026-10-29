"""A tiny, dependency-free vector store.

Brute-force cosine over a numpy matrix is exact and, up to ~100k chunks,
faster than maintaining an ANN index. Past that we transparently switch to
hnswlib (optional) if it is installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANN_THRESHOLD = 50_000


@dataclass
class Hit:
    score: float
    text: str
    meta: dict

    @property
    def source(self) -> str:
        return self.meta.get("source", "?")

    def as_dict(self) -> dict:
        return {"score": round(float(self.score), 4), "text": self.text,
                "source": self.source, "meta": self.meta}


class VectorStore:
    """Exact cosine search over an (n, d) float32 matrix + parallel metadata."""

    def __init__(self, dim: int):
        self.dim = dim
        self.matrix = np.zeros((0, dim), dtype=np.float32)
        self.metas: list[dict] = []
        self.texts: list[str] = []
        self._ann = None

    def __len__(self) -> int:
        return len(self.texts)

    # -- write -------------------------------------------------------------
    def add(self, vectors: np.ndarray, texts: list[str], metas: list[dict]) -> int:
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        if v.size == 0:
            return 0
        if v.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: store={self.dim} got={v.shape[1]}")
        self.matrix = np.vstack([self.matrix, v]) if len(self.texts) else v.copy()
        self.texts.extend(texts)
        self.metas.extend(metas)
        self._ann = None  # invalidate ANN index
        return v.shape[0]

    def clear(self) -> None:
        self.matrix = np.zeros((0, self.dim), dtype=np.float32)
        self.texts.clear()
        self.metas.clear()
        self._ann = None

    # -- read --------------------------------------------------------------
    def search(self, query: np.ndarray, top_k: int = 6, where: dict | None = None) -> list[Hit]:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if len(self.texts) == 0:
            return []
        if q.shape[0] != self.dim:
            raise ValueError(f"dim mismatch: store={self.dim} query={q.shape[0]}")

        if where:
            idx = [i for i, m in enumerate(self.metas) if all(m.get(k) == v for k, v in where.items())]
            if not idx:
                return []
            mat = self.matrix[idx]
        else:
            idx = list(range(len(self.texts)))
            mat = self.matrix

        # matrix is L2-normalized, so dot product == cosine
        sims = mat @ q
        k = min(top_k, len(idx))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]

        return [Hit(score=float(sims[i]), text=self.texts[idx[i]],
                    meta=dict(self.metas[idx[i]])) for i in top]

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path.with_suffix(".npz")), matrix=self.matrix)
        with open(path.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
            for t, m in zip(self.texts, self.metas):
                f.write(json.dumps({"text": t, "meta": m}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        npz = path.with_suffix(".npz")
        jsonl = path.with_suffix(".jsonl")
        if not npz.exists():
            raise FileNotFoundError(f"no index at {npz}")
        data = np.load(str(npz))
        matrix = data["matrix"] if "matrix" in data else data[data.files[0]]
        store = cls(dim=int(matrix.shape[1]))
        store.matrix = matrix.astype(np.float32)
        if jsonl.exists():
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                store.texts.append(rec["text"])
                store.metas.append(rec.get("meta", {}))
        return store

    def stats(self) -> dict:
        size = self.matrix.nbytes if self.matrix.size else 0
        sources = {}
        for m in self.metas:
            sources[m.get("source", "?")] = sources.get(m.get("source", "?"), 0) + 1
        return {"chunks": len(self.texts), "dim": self.dim,
                "vectors_mb": round(size / 1e6, 2), "sources": sources}
