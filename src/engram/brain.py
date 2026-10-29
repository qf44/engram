"""Brain — the public face of engram.

    from engram import Brain
    b = Brain()
    b.ingest("~/notes")
    b.ask("what did I decide about the pricing model?")
"""
from __future__ import annotations

import time
from pathlib import Path

from engram.chunking import chunk_text, estimate_tokens
from engram.config import Config, load_config
from engram.embedding import Embedder
from engram.parsers import Document, parse_file, walk
from engram.store import Hit, VectorStore


class Brain:
    """A local, persistent, queryable index over your own documents."""

    def __init__(self, name: str = "default", cfg: Config | None = None, quiet: bool = False):
        self.name = name
        self.cfg = cfg or load_config()
        self.quiet = quiet
        self.embedder = Embedder(self.cfg, quiet=quiet)
        self.store = VectorStore(dim=384)
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.cfg.brain_dir(self.name) / "index"

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            self.store = VectorStore.load(self.index_path)
            self._loaded = True
            return True
        except FileNotFoundError:
            return False

    def save(self) -> Path:
        self.store.save(self.index_path)
        return self.index_path

    def reset(self) -> None:
        self.store.clear()
        p = self.index_path
        for suf in (".npz", ".jsonl"):
            f = p.with_suffix(suf)
            if f.exists():
                f.unlink()

    # -- ingestion ---------------------------------------------------------
    def ingest(self, path: str | Path, recursive: bool = True) -> dict:
        """Parse, chunk, embed and index everything under *path*."""
        self.load()
        files = walk(path) if recursive else [Path(path)]
        t0 = time.time()

        chunks: list[str] = []
        metas: list[dict] = []
        ok_files, skipped, errors = 0, 0, []

        for f in files:
            doc = parse_file(f)
            if not doc.ok:
                skipped += 1
                if doc.meta.get("error") and "unsupported" not in doc.meta["error"]:
                    errors.append((str(f), doc.meta["error"]))
                continue
            ok_files += 1
            base = {"source": str(f), "title": doc.title, **{k: v for k, v in doc.meta.items()
                                                             if k in ("reader", "mtime", "size")}}
            for c in chunk_text(doc.text, self.cfg.chunk_tokens, self.cfg.chunk_overlap, base):
                chunks.append(c.text)
                metas.append(c.meta)

        if not chunks:
            return {"files": ok_files, "skipped": skipped, "chunks": 0,
                    "errors": errors, "seconds": round(time.time() - t0, 2)}

        vecs = self.embedder.encode(chunks)
        if self.store.dim != vecs.shape[1] or len(self.store) == 0:
            self.store = VectorStore(dim=vecs.shape[1])
        self.store.add(vecs, chunks, metas)
        self.save()

        return {"files": ok_files, "skipped": skipped, "chunks": len(chunks),
                "errors": errors[:10], "seconds": round(time.time() - t0, 2)}

    # -- retrieval ---------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        self.load()
        if len(self.store) == 0:
            return []
        k = top_k or self.cfg.top_k
        vec = self.embedder.encode_query(query)
        return self.store.search(vec, top_k=k)

    def ask(self, question: str, top_k: int | None = None,
            with_answer: bool | None = None) -> dict:
        """Retrieve context for *question*; optionally generate an answer locally."""
        hits = self.search(question, top_k)
        result: dict = {"question": question, "hits": [h.as_dict() for h in hits]}
        want = (self.cfg.llm_endpoint is not None) if with_answer is None else with_answer
        if want and hits:
            try:
                result["answer"] = self._generate(question, hits)
                result["answer_model"] = self.cfg.llm_model
            except Exception as e:  # noqa: BLE001
                result["answer"] = None
                result["answer_error"] = f"{type(e).__name__}: {e}"
        return result

    def _generate(self, question: str, hits: list[Hit]) -> str:
        from engram.llm import generate
        return generate(question, hits, self.cfg)

    # -- introspection -----------------------------------------------------
    def stats(self) -> dict:
        self.load()
        s = self.store.stats()
        s["brain"] = self.name
        s["sources_count"] = len(s.pop("sources", {}))
        return s

    def sources(self) -> list[tuple[str, int]]:
        self.load()
        counts: dict[str, int] = {}
        for m in self.store.metas:
            counts[m.get("source", "?")] = counts.get(m.get("source", "?"), 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    def __len__(self) -> int:
        self.load()
        return len(self.store)
