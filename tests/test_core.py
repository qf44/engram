"""Core tests — run with: pytest tests/"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from engram.chunking import chunk_text, estimate_tokens
from engram.embedding import Embedder, HashingEncoder
from engram.parsers import parse_file, walk
from engram.store import VectorStore


# --------------------------------------------------------------- chunking
def test_chunk_short_text_stays_whole():
    assert len(chunk_text("hello world", max_tokens=100)) == 1


def test_chunk_respects_max_tokens():
    text = "word " * 5000
    chunks = chunk_text(text, max_tokens=100, overlap=10)
    assert len(chunks) > 5
    assert all(c.tokens <= 100 * 1.6 for c in chunks)


def test_chunk_overlaps_tail():
    text = ("Alpha sentence one. Alpha sentence two. " * 5 +
            "\n\n" + "Beta sentence one. Beta sentence two. " * 5)
    chunks = chunk_text(text, max_tokens=60, overlap=20)
    assert len(chunks) >= 2


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_cjk_token_estimate():
    assert estimate_tokens("人工智能是指由人制造出来的系统") > 0
    assert estimate_tokens("") == 0


# --------------------------------------------------------------- fallback encoder
def test_hash_encoder_deterministic():
    e = HashingEncoder(dim=64)
    a, b = e.encode(["the quick brown fox"]), e.encode(["the quick brown fox"])
    assert np.allclose(a, b)


def test_hash_encoder_normalized():
    v = HashingEncoder(dim=64).encode(["some text here"])
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-5)


def test_hash_encoder_similarity():
    e = HashingEncoder(dim=128)
    v = e.encode(["python async await coroutine", "python async await event loop",
                  "banana bread recipe with walnuts"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


# --------------------------------------------------------------- store
def test_store_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(5, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    s = VectorStore(dim=8)
    s.add(v, [f"t{i}" for i in range(5)], [{"source": f"f{i}.md"} for i in range(5)])
    assert len(s) == 5
    hits = s.search(v[2], top_k=2)
    assert hits[0].text == "t2" and hits[0].score > 0.99
    p = tmp_path / "idx"
    s.save(p)
    s2 = VectorStore.load(p)
    assert len(s2) == 5 and s2.search(v[2], 1)[0].text == "t2"


def test_store_filter():
    v = np.eye(3, dtype=np.float32)
    s = VectorStore(dim=3)
    s.add(v, ["a", "b", "c"], [{"source": "x"}, {"source": "y"}, {"source": "x"}])
    assert len(s.search(v[0], 5, where={"source": "x"})) == 2
    assert len(s.search(v[0], 5, where={"source": "zzz"})) == 0


def test_store_empty_search():
    assert VectorStore(dim=4).search(np.ones(4, dtype=np.float32)) == []


# --------------------------------------------------------------- parsers
def test_parse_text_and_markdown(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# My Note\n\nSome body text.", encoding="utf-8")
    d = parse_file(p)
    assert d.ok and "My Note" in d.text and d.title == "My Note"


def test_parse_missing_file():
    assert not parse_file("/definitely/not/here.txt").ok


def test_parse_unsupported_ext(tmp_path):
    p = tmp_path / "a.zzz"
    p.write_text("x", encoding="utf-8")
    assert not parse_file(p).ok


def test_parse_cjk_encoding(tmp_path):
    p = tmp_path / "中文.txt"
    p.write_text("这是一段中文笔记。", encoding="gb18030")
    assert "中文笔记" in parse_file(p).text


def test_walk_skips_noise(tmp_path):
    (tmp_path / "keep.md").write_text("a", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "cfg").write_text("x", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "j.js").write_text("x", encoding="utf-8")
    names = {f.name for f in walk(tmp_path)}
    assert names == {"keep.md"}


# --------------------------------------------------------------- end to end
def test_brain_ingest_and_search(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("ENGRAM_HOME", str(home))

    from engram import Brain
    from engram.config import Config

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "python.md").write_text(
        "# Python\n\nAsyncio uses an event loop to schedule coroutines. "
        "Await suspends a coroutine until the future resolves.", encoding="utf-8")
    (docs / "cooking.txt").write_text(
        "Banana bread needs ripe bananas, flour, sugar and walnuts. "
        "Bake at 175 celsius for one hour.", encoding="utf-8")

    cfg = Config()
    b = Brain(name="test", cfg=cfg, quiet=True)
    res = b.ingest(docs)
    assert res["files"] == 2 and res["chunks"] >= 2

    hits = b.search("asyncio event loop coroutine")
    assert len(hits) >= 1
    assert "python.md" in hits[0].source

    # index survived to disk and reloads
    b2 = Brain(name="test", cfg=cfg, quiet=True)
    assert len(b2) == len(b)
    assert b2.search("banana bread oven temperature")[0].source.endswith("cooking.txt")


def test_brain_ask_returns_hits(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "h"))
    from engram import Brain
    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.md").write_text("The deploy window is Tuesday 02:00 UTC.", encoding="utf-8")
    b = Brain(name="t2", quiet=True)
    b.ingest(docs)
    out = b.ask("when is the deploy window?")
    assert out["hits"] and "Tuesday" in out["hits"][0]["text"]
    assert "answer" not in out  # no LLM configured


def test_embedder_degrades_gracefully(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "hh"))
    from engram.config import Config
    e = Embedder(Config(), quiet=True)
    e.load()
    v = e.encode(["hello there", "general kenobi"])
    assert v.shape == (2, 384)
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-4)
