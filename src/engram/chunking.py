"""Structure-aware text chunking.

Splits on the largest natural boundary that fits (heading -> paragraph ->
sentence -> hard slice) so chunks stay semantically coherent. Overlap keeps
answers that straddle a boundary from falling through the cracks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# CJK has ~1.5 tokens/char; latin ~4 chars/token. Blended estimate.
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(cjk * 0.7 + other / 4.0) + 1


@dataclass
class Chunk:
    text: str
    index: int
    meta: dict

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


_PARA_SPLIT = re.compile(r"\n\s*\n")
_HEADING = re.compile(r"^(#{1,6}\s+.*|.*\n[=-]{3,}\s*)$", re.M)
_SENT_SPLIT = re.compile(r"(?<=[。．.!?？！])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_tokens: int = 320, overlap: int = 48, meta: dict | None = None) -> list[Chunk]:
    """Split *text* into chunks of at most ~max_tokens with *overlap* tail context."""
    meta = dict(meta or {})
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [Chunk(text=text, index=0, meta=meta)]

    # 1. paragraphs (keeps headings attached to their body)
    blocks: list[str] = [b.strip() for b in _PARA_SPLIT.split(text) if b.strip()]
    # 2. oversized paragraphs -> sentences
    refined: list[str] = []
    for b in blocks:
        if estimate_tokens(b) <= max_tokens:
            refined.append(b)
        else:
            refined.extend(_split_sentences(b))

    chunks: list[str] = []
    buf = ""
    for piece in refined:
        if not buf:
            buf = piece
            continue
        if estimate_tokens(buf + "\n" + piece) <= max_tokens:
            buf = buf + "\n" + piece
        else:
            chunks.append(buf)
            # carry tail sentences as overlap
            if overlap > 0:
                tail: list[str] = []
                acc = 0
                for s in reversed(_split_sentences(buf)):
                    t = estimate_tokens(s)
                    if acc + t > overlap:
                        break
                    tail.insert(0, s)
                    acc += t
                buf = "\n".join(tail) + "\n" + piece if tail else piece
            else:
                buf = piece
    if buf.strip():
        chunks.append(buf.strip())

    # 3. last resort: hard slice on a character budget (handles text with no
    #    sentence punctuation at all, e.g. logs, minified code, word salad)
    final: list[str] = []
    for c in chunks:
        if estimate_tokens(c) <= max_tokens * 1.5:
            final.append(c)
            continue
        sents = _split_sentences(c)
        if len(sents) > 1:
            cur = ""
            for s_ in sents:
                if estimate_tokens(cur + s_) > max_tokens and cur:
                    final.append(cur.strip())
                    cur = s_
                else:
                    cur = cur + s_
            if cur.strip():
                final.append(cur.strip())
            continue
        # no sentence boundaries -> slice on a char budget, break on whitespace
        cjk_ratio = len(_CJK.findall(c)) / max(1, len(c))
        budget = int(max_tokens * (1.4 if cjk_ratio > 0.5 else 4.0))
        start = 0
        while start < len(c):
            end = min(start + budget, len(c))
            if end < len(c):
                nl = c.rfind("\n", start, end)
                sp = c.rfind(" ", start, end)
                cut = max(nl, sp)
                if cut > start + budget // 2:
                    end = cut + 1
            piece = c[start:end].strip()
            if piece:
                final.append(piece)
            start = end

    out = []
    for i, c in enumerate(final):
        m = dict(meta)
        m["chunk_index"] = i
        out.append(Chunk(text=c, index=i, meta=m))
    return out
