"""Optional answer generation against a *local* LLM.

engram's retrieval works with zero LLM. This module is opt-in: point
`llm_endpoint` at Ollama / llama.cpp / LM Studio (any OpenAI-compatible
server) and `ask()` will also return a written answer. Nothing leaves your
machine unless you point it at a remote URL yourself.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from engram.config import Config
from engram.store import Hit

PROMPT = """Answer the question using ONLY the context below. If the context does not
contain the answer, say "I don't have that in my notes." Be concise and cite
the source filename in square brackets.

Context:
{context}

Question: {question}
Answer:"""


def generate(question: str, hits: list[Hit], cfg: Config, timeout: int = 120) -> str:
    if not cfg.llm_endpoint:
        raise RuntimeError("no llm_endpoint configured")
    blocks = []
    for i, h in enumerate(hits, 1):
        src = h.meta.get("source", "?")
        blocks.append(f"[{i}] ({Path(src).name})\n{h.text}")
    prompt = PROMPT.format(context="\n\n".join(blocks), question=question)

    url = cfg.llm_endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.llm_model or "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {cfg.llm_api_key}"} if cfg.llm_api_key else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"].strip()
