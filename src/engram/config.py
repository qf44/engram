"""Configuration and on-disk layout for engram."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Quantized ONNX variant — ~23 MB instead of 90 MB, quality loss is negligible.
DEFAULT_ONNX_FILE = "onnx/model_O4_quantized.onnx"
FALLBACK_ONNX_FILE = "onnx/model.onnx"


def home() -> Path:
    """Root directory for all engram state (~/.engram)."""
    override = os.environ.get("ENGRAM_HOME")
    if override:
        p = Path(override).expanduser()
    else:
        p = Path.home() / ".engram"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Config:
    """User-tunable settings. Persisted as JSON under the engram home."""

    model: str = DEFAULT_MODEL
    onnx_file: str = DEFAULT_ONNX_FILE
    # Chunking
    chunk_tokens: int = 320
    chunk_overlap: int = 48
    # Retrieval
    top_k: int = 6
    # Optional local LLM generation (Ollama / any OpenAI-compatible endpoint)
    llm_endpoint: str | None = None     # e.g. "http://localhost:11434/v1"
    llm_model: str | None = None        # e.g. "qwen2.5:7b"
    llm_api_key: str | None = None
    # Embedding batch size
    batch_size: int = 32

    def path(self) -> Path:
        return home() / "config.json"

    def brain_dir(self, name: str = "default") -> Path:
        d = home() / "brains" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def model_dir(self) -> Path:
        d = home() / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config() -> Config:
    p = home() / "config.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {k: v for k, v in data.items() if k in Config.__dataclass_fields__}
            return Config(**known)
        except Exception:
            pass
    return Config()


def save_config(cfg: Config) -> Path:
    p = home() / "config.json"
    p.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
