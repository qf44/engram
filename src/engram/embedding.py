"""Local sentence embeddings via ONNX Runtime — no PyTorch, no cloud.

Primary path loads a sentence-transformers model exported to ONNX, tokenizes
ourselves and mean-pools in numpy (384-d for MiniLM-L6).

If the model cannot be fetched (offline, blocked network, first run before
`engram warm`), engram degrades to a dependency-free hashing encoder instead of
dying. Quality drops from semantic to lexical, but the tool keeps working — and
switches back automatically once the real model is available.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from engram.config import Config

DIM = 384


class EmbedError(RuntimeError):
    pass


class DegradedEmbedder(RuntimeWarning):
    pass


# --------------------------------------------------------------------------
# Fallback: deterministic hashing encoder (no model, no network, no deps)
# --------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


class HashingEncoder:
    """Bag-of-ngrams projected into a fixed dim via signed hashing.

    Lexical rather than semantic, but deterministic, instant and dependency
    free — enough to keep retrieval alive when the neural model is absent.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        t = text.lower()
        toks = _WORD.findall(t)
        cjk = _CJK.findall(t)
        toks.extend(cjk)                       # unigrams
        toks.extend(a + b for a, b in zip(cjk, cjk[1:]))  # bigrams
        return toks

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            for tok in self._tokens(text):
                h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, v in counts.items():
                if v == 0.0:      # positive and negative hashed to same slot
                    continue      # -> cancel out, leave the slot at zero
                # sublinear tf keeps long documents from dominating
                out[i, idx] = np.sign(v) * (1.0 + np.log(abs(v)))
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-12, None)


# --------------------------------------------------------------------------
# Primary: ONNX sentence-transformers
# --------------------------------------------------------------------------
class Embedder:
    """Turns text into normalized vectors using a local ONNX model."""

    def __init__(self, cfg: Config | None = None, quiet: bool = False):
        self.cfg = cfg or Config()
        self.quiet = quiet
        self._sess = None
        self._tok = None
        self.dim = DIM
        self.fallback: HashingEncoder | None = None
        self.degraded_reason: str | None = None

    # -- model acquisition -------------------------------------------------
    def _model_path(self) -> Path:
        from huggingface_hub import hf_hub_download
        kwargs = dict(repo_id=self.cfg.model,
                      local_dir=self.cfg.model_dir() / _safe(self.cfg.model))
        try:
            return Path(hf_hub_download(filename=self.cfg.onnx_file, **kwargs))
        except Exception:
            return Path(hf_hub_download(filename="onnx/model.onnx", **kwargs))

    def _tokenizer_path(self) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(
            repo_id=self.cfg.model, filename="tokenizer.json",
            local_dir=self.cfg.model_dir() / _safe(self.cfg.model),
        ))

    # -- lazy load ---------------------------------------------------------
    def load(self) -> None:
        """Load the ONNX model, or activate the hashing fallback."""
        if self._sess is not None or self.fallback is not None:
            return
        try:
            import onnxruntime as ort
            mpath = self._model_path()
            tpath = self._tokenizer_path()
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(tpath))
            self._tok.enable_truncation(max_length=256)
            self._tok.enable_padding(length=256)

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(
                str(mpath), sess_options=opts, providers=["CPUExecutionProvider"])
            shape = self._sess.get_outputs()[0].shape
            if len(shape) == 3 and isinstance(shape[-1], int):
                self.dim = int(shape[-1])
        except Exception as e:  # noqa: BLE001 - degrade, never crash
            self.fallback = HashingEncoder(DIM)
            self.dim = DIM
            self.degraded_reason = f"{type(e).__name__}: {e}"
            if not self.quiet:
                import warnings
                warnings.warn(
                    f"engram: embedding model unavailable ({self.degraded_reason}). "
                    "Falling back to lexical hashing; run `engram warm` once online "
                    "for full semantic search.", DegradedEmbedder, stacklevel=2)

    @property
    def is_degraded(self) -> bool:
        return self.fallback is not None

    # -- encoding ----------------------------------------------------------
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Encode *texts* -> (n, dim) float32, L2-normalized."""
        self.load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.fallback is not None:
            return self.fallback.encode(texts)
        bs = batch_size or self.cfg.batch_size
        return np.vstack([self._encode_batch(texts[i:i + bs])
                          for i in range(0, len(texts), bs)])

    def _encode_batch(self, batch: list[str]) -> np.ndarray:
        enc = self._tok.encode_batch([str(t) for t in batch])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        tids = np.array([e.type_ids for e in enc], dtype=np.int64)

        names = {i.name for i in self._sess.get_inputs()}
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = tids
        out = self._sess.run(None, feed)
        hidden = out[0]
        return _mean_pool(hidden, mask)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean-pool token embeddings over the attention mask, then L2-normalize."""
    h = np.asarray(hidden, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)[:, :, None]
    summed = (h * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), a_min=1e-9, a_max=None)
    out = summed / counts
    n = np.clip(np.linalg.norm(out, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return (out / n).astype(np.float32)


def _safe(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12] if "/" in s else s
