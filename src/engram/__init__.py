"""engram — your second brain, fully local.

Zero cloud. Zero PyTorch. Zero telemetry.
"""
from __future__ import annotations

__version__ = "0.1.0"

from engram.brain import Brain
from engram.config import Config, load_config, save_config

__all__ = ["Brain", "Config", "load_config", "save_config", "__version__"]
