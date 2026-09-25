"""Process-wide offline guard: PLAN constraint 1 — the runtime never reaches the Hub.

huggingface_hub and transformers freeze these flags when they import, so the
guard has to run before anything loads them — the CLI applies it right after
config load, and `models download` is the one path that opts out.
"""

from __future__ import annotations

import os

OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def ensure_offline() -> None:
    """Set both hub flags so any later hub/transformers call fails fast instead of fetching."""
    for name in OFFLINE_ENV:
        os.environ[name] = "1"
