"""Report local-runtime sovereignty settings without pretending to be a packet sniffer."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SovereigntyStatus:
    offline_env: bool
    localhost_model: bool
    proxy_disabled: bool
    status: str


def check_status(*, model_host: str = "127.0.0.1") -> SovereigntyStatus:
    offline = os.getenv("HF_HUB_OFFLINE") == "1" and os.getenv("TRANSFORMERS_OFFLINE") == "1"
    localhost = model_host in {"127.0.0.1", "localhost", "::1"}
    proxy_disabled = not any(os.getenv(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))
    ok = offline and localhost
    return SovereigntyStatus(offline, localhost, proxy_disabled, "PASS" if ok else "CHECK")
