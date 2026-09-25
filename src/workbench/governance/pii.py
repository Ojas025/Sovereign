"""Conservative, dependency-free PII redaction for local preprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    counts: dict[str, int]


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("aadhaar_like", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    ("phone", re.compile(r"(?<!\d)(?:\+?\d[\d ()_-]{8,}\d)(?!\d)")),
    ("pan_like", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
)


def redact_pii(text: str) -> RedactionResult:
    counts: dict[str, int] = {}
    redacted = text
    for kind, pattern in _PATTERNS:
        redacted, count = pattern.subn(f"[{kind.upper()}]", redacted)
        if count:
            counts[kind] = count
    return RedactionResult(redacted, counts)
