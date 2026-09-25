"""Projection of the typed event stream for the status line and /commands.

The TUI never peeks into loop internals: the single event consumer (ruling
M6.2) folds every event into this pure-data state, per-turn counters reset
on ``turn_start`` exactly like the loop's own budgets so the status line
shows the current turn, not the session average.
"""

from __future__ import annotations

from collections.abc import Mapping

from workbench.core.events import Event


def _text(data: Mapping[str, object], key: str, default: str = "") -> str:
    value = data.get(key, default)
    return value if isinstance(value, str) else default


def _number(data: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _count(data: Mapping[str, object], key: str, default: int = 0) -> int:
    value = data.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


class TuiState:
    """Accumulated display state; mutated only by :meth:`apply` in stream order."""

    def __init__(self) -> None:
        # session-level
        self.session_id: str = ""
        self.tier: str = ""
        self.model: str = ""
        self.intent: str = ""
        self.difficulty: float = 0.0
        self.confidence: float = 0.0
        self.backend: str = ""
        self.reason: str = ""
        self.suggest_plan: bool = False
        # per-turn counters (reset on turn_start)
        self.rounds: int = 0
        self.tokens: int = 0
        self.escalations: int = 0
        # last stream facts
        self.prompt_tokens: int = 0
        self.ttft_s: float | None = None
        self.last_turn: Mapping[str, object] = {}
        self.last_error: str = ""

    def apply(self, event: Event) -> None:
        """Fold one bus event into the display state."""
        kind = event.kind
        data = event.data
        if kind == "agent_start":
            self.session_id = _text(data, "session_id")
        elif kind == "turn_start":
            self.rounds = 0
            self.tokens = 0
            self.escalations = 0
            self.last_error = ""
        elif kind == "route_decided":
            self.tier = _text(data, "tier")
            self.model = _text(data, "model")
            self.intent = _text(data, "intent")
            self.difficulty = _number(data, "difficulty")
            self.confidence = _number(data, "confidence")
            self.backend = _text(data, "backend")
            self.reason = _text(data, "reason")
            self.suggest_plan = bool(data.get("suggest_plan", False))
        elif kind == "route_escalated":
            self.escalations += 1
        elif kind == "message_start":
            self.rounds += 1
        elif kind == "message_end":
            self._absorb_usage(data)
        elif kind == "turn_end":
            self.last_turn = dict(data)
        elif kind == "error":
            self.last_error = _text(data, "message")

    def _absorb_usage(self, data: Mapping[str, object]) -> None:
        usage = data.get("usage")
        if isinstance(usage, Mapping):
            prompt = _count(usage, "prompt_tokens")
            completion = _count(usage, "completion_tokens")
            self.tokens += prompt + completion
            self.prompt_tokens = prompt
        ttft = data.get("ttft_s")
        if isinstance(ttft, (int, float)):
            self.ttft_s = float(ttft)
