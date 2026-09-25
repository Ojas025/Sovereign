"""Typed in-process event bus with a pi-style vocabulary.

The bus is deliberately dumb: handlers run inline, handler failures are
logged and swallowed, and emit() never raises — the agent loop must not be
breakable by an observer (metrics, JSONL logger, TUI renderer).
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import time
from typing import Literal

logger = logging.getLogger("workbench.events")

EventKind = Literal[
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
    "route_decided",
    "route_escalated",
    "plan_proposed",
    "plan_approved",
    "plan_rejected",
    "reflection",
    "budget_exceeded",
    "pii_redacted",
    "server_started",
    "server_stopped",
    "error",
]

Handler = Callable[["Event"], None]


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    data: Mapping[str, object] = field(default_factory=dict)
    ts: float = field(default_factory=time)


class EventBus:
    """Subscription-ordered bus; unsubscribe is idempotent."""

    def __init__(self) -> None:
        # Single ordered list of (kind, handler) so emit() preserves
        # subscription order across all kinds.
        self._subscriptions: list[tuple[EventKind | None, Handler]] = []

    def subscribe(
        self, kind: EventKind | None, handler: Handler
    ) -> Callable[[], None]:
        """Register a handler for one kind (None = wildcard); return unsubscribe."""
        self._subscriptions.append((kind, handler))

        def unsubscribe() -> None:
            try:
                self._subscriptions.remove((kind, handler))
            except ValueError:
                pass

        return unsubscribe

    def emit(self, event: Event) -> None:
        """Dispatch to matching handlers; never raise, log handler failures."""
        # Snapshot under no lock (single-threaded asyncio loop is the caller);
        # run outside the list so handlers may subscribe/unsubscribe re-entrantly.
        matching = [
            handler
            for subscribed_kind, handler in self._subscriptions
            if subscribed_kind is None or subscribed_kind == event.kind
        ]
        for handler in matching:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - observers must not break the loop
                logger.exception("event handler failed for %s", event.kind)
