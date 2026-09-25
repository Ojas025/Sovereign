"""Bus → metrics: one subscriber mapping the event stream onto PLAN §8.3 (M7.2).

The bus already flows through every layer (PLAN §7: events → TUI render +
JSONL log + Prometheus metrics), so instrumentation subscribes instead of
being sprinkled through call sites. Derivations that aren't in any payload
live here too (M7.3): confidence band, the model→tier reverse map, fallback
detection, and the single-in-flight pairing of message_start/message_end.

Handlers run inline on the emitting thread (the TUI agent worker, or the
main loop for app-level events); those streams are serialized in practice
and prometheus_client instruments are thread-safe, so no extra locking.
"""

from __future__ import annotations

from collections.abc import Mapping

from workbench.agent.loop import resolve_model
from workbench.config import Config
from workbench.core.events import Event, EventBus
from workbench.observability.metrics import Metrics

# M7.3 confidence bands: bounded label values, no unbounded raw scores.
_BAND_LOW_MAX = 0.5  # below this the router is guessing
_BAND_MID_MAX = 0.8  # below this it is confident, not certain


def _band(confidence: float) -> str:
    """Bucket a confidence score into the bounded `band` label."""
    if confidence < _BAND_LOW_MAX:
        return "low"
    if confidence < _BAND_MID_MAX:
        return "mid"
    return "high"


def _num(data: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """Read a numeric payload field without letting a surprise crash the bus."""
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


def _opt_num(data: Mapping[str, object], key: str) -> float | None:
    """Numeric payload field that may legitimately be absent or None (ttft)."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _text(data: Mapping[str, object], key: str, default: str = "unknown") -> str:
    """Read a string payload field; defaults keep label values bounded."""
    value = data.get(key, default)
    return value if isinstance(value, str) else default


class MetricsCollector:
    """Wildcard bus subscriber feeding every §8.3 instrument (see module docstring)."""

    def __init__(self, bus: EventBus, config: Config, metrics: Metrics) -> None:
        self._metrics = metrics
        self._backend = config.routing.backend
        # Reverse map so llm_* metrics can label by tier from the model id
        # that actually streamed (the events carry model, not tier — M7.3).
        self._tier_by_model = {
            resolve_model(config, tier): tier for tier in config.routing.tiers
        }
        self._pending: tuple[str, float] | None = None  # (model, message_start ts)
        self._unsubscribe = bus.subscribe(None, self.on_event)

    def stop(self) -> None:
        """Detach from the bus; idempotent (the bus's unsubscribe is too)."""
        self._unsubscribe()

    # --- dispatch -------------------------------------------------------------

    def on_event(self, event: Event) -> None:
        """Route one event to its instrument; event payloads stay loosely typed."""
        kind = event.kind
        data = event.data
        if kind == "route_decided":
            backend = _text(data, "backend")
            self._metrics.record_route(
                intent=_text(data, "intent"),
                tier=_text(data, "tier"),
                backend=backend,
                band=_band(_num(data, "confidence")),
                duration_s=_num(data, "duration_s"),
            )
            # A backend other than the configured one IS the fallback (M7.3).
            if backend != self._backend:
                self._metrics.record_route_fallback()
        elif kind == "route_escalated":
            self._metrics.record_route_escalated()
        elif kind == "message_start":
            self._pending = (_text(data, "model"), event.ts)
        elif kind == "message_end":
            self._settle_streamed(event, data)
        elif kind == "error" and data.get("stage") == "llm":
            self._settle_inflight(event, status="error")
        elif kind == "turn_end":
            # Cancellations raise out of run_turn without a message_end; the
            # turn is over either way, so close the request and count the task.
            self._settle_inflight(event, status="aborted")
            self._metrics.record_task(
                outcome=_text(data, "outcome", "unknown"),
                rounds=int(_num(data, "rounds")),
            )
        elif kind == "agent_end":
            self._settle_inflight(event, status="aborted")
        elif kind == "reflection":
            self._metrics.record_reflection(verdict=_text(data, "verdict"))
        elif kind == "plan_approved":
            self._metrics.record_plan_approval(decision="approved")
        elif kind == "plan_rejected":
            self._metrics.record_plan_approval(decision=_text(data, "decision"))
        elif kind == "tool_execution_end":
            self._metrics.record_tool(
                tool=_text(data, "tool"),
                status=_text(data, "status"),
                duration_s=_num(data, "duration_s"),
            )
            reason = data.get("blocked_reason")
            if isinstance(reason, str) and reason:
                self._metrics.record_tool_blocked(reason=reason)
        elif kind == "server_started":
            self._metrics.record_server_started(
                ready_s=_num(data, "ready_s"),
                restart=data.get("restart") is True,
            )
        elif kind == "server_stopped":
            self._metrics.record_server_stopped()

    # --- llm request pairing ---------------------------------------------------

    def _settle_streamed(self, event: Event, data: Mapping[str, object]) -> None:
        """A request that streamed its end: stop_reason is the status (M7.3)."""
        pending = self._pending
        self._pending = None
        model, started = pending if pending is not None else ("unknown", event.ts)
        duration_s = _opt_num(data, "duration_s")
        if duration_s is None:
            duration_s = max(0.0, event.ts - started)
        usage = data.get("usage")
        prompt_tokens = completion_tokens = 0.0
        if isinstance(usage, Mapping):
            prompt_tokens = _num(usage, "prompt_tokens")
            completion_tokens = _num(usage, "completion_tokens")
        self._metrics.record_llm(
            model=model,
            tier=self._tier_by_model.get(model, "unknown"),
            status=_text(data, "stop_reason"),
            ttft_s=_opt_num(data, "ttft_s"),
            duration_s=duration_s,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )

    def _settle_inflight(self, event: Event, *, status: str) -> None:
        """Close a request that never streamed its end (error/abort/quit)."""
        if self._pending is None:
            return
        model, started = self._pending
        self._pending = None
        self._metrics.record_llm(
            model=model,
            tier=self._tier_by_model.get(model, "unknown"),
            status=status,
            ttft_s=None,
            duration_s=max(0.0, event.ts - started),
            prompt_tokens=0,
            completion_tokens=0,
        )
