"""The PLAN §8.3 metric vocabulary, constructed in exactly one place.

Ruling M7.1: prometheus-client sits behind this facade — callers and tests
speak only the named ``record_*`` methods and ``render()``, never
prometheus_client types, and every instance owns its CollectorRegistry so
runs and tests never share instrument state.

Each method's docstring names the operational question the metric answers —
telemetry without a question is noise.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)

# Latency-ish buckets for everything timed in seconds: local rounds are usually
# sub-second, but plan/reflection/model loads stretch into tens of seconds.
_DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
# Scheduler slip is normally milliseconds; anything approaching seconds is an
# incident, so the buckets hug zero to make that visible.
_LAG_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
# Output token latency buckets in seconds (0.005s to 0.5s, i.e. 2 to 200 tok/s).
_TPOT_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5)


class Metrics:
    """One process run's instruments, isolated on their own registry."""

    def __init__(self) -> None:
        self._registry = CollectorRegistry()
        # The PLAN §8.3 "System" row wants process metrics; registering them on
        # *our* registry (not the global default) keeps instances independent.
        ProcessCollector(registry=self._registry)

        self._route_decisions = Counter(
            "route_decisions_total",
            "Routing decisions announced to the user, by shape of the decision.",
            ["intent", "tier", "backend", "band"],
            registry=self._registry,
        )
        self._route_fallback = Counter(
            "route_fallback_total",
            "Decisions that fell back to a non-configured classification backend.",
            registry=self._registry,
        )
        self._route_escalated = Counter(
            "route_escalated_total",
            "Tier escalations triggered by consecutive tool failures.",
            registry=self._registry,
        )
        self._route_duration = Histogram(
            "route_duration_seconds",
            "Wall time from classify start to announced routing decision.",
            buckets=_DURATION_BUCKETS,
            registry=self._registry,
        )
        self._llm_requests = Counter(
            "llm_requests_total",
            "Model requests by terminal status (stop_reason, error, or aborted).",
            ["model", "tier", "status"],
            registry=self._registry,
        )
        self._llm_ttft = Histogram(
            "llm_ttft_seconds",
            "Time to first streamed token per model request.",
            buckets=_DURATION_BUCKETS,
            registry=self._registry,
        )
        self._llm_tpot = Histogram(
            "llm_tpot_seconds",
            "Time per output token for generated completions.",
            buckets=_TPOT_BUCKETS,
            registry=self._registry,
        )
        self._llm_duration = Histogram(
            "llm_duration_seconds",
            "Wall time of a full model stream.",
            buckets=_DURATION_BUCKETS,
            registry=self._registry,
        )
        self._llm_tokens = Counter(
            "llm_tokens_total",
            "Tokens consumed from and produced by local models.",
            ["direction", "model"],
            registry=self._registry,
        )
        self._agent_rounds = Counter(
            "agent_rounds_total",
            "Model rounds executed across tasks (turns).",
            registry=self._registry,
        )
        self._agent_tasks = Counter(
            "agent_tasks_total",
            "Turns by final outcome.",
            ["outcome"],
            registry=self._registry,
        )
        self._agent_reflections = Counter(
            "agent_reflections_total",
            "Reflection rounds by verdict.",
            ["verdict"],
            registry=self._registry,
        )
        self._agent_plan_approvals = Counter(
            "agent_plan_approvals_total",
            "Plan approval decisions (approved vs rejected/cancelled/...).",
            ["decision"],
            registry=self._registry,
        )
        self._tool_calls = Counter(
            "tool_calls_total",
            "Tool executions by tool and status.",
            ["tool", "status"],
            registry=self._registry,
        )
        self._tool_duration = Histogram(
            "tool_duration_seconds",
            "Wall time per tool execution.",
            ["tool"],
            buckets=_DURATION_BUCKETS,
            registry=self._registry,
        )
        self._tool_blocked = Counter(
            "tool_blocked_total",
            "Policy denials by stable reason token (not runtime errors).",
            ["reason"],
            registry=self._registry,
        )
        self._server_starts = Counter(
            "server_starts_total",
            "llama-server startups that reached health.",
            registry=self._registry,
        )
        self._server_ready = Histogram(
            "server_ready_seconds",
            "Spawn-to-health time for llama-server.",
            buckets=_DURATION_BUCKETS,
            registry=self._registry,
        )
        self._server_active = Gauge(
            "server_active",
            "1 while llama-server is up, 0 otherwise.",
            registry=self._registry,
        )
        self._server_restarts = Counter(
            "server_restarts_total",
            "Successful crash-restarts of llama-server.",
            registry=self._registry,
        )
        self._event_loop_lag = Histogram(
            "event_loop_lag_seconds",
            "Event-loop scheduler slip per sampling tick.",
            buckets=_LAG_BUCKETS,
            registry=self._registry,
        )

    @property
    def registry(self) -> CollectorRegistry:
        """The registry the exposition server scrapes (and tests render)."""
        return self._registry

    def render(self) -> bytes:
        """Current exposition in Prometheus text format — the test-facing view."""
        return generate_latest(self._registry)

    # --- routing (which decisions, how fast, when did we degrade) ------------

    def record_route(
        self, *, intent: str, tier: str, backend: str, band: str, duration_s: float
    ) -> None:
        """Which decisions did the router make, and how long did deciding take?"""
        self._route_decisions.labels(
            intent=intent, tier=tier, backend=backend, band=band
        ).inc()
        self._route_duration.observe(duration_s)

    def record_route_fallback(self) -> None:
        """How often did the configured backend fail over to another one?"""
        self._route_fallback.inc()

    def record_route_escalated(self) -> None:
        """How often did tool failures push us up a tier?"""
        self._route_escalated.inc()

    # --- llm (is the model healthy, fast, and cheap?) -------------------------

    def record_llm(
        self,
        *,
        model: str,
        tier: str,
        status: str,
        ttft_s: float | None,
        duration_s: float,
        prompt_tokens: int,
        completion_tokens: int,
        tpot_s: float | None = None,
    ) -> None:
        """Did a request finish cleanly, how fast did it stream, what did it cost?"""
        self._llm_requests.labels(model=model, tier=tier, status=status).inc()
        if ttft_s is not None:
            self._llm_ttft.observe(ttft_s)
        if tpot_s is not None:
            self._llm_tpot.observe(tpot_s)
        self._llm_duration.observe(duration_s)
        if prompt_tokens or completion_tokens:
            self._llm_tokens.labels(direction="prompt", model=model).inc(prompt_tokens)
            self._llm_tokens.labels(direction="completion", model=model).inc(
                completion_tokens
            )

    # --- agent (did tasks complete, and did plans/reflections behave?) --------

    def record_task(self, *, outcome: str, rounds: int) -> None:
        """What outcomes do turns end in, and how many rounds do they burn?"""
        self._agent_tasks.labels(outcome=outcome).inc()
        if rounds:
            self._agent_rounds.inc(rounds)

    def record_reflection(self, *, verdict: str) -> None:
        """Do reflections conclude tasks or keep the loop going?"""
        self._agent_reflections.labels(verdict=verdict).inc()

    def record_plan_approval(self, *, decision: str) -> None:
        """Do users (or the headless approver) accept proposed plans?"""
        self._agent_plan_approvals.labels(decision=decision).inc()

    # --- tools (what runs, how slow, what gets blocked?) ----------------------

    def record_tool(self, *, tool: str, status: str, duration_s: float) -> None:
        """Which tools run, fail, and eat time?"""
        self._tool_calls.labels(tool=tool, status=status).inc()
        self._tool_duration.labels(tool=tool).observe(duration_s)

    def record_tool_blocked(self, *, reason: str) -> None:
        """Which policy denials fire — the stable M4 reason tokens?"""
        self._tool_blocked.labels(reason=reason).inc()

    # --- server (is llama-server up, ready fast, restarting?) -----------------

    def record_server_started(self, *, ready_s: float, restart: bool) -> None:
        """How long until llama-server serves, and is this a crash-restart?"""
        self._server_starts.inc()
        self._server_ready.observe(ready_s)
        self._server_active.set(1)
        if restart:
            self._server_restarts.inc()

    def record_server_stopped(self) -> None:
        """Keep the active gauge honest across stops and idle-shutdowns."""
        self._server_active.set(0)

    # --- system (is the host itself struggling?) ------------------------------

    def observe_lag(self, seconds: float) -> None:
        """Is the event loop being starved (scheduler slip per tick)?"""
        self._event_loop_lag.observe(seconds)
