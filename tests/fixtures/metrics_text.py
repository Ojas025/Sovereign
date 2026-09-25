"""Parse Prometheus text exposition so tests can assert names, labels and values."""

from __future__ import annotations

import re

_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})? (?P<value>\S+)$"
)
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

# PLAN §8.3 — every metric name the observability milestone promises users.
# Shared by the facade unit tests and the e2e "scripted run exposes /metrics"
# test so both assert the same canonical vocabulary.
PLAN_METRICS = (
    "route_decisions_total",
    "route_fallback_total",
    "route_escalated_total",
    "route_duration_seconds",
    "llm_requests_total",
    "llm_ttft_seconds",
    "llm_tpot_seconds",
    "llm_duration_seconds",
    "llm_tokens_total",
    "agent_rounds_total",
    "agent_tasks_total",
    "agent_reflections_total",
    "agent_plan_approvals_total",
    "tool_calls_total",
    "tool_duration_seconds",
    "tool_blocked_total",
    "server_starts_total",
    "server_ready_seconds",
    "server_active",
    "server_restarts_total",
    "event_loop_lag_seconds",
    "process_cpu_seconds_total",  # process metrics (PLAN §8.3 System row)
)


def samples(text: str, name: str, **labels: str) -> list[float]:
    """Values of series `name` whose labels contain every given key=value pair.

    Unlabelled series match on name alone; the exposition escapes nothing in
    our label sets, so a plain parse is enough for test assertions.
    """
    values: list[float] = []
    for line in text.splitlines():
        match = _LINE.match(line)
        if match is None or match["name"] != name:
            continue
        found = dict(_LABEL.findall(match["labels"] or ""))
        if all(found.get(key) == value for key, value in labels.items()):
            values.append(float(match["value"]))
    return values


def has_metric(text: str, name: str) -> bool:
    """True when `name`'s family appears in the exposition.

    Counters/gauges match their series name exactly; histogram families match
    through the `_bucket`/`_sum`/`_count` suffixes Prometheus exposes them as.
    """
    for line in text.splitlines():
        match = _LINE.match(line)
        if match is None:
            continue
        series = match["name"]
        if series == name or series.startswith(f"{name}_"):
            return True
    return False
