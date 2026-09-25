"""T7.2 — bus subscriber maps events onto the PLAN §8.3 instruments (M7.2/M7.3)."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fixtures.fake_llama_server import free_port
from fixtures.metrics_text import samples
from workbench.config import Config, load_config
from workbench.core.events import Event, EventBus
from workbench.observability import Observability, start_observability
from workbench.observability.collector import MetricsCollector
from workbench.observability.metrics import Metrics

_QWEN_STEM = "Qwen2.5-Coder-7B-Instruct-Q4_K_M"
_MINICPM_STEM = "MiniCPM5-2B-Q4_K_M"


def base_config(**overrides: str) -> Config:
    """Default config (routing.backend=laya), overridable CLI-style."""
    return load_config(
        home=Path("/nonexistent"),
        project_dir=Path("/nonexistent"),
        cli_overrides=overrides,
    )


def profile_config(tmp_path: Path) -> Config:
    """Two tiers mapped to two distinct model stems — feeds the reverse map."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".workbench.toml").write_text(
        '[routing.tiers]\nsmall = "coder"\nmid = "base"\n\n'
        '[models.profiles.coder]\npath = "/models/'
        + _QWEN_STEM
        + '.gguf"\n\n[models.profiles.base]\npath = "/models/'
        + _MINICPM_STEM
        + '.gguf"\n',
        encoding="utf-8",
    )
    return load_config(home=tmp_path / "home", project_dir=project)


def wired(config: Config) -> tuple[EventBus, Metrics, MetricsCollector]:
    """A bus with the production collector attached and its own Metrics."""
    bus = EventBus()
    metrics = Metrics()
    collector = MetricsCollector(bus, config, metrics)
    return bus, metrics, collector


def route_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "tier": "small",
        "reason": "easy",
        "intent": "chat",
        "difficulty": 0.1,
        "confidence": 0.9,
        "backend": "laya",
        "model": "MiniCPM5-2B-Q4_K_M",
        "suggest_plan": False,
        "duration_s": 0.42,
    }
    data.update(overrides)
    return data


def test_route_decision_counts_with_band_and_duration() -> None:
    bus, metrics, _ = wired(base_config())
    bus.emit(Event("route_decided", route_data(intent="planning", tier="mid")))
    text = metrics.render().decode()
    assert samples(
        text,
        "route_decisions_total",
        intent="planning",
        tier="mid",
        backend="laya",
        band="high",
    ) == [1.0]
    assert samples(text, "route_duration_seconds_sum") == [pytest.approx(0.42)]
    assert samples(text, "route_fallback_total") == [0.0]


@pytest.mark.parametrize(
    ("confidence", "band"),
    [(0.49, "low"), (0.5, "mid"), (0.79, "mid"), (0.8, "high")],
)
def test_confidence_band_boundaries(confidence: float, band: str) -> None:
    bus, metrics, _ = wired(base_config())
    bus.emit(Event("route_decided", route_data(confidence=confidence)))
    text = metrics.render().decode()
    assert samples(
        text, "route_decisions_total", intent="chat", tier="small",
        backend="laya", band=band,
    ) == [1.0]


def test_fallback_counts_only_non_config_backend() -> None:
    bus, metrics, _ = wired(base_config())
    bus.emit(Event("route_decided", route_data()))
    assert samples(metrics.render().decode(), "route_fallback_total") == [0.0]

    bus.emit(Event("route_decided", route_data(backend="heuristic")))
    assert samples(metrics.render().decode(), "route_fallback_total") == [1.0]

    # When heuristic IS the configured backend, it is not a fallback (M7.3).
    bus2, metrics2, _ = wired(base_config(**{"routing.backend": "heuristic"}))
    bus2.emit(Event("route_decided", route_data(backend="heuristic")))
    assert samples(metrics2.render().decode(), "route_fallback_total") == [0.0]


def test_route_escalated_counts() -> None:
    bus, metrics, _ = wired(base_config())
    bus.emit(Event("route_escalated", {"from_tier": "small", "to_tier": "mid", "failures": 3}))
    assert samples(metrics.render().decode(), "route_escalated_total") == [1.0]


def test_llm_success_maps_model_to_tier_and_records_usage(tmp_path: Path) -> None:
    bus, metrics, _ = wired(profile_config(tmp_path))
    bus.emit(Event("message_start", {"model": _MINICPM_STEM, "role": "assistant"}))
    bus.emit(
        Event(
            "message_end",
            {
                "stop_reason": "tool_calls",
                "ttft_s": 0.1,
                "duration_s": 1.25,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
    )
    text = metrics.render().decode()
    assert samples(
        text,
        "llm_requests_total",
        model=_MINICPM_STEM,
        tier="mid",
        status="tool_calls",
    ) == [1.0]
    assert samples(text, "llm_ttft_seconds_sum") == [pytest.approx(0.1)]
    assert samples(text, "llm_tpot_seconds_sum") == [pytest.approx(0.23)]
    assert samples(text, "llm_duration_seconds_sum") == [pytest.approx(1.25)]
    assert samples(text, "llm_tokens_total", direction="prompt", model=_MINICPM_STEM) == [10.0]
    assert samples(text, "llm_tokens_total", direction="completion", model=_MINICPM_STEM) == [5.0]


def test_llm_error_settles_the_pending_message(tmp_path: Path) -> None:
    bus, metrics, _ = wired(profile_config(tmp_path))
    bus.emit(Event("message_start", {"model": _QWEN_STEM, "role": "assistant"}, ts=100.0))
    bus.emit(Event("error", {"stage": "llm", "message": "boom"}, ts=100.5))
    text = metrics.render().decode()
    assert samples(
        text, "llm_requests_total", model=_QWEN_STEM, tier="small", status="error"
    ) == [1.0]
    assert samples(text, "llm_duration_seconds_sum") == [pytest.approx(0.5)]
    # the error was not an llm stage — nothing to settle
    bus.emit(Event("message_start", {"model": _QWEN_STEM, "role": "assistant"}))
    bus.emit(Event("error", {"stage": "tool", "message": "x"}))
    bus.emit(
        Event(
            "message_end",
            {"stop_reason": "stop", "ttft_s": None, "duration_s": 0.3, "usage": None},
        )
    )
    assert samples(
        metrics.render().decode(),
        "llm_requests_total",
        model=_QWEN_STEM,
        tier="small",
        status="stop",
    ) == [1.0]


def test_turn_end_settles_inflight_as_aborted_and_counts_task(tmp_path: Path) -> None:
    bus, metrics, _ = wired(profile_config(tmp_path))
    bus.emit(Event("message_start", {"model": _MINICPM_STEM, "role": "assistant"}, ts=10.0))
    bus.emit(
        Event("turn_end", {"outcome": "completed", "rounds": 3, "duration_s": 5.0}, ts=12.0)
    )
    text = metrics.render().decode()
    assert samples(
        text, "llm_requests_total", model=_MINICPM_STEM, tier="mid", status="aborted"
    ) == [1.0]
    assert samples(text, "llm_duration_seconds_sum") == [pytest.approx(2.0)]
    assert samples(text, "agent_tasks_total", outcome="completed") == [1.0]
    assert samples(text, "agent_rounds_total") == [3.0]


def test_agent_end_settles_a_leaked_inflight_message(tmp_path: Path) -> None:
    bus, metrics, _ = wired(profile_config(tmp_path))
    bus.emit(Event("message_start", {"model": _MINICPM_STEM, "role": "assistant"}, ts=1.0))
    bus.emit(Event("agent_end", {"outcome": "quit"}, ts=2.0))
    text = metrics.render().decode()
    assert samples(
        text, "llm_requests_total", model=_MINICPM_STEM, tier="mid", status="aborted"
    ) == [1.0]


def test_unknown_model_lands_in_the_unknown_tier() -> None:
    bus, metrics, _ = wired(base_config())
    bus.emit(Event("message_start", {"model": "mystery-model", "role": "assistant"}))
    bus.emit(
        Event(
            "message_end",
            {"stop_reason": "stop", "ttft_s": 0.2, "duration_s": 0.4, "usage": None},
        )
    )
    assert samples(
        metrics.render().decode(), "llm_requests_total", model="mystery-model", tier="unknown"
    ) == [1.0]


def test_reflection_and_plan_decisions_count() -> None:
    bus, metrics, _ = wired(base_config())
    bus.emit(Event("reflection", {"verdict": "done", "completed": [1]}))
    bus.emit(Event("plan_approved", {"attempt": 1}))
    bus.emit(Event("plan_rejected", {"decision": "cancelled", "attempt": 1, "feedback": ""}))
    text = metrics.render().decode()
    assert samples(text, "agent_reflections_total", verdict="done") == [1.0]
    assert samples(text, "agent_plan_approvals_total", decision="approved") == [1.0]
    assert samples(text, "agent_plan_approvals_total", decision="cancelled") == [1.0]


def test_tool_end_counts_and_blocked_reason_only_when_present() -> None:
    bus, metrics, _ = wired(base_config())
    bus.emit(
        Event(
            "tool_execution_end",
            {"call_id": "c1", "tool": "bash", "status": "ok", "duration_s": 0.4,
             "output": "hi", "blocked_reason": None},
        )
    )
    bus.emit(
        Event(
            "tool_execution_end",
            {"call_id": "c2", "tool": "bash", "status": "error", "duration_s": 0.1,
             "output": "boom", "blocked_reason": "denied_pattern"},
        )
    )
    text = metrics.render().decode()
    assert samples(text, "tool_calls_total", tool="bash", status="ok") == [1.0]
    assert samples(text, "tool_calls_total", tool="bash", status="error") == [1.0]
    assert samples(text, "tool_duration_seconds_sum", tool="bash") == [pytest.approx(0.5)]
    assert samples(text, "tool_blocked_total", reason="denied_pattern") == [1.0]


def test_server_lifecycle_series() -> None:
    bus, metrics, _ = wired(base_config())
    bus.emit(Event("server_started", {"port": 8080, "pid": 1, "ready_s": 2.5, "restart": False}))
    text = metrics.render().decode()
    assert samples(text, "server_starts_total") == [1.0]
    assert samples(text, "server_restarts_total") == [0.0]  # unlabeled = visible at zero
    assert samples(text, "server_ready_seconds_sum") == [pytest.approx(2.5)]
    assert samples(text, "server_active") == [1.0]
    bus.emit(Event("server_started", {"port": 8080, "pid": 2, "ready_s": 1.0, "restart": True}))
    bus.emit(Event("server_stopped", {"reason": "explicit"}))
    text = metrics.render().decode()
    assert samples(text, "server_starts_total") == [2.0]
    assert samples(text, "server_restarts_total") == [1.0]
    assert samples(text, "server_active") == [0.0]


def test_collector_stop_unsubscribes() -> None:
    bus, metrics, collector = wired(base_config())
    bus.emit(Event("route_escalated", {"from_tier": "small", "to_tier": "mid", "failures": 1}))
    collector.stop()
    bus.emit(Event("route_escalated", {"from_tier": "small", "to_tier": "mid", "failures": 2}))
    assert samples(metrics.render().decode(), "route_escalated_total") == [1.0]
    collector.stop()  # idempotent


def test_start_observability_port_zero_collects_in_process() -> None:
    config = base_config(**{"observability.metrics_port": "0"})
    bus = EventBus()
    handle = start_observability(config, bus)
    try:
        assert isinstance(handle, Observability)
        assert handle.server.running is False
        assert handle.sampler is None  # no exposition -> nothing to sample for
        bus.emit(Event("route_escalated", {"from_tier": "small", "to_tier": "mid", "failures": 1}))
        assert samples(handle.metrics.render().decode(), "route_escalated_total") == [1.0]
    finally:
        handle.stop()
    bus.emit(Event("route_escalated", {"from_tier": "small", "to_tier": "mid", "failures": 2}))
    assert samples(handle.metrics.render().decode(), "route_escalated_total") == [1.0]


def test_start_observability_serves_then_releases_on_stop() -> None:
    port = free_port()
    config = base_config(**{"observability.metrics_port": str(port)})
    bus = EventBus()
    handle = start_observability(config, bus)
    try:
        assert handle.server.running is True
        assert handle.sampler is not None
        bus.emit(Event("route_escalated", {"from_tier": "small", "to_tier": "mid", "failures": 1}))
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            body = response.read().decode()
        assert "route_escalated_total" in body
    finally:
        handle.stop()
    assert handle.server.running is False
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1)
