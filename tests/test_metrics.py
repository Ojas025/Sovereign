"""T7.1 — metrics facade renders the PLAN §8.3 vocabulary; exposition on localhost."""

from __future__ import annotations

import logging
import socket
import time
import urllib.error
import urllib.request

import pytest

from fixtures.fake_llama_server import free_port
from fixtures.metrics_text import PLAN_METRICS, has_metric, samples
from workbench.observability.exposition import LagSampler, MetricsServer
from workbench.observability.metrics import Metrics


def filled() -> Metrics:
    """Exercise every record_* method once — the vocabulary smoke fill."""
    metrics = Metrics()
    metrics.record_route(
        intent="planning", tier="small", backend="laya", band="high", duration_s=0.25
    )
    metrics.record_route_fallback()
    metrics.record_route_escalated()
    metrics.record_llm(
        model="MiniCPM5-2B-Q4_K_M",
        tier="small",
        status="stop",
        ttft_s=0.1,
        duration_s=1.5,
        prompt_tokens=10,
        completion_tokens=20,
    )
    metrics.record_task(outcome="completed", rounds=3)
    metrics.record_reflection(verdict="done")
    metrics.record_plan_approval(decision="approved")
    metrics.record_tool(tool="bash", status="ok", duration_s=0.4)
    metrics.record_tool_blocked(reason="denied_pattern")
    metrics.record_server_started(ready_s=2.5, restart=False)
    metrics.record_server_stopped()
    metrics.observe_lag(0.002)
    return metrics


def test_render_lists_every_plan_metric() -> None:
    text = filled().render().decode()
    missing = [name for name in PLAN_METRICS if not has_metric(text, name)]
    assert missing == []


def test_route_series_values() -> None:
    text = filled().render().decode()
    assert samples(
        text,
        "route_decisions_total",
        intent="planning",
        tier="small",
        backend="laya",
        band="high",
    ) == [1.0]
    assert samples(text, "route_fallback_total") == [1.0]
    assert samples(text, "route_escalated_total") == [1.0]
    assert samples(text, "route_duration_seconds_sum") == [pytest.approx(0.25)]
    assert samples(text, "route_duration_seconds_count") == [1.0]


def test_llm_series_values() -> None:
    text = filled().render().decode()
    assert samples(
        text,
        "llm_requests_total",
        model="MiniCPM5-2B-Q4_K_M",
        tier="small",
        status="stop",
    ) == [1.0]
    assert samples(text, "llm_ttft_seconds_sum") == [pytest.approx(0.1)]
    assert samples(text, "llm_duration_seconds_sum") == [pytest.approx(1.5)]
    assert samples(text, "llm_tokens_total", direction="prompt",
                   model="MiniCPM5-2B-Q4_K_M") == [10.0]
    assert samples(text, "llm_tokens_total", direction="completion",
                   model="MiniCPM5-2B-Q4_K_M") == [20.0]


def test_agent_series_values() -> None:
    text = filled().render().decode()
    assert samples(text, "agent_tasks_total", outcome="completed") == [1.0]
    assert samples(text, "agent_rounds_total") == [3.0]
    assert samples(text, "agent_reflections_total", verdict="done") == [1.0]
    assert samples(text, "agent_plan_approvals_total", decision="approved") == [1.0]


def test_tool_series_values() -> None:
    text = filled().render().decode()
    assert samples(text, "tool_calls_total", tool="bash", status="ok") == [1.0]
    assert samples(text, "tool_duration_seconds_sum", tool="bash") == [pytest.approx(0.4)]
    assert samples(text, "tool_blocked_total", reason="denied_pattern") == [1.0]


def test_server_series_values() -> None:
    metrics = filled()
    metrics.record_server_started(ready_s=0.5, restart=True)  # crash-restart
    text = metrics.render().decode()
    assert samples(text, "server_starts_total") == [2.0]
    assert samples(text, "server_restarts_total") == [1.0]
    assert samples(text, "server_ready_seconds_sum") == [pytest.approx(3.0)]
    assert samples(text, "server_active") == [1.0]  # restarted = active again
    metrics.record_server_stopped()
    assert samples(metrics.render().decode(), "server_active") == [0.0]


def test_instances_are_isolated() -> None:
    one, two = Metrics(), Metrics()
    one.record_task(outcome="completed", rounds=1)
    assert samples(one.render().decode(), "agent_tasks_total", outcome="completed") == [1.0]
    assert samples(two.render().decode(), "agent_tasks_total", outcome="completed") == []


def test_exposition_serves_on_localhost() -> None:
    metrics = filled()
    port = free_port()
    server = MetricsServer(metrics, port=port)
    assert server.start() is True
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/plain")
            body = response.read().decode()
        assert has_metric(body, "route_decisions_total")
        assert has_metric(body, "process_cpu_seconds_total")
    finally:
        server.stop()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1)


def test_port_zero_disables_exposition() -> None:
    server = MetricsServer(Metrics(), port=0)
    assert server.start() is False
    assert server.running is False
    server.stop()  # idempotent on a never-started server


def test_busy_port_degrades_instead_of_raising(caplog: pytest.LogCaptureFixture) -> None:
    port = free_port()
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        with caplog.at_level(logging.WARNING):
            server = MetricsServer(Metrics(), port=port)
            assert server.start() is False  # run must continue exposition-less
        assert any("metrics" in record.message for record in caplog.records)
    finally:
        blocker.close()


def test_stop_frees_the_port_for_a_restart() -> None:
    port = free_port()
    first = MetricsServer(Metrics(), port=port)
    assert first.start() is True
    first.stop()
    second = MetricsServer(Metrics(), port=port)
    assert second.start() is True  # no TIME_WAIT limbo after stop()
    second.stop()


def test_lag_sampler_reports_and_stops() -> None:
    observed: list[float] = []
    sampler = LagSampler(observed.append, interval_s=0.01)
    sampler.start()
    deadline = time.monotonic() + 2.0
    while not observed and time.monotonic() < deadline:
        time.sleep(0.01)
    sampler.stop()
    assert observed, "lag sampler never observed a tick"
    assert observed[0] >= 0.0
    count = len(observed)
    time.sleep(0.05)
    assert len(observed) == count  # stop() actually stops the thread
