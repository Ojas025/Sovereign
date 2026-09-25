"""llama-serve manager: lazy spawn, health polling, idle-stop, crash restart."""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from fixtures.fake_llama_server import fake_server_command, free_port
from workbench.config import Config, load_config
from workbench.core.events import Event, EventBus
from workbench.llm.server import ServerError, ServerManager


def make_config(**overrides: str) -> Config:
    cli_overrides = {"server.port": str(free_port()), **overrides}
    return load_config(
        home=Path("/nonexistent"),
        project_dir=Path("/nonexistent"),
        cli_overrides=cli_overrides,
    )


async def wait_until(
    predicate: Callable[[], bool], *, timeout_s: float = 5.0, interval_s: float = 0.05
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval_s)
    raise AssertionError(f"condition not met within {timeout_s}s")


def collect_events(bus: EventBus) -> list[Event]:
    seen: list[Event] = []
    bus.subscribe(None, seen.append)
    return seen


async def test_ensure_running_spawns_waits_for_health_and_is_idempotent() -> None:
    bus = EventBus()
    seen = collect_events(bus)
    config = make_config()
    manager = ServerManager(
        config, bus, command=fake_server_command(config.server.port, ready_after=2)
    )
    try:
        await manager.ensure_running()

        assert manager.state == "ready"
        assert manager.process is not None and manager.process.returncode is None
        assert "server_started" in [event.kind for event in seen]

        # Idempotent: a second call must not respawn the child.
        pid = manager.process.pid
        await manager.ensure_running()
        assert manager.process.pid == pid

        assert await manager.list_models() == ["fake-model"]
    finally:
        await manager.stop()

    assert manager.state == "stopped"
    assert manager.process is None
    assert "server_stopped" in [event.kind for event in seen]


async def test_idle_timeout_stops_server_without_activity() -> None:
    bus = EventBus()
    seen = collect_events(bus)
    config = make_config(**{"server.idle_timeout_s": "0.3"})
    manager = ServerManager(config, bus, command=fake_server_command(config.server.port))
    try:
        await manager.ensure_running()
        await wait_until(lambda: manager.state == "stopped")
    finally:
        await manager.stop()

    stop_events = [e for e in seen if e.kind == "server_stopped"]
    assert stop_events and stop_events[0].data["reason"] == "idle"
    assert manager.process is None


async def test_touch_delays_idle_stop() -> None:
    bus = EventBus()
    config = make_config(**{"server.idle_timeout_s": "0.5"})
    manager = ServerManager(config, bus, command=fake_server_command(config.server.port))
    try:
        await manager.ensure_running()
        for _ in range(4):
            await asyncio.sleep(0.2)
            manager.touch()
        assert manager.state == "ready"
    finally:
        await manager.stop()


async def test_crash_triggers_automatic_restart() -> None:
    bus = EventBus()
    config = make_config()
    manager = ServerManager(config, bus, command=fake_server_command(config.server.port))
    try:
        await manager.ensure_running()
        first_pid = manager.process.pid

        manager.process.kill()
        await wait_until(
            lambda: manager.process is not None
            and manager.process.pid != first_pid
            and manager.state == "ready"
        )
        assert manager.process.returncode is None
    finally:
        await manager.stop()


async def test_server_started_payload_reports_readiness_and_restart_flag() -> None:
    """M7.2: ready_s feeds server_ready_seconds, restart feeds server_restarts_total."""
    bus = EventBus()
    seen = collect_events(bus)
    config = make_config()
    manager = ServerManager(config, bus, command=fake_server_command(config.server.port))
    try:
        await manager.ensure_running()
        first = [e for e in seen if e.kind == "server_started"][-1]
        assert isinstance(first.data["ready_s"], float)
        assert first.data["ready_s"] >= 0.0
        assert first.data["restart"] is False  # a manual start is no crash-restart

        manager.process.kill()
        await wait_until(
            lambda: manager.state == "ready"
            and len([e for e in seen if e.kind == "server_started"]) == 2
        )
        second = [e for e in seen if e.kind == "server_started"][-1]
        assert second.data["restart"] is True  # the child came back by itself
    finally:
        await manager.stop()


async def test_crash_without_restart_budget_fails_and_reports_error() -> None:
    bus = EventBus()
    seen = collect_events(bus)
    config = make_config(**{"server.restart_max": "0"})
    manager = ServerManager(config, bus, command=fake_server_command(config.server.port))
    try:
        await manager.ensure_running()
        manager.process.kill()
        await wait_until(lambda: manager.state == "failed")

        assert any(event.kind == "error" for event in seen)
    finally:
        await manager.stop()


async def test_unspawnable_command_raises_server_error() -> None:
    bus = EventBus()
    seen = collect_events(bus)
    manager = ServerManager(make_config(), bus, command=["/nonexistent/llama-serve"])

    with pytest.raises(ServerError, match="nonexistent"):
        await manager.ensure_running()

    assert manager.state == "failed"
    assert any(event.kind == "error" for event in seen)
    await manager.stop()


async def test_stop_is_idempotent_without_start() -> None:
    manager = ServerManager(make_config(), EventBus(), command=["/bin/false"])

    await manager.stop()
    await manager.stop()

    assert manager.state == "stopped"


def test_default_command_enforces_single_active_gpu() -> None:
    """llama-server defaults to loading 4 models; the 6GB budget requires 1."""
    manager = ServerManager(make_config(), EventBus())

    assert "--models-max" in manager.command
    assert manager.command[manager.command.index("--models-max") + 1] == "1"
    assert "--router" in manager.command
    assert "--models-dir" in manager.command
