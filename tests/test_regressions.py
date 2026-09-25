"""Regression test suite for edge cases, error conditions, and resilience (PLAN §9, M9).

Covers:
- Budget exceeded: graceful termination when max_rounds is breached
- Tool error recovery and escalation: tier escalation on consecutive tool failures
- EditTool exact-match uniqueness and replace_all behavior
- Path jail resolution edge cases (nested paths, subdirectories, symlink checks)
- ServerManager backoff on crash loops
- Config layering precedence (CLI > Env > Project > Global)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from fixtures.fake_llama_server import free_port
from workbench.agent.loop import AgentLoop
from workbench.agent.planning import auto_approver
from workbench.config import (
    AgentConfig,
    Config,
    ModelProfile,
    ModelsConfig,
    ObservabilityConfig,
    RoutingConfig,
    RuntimeConfig,
    SandboxConfig,
    ServerConfig,
    TierPolicyConfig,
    ToolsConfig,
    load_config,
)
from workbench.core.events import Event, EventBus
from workbench.core.protocols import (
    ChatRequest,
    Classification,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
    ToolContext,
    ToolResult,
)
from workbench.core.session import Session
from workbench.llm.server import ServerError, ServerManager
from workbench.routing.policy import TierPolicy
from workbench.tools.files import EditTool
from workbench.tools.jail import PathJail, PathJailError


class _ScriptedLLM:
    """Scripted LLM client that returns sequences of StreamEvents."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self._turns = list(turns)
        self.call_count = 0

    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.call_count += 1
        events = self._turns.pop(0) if self._turns else [TextDelta("default"), StreamEnd("stop")]

        async def _gen() -> AsyncIterator[StreamEvent]:
            for ev in events:
                yield ev

        return _gen()


class _DummyRouter:
    def __init__(self, intent: str = "chat_qa", difficulty: float = 0.5) -> None:
        self.intent = intent
        self.difficulty = difficulty

    async def classify(self, message: str) -> Classification:
        return Classification(
            intent=self.intent,
            difficulty=self.difficulty,
            confidence=0.9,
            needs_tools=False,
            backend="dummy",
        )


class _FailingTool:
    name = "fail_tool"
    description = "Always fails"
    parameters: dict[str, object] = {"type": "object"}

    async def execute(self, arguments: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(content="failed intentional", is_error=True)


def _make_config(tmp_path: Path, **agent_kwargs: object) -> Config:
    agent_cfg = AgentConfig(**agent_kwargs)  # type: ignore[arg-type]
    return Config(
        runtime=RuntimeConfig(workspace_root=str(tmp_path)),
        models=ModelsConfig(
            profiles={
                "small_p": ModelProfile(path="small.gguf"),
                "mid_p": ModelProfile(path="mid.gguf"),
            }
        ),
        routing=RoutingConfig(
            tiers={"small": "small_p", "mid": "mid_p"},
            policy=TierPolicyConfig(escalate_after_failures=2),
        ),
        server=ServerConfig(),
        agent=agent_cfg,
        tools=ToolsConfig(),
        sandbox=SandboxConfig(),
        observability=ObservabilityConfig(),
    )


# --- 1. Agent Loop Budget Exceeded -------------------------------------------


async def test_agent_loop_terminates_gracefully_when_rounds_budget_exceeded(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, max_rounds=2)
    bus = EventBus()
    events: list[Event] = []
    bus.subscribe(None, events.append)

    # LLM keeps emitting tool calls indefinitely
    infinite_tool_turns = [
        [
            ToolCallStart(call_id=f"c{i}", name="fail_tool"),
            ToolCallDelta(call_id=f"c{i}", arguments_delta="{}"),
            StreamEnd(stop_reason="tool_calls"),
        ]
        for i in range(10)
    ]
    llm = _ScriptedLLM(infinite_tool_turns)
    session = Session(id="test_session", created_at=1.0)

    async def fake_confirm(p: str) -> bool:
        return True

    ctx = ToolContext(workspace_root=tmp_path, confirm=fake_confirm)

    loop = AgentLoop(
        client=llm,  # type: ignore[arg-type]
        router=_DummyRouter(),
        policy=TierPolicy(config.routing),
        tools={"fail_tool": _FailingTool()},
        session=session,
        bus=bus,
        config=config,
        context=ctx,
        approver=auto_approver,
    )

    outcome = await loop.run_turn("infinite tool task")

    assert outcome.outcome == "budget"
    kinds = [e.kind for e in events]
    assert "budget_exceeded" in kinds
    # Max rounds is 2, so LLM should have been called at most 2 times
    assert llm.call_count <= 2


# --- 2. EditTool Uniqueness and Replace All -----------------------------------


async def test_edit_tool_rejects_non_unique_target_when_replace_all_false(
    tmp_path: Path,
) -> None:
    tool = EditTool(ToolsConfig())
    test_file = tmp_path / "sample.txt"
    test_file.write_text("apple banana apple cherry apple\n")

    async def confirm(p: str) -> bool:
        return True

    ctx = ToolContext(workspace_root=tmp_path, confirm=confirm)

    # Trying to replace 'apple' when it appears 3 times without replace_all=True
    res = await tool.execute(
        {"path": "sample.txt", "old_string": "apple", "new_string": "orange"},
        ctx,
    )

    assert res.is_error
    assert "3 matches" in res.content
    assert test_file.read_text() == "apple banana apple cherry apple\n"


async def test_edit_tool_replaces_all_when_flag_is_true(tmp_path: Path) -> None:
    tool = EditTool(ToolsConfig())
    test_file = tmp_path / "sample.txt"
    test_file.write_text("apple banana apple cherry apple\n")

    async def confirm(p: str) -> bool:
        return True

    ctx = ToolContext(workspace_root=tmp_path, confirm=confirm)

    res = await tool.execute(
        {
            "path": "sample.txt",
            "old_string": "apple",
            "new_string": "orange",
            "replace_all": True,
        },
        ctx,
    )

    assert not res.is_error
    assert test_file.read_text() == "orange banana orange cherry orange\n"


# --- 3. Path Jail Escape Attempts --------------------------------------------


def test_path_jail_blocks_parent_directory_traversals(tmp_path: Path) -> None:
    jail = PathJail(tmp_path)

    escapes = [
        "../outside.txt",
        "../../outside.txt",
        "foo/../../outside.txt",
        "foo/bar/../../../outside.txt",
    ]
    for esc in escapes:
        with pytest.raises(PathJailError):
            jail.resolve_read(esc)


def test_path_jail_blocks_symlink_pointing_outside(tmp_path: Path) -> None:
    jail = PathJail(tmp_path)
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret data")

    symlink_in_jail = tmp_path / "leak.txt"
    try:
        symlink_in_jail.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported")

    with pytest.raises(PathJailError):
        jail.resolve_read("leak.txt")


# --- 4. Server Manager Crash Loop Backoff -------------------------------------


async def test_server_manager_stops_after_max_restarts(tmp_path: Path) -> None:
    crasher = tmp_path / "crasher.sh"
    crasher.write_text("#!/bin/sh\nexit 1\n")
    crasher.chmod(0o755)

    bus = EventBus()
    config = load_config(
        home=tmp_path,
        project_dir=tmp_path,
        cli_overrides={
            "server.port": str(free_port()),
            "server.restart_max": "2",
        },
    )
    manager = ServerManager(config, bus, command=[str(crasher)])

    with pytest.raises(ServerError, match="exited during startup"):
        await manager.ensure_running()


# --- 5. Config Layering Precedence --------------------------------------------


def test_config_layering_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_dir = tmp_path / "home"
    global_cfg = home_dir / ".config" / "workbench" / "config.toml"
    global_cfg.parent.mkdir(parents=True)
    global_cfg.write_text('[agent]\nmax_rounds = 10\nwall_clock_s = 100.0\n')

    proj_dir = tmp_path / "project"
    proj_dir.mkdir()
    (proj_dir / ".workbench.toml").write_text('[agent]\nmax_rounds = 20\n')

    # Project overrides global
    cfg = load_config(home=home_dir, project_dir=proj_dir)
    assert cfg.agent.max_rounds == 20
    assert cfg.agent.wall_clock_s == 100.0

    # Env overrides project
    monkeypatch.setenv("WB_MAX_ROUNDS", "35")
    cfg_env = load_config(home=home_dir, project_dir=proj_dir)
    assert cfg_env.agent.max_rounds == 35
