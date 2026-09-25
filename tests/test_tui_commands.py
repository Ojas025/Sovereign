"""Slash commands: built-in vocabulary + plugin registration (PLAN §7, ruling M6.9)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from rich.console import Console

from workbench.config import Config, ModelProfile, ModelsConfig, RoutingConfig
from workbench.core.protocols import ChatMessage
from workbench.core.registry import Registry
from workbench.core.session import Plan, PlanStep, Session, SessionStore
from workbench.tui.commands import CommandContext, register_commands
from workbench.tui.state import TuiState

Handler = Any  # async (ui, args) — tests call handlers directly


class FakeUI:
    """The agentui Session surface commands touch: print + register_command."""

    def __init__(self) -> None:
        self.prints: list[object] = []
        self.commands: dict[str, tuple[Handler, str]] = {}

    def print(self, *objects: object, **kwargs: object) -> None:
        self.prints.extend(objects)

    def register_command(self, name: str, handler: Handler, *, help_text: str = "") -> None:
        self.commands[name] = (handler, help_text)


def printed_text(ui: FakeUI) -> str:
    out = []
    for obj in ui.prints:
        if isinstance(obj, str):
            out.append(obj)
        else:
            console = Console(width=100, force_terminal=False, record=True)
            console.print(obj)
            out.append(console.export_text())
    return "\n".join(out)


def make_config() -> Config:
    return Config(
        routing=RoutingConfig(tiers={"small": "small", "mid": "coder"}),
        models=ModelsConfig(
            profiles={
                "small": ModelProfile(path="~/models/small.gguf", ctx_len=8192),
                "coder": ModelProfile(path="~/models/coder.gguf", ctx_len=32768),
            }
        ),
    )


def make_context(tmp_path: Path, **overrides: object) -> tuple[CommandContext, Session]:
    store = SessionStore(tmp_path / "sessions")
    session = Session.create(store=store, workspace=str(tmp_path / "ws"))
    context = CommandContext(
        config=make_config(),
        state=TuiState(),
        store=store,
        registry=Registry(),
        session=lambda: session,
        plan=lambda: None,
        resume=lambda _loaded: None,
    )
    for key, value in overrides.items():
        setattr(context, key, value)
    return context, session


async def arun(ui: FakeUI, command: str, args: str = "") -> None:
    handler, _help = ui.commands[command]
    await handler(ui, args)


# --- registration -----------------------------------------------------------------


def test_builtins_register_with_help_text(tmp_path: Path) -> None:
    context, _session = make_context(tmp_path)
    ui = FakeUI()

    register_commands(ui, context)

    for name in ("model", "models", "plan", "stats", "router", "sessions", "resume"):
        assert name in ui.commands, name
        assert ui.commands[name][1], f"/{name} missing help text"


def test_plugin_command_registers_and_clash_keeps_the_builtin(
    tmp_path: Path, caplog: Any
) -> None:
    context, _session = make_context(tmp_path)
    registry = Registry()

    async def hello(_ui: object, _args: str) -> None:
        """plugin command"""

    async def hijack(_ui: object, _args: str) -> None:
        """attempted override"""

    registry.register("commands", "hello", hello)
    registry.register("commands", "model", hijack)
    context.registry = registry
    ui = FakeUI()

    with caplog.at_level(logging.WARNING):
        register_commands(ui, context)

    assert "hello" in ui.commands
    assert ui.commands["model"][0] is not hijack  # built-in wins (ruling M6.9)
    assert any(
        "model" in record.getMessage() and "clash" in record.getMessage()
        for record in caplog.records
    )


# --- /model ------------------------------------------------------------------------


async def test_model_pin_sets_tier_and_reports(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "model", "mid")

    assert session.pinned_tier == "mid"
    assert session.tier == "mid"
    assert "pinned" in printed_text(ui)


async def test_model_auto_clears_the_pin(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    session.pinned_tier = "mid"
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "model", "auto")

    assert session.pinned_tier is None
    assert "automatic" in printed_text(ui)


async def test_model_accepts_a_profile_name_via_the_tier_map(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "model", "coder")  # profile name; only tier "mid" maps to it

    assert session.pinned_tier == "mid"


async def test_model_unknown_tier_lists_the_known_ones(tmp_path: Path) -> None:
    context, _session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "model", "huge")

    text = printed_text(ui)
    assert "unknown" in text
    assert "small" in text and "mid" in text


async def test_model_without_args_shows_current_tier_and_pin(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    session.tier = "small"
    session.pinned_tier = "small"
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "model", "")

    text = printed_text(ui)
    assert "small" in text
    assert "pinned" in text
    assert "coder" in text


async def test_model_accepts_model_stem_and_filename(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "model", "coder.gguf")

    assert session.pinned_tier == "mid"
    assert session.tier == "mid"
    assert context.state.tier == "mid"
    assert context.state.model == "coder"


async def test_model_accepts_fuzzy_substring(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "model", "code")

    assert session.pinned_tier == "mid"
    assert context.state.model == "coder"


async def test_model_pin_updates_state_model_and_tier_immediately(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "model", "small")

    assert session.pinned_tier == "small"
    assert context.state.tier == "small"
    assert context.state.model == "small"



# --- /models, /plan, /stats, /router --------------------------------------------------


async def test_models_lists_profiles_with_ctx(tmp_path: Path) -> None:
    context, _session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "models", "")

    text = printed_text(ui)
    assert "coder:" in text and "ctx 32768" in text
    assert "small:" in text and "ctx 8192" in text


async def test_plan_reports_absence_then_renders_the_block(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "plan", "")
    assert "no active plan" in printed_text(ui)

    ui.prints.clear()
    plan = Plan(steps=[PlanStep(index=1, text="do the thing")])
    session.set_plan(plan)
    context.plan = lambda: plan

    await arun(ui, "plan", "")

    assert "○ 1. do the thing" in printed_text(ui)


async def test_stats_reports_turns_and_last_outcome(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    session.turn_count = 3
    session.tier = "mid"
    context.state.last_turn = {"outcome": "completed", "rounds": 4, "duration_s": 11.1}
    context.state.rounds = 1
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "stats", "")

    text = printed_text(ui)
    assert "turns 3" in text
    assert "completed" in text
    assert "4 rounds" in text
    assert "rounds 1/15" in text


async def test_router_explains_the_last_decision_or_says_none(tmp_path: Path) -> None:
    context, _session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "router", "")
    assert "no route decided yet" in printed_text(ui)

    ui.prints.clear()
    context.state.tier = "mid"
    context.state.reason = "needs_tools"
    context.state.intent = "code_gen"
    context.state.difficulty = 2.5
    context.state.confidence = 0.9
    context.state.backend = "fake"
    context.state.model = "coder7b"

    await arun(ui, "router", "")

    text = printed_text(ui)
    assert "needs_tools" in text
    assert "coder7b" in text
    assert "2.5" in text


# --- /sessions, /resume ----------------------------------------------------------------


async def test_sessions_lists_saved_ids_and_current_is_marked(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    context.store.append("other123", {"type": "meta", "session_id": "other123"})
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "sessions", "")

    text = printed_text(ui)
    assert session.id in text
    assert "other123" in text


async def test_sessions_empty_store_hints(tmp_path: Path) -> None:
    context, _session = make_context(tmp_path)
    ui = FakeUI()
    # a store that has never been written to
    context.store = SessionStore(tmp_path / "never")
    register_commands(ui, context)

    await arun(ui, "sessions", "")

    assert "no saved sessions" in printed_text(ui)


async def test_resume_hands_the_loaded_session_to_the_app(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    session.append_message(ChatMessage(role="user", content="remember me"))
    resumed: list[Session] = []
    context.resume = resumed.append  # type: ignore[assignment]
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "resume", session.id)

    assert len(resumed) == 1
    assert resumed[0].id == session.id
    assert [m.content for m in resumed[0].messages] == ["remember me"]
    assert "resumed" in printed_text(ui)


async def test_resume_unknown_id_is_an_error_not_a_crash(tmp_path: Path) -> None:
    context, _session = make_context(tmp_path)
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "resume", "doesnotexist")

    assert "unknown session" in printed_text(ui)


async def test_resume_without_args_takes_the_most_recent_file(tmp_path: Path) -> None:
    context, session = make_context(tmp_path)
    old_id = "aaaaaaaaaaaa"
    context.store.append(old_id, {"type": "meta", "session_id": old_id})
    os.utime(context.store.path(old_id), (1_000_000, 1_000_000))
    os.utime(context.store.path(session.id), (2_000_000, 2_000_000))
    resumed: list[Session] = []
    context.resume = resumed.append  # type: ignore[assignment]
    ui = FakeUI()
    register_commands(ui, context)

    await arun(ui, "resume", "")

    assert len(resumed) == 1
    assert resumed[0].id == session.id  # newest mtime wins, not lexicographic order
