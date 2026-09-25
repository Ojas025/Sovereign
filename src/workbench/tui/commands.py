"""Slash commands: the fixed workbench vocabulary plus plugin commands.

Built-ins register first so a ``workbench.commands`` plugin can never shadow
one of them (ruling M6.9) — a clash logs a warning and keeps the built-in.
Every handler is a plain async ``(ui, args)`` closure over a
:class:`CommandContext` the app builds, so tests drive them with fakes.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentui import Session

from workbench.agent.loop import resolve_model
from workbench.config import Config
from workbench.core.registry import Registry
from workbench.core.session import Plan, SessionStore
from workbench.core.session import Session as SessionState
from workbench.tui.plan import render_plan
from workbench.tui.state import TuiState

logger = logging.getLogger("workbench.tui")

CommandHandler = Callable[[Any, str], Awaitable[None]]

# agentui's own builtins (session._register_builtins): plugins may not shadow them either.
_UI_BUILTINS = frozenset({"help", "quit", "exit", "clear"})


@dataclass
class CommandContext:
    """Everything the built-in commands need; the app builds it, tests fake the parts."""

    config: Config
    state: TuiState
    store: SessionStore
    registry: Registry
    session: Callable[[], SessionState]  # current session (swapped by /resume)
    plan: Callable[[], Plan | None]  # transcript's live plan, else the session's
    resume: Callable[[SessionState], None]  # hand a loaded session back to the app


def register_commands(ui: Session, ctx: CommandContext) -> None:
    """Register built-in slash commands, then plugin commands."""
    for name, handler, help_text in _builtins(ctx):
        ui.register_command(name, handler, help_text=help_text)
    _register_plugins(ui, ctx.registry)


def _builtins(ctx: CommandContext) -> list[tuple[str, CommandHandler, str]]:
    async def model(ui: Session, args: str) -> None:
        tiers = ctx.config.routing.tiers
        session = ctx.session()
        name = args.strip()
        if not name:
            pin = f" (pinned: {session.pinned_tier})" if session.pinned_tier else ""
            active_tier = session.tier or "-"
            active_model = ctx.state.model or (
                resolve_model(ctx.config, session.tier) if session.tier else "-"
            )
            known_tiers = ", ".join(sorted(tiers))
            avail: list[str] = ["auto"]
            for prof_name, profile in sorted(ctx.config.models.profiles.items()):
                stem = Path(profile.path).stem if profile.path else prof_name
                mapped_tier = next((t for t, p in tiers.items() if p == prof_name), None)
                tier_info = f" [{mapped_tier}]" if mapped_tier else ""
                avail.append(f"{prof_name} ({stem}){tier_info}")
            ui.print(
                f"[dim]tier: {active_tier}{pin} · model: {active_model} · "
                f"available: {', '.join(avail)} (tiers: {known_tiers})[/dim]"
            )
            return
        if name == "auto":
            session.pinned_tier = None
            ui.print("[dim]pin cleared — routing is automatic again[/dim]")
            return
        target = _resolve_target(ctx.config, name)
        if target is None:
            profiles = ", ".join(sorted(ctx.config.models.profiles)) or "(none)"
            stems = ", ".join(
                sorted(
                    Path(p.path).stem
                    for p in ctx.config.models.profiles.values()
                    if p.path
                )
            )
            stem_info = f" · models: {stems}" if stems else ""
            ui.print(
                f"[red]unknown model {name!r}[/red] [dim]— tiers: "
                f"{', '.join(sorted(tiers))} · profiles: {profiles}{stem_info} · "
                f"/model auto clears[/dim]"
            )
            return
        tier, profile_name, model_stem = target
        session.pinned_tier = tier
        session.tier = tier
        ctx.state.tier = tier
        ctx.state.model = model_stem
        ui.print(
            f"[yellow]pinned[/yellow] [dim]{tier} → model {model_stem} "
            f"(profile: {profile_name}) (/model auto to clear)[/dim]"
        )

    async def models(ui: Session, args: str) -> None:
        profiles = ctx.config.models.profiles
        if not profiles:
            ui.print(
                "[dim]no model profiles configured "
                "(add [models.profiles.<name>] to .workbench.toml)[/dim]"
            )
            return
        for name, profile in sorted(profiles.items()):
            ui.print(
                f"[dim]{name}: {profile.path or '(no path)'} "
                f"(ctx {profile.ctx_len}, tool_calling {profile.tool_calling})[/dim]"
            )

    async def plan(ui: Session, args: str) -> None:
        current = ctx.plan()
        if current is None:
            ui.print("[dim]no active plan[/dim]")
            return
        ui.print(render_plan(current))

    async def stats(ui: Session, args: str) -> None:
        session = ctx.session()
        state = ctx.state
        pin = " (pinned)" if session.pinned_tier else ""
        ui.print(
            f"[dim]session {session.id} · turns {session.turn_count} · "
            f"tier {session.tier or '-'}{pin}[/dim]"
        )
        last = state.last_turn
        if last:
            duration = last.get("duration_s")
            duration_text = f"{float(duration):.1f}s" if isinstance(duration, (int, float)) else "—"
            ui.print(
                f"[dim]last turn: {last.get('outcome', '-')} · "
                f"{last.get('rounds', 0)} rounds · {duration_text}[/dim]"
            )
        ttft = state.ttft_s
        ttft_text = f"{ttft:.2f}s" if ttft is not None else "—"
        ui.print(
            f"[dim]this turn: rounds {state.rounds}/{ctx.config.agent.max_rounds} · "
            f"tok {state.tokens} · ttft {ttft_text}[/dim]"
        )

    async def router(ui: Session, args: str) -> None:
        state = ctx.state
        if not state.tier and not state.reason:
            ui.print("[dim]no route decided yet[/dim]")
            return
        ui.print(
            f"[dim]tier: {state.tier or '-'} · reason: {state.reason or '-'} · "
            f"intent: {state.intent or '-'} · difficulty: {state.difficulty:.2f} · "
            f"confidence: {state.confidence:.2f} · backend: {state.backend or '-'} · "
            f"model: {state.model or '-'} · suggest_plan: {state.suggest_plan}[/dim]"
        )

    async def sessions(ui: Session, args: str) -> None:
        ids = ctx.store.list_ids()
        if not ids:
            ui.print("[dim]no saved sessions[/dim]")
            return
        current = ctx.session().id
        for session_id in ids:
            marker = " *" if session_id == current else ""
            ui.print(f"[dim]{session_id}{marker}[/dim]")

    async def resume(ui: Session, args: str) -> None:
        ids = ctx.store.list_ids()
        name = args.strip()
        if not ids:
            ui.print("[dim]no saved sessions to resume[/dim]")
            return
        if name:
            if name not in ids:
                ui.print(
                    f"[red]unknown session {name!r}[/red] [dim]— /sessions lists saved ids[/dim]"
                )
                return
        else:
            # most recently touched file, not lexicographically last id
            name = max(ids, key=lambda sid: ctx.store.path(sid).stat().st_mtime)
        loaded = ctx.store.load(name)
        ctx.resume(loaded)
        ui.print(f"[dim]resumed session {loaded.id} · {len(loaded.messages)} messages[/dim]")

    return [
        ("model", model, "Show or pin the active model (/model <name> or /model auto)"),
        ("models", models, "List configured model profiles"),
        ("plan", plan, "Show the current plan and step statuses"),
        ("stats", stats, "Session and turn statistics"),
        ("router", router, "Explain the last routing decision"),
        ("sessions", sessions, "List saved sessions"),
        ("resume", resume, "Resume a saved session (/resume [id])"),
    ]


def _resolve_target(config: Config, name: str) -> tuple[str, str, str] | None:
    """Resolve a tier, profile, model stem, or unique substring to (tier, profile, stem)."""
    q = name.strip()
    if not q:
        return None
    tiers = config.routing.tiers
    profiles = config.models.profiles

    # 1. Exact tier match
    if q.lower() in tiers:
        tier = q.lower()
        prof = tiers[tier]
        stem = resolve_model(config, tier)
        return (tier, prof, stem)

    # 2. Exact profile match
    for prof_name, profile in profiles.items():
        if q.lower() == prof_name.lower():
            tier = next((t for t, p in tiers.items() if p == prof_name), prof_name)
            stem = Path(profile.path).stem if profile.path else prof_name
            return (tier, prof_name, stem)

    # 3. Exact model stem match (with or without .gguf)
    q_stem = q[:-5] if q.lower().endswith(".gguf") else q
    for prof_name, profile in profiles.items():
        stem = Path(profile.path).stem if profile.path else prof_name
        if q_stem.lower() == stem.lower():
            tier = next((t for t, p in tiers.items() if p == prof_name), prof_name)
            return (tier, prof_name, stem)

    # 4. Fuzzy/substring match across profile names and model stems
    matches: list[tuple[str, str, str]] = []
    for prof_name, profile in profiles.items():
        stem = Path(profile.path).stem if profile.path else prof_name
        tier = next((t for t, p in tiers.items() if p == prof_name), prof_name)
        if q.lower() in prof_name.lower() or (profile.path and q.lower() in stem.lower()):
            matches.append((tier, prof_name, stem))
    if len(matches) == 1:
        return matches[0]

    return None


def _resolve_tier(config: Config, name: str) -> str | None:
    """Accept a tier key, profile name, or model stem, returning the tier name."""
    target = _resolve_target(config, name)
    return target[0] if target is not None else None


def _register_plugins(ui: Session, registry: Registry) -> None:
    """workbench.commands entries: callable only, never over a built-in name."""
    for name in registry.entries("commands"):
        plugin = registry.get("commands", name)
        if not callable(plugin):
            logger.warning("command plugin %r is not callable; skipped", name)
            continue
        if name in _BUILTIN_COMMANDS or name in _UI_BUILTINS:
            logger.warning(
                "command plugin %r clashes with built-in /%s; keeping the built-in",
                name,
                name,
            )
            continue
        ui.register_command(name, plugin, help_text="plugin command")


_BUILTIN_COMMANDS = frozenset({"model", "models", "plan", "stats", "router", "sessions", "resume"})
