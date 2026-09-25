"""Command-line interface — the workbench's only interface."""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from workbench import __version__
from workbench.agent.loop import AgentLoop
from workbench.agent.planning import auto_approver
from workbench.config import (
    ENV_OVERRIDES,
    Config,
    ConfigError,
    layer_paths,
    load_config,
    set_config,
)
from workbench.core.bootstrap import build_registry, build_tools, resolve_router
from workbench.core.events import Event, EventBus
from workbench.core.protocols import ToolContext
from workbench.core.registry import RegistryError
from workbench.core.session import Session, SessionStore
from workbench.llm.client import LLMClientHttp
from workbench.llm.server import ServerError, ServerManager
from workbench.logging_setup import configure_logging
from workbench.models import DownloadError, Store, human_size
from workbench.models.download import run_download
from workbench.models.offline import ensure_offline
from workbench.observability import start_observability
from workbench.routing.policy import TierPolicy
from workbench.tools import headless_confirm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workbench",
        description=(
            "Local AI workbench: offline llama.cpp runtime, routed agent loop, sandboxed tools."
        ),
    )
    parser.add_argument("--version", action="version", version=f"workbench {__version__}")
    parser.add_argument(
        "--config", metavar="PATH", help="explicit config file (replaces the global file)"
    )
    parser.add_argument("--workspace", metavar="DIR", help="project workspace root")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose console logging")
    parser.add_argument(
        "-p",
        "--print",
        dest="print_prompt",
        metavar="PROMPT",
        help="run one task headlessly and print the final answer",
    )
    parser.add_argument(
        "--json",
        dest="json_stream",
        action="store_true",
        help="with -p: emit the agent event stream as NDJSON on stdout",
    )

    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser("config", help="inspect the effective configuration")
    config_sub = config_parser.add_subparsers(dest="config_command")
    show_parser = config_sub.add_parser("show", help="print the effective configuration")
    show_parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    config_sub.add_parser("debug", help="show which config layers and env vars are in effect")

    models_parser = subparsers.add_parser("models", help="manage local models")
    models_sub = models_parser.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="list configured model profiles")
    download_parser = models_sub.add_parser(
        "download", help="fetch a model into the store (the only command that uses the network)"
    )
    download_parser.add_argument(
        "spec",
        metavar="SPEC",
        help="hf:org/repo[/file.gguf], ollama:name[:tag], or a bare spec auto-detected by shape",
    )

    obs_parser = subparsers.add_parser(
        "obs", help="local observability stack (Prometheus + Grafana)"
    )
    obs_sub = obs_parser.add_subparsers(dest="obs_command")
    obs_sub.required = True  # a bare `obs` is a usage error, never a half-start (M7.6)
    obs_sub.add_parser("up", help="start Prometheus and Grafana")
    obs_sub.add_parser("down", help="stop Prometheus and Grafana")

    return parser


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _show_config(config: Config, as_json: bool) -> None:
    data = asdict(config)
    if as_json:
        print(json.dumps(data, indent=2))
        return
    for path, value in sorted(_flatten(data).items()):
        print(f"{path} = {value}")


def _list_models(config: Config) -> None:
    """Join profiles against routing tiers and the local store (PLAN §6.1)."""
    if not config.models.profiles:
        print("no model profiles configured (add [models.profiles.<name>] to .workbench.toml)")
        return
    tiers_by_profile: dict[str, list[str]] = {}
    for tier, profile_name in config.routing.tiers.items():
        tiers_by_profile.setdefault(profile_name, []).append(tier)
    entries = Store(config.models.search_paths).scan()
    by_path = {entry.path.resolve(): entry for entry in entries}
    referenced: set[Path] = set()
    for name, profile in sorted(config.models.profiles.items()):
        tiers = ",".join(sorted(tiers_by_profile.get(name, []))) or "-"
        entry = by_path.get(Path(profile.path).expanduser().resolve()) if profile.path else None
        if entry is None:
            note = "not in store"
        else:
            referenced.add(entry.path.resolve())
            quant = f" {entry.quant}" if entry.quant else ""
            note = f"store {entry.family}{quant} {human_size(entry.size)}"
        print(
            f"{name}: {profile.path} (ctx {profile.ctx_len}, "
            f"tool_calling {profile.tool_calling}, tier {tiers}, {note})"
        )
    for entry in entries:
        if entry.path.resolve() not in referenced:
            quant = f", {entry.quant}" if entry.quant else ""
            detail = f"({entry.family}{quant}, {human_size(entry.size)})"
            print(f"store: {entry.path.name} {detail} at {entry.path}")


def _download_models(config: Config, spec: str) -> int:
    """`models download` — fetch, checksum into the store, warm the laya checkpoint."""
    # Big pulls are silent until done (T8.3); announce the start so it never looks hung.
    print(f"downloading {spec} ...")
    try:
        report = run_download(config, spec)
    except (DownloadError, RegistryError) as error:
        print(f"workbench: {error}", file=sys.stderr)
        return 1
    sha = report.entry.sha or "unknown"
    print(f"downloaded {report.path} ({human_size(report.entry.size)}, sha {sha[:12]})")
    print(f"laya checkpoint: {report.laya}")
    return 0


def _debug_config() -> None:
    """Show the layering inputs so 'which file won?' is answerable at a glance."""
    for path in layer_paths():
        state = "loaded" if path.is_file() else "missing"
        print(f"file: {path} ({state})")
    active = [f"{name}={os.environ[name]}" for name in sorted(ENV_OVERRIDES) if name in os.environ]
    print(f"env: {', '.join(active) if active else '(no WB_* overrides)'}")


def _run_print(config: Config, prompt: str, *, json_stream: bool) -> int:
    """Headless entry: one task against the full stack, answer on stdout."""
    return asyncio.run(_print_session(config, prompt, json_stream=json_stream))


def _print_event(event: Event) -> None:
    """NDJSON line per event — the machine interface for scripts and e2e tests."""
    print(
        json.dumps(
            {"kind": event.kind, "ts": event.ts, "data": dict(event.data)},
            ensure_ascii=False,
        ),
        flush=True,
    )


async def _print_session(config: Config, prompt: str, *, json_stream: bool) -> int:
    """Own the run's observability stack: metrics live exactly as long as -p (M7.4)."""
    bus = EventBus()
    if json_stream:
        bus.subscribe(None, _print_event)
    observability = start_observability(config, bus)
    try:
        return await _run_session(config, bus, prompt, json_stream=json_stream)
    finally:
        observability.stop()


async def _run_session(
    config: Config, bus: EventBus, prompt: str, *, json_stream: bool
) -> int:
    """Wire registry/router/policy/loop/server, run one turn, tear the server down."""
    registry = build_registry(config)
    workspace = Path(config.runtime.workspace_root).expanduser()
    session = Session.create(
        store=SessionStore(Path(config.agent.session_dir).expanduser()),
        workspace=str(workspace),
    )
    manager = ServerManager(config, bus)
    loop = AgentLoop(
        client=LLMClientHttp(manager.base_url),
        router=resolve_router(registry, config),
        policy=TierPolicy(config.routing),
        tools=build_tools(registry),
        session=session,
        bus=bus,
        config=config,
        context=ToolContext(workspace_root=workspace, confirm=headless_confirm()),
        approver=auto_approver,
    )
    bus.emit(Event("agent_start", {"session_id": session.id, "mode": "print"}))
    try:
        await manager.ensure_running()
    except ServerError as error:
        bus.emit(Event("error", {"stage": "server", "message": str(error)}))
        bus.emit(Event("agent_end", {"outcome": "error"}))
        print(f"workbench: server failed: {error}", file=sys.stderr)
        return 1
    try:
        outcome = await loop.run_turn(prompt)
    finally:
        await manager.stop(reason="print finished")
    bus.emit(Event("agent_end", {"outcome": outcome.outcome, "rounds": outcome.rounds}))
    if not json_stream:
        print(outcome.text)
    return 1 if outcome.outcome == "error" else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.json_stream and not args.print_prompt:
        parser.error("--json requires -p")
    if args.print_prompt and args.command is not None:
        parser.error("-p cannot be combined with subcommands")

    cli_overrides: dict[str, str] = {}
    if getattr(args, "verbose", False):
        cli_overrides["runtime.verbose"] = "true"
    if getattr(args, "workspace", None):
        cli_overrides["runtime.workspace_root"] = args.workspace

    try:
        config = load_config(
            config_path=Path(args.config) if args.config else None,
            cli_overrides=cli_overrides,
        )
    except ConfigError as exc:
        print(f"workbench: config error: {exc}", file=sys.stderr)
        return 2

    set_config(config)
    configure_logging(
        log_dir=Path(config.observability.log_dir).expanduser(),
        level=config.observability.log_level,
        console=args.verbose,
    )
    if not (args.command == "models" and args.models_command == "download"):
        # Constraint 1: everything but `models download` runs offline; the hub
        # freezes its flags at import, so this must precede any hub usage.
        ensure_offline()

    if args.print_prompt:
        return _run_print(config, args.print_prompt, json_stream=args.json_stream)

    if args.command is None:
        # Bare invocation starts the interactive TUI (ruling M6.11); imported
        # here so -p/config/models paths never pay for agentui + prompt_toolkit.
        from workbench.tui import run_tui

        return run_tui(config)

    if args.command == "config":
        if args.config_command == "show":
            _show_config(config, as_json=getattr(args, "json", False))
            return 0
        if args.config_command == "debug":
            _debug_config()
            return 0
    if args.command == "models":
        if args.models_command == "list":
            _list_models(config)
            return 0
        if args.models_command == "download":
            return _download_models(config, args.spec)
    if args.command == "obs":
        from workbench.observability.stack import ObsError, run_stack

        try:
            return run_stack(args.obs_command)
        except ObsError as error:
            print(f"workbench: {error}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
