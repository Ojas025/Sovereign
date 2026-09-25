"""Command-line interface — the workbench's only interface."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from workbench import __version__
from workbench.config import (
    ENV_OVERRIDES,
    Config,
    ConfigError,
    layer_paths,
    load_config,
    set_config,
)
from workbench.logging_setup import configure_logging


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

    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser("config", help="inspect the effective configuration")
    config_sub = config_parser.add_subparsers(dest="config_command")
    show_parser = config_sub.add_parser("show", help="print the effective configuration")
    show_parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    config_sub.add_parser("debug", help="show which config layers and env vars are in effect")

    models_parser = subparsers.add_parser("models", help="manage local models")
    models_sub = models_parser.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="list configured model profiles")

    obs_parser = subparsers.add_parser(
        "obs", help="local observability stack (Prometheus + Grafana)"
    )
    obs_sub = obs_parser.add_subparsers(dest="obs_command")
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
    if not config.models.profiles:
        print("no model profiles configured (add [models.profiles.<name>] to .workbench.toml)")
        return
    for name, profile in sorted(config.models.profiles.items()):
        print(
            f"{name}: {profile.path} (ctx {profile.ctx_len}, tool_calling {profile.tool_calling})"
        )


def _debug_config() -> None:
    """Show the layering inputs so 'which file won?' is answerable at a glance."""
    for path in layer_paths():
        state = "loaded" if path.is_file() else "missing"
        print(f"file: {path} ({state})")
    active = [f"{name}={os.environ[name]}" for name in sorted(ENV_OVERRIDES) if name in os.environ]
    print(f"env: {', '.join(active) if active else '(no WB_* overrides)'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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

    if args.command is None:
        parser.print_help()
        return 0

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
    if args.command == "obs":
        print(
            f"workbench obs {args.obs_command}: not available yet (see PLAN.md, milestone 7)",
            file=sys.stderr,
        )
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
