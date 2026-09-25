"""Core wiring: a fresh registry with entry-point plugins plus core defaults.

Order matters: entry-point plugins load first so config-selected backends they
provide resolve normally, then core registers its fixed tool set — a plugin
claiming a core name fails loudly instead of shadowing it.
"""

from __future__ import annotations

from pathlib import Path

from workbench.config import Config, ConfigError
from workbench.core.protocols import Router, Sandbox, Tool
from workbench.core.registry import Registry, RegistryError
from workbench.routing.heuristic import HeuristicRouter
from workbench.routing.laya_router import LayaRouter
from workbench.sandbox.bwrap import BwrapSandbox
from workbench.tools.bash import BashTool
from workbench.tools.files import EditTool, ReadTool, WriteTool

_CORE_SANDBOX = "bwrap"


def build_registry(config: Config) -> Registry:
    """Build the process registry: plugins first, then core tools, sandbox, routers."""
    registry = Registry()
    registry.load_entry_points()

    registry.register("routers", "heuristic", HeuristicRouter())
    registry.register("routers", "laya", LayaRouter(HeuristicRouter(), config.routing))
    registry.register("sandboxes", _CORE_SANDBOX, _core_sandbox(config))
    sandbox = _resolve_sandbox(registry, config)
    tools: tuple[Tool, ...] = (
        ReadTool(config.tools),
        WriteTool(config.tools),
        EditTool(config.tools),
        BashTool(sandbox, config.sandbox),
    )
    for tool in tools:
        registry.register("tools", tool.name, tool)
    return registry


def _core_sandbox(config: Config) -> Sandbox:
    # workspace path comes from config (the CLI resolves --workspace at startup);
    # file tools get theirs per call from ToolContext.
    return BwrapSandbox(Path(config.runtime.workspace_root).expanduser())


def resolve_router(registry: Registry, config: Config) -> Router:
    """Look up the configured classification backend (mirrors sandbox resolution)."""
    name = config.routing.backend
    try:
        router = registry.get("routers", name)
    except RegistryError as exc:
        known = ", ".join(registry.entries("routers")) or "(none)"
        raise ConfigError(
            f"routing.backend: unknown router {name!r} (registered: {known})"
        ) from exc
    if not isinstance(router, Router):
        raise ConfigError(f"routing.backend: {name!r} does not implement the Router protocol")
    return router


def _resolve_sandbox(registry: Registry, config: Config) -> Sandbox:
    name = config.sandbox.backend
    try:
        sandbox = registry.get("sandboxes", name)
    except RegistryError as exc:
        known = ", ".join(registry.entries("sandboxes")) or "(none)"
        raise ConfigError(
            f"sandbox.backend: unknown sandbox backend {name!r} (registered: {known})"
        ) from exc
    if not isinstance(sandbox, Sandbox):
        raise ConfigError(f"sandbox.backend: {name!r} does not implement the Sandbox protocol")
    return sandbox
