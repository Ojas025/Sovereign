"""Central configuration: layered TOML merged into one frozen model.

Precedence (low -> high): built-in defaults, global file, project file,
``WB_*`` environment variables, CLI flags. Unknown keys and type mismatches
are hard errors: a typo must never be silently ignored.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

ENV_OVERRIDES = {
    "WB_VERBOSE": "runtime.verbose",
    "WB_WORKSPACE": "runtime.workspace_root",
    "WB_ROUTING_BACKEND": "routing.backend",
    "WB_SERVER_BINARY": "server.binary",
    "WB_IDLE_TIMEOUT_S": "server.idle_timeout_s",
    "WB_MAX_ROUNDS": "agent.max_rounds",
    "WB_SANDBOX_BACKEND": "sandbox.backend",
    "WB_METRICS_PORT": "observability.metrics_port",
    "WB_LOG_LEVEL": "observability.log_level",
    "WB_LOG_DIR": "observability.log_dir",
}

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


class ConfigError(Exception):
    """Unknown key, type mismatch, unreadable file, or bad override value."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    verbose: bool = False
    workspace_root: str = "."


@dataclass(frozen=True, slots=True)
class TierPolicyConfig:
    high_threshold: float = 2.0
    confidence_gate: float = 0.6
    default_tier: str = "mid"
    hysteresis_delta: float = 0.3
    escalate_after_failures: int = 2


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    backend: str = "laya"
    checkpoint: str = "laya"
    tiers: dict[str, str] = field(default_factory=lambda: {"small": "small", "mid": "mid"})
    policy: TierPolicyConfig = field(default_factory=TierPolicyConfig)
    intent_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # a typo'd tier name would silently never fire — reject it at load time
        for intent, tier in self.intent_overrides.items():
            if tier not in self.tiers:
                raise ConfigError(
                    f"routing.intent_overrides.{intent}: unknown tier {tier!r} "
                    f"(known tiers: {', '.join(self.tiers)})"
                )


@dataclass(frozen=True, slots=True)
class ModelProfile:
    path: str = ""
    ctx_len: int = 8192
    n_gpu_layers: int | str = "auto"
    tool_calling: str = "auto"
    reasoning_effort: str = ""


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    profiles: dict[str, ModelProfile] = field(default_factory=dict)
    search_paths: tuple[str, ...] = ("~/models", "~/.local/share/workbench/models")


@dataclass(frozen=True, slots=True)
class ServerConfig:
    binary: str = "llama-serve"
    host: str = "127.0.0.1"
    port: int = 8080
    idle_timeout_s: float = 300.0
    restart_max: int = 3
    models_max: int = 1  # llama-server defaults to 4; single-active GPU budget


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_rounds: int = 15
    wall_clock_s: float = 600.0
    reflection_nudge_cap: int = 3
    plan_max_retries: int = 2
    token_cap_per_turn: int = 16384


@dataclass(frozen=True, slots=True)
class ToolsConfig:
    """Caps for the file tools (plan §5.2) plus declared extra read roots (§5.3.1)."""

    read_max_bytes: int = 65_536  # one read/edit call never loads more than this
    write_max_bytes: int = 1_048_576  # a single write stays well below context size
    read_roots: tuple[str, ...] = ()  # extra roots readable from outside the workspace


_SANDBOX_MODES = ("confirm", "auto", "strict")


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    backend: str = "bwrap"
    bash_timeout_s: float = 60.0
    output_max_bytes: int = 20000
    mode: str = "confirm"
    allowlist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # a typo'd mode would silently change the confirmation contract — reject at load
        if self.mode not in _SANDBOX_MODES:
            raise ConfigError(
                f"sandbox.mode: unknown mode {self.mode!r} (known: {', '.join(_SANDBOX_MODES)})"
            )


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    metrics_port: int = 9600
    log_level: str = "INFO"
    log_dir: str = "~/.local/state/workbench/logs"


@dataclass(frozen=True, slots=True)
class Config:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)


def _build_section(section_type: type, data: Mapping[str, object], path: str) -> object:
    """Validate a config table against its dataclass, raising ConfigError on any mismatch."""
    hints = get_type_hints(section_type)
    known = {f.name for f in fields(section_type)}
    kwargs: dict[str, object] = {}
    for key, value in data.items():
        key_path = f"{path}.{key}" if path else str(key)
        if key not in known:
            raise ConfigError(f"unknown config key: {key_path}")
        kwargs[key] = _coerce(hints[key], value, key_path)
    return section_type(**kwargs)


def _coerce(hint: object, value: object, path: str) -> object:
    origin = get_origin(hint)

    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a table, got {value!r}")
        _, item_hint = get_args(hint)
        if isinstance(item_hint, type) and is_dataclass(item_hint):
            return {
                str(key): _build_section(item_hint, item, f"{path}.{key}")
                for key, item in value.items()
            }
        for key, item in value.items():
            if not isinstance(item, item_hint):
                raise ConfigError(f"{path}.{key}: expected {item_hint}, got {item!r}")
        return dict(value)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected an array, got {value!r}")
        item_hint = get_args(hint)[0]
        for item in value:
            if not isinstance(item, item_hint):
                raise ConfigError(f"{path}: expected {item_hint} items, got {item!r}")
        return tuple(value)

    if origin in (UnionType, Union):
        if any(_matches(arg, value) for arg in get_args(hint)):
            return value
        raise ConfigError(f"{path}: expected {hint}, got {value!r}")

    if isinstance(hint, type) and is_dataclass(hint):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a table, got {value!r}")
        return _build_section(hint, value, path)

    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    if hint in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected {hint}, got {value!r}")
        return float(value) if hint is float else value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {value!r}")
        return value
    return value


def _matches(hint: object, value: object) -> bool:
    return isinstance(value, hint) if isinstance(hint, type) else False


def _coerce_scalar(raw: str, default: object, path: str) -> object:
    """Convert a raw env/CLI string to the type of the default at that path."""
    if isinstance(default, bool):
        lowered = raw.lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
        raise ConfigError(f"{path}: expected a boolean, got {raw!r}")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"{path}: expected an integer, got {raw!r}") from exc
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"{path}: expected a number, got {raw!r}") from exc
    return raw


def _default_at(base: Mapping[str, object], path: str) -> object:
    node: object = base
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"unknown override: {path}")
        node = node[part]
    return node


def _set_dotted(target: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    node = target
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        elif not isinstance(child, dict):
            raise ConfigError(f"unknown override: {path}")
        node = child
    node[parts[-1]] = value


def _deep_merge(base: dict[str, object], overlay: Mapping[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, object]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def layer_paths(
    *,
    home: Path | None = None,
    project_dir: Path | None = None,
    config_path: Path | None = None,
) -> list[Path]:
    """Config file layers in precedence order (global file first, project file last)."""
    home = home if home is not None else Path.home()
    project_dir = project_dir if project_dir is not None else Path.cwd()
    global_path = (
        config_path if config_path is not None else home / ".config" / "workbench" / "config.toml"
    )
    return [global_path, project_dir / ".workbench.toml"]


def load_config(
    *,
    home: Path | None = None,
    project_dir: Path | None = None,
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, str] | None = None,
) -> Config:
    """Merge every configuration layer in precedence order and validate the result."""
    home = home if home is not None else Path.home()
    project_dir = project_dir if project_dir is not None else Path.cwd()
    env_map = env if env is not None else os.environ

    base = asdict(Config())
    merged = base

    for layer_path in layer_paths(home=home, project_dir=project_dir, config_path=config_path):
        if layer_path.is_file():
            merged = _deep_merge(merged, _read_toml(layer_path))

    overrides: dict[str, object] = {}
    for var, dotted in ENV_OVERRIDES.items():
        if var in env_map:
            default = _default_at(base, dotted)
            _set_dotted(overrides, dotted, _coerce_scalar(env_map[var], default, dotted))
    for dotted, raw in (cli_overrides or {}).items():
        default = _default_at(base, dotted)
        _set_dotted(overrides, dotted, _coerce_scalar(raw, default, dotted))
    merged = _deep_merge(merged, overrides)

    return _build_section(Config, merged, "")  # type: ignore[return-value]


_config: Config | None = None


def set_config(config: Config) -> None:
    """Install the process-wide configuration (called once at startup)."""
    global _config
    _config = config


def get_config() -> Config:
    """Return the installed configuration; fail loudly if startup wiring was skipped."""
    if _config is None:
        raise ConfigError("configuration not loaded; call set_config(load_config()) first")
    return _config
