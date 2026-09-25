"""CLI contract: the workbench entry point exposes version and help output."""

import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "workbench", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_version_flag_prints_package_version() -> None:
    result = run_cli("--version")

    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_help_flag_lists_subcommands_and_exits_zero() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    # Core subcommands promised by the plan must be discoverable from --help.
    for subcommand in ("models", "obs", "config"):
        assert subcommand in result.stdout


def test_config_show_reflects_environment_override(tmp_path) -> None:
    import json
    import os

    env = {**os.environ, "WB_METRICS_PORT": "9902"}
    result = subprocess.run(
        [sys.executable, "-m", "workbench", "config", "show", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["observability"]["metrics_port"] == 9902


def test_config_debug_lists_layers_and_active_env(tmp_path) -> None:
    import os

    env = {**os.environ, "WB_METRICS_PORT": "9601"}
    result = subprocess.run(
        [sys.executable, "-m", "workbench", "config", "debug"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    # Which files were consulted, and which WB_* vars are actually in effect.
    assert ".workbench.toml" in result.stdout
    assert "config.toml" in result.stdout
    assert "WB_METRICS_PORT=9601" in result.stdout
