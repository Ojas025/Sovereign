"""T7.3 — local observability stack: compose, provisioning, dashboards, obs cmd (M7.6)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
COMPOSE_FILE = REPO_ROOT / "docker" / "observability" / "docker-compose.yml"
PROMETHEUS_CONFIG = REPO_ROOT / "docker" / "observability" / "prometheus.yml"
GRAFANA_PROVISIONING = REPO_ROOT / "docker" / "grafana" / "provisioning"
DASHBOARDS_DIR = REPO_ROOT / "docker" / "grafana" / "dashboards"

_EXPECTED_DASHBOARD_TITLES = {"Overview", "Routing", "Agent & Tools", "Server"}


def run_cli(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "workbench", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def fake_docker_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """PATH-shadowed docker that records its argv; no containers are touched."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "docker"
    script.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$@" > "$FAKE_DOCKER_ARGS"\n'
        'exit "${FAKE_DOCKER_RC:-0}"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_DOCKER_ARGS"] = str(tmp_path / "docker_args.txt")
    return env, Path(env["FAKE_DOCKER_ARGS"])


def test_compose_file_resolves_from_the_package_location() -> None:
    from workbench.observability.stack import compose_file

    resolved = compose_file()
    assert resolved == COMPOSE_FILE
    assert resolved.is_file()


def test_compose_command_builds_the_expected_argv() -> None:
    from workbench.observability.stack import compose_command

    prefix = ["docker", "compose", "-f", str(COMPOSE_FILE)]
    assert compose_command("up") == [*prefix, "up", "-d"]
    assert compose_command("down") == [*prefix, "down"]


def test_docker_compose_config_parses_cleanly() -> None:
    """PLAN §9 CI-safe validity check: the compose file must resolve."""
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "prometheus" in result.stdout and "grafana" in result.stdout
    # M7.7 (VAL finding): loopback-bound host services are only reachable with
    # host networking — both services must run in it for scrapes to land.
    assert result.stdout.count("network_mode: host") == 2


def test_prometheus_scrapes_workbench_and_llama_server() -> None:
    text = PROMETHEUS_CONFIG.read_text(encoding="utf-8")
    assert "scrape_interval" in text
    assert "127.0.0.1:9600" in text  # our loopback metrics port (host mode, M7.7)
    assert "127.0.0.1:8080" in text  # llama-serve's own /metrics


def test_grafana_provisioning_wires_datasource_and_dashboards() -> None:
    datasource = (
        GRAFANA_PROVISIONING / "datasources" / "prometheus.yml"
    ).read_text(encoding="utf-8")
    assert "http://127.0.0.1:9090" in datasource  # host networking (M7.7)
    provider = (GRAFANA_PROVISIONING / "dashboards" / "dashboards.yml").read_text(
        encoding="utf-8"
    )
    assert "/var/lib/grafana/dashboards" in provider


def test_dashboards_are_valid_json_and_query_real_metrics() -> None:
    """Dashboards as JSON code (PLAN §8.4) — parse them and check the queries."""
    files = sorted(DASHBOARDS_DIR.glob("*.json"))
    titles: set[str] = set()
    expressions: list[str] = []
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        titles.add(str(document["title"]))
        for panel in document.get("panels", []):
            for target in panel.get("targets", []):
                if "expr" in target:
                    expressions.append(str(target["expr"]))
    assert titles == _EXPECTED_DASHBOARD_TITLES, titles
    joined = "\n".join(expressions)
    for metric in (
        "route_decisions_total",
        "llm_ttft_seconds",
        "agent_tasks_total",
        "tool_calls_total",
        "server_active",
    ):
        assert metric in joined, f"{metric} never queried by any dashboard"


def test_obs_up_and_down_shell_docker_compose(tmp_path: Path) -> None:
    env, args_file = fake_docker_env(tmp_path)

    up = run_cli(env, "obs", "up")
    assert up.returncode == 0, up.stderr
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "up",
        "-d",
    ]

    down = run_cli(env, "obs", "down")
    assert down.returncode == 0, down.stderr
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "down",
    ]


def test_obs_propagates_docker_failure(tmp_path: Path) -> None:
    env, _ = fake_docker_env(tmp_path)
    env["FAKE_DOCKER_RC"] = "3"

    result = run_cli(env, "obs", "up")

    assert result.returncode == 3  # docker's exit code reaches scripts intact


def test_obs_without_a_subcommand_is_a_usage_error(tmp_path: Path) -> None:
    env, args_file = fake_docker_env(tmp_path)

    result = run_cli(env, "obs")

    assert result.returncode == 2  # argparse, never a half-started stack
    assert not args_file.exists()
