"""Drive the local Prometheus + Grafana compose stack (PLAN §8.4, ruling M7.6).

A dev/ops convenience bound to this checkout: paths resolve from the package
location (not the CWD), docker compose talks straight to the terminal, and
nothing reaches the network beyond the image pulls the user asked for.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from workbench.logging_setup import get_logger

logger = get_logger("observability")

# src/workbench/observability/stack.py -> repo root holds docker/ (checkout-only).
_COMPOSE_RELATIVE = Path("docker/observability/docker-compose.yml")

ComposeAction = Literal["up", "down"]


class ObsError(RuntimeError):
    """User-facing stack failure: missing checkout or missing docker."""


def compose_file() -> Path:
    """Resolve the compose file from the package location (checkout-only, M7.6)."""
    candidate = Path(__file__).resolve().parents[3] / _COMPOSE_RELATIVE
    if not candidate.is_file():
        raise ObsError(
            f"compose file not found at {candidate} — 'workbench obs' runs from a "
            "source checkout (docker/observability/docker-compose.yml)"
        )
    return candidate


def compose_command(action: ComposeAction) -> list[str]:
    """argv for docker compose: detached up, plain down."""
    tail = ["up", "-d"] if action == "up" else ["down"]
    return ["docker", "compose", "-f", str(compose_file()), *tail]


def run_stack(action: ComposeAction) -> int:
    """Run the compose action with inherited stdio; return docker's exit code."""
    command = compose_command(action)
    logger.info("running %s", " ".join(command))
    try:
        return subprocess.run(command).returncode
    except FileNotFoundError as exc:
        raise ObsError("docker does not seem to be installed (or is not on PATH)") from exc
