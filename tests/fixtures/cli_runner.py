"""Run cli.main in a fresh interpreter so process-wide state stays out of the tests.

The offline guard mutates os.environ and huggingface_hub freezes its flags at
import time, so both are observed from inside the child (a printed state line
taken after main() returns) — never by calling main() in-process, where the
env/config/logging globals would leak into sibling tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

STATE_PREFIX = "WBSTATE "


def run_cli(
    argv: list[str],
    *,
    setup: str = "",
    cwd: Path | None = None,
) -> tuple[dict[str, object], subprocess.CompletedProcess[str]]:
    """Dispatch cli.main(argv) in a child process; return (state, result).

    `setup` runs before main() — tests use it to stub workbench.cli internals
    (e.g. run_download) so no network-capable code path can execute.
    """
    snippet = (
        "import json, os, sys\n"
        f"{setup}"
        "from workbench import cli\n"
        "try:\n"
        "    code = cli.main(sys.argv[1:])\n"
        "except SystemExit as exc:\n"
        "    code = int(exc.code or 0)\n"
        "import huggingface_hub.constants as hc\n"
        f"print({STATE_PREFIX!r} + json.dumps({{\n"
        "    'code': code,\n"
        "    'hf': os.environ.get('HF_HUB_OFFLINE'),\n"
        "    'tf': os.environ.get('TRANSFORMERS_OFFLINE'),\n"
        "    'hub': hc.HF_HUB_OFFLINE,\n"
        "}))\n"
    )
    # The child must start with a clean slate: an exported offline flag from the
    # developer shell would mask what the CLI itself decides to set.
    env = dict(os.environ)
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    result = subprocess.run(
        [sys.executable, "-c", snippet, *argv],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=cwd,
    )
    states = [line for line in result.stdout.splitlines() if line.startswith(STATE_PREFIX)]
    if not states:
        raise AssertionError(
            f"no {STATE_PREFIX.strip()} line\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    state = json.loads(states[-1][len(STATE_PREFIX) :])
    return state, result
