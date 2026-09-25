"""bubblewrap sandbox: one-shot namespace jail per command (plan §5.3.2).

Guarantees (verified against real bwrap):
- whole host filesystem visible but read-only; workspace bound rw on top
  (workspace mounts must come *after* the private /tmp so a workspace living
  under /tmp — e.g. pytest tmp dirs — is not shadowed by it)
- ``--unshare-net``: no network at the OS level (offline enforced, plan §1)
- private /tmp, fresh proc/dev, cleared environment (secret-free both before
  and inside bwrap), all capabilities dropped, ``--die-with-parent`` so no
  command outlives the workbench
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from workbench.core.protocols import SandboxResult

_TIMEOUT_EXIT_CODE = 124  # GNU timeout convention
_WAIT_AFTER_KILL_S = 5.0

# Minimal secret-free environment: everything else (tokens, WB_*, DBus, X11) is dropped.
_SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "TZ")
_DEFAULT_PATH = "/usr/bin:/bin"


def _scrubbed_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _SAFE_ENV_KEYS if key in os.environ}
    env.setdefault("PATH", _DEFAULT_PATH)
    return env


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Kill the sandbox and every process inside it (start_new_session = own group)."""
    try:
        if sys.platform == "win32":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # exited between the timeout and the kill
    except PermissionError:
        proc.kill()
    proc.wait(timeout=_WAIT_AFTER_KILL_S)


class BwrapSandbox:
    """Executes commands inside bubblewrap with the isolation profile from §5.3.2."""

    def __init__(self, workspace_root: Path, binary: str = "bwrap") -> None:
        self._workspace = workspace_root.expanduser().resolve()
        self._binary = binary

    def run(self, command: str, timeout_s: float) -> SandboxResult:
        env = _scrubbed_env()  # bwrap itself never sees secrets either
        proc = subprocess.Popen(
            self._argv(command, env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,  # own process group → whole jail dies on timeout
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            stdout, stderr = proc.communicate()  # retry keeps output produced so far
            return SandboxResult(_TIMEOUT_EXIT_CODE, stdout, stderr, timed_out=True)
        return SandboxResult(proc.returncode or 0, stdout, stderr)

    def _argv(self, command: str, env: dict[str, str]) -> list[str]:
        workspace = str(self._workspace)
        argv = [
            self._binary,
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--tmpfs",  # before the workspace bind: a /tmp-based workspace must stay visible
            "/tmp",
            "--bind",
            workspace,
            workspace,
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            workspace,
            "--clearenv",
        ]
        for key in sorted(env):
            argv += ["--setenv", key, env[key]]
        argv += ["--cap-drop", "ALL", "/bin/bash", "-c", command]
        return argv

