"""Example sandbox backend plugin proving the ``workbench.sandboxes`` extension point (PLAN §3).

Defines an example passthrough/mock sandbox implementing the Sandbox protocol.
Configured via:
    [sandbox]
    backend = "passthrough"
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from workbench.core.protocols import SandboxResult
from workbench.core.registry import Registry


class PassthroughSandbox:
    """Non-isolating sandbox for controlled test environments."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def run(self, command: str, timeout_s: float) -> SandboxResult:
        try:
            process = subprocess.run(
                ["bash", "-c", command],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return SandboxResult(
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return SandboxResult(
                exit_code=-1,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )


def register(registry: Registry) -> None:
    """Registration hook when loaded from ~/.config/workbench/plugins/."""
    registry.register("sandboxes", "passthrough", PassthroughSandbox(Path.cwd()))
