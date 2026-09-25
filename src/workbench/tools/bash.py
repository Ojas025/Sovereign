"""bash tool: guarded shell execution behind guardrails + sandbox (plan §5.2/§5.3).

Flow per call: validate args → guardrail classify (deny/confirm/allow) →
confirmation prompt when flagged → run in the sandbox with a config-capped
timeout → truncate output → format for the model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from workbench.config import SandboxConfig
from workbench.core.protocols import Sandbox, SandboxResult, ToolContext, ToolResult
from workbench.tools.guardrails import classify

_BASH_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "shell command to run"},
        "timeout": {
            "type": "number",
            "description": "seconds before the command is killed (capped by config)",
        },
    },
    "required": ["command"],
}


def _truncate(text: str, cap_bytes: int) -> str:
    """Bound one output stream to the configured byte cap, marking what was cut."""
    data = text.encode("utf-8")
    if len(data) <= cap_bytes:
        return text
    kept = data[:cap_bytes].decode("utf-8", errors="ignore")
    return kept + f"\n[output truncated: {len(data) - cap_bytes} bytes dropped]"


def _format(result: SandboxResult, timeout_s: float, cap_bytes: int) -> str:
    parts: list[str] = []
    if result.timed_out:
        parts.append(f"timed out after {timeout_s:g}s")
    elif result.exit_code != 0:
        parts.append(f"exit code {result.exit_code}")
    stdout = _truncate(result.stdout, cap_bytes)
    stderr = _truncate(result.stderr, cap_bytes)
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(stderr)
    return "\n".join(parts)


class BashTool:
    name: str = "bash"
    description: str = (
        "Run a shell command in the sandbox: workspace as cwd, read-only system "
        "filesystem, no network access."
    )
    parameters: Mapping[str, object] = _BASH_PARAMETERS

    def __init__(self, sandbox: Sandbox, config: SandboxConfig) -> None:
        self._sandbox = sandbox
        self._config = config

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult("command must be a non-empty string", is_error=True)
        timeout = arguments.get("timeout")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            return ToolResult("timeout must be a positive number of seconds", is_error=True)

        decision = classify(
            command, mode=self._config.mode, allowlist=self._config.allowlist
        )
        if decision.verdict == "deny":
            return ToolResult(
                f"blocked: command rejected by guardrails ({decision.reason})", is_error=True
            )
        if decision.verdict == "confirm":
            prompt = f"Run command? [{decision.reason}] {command}"
            if not await context.confirm(prompt):
                return ToolResult(
                    f"blocked: confirmation declined ({decision.reason})", is_error=True
                )

        cap = self._config.bash_timeout_s
        effective_timeout = cap if timeout is None else min(float(timeout), cap)
        result = await asyncio.to_thread(self._sandbox.run, command, effective_timeout)
        return ToolResult(
            _format(result, effective_timeout, self._config.output_max_bytes),
            is_error=result.timed_out or result.exit_code != 0,
        )
