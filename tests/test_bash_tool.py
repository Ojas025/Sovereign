"""Bash tool: guardrail gating, confirmation policy, timeout caps, output truncation."""

import pytest

from workbench.config import SandboxConfig
from workbench.core.protocols import SandboxResult, ToolContext
from workbench.tools.bash import BashTool

_OUTPUT_CAP = 20_000  # default sandbox.output_max_bytes — keeps model context bounded


class FakeSandbox:
    """Scripted Sandbox double: records every call, replays a canned result."""

    def __init__(self, result: SandboxResult) -> None:
        self.calls: list[tuple[str, float]] = []
        self._result = result

    def run(self, command: str, timeout_s: float) -> SandboxResult:
        self.calls.append((command, timeout_s))
        return self._result


class RecordingConfirmer:
    def __init__(self, approve: bool) -> None:
        self.approve = approve
        self.prompts: list[str] = []

    async def confirm(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.approve


def make_context(workspace, approve: bool = True) -> tuple[ToolContext, RecordingConfirmer]:
    confirmer = RecordingConfirmer(approve)
    context = ToolContext(workspace_root=workspace, confirm=confirmer.confirm)
    return context, confirmer


OK = SandboxResult(exit_code=0, stdout="", stderr="")


class TestExecution:
    async def test_runs_command_and_returns_stdout(self, tmp_path) -> None:
        sandbox = FakeSandbox(SandboxResult(exit_code=0, stdout="hello", stderr=""))
        tool = BashTool(sandbox, SandboxConfig())
        context, _ = make_context(tmp_path)

        result = await tool.execute({"command": "echo hello"}, context)

        assert result.is_error is False
        assert result.content == "hello"
        assert sandbox.calls == [("echo hello", 60.0)]

    async def test_stderr_is_included(self, tmp_path) -> None:
        sandbox = FakeSandbox(SandboxResult(exit_code=0, stdout="out", stderr="warn"))
        tool = BashTool(sandbox, SandboxConfig())
        context, _ = make_context(tmp_path)

        result = await tool.execute({"command": "x"}, context)

        assert result.content == "out\nwarn"

    async def test_nonzero_exit_marks_error_with_status_line(self, tmp_path) -> None:
        sandbox = FakeSandbox(SandboxResult(exit_code=3, stdout="partial", stderr="bad"))
        tool = BashTool(sandbox, SandboxConfig())
        context, _ = make_context(tmp_path)

        result = await tool.execute({"command": "x"}, context)

        assert result.is_error is True
        assert result.content == "exit code 3\npartial\nbad"

    async def test_timeout_result_marks_error(self, tmp_path) -> None:
        sandbox = FakeSandbox(SandboxResult(exit_code=124, stdout="", stderr="", timed_out=True))
        tool = BashTool(sandbox, SandboxConfig(bash_timeout_s=2.5))
        context, _ = make_context(tmp_path)

        result = await tool.execute({"command": "sleep 10"}, context)

        assert result.is_error is True
        assert result.content == "timed out after 2.5s"


class TestTimeoutPolicy:
    async def test_default_timeout_comes_from_config(self, tmp_path) -> None:
        sandbox = FakeSandbox(OK)
        tool = BashTool(sandbox, SandboxConfig(bash_timeout_s=45.0))
        context, _ = make_context(tmp_path)

        await tool.execute({"command": "x"}, context)

        assert sandbox.calls[0][1] == 45.0

    async def test_timeout_argument_is_clamped_to_config_cap(self, tmp_path) -> None:
        sandbox = FakeSandbox(OK)
        tool = BashTool(sandbox, SandboxConfig(bash_timeout_s=60.0))
        context, _ = make_context(tmp_path)

        await tool.execute({"command": "x", "timeout": 9999}, context)

        assert sandbox.calls[0][1] == 60.0

    async def test_invalid_timeout_is_rejected(self, tmp_path) -> None:
        sandbox = FakeSandbox(OK)
        tool = BashTool(sandbox, SandboxConfig())
        context, _ = make_context(tmp_path)

        for bad in (-1, "soon", True):
            result = await tool.execute({"command": "x", "timeout": bad}, context)

            assert result.is_error is True
            assert "timeout" in result.content
        assert sandbox.calls == []


class TestOutputTruncation:
    async def test_stdout_over_cap_is_truncated_with_marker(self, tmp_path) -> None:
        big = "A" * (_OUTPUT_CAP + 5_000)
        sandbox = FakeSandbox(SandboxResult(exit_code=0, stdout=big, stderr=""))
        tool = BashTool(sandbox, SandboxConfig())
        context, _ = make_context(tmp_path)

        result = await tool.execute({"command": "x"}, context)

        assert "[output truncated" in result.content
        body = result.content.split("\n[output truncated")[0]
        assert len(body.encode()) == _OUTPUT_CAP

    async def test_stderr_is_truncated_independently(self, tmp_path) -> None:
        big_err = "E" * (_OUTPUT_CAP + 500)
        sandbox = FakeSandbox(SandboxResult(exit_code=1, stdout="o", stderr=big_err))
        tool = BashTool(sandbox, SandboxConfig())
        context, _ = make_context(tmp_path)

        result = await tool.execute({"command": "x"}, context)

        assert result.content.startswith("exit code 1\no\n")
        assert "[output truncated" in result.content
        assert len(result.content.encode()) < _OUTPUT_CAP * 2 + 200


class TestGuardrails:
    async def test_denied_command_never_reaches_sandbox(self, tmp_path) -> None:
        sandbox = FakeSandbox(OK)
        tool = BashTool(sandbox, SandboxConfig())
        context, confirmer = make_context(tmp_path)

        result = await tool.execute({"command": "rm -rf /"}, context)

        assert result.is_error is True
        assert "blocked" in result.content
        assert "deny:rm_root" in result.content
        assert sandbox.calls == []
        assert confirmer.prompts == []

    async def test_flagged_command_prompts_and_runs_when_approved(self, tmp_path) -> None:
        sandbox = FakeSandbox(SandboxResult(exit_code=0, stdout="ok", stderr=""))
        tool = BashTool(sandbox, SandboxConfig())
        context, confirmer = make_context(tmp_path, approve=True)

        result = await tool.execute({"command": "sudo systemctl restart nginx"}, context)

        assert result.is_error is False
        assert result.content == "ok"
        assert len(confirmer.prompts) == 1
        assert "sudo systemctl restart nginx" in confirmer.prompts[0]
        assert "risk:privilege" in confirmer.prompts[0]

    async def test_declined_confirmation_blocks_without_running(self, tmp_path) -> None:
        sandbox = FakeSandbox(OK)
        tool = BashTool(sandbox, SandboxConfig())
        context, _ = make_context(tmp_path, approve=False)

        result = await tool.execute({"command": "curl http://x"}, context)

        assert result.is_error is True
        assert "declined" in result.content
        assert sandbox.calls == []

    async def test_safe_command_never_prompts(self, tmp_path) -> None:
        sandbox = FakeSandbox(OK)
        tool = BashTool(sandbox, SandboxConfig())
        context, confirmer = make_context(tmp_path)

        await tool.execute({"command": "ls -la"}, context)

        assert confirmer.prompts == []

    async def test_auto_mode_allows_risky_command_without_prompting(self, tmp_path) -> None:
        sandbox = FakeSandbox(OK)
        tool = BashTool(sandbox, SandboxConfig(mode="auto"))
        context, confirmer = make_context(tmp_path)

        await tool.execute({"command": "sudo reboot"}, context)

        assert confirmer.prompts == []
        assert sandbox.calls != []


class TestArgumentValidation:
    async def test_missing_command_reports_error(self, tmp_path) -> None:
        tool = BashTool(FakeSandbox(OK), SandboxConfig())
        context, _ = make_context(tmp_path)

        result = await tool.execute({}, context)

        assert result.is_error is True
        assert "command" in result.content

    @pytest.mark.parametrize("command", ["", "   ", 42, None])
    async def test_blank_or_non_string_command_reports_error(self, tmp_path, command) -> None:
        sandbox = FakeSandbox(OK)
        tool = BashTool(sandbox, SandboxConfig())
        context, _ = make_context(tmp_path)

        result = await tool.execute({"command": command}, context)

        assert result.is_error is True
        assert "command" in result.content
        assert sandbox.calls == []


class TestToolMetadata:
    def test_exposes_openai_style_parameters(self) -> None:
        tool = BashTool(FakeSandbox(OK), SandboxConfig())

        assert tool.name == "bash"
        assert tool.parameters["type"] == "object"
        properties = tool.parameters["properties"]
        assert isinstance(properties, dict)
        assert "command" in properties
        assert tool.parameters["required"] == ["command"]
