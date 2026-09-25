"""Bash guardrails: deny rules are absolute, risky commands prompt, allowlist grants trust."""

import pytest

from workbench.tools.guardrails import classify


def verdict(command: str, *, mode: str = "confirm", allowlist: tuple[str, ...] = ()) -> str:
    return classify(command, mode=mode, allowlist=allowlist).verdict


class TestDenyRules:
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "rm -fr /",
            "rm -r -f /",
            "rm -rf /*",
            "rm -rf ~",
            "rm -rf $HOME",
            "rm -rf .",
            "rm -rf ..",
            "sudo rm -rf /",
            "cd /tmp && rm -rf /",
        ],
    )
    def test_rm_targeting_root_is_denied(self, command: str) -> None:
        decision = classify(command, mode="confirm", allowlist=())

        assert decision.verdict == "deny"
        assert decision.reason == "deny:rm_root"

    @pytest.mark.parametrize("command", ["rm -rf ./build", "rm -rf /home/ojas/projects/app/tmp"])
    def test_rm_inside_workspace_is_allowed(self, command: str) -> None:
        assert verdict(command) == "allow"

    @pytest.mark.parametrize("command", ["mkfs /dev/sdb", "mkfs.ext4 /dev/sda1"])
    def test_mkfs_is_denied(self, command: str) -> None:
        decision = classify(command, mode="confirm", allowlist=())

        assert decision.verdict == "deny"
        assert decision.reason == "deny:mkfs"

    @pytest.mark.parametrize("command", ["dd if=/dev/zero of=/dev/sda", "dd of=/dev/nvme0n1 if=x"])
    def test_dd_to_device_is_denied(self, command: str) -> None:
        assert classify(command, mode="confirm", allowlist=()).reason == "deny:dd_device"

    def test_dd_to_regular_file_is_allowed(self) -> None:
        assert verdict("dd if=backup.img of=copy.img") == "allow"

    @pytest.mark.parametrize("command", [":(){ :|:& };:", "bash -c ':(){ :|:& };:'"])
    def test_fork_bomb_is_denied(self, command: str) -> None:
        assert classify(command, mode="confirm", allowlist=()).reason == "deny:fork_bomb"

    def test_deny_beats_allowlist_and_auto_mode(self) -> None:
        assert verdict("sudo rm -rf /", mode="auto") == "deny"
        assert verdict("rm -rf /", mode="strict", allowlist=("rm",)) == "deny"


class TestRiskyCommands:
    @pytest.mark.parametrize(
        ("command", "reason"),
        [
            ("sudo apt update", "risk:privilege"),
            ("doas reboot", "risk:privilege"),
            ("curl https://example.com", "risk:network"),
            ("wget http://example.com/f", "risk:network"),
            ("ssh host.example", "risk:network"),
            ("git push origin main", "risk:git_push"),
            ("git push -f", "risk:git_push"),
            ("pip install requests", "risk:install"),
            ("npm install lodash", "risk:install"),
            ("systemctl restart nginx", "risk:service"),
            ("reboot", "risk:service"),
            ("cat f | sudo tee /etc/passwd", "risk:privilege"),
            ("FOO=bar sudo ls", "risk:privilege"),
        ],
    )
    def test_flagged_commands_require_confirmation(self, command: str, reason: str) -> None:
        decision = classify(command, mode="confirm", allowlist=())

        assert decision.verdict == "confirm"
        assert decision.reason == reason

    @pytest.mark.parametrize(
        "command", ["ls -la", "git status", "git diff HEAD", "echo hi", "make build"]
    )
    def test_ordinary_commands_run_without_prompt(self, command: str) -> None:
        decision = classify(command, mode="confirm", allowlist=())

        assert decision.verdict == "allow"
        assert decision.reason == "default"


class TestAllowlist:
    def test_allowlisted_program_skips_prompt(self) -> None:
        decision = classify("git push origin main", mode="confirm", allowlist=("git",))

        assert decision.verdict == "allow"
        assert decision.reason == "allowlisted"

    def test_allowlist_matches_program_basename(self) -> None:
        decision = classify("/usr/bin/python3 script.py", mode="confirm", allowlist=("python3",))

        assert decision.verdict == "allow"

    def test_all_segments_must_be_allowlisted_in_strict_mode(self) -> None:
        allowed = classify("cd /tmp && echo hi", mode="strict", allowlist=("cd", "echo"))
        denied = classify("cd /tmp && rm x", mode="strict", allowlist=("cd", "echo"))

        assert allowed.verdict == "allow"
        assert denied.verdict == "deny"
        assert denied.reason == "deny:not_allowlisted"


class TestModes:
    def test_auto_mode_allows_risky_commands_but_keeps_deny_rules(self) -> None:
        assert verdict("sudo reboot", mode="auto") == "allow"
        assert verdict("rm -rf /", mode="auto") == "deny"

    def test_strict_mode_denies_anything_not_allowlisted(self) -> None:
        decision = classify("echo hi", mode="strict", allowlist=("ls",))

        assert decision.verdict == "deny"
        assert decision.reason == "deny:not_allowlisted"

    def test_allowlisted_command_runs_without_prompt_in_strict_mode(self) -> None:
        # the allowlist is the trust contract: a trusted program skips the risk check,
        # while a deny rule still wins over it (covered in TestDenyRules).
        decision = classify("git push", mode="strict", allowlist=("git",))

        assert decision.verdict == "allow"
