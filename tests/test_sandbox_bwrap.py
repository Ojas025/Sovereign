"""Real bubblewrap execution: isolation guarantees end-to-end (local only, skipped in CI)."""

import subprocess
import time
from pathlib import Path

import pytest

from workbench.sandbox.bwrap import BwrapSandbox

pytestmark = pytest.mark.sandbox


def sandbox_for(workspace: Path) -> BwrapSandbox:
    return BwrapSandbox(workspace)


class TestBasics:
    def test_captures_stdout_and_exit_code(self, tmp_path: Path) -> None:
        result = sandbox_for(tmp_path).run("echo hello-sandbox", timeout_s=10.0)

        assert result.exit_code == 0
        assert result.stdout.strip() == "hello-sandbox"
        assert result.timed_out is False

    def test_nonzero_exit_propagates(self, tmp_path: Path) -> None:
        result = sandbox_for(tmp_path).run("exit 7", timeout_s=10.0)

        assert result.exit_code == 7

    def test_working_directory_is_the_workspace(self, tmp_path: Path) -> None:
        result = sandbox_for(tmp_path).run("pwd", timeout_s=10.0)

        assert result.stdout.strip() == str(tmp_path)

    def test_workspace_is_writable_from_inside(self, tmp_path: Path) -> None:
        result = sandbox_for(tmp_path).run("echo made-inside > created.txt", timeout_s=10.0)

        assert result.exit_code == 0
        assert (tmp_path / "created.txt").read_text(encoding="utf-8").strip() == "made-inside"


class TestIsolation:
    def test_root_filesystem_is_read_only(self, tmp_path: Path) -> None:
        result = sandbox_for(tmp_path).run("touch /etc/should-not-exist", timeout_s=10.0)

        assert result.exit_code != 0
        assert not Path("/etc/should-not-exist").exists()

    def test_network_is_unreachable(self, tmp_path: Path) -> None:
        probe = (
            "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 80), 3)\""
        )
        result = sandbox_for(tmp_path).run(probe, timeout_s=15.0)

        assert result.exit_code != 0
        assert "unreachable" in (result.stdout + result.stderr).lower()

    def test_environment_is_scrubbed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WB_SECRET_TOKEN", "hunter2")

        result = sandbox_for(tmp_path).run(
            "echo secret=${WB_SECRET_TOKEN:-unset}; echo path=${PATH:-empty}", timeout_s=10.0
        )

        assert "secret=unset" in result.stdout
        assert "path=empty" not in result.stdout

    def test_private_tmp_hides_host_tmp(self, tmp_path: Path) -> None:
        marker = "/tmp/wb-m4-sandbox-private-tmp-marker"
        Path(marker).unlink(missing_ok=True)

        result = sandbox_for(tmp_path).run(f"touch {marker}; ls {marker}", timeout_s=10.0)

        assert result.exit_code == 0
        assert not Path(marker).exists()

    def test_capabilities_are_dropped(self, tmp_path: Path) -> None:
        result = sandbox_for(tmp_path).run("grep CapEff /proc/self/status", timeout_s=10.0)

        assert "0000000000000000" in result.stdout


class TestTimeouts:
    def test_overrunning_command_is_killed_and_flagged(self, tmp_path: Path) -> None:
        sandbox = sandbox_for(tmp_path)
        started = time.monotonic()

        result = sandbox.run("sleep 30", timeout_s=0.5)

        elapsed = time.monotonic() - started
        assert result.timed_out is True
        assert result.exit_code == 124
        assert elapsed < 5.0

    def test_timeout_leaves_no_stray_processes(self, tmp_path: Path) -> None:
        # unique duration → the host-side match cannot collide with unrelated processes
        result = sandbox_for(tmp_path).run("sleep 37.13 & echo started; wait", timeout_s=0.5)
        assert result.timed_out is True

        time.sleep(0.3)  # allow init to reap the SIGKILLed orphans
        check = subprocess.run(
            ["pgrep", "-f", "^sleep 37.13"], capture_output=True, text=True, check=False
        )
        assert check.returncode != 0, f"stray processes survived the timeout: {check.stdout}"
