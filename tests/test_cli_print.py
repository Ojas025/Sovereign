"""Headless -p / --json: fake llama-server, real registry, real files on disk.

The wrapper script bakes the fake server's port and scripted responses into a
launcher that ignores the manager's router flags — so the full production
wiring (manager → client → router → policy → loop → tools) runs unmodified.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fixtures.fake_llama_server import free_port

FIXTURE_DIR = Path(__file__).parent / "fixtures"

_PLAN = json.dumps({"steps": ["write the marker file", "verify the marker exists"]})
_FINAL = "Marker written - task complete."
_REFLECT = "completed: 1, 2\nverdict: DONE"


def _wrapper_script(
    *,
    port: int,
    responses: tuple[str, ...],
    tool_name: str,
    tool_arguments: str,
) -> str:
    """Self-contained fake-server launcher; never parses its argv."""
    return (
        f"#!{sys.executable}\n"
        "import sys\n"
        "import threading\n"
        f"sys.path.insert(0, {str(FIXTURE_DIR)!r})\n"
        "from fake_llama_server import FakeBehavior, FakeLlamaServer\n"
        "behavior = FakeBehavior(\n"
        f"    responses={responses!r},\n"
        f"    tool_name={tool_name!r},\n"
        f"    tool_arguments={tool_arguments!r},\n"
        ")\n"
        f"server = FakeLlamaServer(behavior, host='127.0.0.1', port={port})\n"
        "server.start()\n"
        "threading.Event().wait()\n"
    )


def _prepare(
    tmp_path: Path,
    responses: tuple[str, ...],
    *,
    tool_name: str = "write",
    tool_arguments: str = "{}",
    sandbox_mode: str | None = None,
) -> tuple[Path, Path]:
    port = free_port()
    wrapper = tmp_path / "fake_wrapper.py"
    wrapper.write_text(
        _wrapper_script(
            port=port,
            responses=responses,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    project = tmp_path / "project"
    project.mkdir()
    toml = (
        f'[server]\nbinary = "{wrapper}"\nport = {port}\n\n'
        '[routing]\nbackend = "heuristic"\n\n'
        f'[agent]\nsession_dir = "{tmp_path / "sessions"}"\n'
    )
    if sandbox_mode is not None:
        toml += f'\n[sandbox]\nmode = "{sandbox_mode}"\n'
    (project / ".workbench.toml").write_text(toml, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return project, workspace


def _run(project: Path, workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "workbench", "--workspace", str(workspace), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=project,
    )


def _events(result: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def test_print_plain_turn_prints_final_answer(tmp_path: Path) -> None:
    project, workspace = _prepare(tmp_path, ("text:Hello from the fake model.",))

    result = _run(project, workspace, "-p", "hello there friend")

    assert result.returncode == 0, result.stderr
    assert result.stdout == "Hello from the fake model.\n"
    assert "Traceback" not in result.stderr


def test_json_print_streams_the_full_plan_cycle(tmp_path: Path) -> None:
    project, workspace = _prepare(
        tmp_path,
        (f"text:{_PLAN}", "tool_call", f"text:{_FINAL}", f"text:{_REFLECT}"),
        tool_name="write",
        tool_arguments='{"path": "e2e_marker.txt", "content": "milestone 5 ok"}',
    )

    result = _run(
        project,
        workspace,
        "-p",
        "Plan the steps to create a marker file",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    events = _events(result)
    kinds = [event["kind"] for event in events]
    for expected in (
        "agent_start",
        "turn_start",
        "route_decided",
        "plan_proposed",
        "plan_approved",
        "tool_execution_start",
        "tool_execution_end",
        "reflection",
        "turn_end",
        "agent_end",
    ):
        assert expected in kinds, f"missing {expected} in {kinds}"
    by_kind = {event["kind"]: event for event in events}
    assert by_kind["route_decided"]["data"]["tier"] == "mid"
    assert by_kind["route_decided"]["data"]["reason"] == "planning"
    assert by_kind["plan_proposed"]["data"]["steps"] == [
        "write the marker file",
        "verify the marker exists",
    ]
    tool_end = by_kind["tool_execution_end"]["data"]
    assert tool_end["tool"] == "write"
    assert tool_end["status"] == "ok"
    assert by_kind["reflection"]["data"] == {"verdict": "done", "completed": [1, 2]}
    assert by_kind["turn_end"]["data"]["outcome"] == "completed"
    assert by_kind["agent_end"]["data"]["outcome"] == "completed"
    # real files: the write tool went through the path jail into the workspace
    assert (workspace / "e2e_marker.txt").read_text(encoding="utf-8") == "milestone 5 ok"
    # plan artifact persisted for /sessions and /resume (M6)
    session_files = list((tmp_path / "sessions").glob("*.jsonl"))
    assert len(session_files) == 1
    persisted = session_files[0].read_text(encoding="utf-8")
    assert '"type": "plan"' in persisted
    assert '"outcome": "completed"' in persisted


@pytest.mark.sandbox
def test_print_runs_bash_through_the_real_sandbox(tmp_path: Path) -> None:
    plan = json.dumps({"steps": ["create the file with echo"]})
    project, workspace = _prepare(
        tmp_path,
        (f"text:{plan}", "tool_call", "text:bash finished", "text:completed: 1\nverdict: DONE"),
        tool_name="bash",
        tool_arguments='{"command": "echo milestone5 > bash_marker.txt"}',
        sandbox_mode="auto",
    )

    result = _run(
        project,
        workspace,
        "-p",
        "run a shell echo to create the file",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    assert (workspace / "bash_marker.txt").read_text(encoding="utf-8").strip() == "milestone5"
    events = _events(result)
    tool_end = [event for event in events if event["kind"] == "tool_execution_end"][0]
    assert tool_end["data"]["status"] == "ok"
    turn_end = [event for event in events if event["kind"] == "turn_end"][0]
    assert turn_end["data"]["outcome"] == "completed"
