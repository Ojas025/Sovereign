"""Interactive TUI under a real pty — end-to-end without a GPU (M6 gate VAL).

Drives `workbench` with a pty as stdin/stdout: a plan is proposed, approved
with ``y``, the write tool runs with an inline diff, the model reflects, and
``/quit`` exits 0. Uses the same wrapper-script fake server as the headless
e2e tests (no model/sandbox markers needed — CI-safe).

A shared ``PtySession`` owns the buffer with a **search cursor**: each
``read_until`` only matches text it has not returned yet, so a marker like the
status toolbar (``tier:``) can be asserted once per expected appearance.
"""

from __future__ import annotations

import fcntl
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

from test_cli_print import _prepare

_READ_CHUNK = 4096
_READ_TIMEOUT_S = 30.0
_EXIT_TIMEOUT_S = 30.0
_EXIT_GRACE_S = 3.0
_READ_POLL_S = 0.5
# prompt_toolkit asks "cursor where?" before its first paint; a real terminal
# answers — the harness must too, or the prompt renders in CPR-fallback mode.
_CPR_REQUEST = b"\x1b[6n"
_CPR_REPLY = b"\x1b[1;1R"
# A fresh pty reports 0x0, which prompt_toolkit renders as scrambled geometry;
# give it a real 24x100 window like any terminal emulator would.
_WIN_ROWS = 24
_WIN_COLS = 100
# Rendered through Rich markup: plain-English markers only (never ANSI-fragile).
_MARKER_STATUS = b"tier:"
_MARKER_PLAN = b"Approve this plan (2 steps)?"
_MARKER_DONE = b"DONE"

_PLAN = '{"steps": ["write the marker file", "verify the marker exists"]}'
_FINAL = "Marker written - task complete."
_REFLECT = "completed: 1, 2\nverdict: DONE"
_PROMPT = "Plan the steps to create a marker file"


class PtySession:
    """One subprocess attached to a pty, with cursor-ordered reads."""

    def __init__(self, project: Path, workspace: Path) -> None:
        self._master, slave = pty.openpty()
        winsize = struct.pack("HHHH", _WIN_ROWS, _WIN_COLS, 0, 0)
        fcntl.ioctl(self._master, termios.TIOCSWINSZ, winsize)
        env = {**os.environ, "TERM": "xterm-256color", "COLUMNS": str(_WIN_COLS)}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "workbench", "--workspace", str(workspace)],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=project,
            env=env,
        )
        os.close(slave)
        self.raw = b""
        self._cursor = 0
        self._cpr_answered = 0

    def _answer_cpr(self) -> None:
        """Reply to every cursor-position request seen so far (row 1, col 1)."""
        pending = self.raw.count(_CPR_REQUEST) - self._cpr_answered
        for _ in range(pending):
            os.write(self._master, _CPR_REPLY)
            self._cpr_answered += 1

    def read_until(self, marker: bytes, timeout: float = _READ_TIMEOUT_S) -> bytes:
        """Block until `marker` appears AFTER every previously consumed byte."""
        deadline = time.monotonic() + timeout
        while True:
            found = self.raw.find(marker, self._cursor)
            if found != -1:
                self._cursor = found + len(marker)
                return self.raw
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"marker {marker!r} never appeared; tail: {self.raw[-2000:]!r}"
                )
            ready, _, _ = select.select([self._master], [], [], min(remaining, _READ_POLL_S))
            if not ready:
                continue
            try:
                chunk = os.read(self._master, _READ_CHUNK)
            except OSError:  # slave side closed: process is exiting
                chunk = b""
            if not chunk:
                if self.proc.poll() is not None:
                    raise AssertionError(
                        f"process exited (rc={self.proc.returncode}) before "
                        f"{marker!r}; tail: {self.raw[-2000:]!r}"
                    )
                continue
            self.raw += chunk
            self._answer_cpr()

    def type_line(self, text: str) -> None:
        """Send one line of input the way a user would."""
        os.write(self._master, text.encode() + b"\n")

    def text(self) -> str:
        return self.raw.decode("utf-8", errors="replace")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=_EXIT_TIMEOUT_S)
        os.close(self._master)


def _finish(session: PtySession) -> None:
    """Feed /quit, require a clean exit, and collect anything still buffered."""
    session.type_line("/quit")
    assert session.proc.wait(timeout=_EXIT_TIMEOUT_S) == 0, session.text()
    deadline = time.monotonic() + _EXIT_GRACE_S
    while time.monotonic() < deadline:
        ready, _, _ = select.select([session._master], [], [], 0.1)  # noqa: SLF001
        if not ready:
            continue
        try:
            chunk = os.read(session._master, _READ_CHUNK)  # noqa: SLF001
        except OSError:
            break
        if not chunk:
            break
        session.raw += chunk
    session._answer_cpr()


def test_tui_full_session_under_a_real_pty(tmp_path: Path) -> None:
    responses = (f"text:{_PLAN}", "tool_call", f"text:{_FINAL}", f"text:{_REFLECT}")
    project, workspace = _prepare(
        tmp_path,
        responses,
        tool_name="write",
        tool_arguments='{"path": "marker.txt", "content": "hello world"}',
    )

    session = PtySession(project, workspace)
    try:
        # prompt_toolkit splits "> " across lines in its geometry dance, so the
        # stable markers are: each prompt's status toolbar, the confirm line,
        # and the reflection footer (all verified contiguous in the raw stream).
        session.read_until(_MARKER_STATUS)  # prompt #1 + toolbar painted

        session.type_line(_PROMPT)
        session.read_until(_MARKER_PLAN)  # plan panel rendered, y/N prompt shown

        session.type_line("y")
        session.read_until(_MARKER_DONE)  # approval, tool, reflection done
        session.read_until(_MARKER_STATUS)  # prompt #2: back at the prompt

        _finish(session)

        out = session.text()
        assert session.proc.returncode == 0, out
        assert "write the marker file" in out  # plan block
        assert "marker.txt" in out  # tool/diff block rendered
        assert "hello world" in out  # the write diff content
        assert (workspace / "marker.txt").read_text() == "hello world"
        assert "DONE" in out  # reflection footer
        assert "Traceback" not in out
    finally:
        session.close()
