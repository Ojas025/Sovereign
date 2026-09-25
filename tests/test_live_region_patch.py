"""Test patch_live_region: in-place streaming markdown without line accumulation."""

from __future__ import annotations

from io import StringIO

from agentui import Session
from rich.console import Console
from rich.markdown import Markdown

from workbench.tui.patch import patch_live_region


def test_live_region_patch_overwrites_single_line_in_place() -> None:
    patch_live_region()

    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    session = Session()
    session._console = console
    session._live_region._console = console

    turn = session.assistant_turn()
    turn._live_region.open()

    # Capture raw output across incremental streaming chunks
    writes: list[str] = []
    turn._live_region._raw_write = lambda data: writes.append(data)

    turn.append_markdown_sync("Your")
    turn._flush_all()
    turn.append_markdown_sync(" name")
    turn._flush_all()
    turn.append_markdown_sync(" is Ojas")
    turn._flush_all()

    # There should have been updates
    assert len(writes) >= 3

    # Update 1 wrote "Your" (no newline, cursor stays on current line)
    assert "Your" in writes[0]

    # Subsequent updates MUST include \r and \x1b[2K (erase line) to overwrite in place
    for w in writes[1:]:
        assert "\r" in w, f"Expected carriage return in update: {w!r}"
        assert "\x1b[2K" in w, f"Expected erase line sequence in update: {w!r}"

    # Commit must emit \n so the final completed message lands in scrollback
    turn._live_region.commit()
    assert "\n" in writes[-2] or "\n" in writes[-1]


def test_live_region_patch_handles_multiline_markdown_erasing() -> None:
    patch_live_region()

    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    session = Session()
    session._console = console
    session._live_region._console = console

    turn = session.assistant_turn()
    turn._live_region.open()

    writes: list[str] = []
    turn._live_region._raw_write = lambda data: writes.append(data)

    turn._live_region.update(Markdown("Line 1\n\nLine 2"))
    # Two content lines (with a blank line between) = 3 terminal lines
    assert turn._live_region._last_line_count == 3

    # Next update must move up 2 lines (\x1b[2A) to the start of the 3-line block
    turn._live_region.update(Markdown("Line 1\n\nLine 2 updated"))
    assert any("\x1b[2A" in w for w in writes[1:])

    turn._live_region.commit()
