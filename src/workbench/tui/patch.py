"""Patch agentui.LiveRegion to enable proper in-place streaming updates.

agentui's LiveRegion implementation assumes the cursor stays on the last content
line after rendering. However, Rich block renderables (Markdown, Group) emit a
trailing newline from Console.print(). Writing that trailing newline causes the
terminal cursor to advance to a new line, which causes the subsequent cursor-up
and erase calculations to miss by one line, resulting in each streaming chunk
appearing on a separate line.
"""

from __future__ import annotations

from typing import Any


def patch_live_region() -> None:
    """Apply monkeypatch to agentui.LiveRegion for flicker-free in-place streaming."""
    from agentui._live_region import LiveRegion
    from rich.console import RenderableType

    if getattr(LiveRegion, "_workbench_patched", False):
        return

    def update(self: Any, renderable: RenderableType) -> None:
        if not self._active:
            return

        rendered = self._capture(renderable).rstrip("\r\n")
        new_line_count = self._count_rendered_lines(rendered)

        parts: list[str] = ["\x1b[?2026h"]

        if self._last_line_count > 0:
            parts.append("\r")
            if self._last_line_count > 1:
                parts.append(f"\x1b[{self._last_line_count - 1}A")
            for i in range(self._last_line_count):
                parts.append("\x1b[2K")
                if i < self._last_line_count - 1:
                    parts.append("\n")
            if self._last_line_count > 1:
                parts.append("\r")
                parts.append(f"\x1b[{self._last_line_count - 1}A")

        parts.append(rendered)
        parts.append("\x1b[?2026l")

        self._raw_write("".join(parts))
        self._last_line_count = new_line_count

    def commit(self: Any) -> None:
        if not self._active:
            return
        self._active = False
        if self._last_line_count > 0:
            self._raw_write("\n")
        self._last_line_count = 0
        self._raw_write("\x1b[?25h")

    LiveRegion.update = update  # type: ignore[method-assign]
    LiveRegion.commit = commit  # type: ignore[method-assign]
    LiveRegion._workbench_patched = True  # type: ignore[attr-defined]
