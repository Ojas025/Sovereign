"""Plan block and reflection footer: the plan's visible contract (PLAN §7).

Scrollback-printed rather than live-updated (ruling M6.5): the loop emits no
per-step events, so the block re-renders at proposal and at every reflection
with the statuses known so far, and ``/plan`` re-renders on demand.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from workbench.core.session import Plan

# Glyph vocabulary mirrors PLAN §7 (pending → active → done); "failed" is not a
# PlanStep status — reflection only ever marks steps done or leaves them pending.
_STATUS_GLYPHS: dict[str, tuple[str, str]] = {
    "pending": ("○", "dim"),
    "active": ("●", "yellow"),
    "done": ("✓", "green"),
}


def render_plan(plan: Plan) -> Panel:
    """Numbered steps with status glyphs, bordered and titled "plan"."""
    lines: list[Text] = []
    for step in plan.steps:
        glyph, style = _STATUS_GLYPHS.get(step.status, _STATUS_GLYPHS["pending"])
        line = Text()
        line.append(f"{glyph} ", style=style)
        line.append(f"{step.index}. ", style="bold")
        line.append(step.text)
        lines.append(line)
    return Panel(Group(*lines), title="plan", border_style="dim", padding=(0, 1))


def reflection_footer(verdict: str, completed: Sequence[int]) -> Text:
    """Dim one-liner under the re-rendered plan: what the self-check concluded."""
    steps = ",".join(str(index) for index in completed) or "—"
    verdict_str = "WAITING FOR USER" if verdict.lower() == "wait_user" else verdict.upper()
    return Text(f"↳ {verdict_str} · completed {steps}", style="dim italic")
