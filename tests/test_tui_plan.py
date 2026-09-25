"""Plan block and reflection footer rendering (PLAN §7, ruling M6.5)."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from workbench.core.session import Plan, PlanStep
from workbench.tui.plan import reflection_footer, render_plan


def _rendered(renderable: object) -> str:
    console = Console(width=80, force_terminal=False, record=True)
    console.print(renderable)
    return console.export_text()


def test_pending_plan_shows_numbered_circle_steps() -> None:
    plan = Plan(
        steps=[
            PlanStep(index=1, text="create hello.txt"),
            PlanStep(index=2, text="verify content"),
        ]
    )

    text = _rendered(render_plan(plan))

    assert "plan" in text
    assert "○ 1. create hello.txt" in text
    assert "○ 2. verify content" in text


def test_statuses_render_as_active_and_done_glyphs() -> None:
    plan = Plan(
        steps=[
            PlanStep(index=1, text="first", status="done"),
            PlanStep(index=2, text="second", status="active"),
            PlanStep(index=3, text="third"),
        ]
    )

    text = _rendered(render_plan(plan))

    assert "✓ 1. first" in text
    assert "● 2. second" in text
    assert "○ 3. third" in text


def test_reflection_footer_is_a_dim_note_with_verdict_and_completed() -> None:
    footer = reflection_footer("done", (1, 2))

    assert isinstance(footer, Text)
    assert "DONE" in footer.plain
    assert "1,2" in footer.plain
    assert "dim" in str(footer.style)


def test_reflection_footer_without_completed_steps_reads_done() -> None:
    footer = reflection_footer("continue", ())

    assert isinstance(footer, Text)
    assert "CONTINUE" in footer.plain


def test_reflection_footer_wait_user() -> None:
    footer = reflection_footer("wait_user", (1,))

    assert isinstance(footer, Text)
    assert "WAITING FOR USER" in footer.plain
    assert "1" in footer.plain

