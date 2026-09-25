"""Event stream → agentui rendering: the TUI's single event consumer.

Lifecycle rule (ruling M6.4): a Turn owns the live region only while text or
tool blocks are on screen. Everything that lands in permanent scrollback —
plan blocks, notices, prompts — happens after the open turn is committed, so
Rich never interleaves with the region and a prompt can never corrupt it. A
tool whose start-block was committed by an interrupting prompt is re-created
at its end event from the pending call record.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager

from agentui import Session, Turn
from agentui._tool_call import ToolCallContext  # not re-exported by agentui
from rich.console import Group
from rich.markup import escape

from workbench.core.events import Event
from workbench.core.session import Plan, PlanStep
from workbench.tui.blocks import build_diff, compact_arguments
from workbench.tui.plan import reflection_footer, render_plan
from workbench.tui.state import TuiState

_BASH_OUTPUT_FENCE = "```text"


class TranscriptController:
    """Folds bus events into transcript renders and status state, in stream order."""

    def __init__(
        self,
        ui: Session,
        state: TuiState,
        *,
        on_turn_opened: Callable[[], None] | None = None,
    ) -> None:
        self._ui = ui
        self._state = state
        self._on_turn_opened = on_turn_opened
        self._turn: Turn | None = None
        self._turn_cm: AbstractAsyncContextManager[Turn] | None = None
        self._plan: Plan | None = None
        # call_id → (tool, raw args JSON) kept across turn commits so a block
        # committed by a prompt can still be re-created at its end event.
        self._pending: dict[str, tuple[str, str]] = {}
        self._blocks: dict[str, ToolCallContext] = {}
        self._thinking_seg: object | None = None

    # --- public API ---------------------------------------------------------

    async def handle(self, event: Event) -> None:
        """Apply one event in stream order; state-only kinds fall through."""
        self._state.apply(event)
        kind = event.kind
        data = event.data
        if kind == "message_start":
            await self._open()
            self._show_thinking()
        elif kind == "message_update":
            await self._open()
            self._clear_thinking()
            assert self._turn is not None
            self._turn.append_markdown_sync(str(data.get("text", "")))
        elif kind == "tool_execution_start":
            self._clear_thinking()
            await self._start_tool(data)
        elif kind == "tool_execution_end":
            await self._end_tool(data)
        elif kind == "plan_proposed":
            await self._plan_proposed(data)
        elif kind == "plan_approved":
            await self._close()
            steps = len(self._plan.steps) if self._plan is not None else 0
            self._ui.print(f"[green]✓ plan approved[/green] [dim]· {steps} steps[/dim]")
        elif kind == "plan_rejected":
            await self._close()
            self._print_plan_rejection(data)
        elif kind == "reflection":
            await self._reflection(data)
        elif kind == "route_escalated":
            await self._close()
            self._ui.print(
                "[yellow]↷ escalated "
                f"{data.get('from_tier')}→{data.get('to_tier')}: "
                f"{data.get('failures')} tool failures[/yellow]"
            )
        elif kind == "error":
            await self._close()
            self._ui.print(
                f"[red]error ({escape(str(data.get('stage')))}): "
                f"{escape(str(data.get('message')))}[/red]"
            )
        elif kind == "turn_end":
            await self._close()

    async def close(self) -> None:
        """Commit any open turn — called before every bridged prompt (M6.4)."""
        await self._close()

    @property
    def plan(self) -> Plan | None:
        """The plan this transcript last rendered (proposal or reflection)."""
        return self._plan

    def clear_plan(self) -> None:
        """Forget the rendered plan — used when /resume swaps the session."""
        self._plan = None

    # --- turn lifecycle -------------------------------------------------------

    async def _open(self) -> None:
        if self._turn_cm is None:
            cm = self._ui.assistant_turn()
            turn = await cm.__aenter__()
            self._turn_cm = cm
            self._turn = turn
            if self._on_turn_opened is not None:
                # agentui installed ITS SIGINT handler in __aenter__; the app's
                # hook puts back the handler that aborts the turn (M6.15).
                self._on_turn_opened()

    async def _close(self) -> None:
        self._clear_thinking()
        cm = self._turn_cm
        if cm is None:
            return
        self._turn_cm = None
        self._turn = None
        # Blocks die with the turn that rendered them; pending call records
        # survive so the end event can re-create a committed block.
        self._blocks.clear()
        await cm.__aexit__(None, None, None)

    def _show_thinking(self) -> None:
        if self._turn is not None and hasattr(self._turn, "_segments"):
            from rich.text import Text

            seg = Text("⠋ thinking...", style="dim italic")
            self._thinking_seg = seg
            self._turn._segments.append(seg)
            if hasattr(self._turn, "_rerender"):
                self._turn._rerender()

    def _clear_thinking(self) -> None:
        if self._thinking_seg is not None and self._turn is not None:
            if hasattr(self._turn, "_segments") and self._thinking_seg in self._turn._segments:
                self._turn._segments.remove(self._thinking_seg)
            self._thinking_seg = None

    # --- tools -----------------------------------------------------------------

    async def _start_tool(self, data: Mapping[str, object]) -> None:
        call_id = str(data.get("call_id", ""))
        tool = str(data.get("tool", ""))
        self._pending[call_id] = (tool, str(data.get("arguments", "")))
        if self._turn is None:
            return  # no open turn: re-create at the end event instead
        ctx = self._turn.tool_call(tool, compact_arguments(tool, self._pending[call_id][1]))
        await ctx.__aenter__()  # running badge; the block spans events, not a `with`
        self._blocks[call_id] = ctx

    async def _end_tool(self, data: Mapping[str, object]) -> None:
        call_id = str(data.get("call_id", ""))
        tool, arguments_json = self._pending.pop(call_id, (str(data.get("tool", "")), ""))
        status = str(data.get("status", "error"))
        output = str(data.get("output", ""))
        if self._turn is None:
            await self._open()
        assert self._turn is not None
        ctx = self._blocks.pop(call_id, None)
        if ctx is None:
            ctx = self._turn.tool_call(tool, compact_arguments(tool, arguments_json))
            await ctx.__aenter__()
        if status != "ok":
            await ctx.error(output)
            return
        await ctx.complete()
        if tool == "bash":
            # §7: bash shows its (already size-capped) output — as a code block
            # under the collapsed header, never as raw JSON args.
            self._turn.append_markdown_sync(f"{_BASH_OUTPUT_FENCE}\n{output}\n```")
        else:
            diff = build_diff(tool, arguments_json)
            if diff is not None:
                self._turn.diff(*diff)

    # --- plan / reflection -------------------------------------------------------

    async def _plan_proposed(self, data: Mapping[str, object]) -> None:
        await self._close()
        raw_steps = data.get("steps")
        steps = raw_steps if isinstance(raw_steps, list) else []
        self._plan = Plan(
            steps=[
                PlanStep(index=index, text=str(text))
                for index, text in enumerate(steps, start=1)
            ]
        )
        self._ui.print(render_plan(self._plan))

    def _print_plan_rejection(self, data: Mapping[str, object]) -> None:
        decision = data.get("decision")
        feedback = data.get("feedback")
        note = f"[yellow]plan {escape(str(decision))}[/yellow]"
        if feedback:
            note += f" [dim]— {escape(str(feedback))}[/dim]"
        self._ui.print(note)

    async def _reflection(self, data: Mapping[str, object]) -> None:
        await self._close()
        verdict = str(data.get("verdict", ""))
        raw_completed = data.get("completed")
        completed = (
            [int(index) for index in raw_completed] if isinstance(raw_completed, list) else []
        )
        if self._plan is None:
            self._ui.print(reflection_footer(verdict, completed))
            return
        for step in self._plan.steps:
            if step.index in completed:
                step.status = "done"
        pending = self._plan.first_pending()
        if pending is not None and pending.status == "pending":
            pending.status = "active"
        self._ui.print(Group(render_plan(self._plan), reflection_footer(verdict, completed)))
