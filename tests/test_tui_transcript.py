"""Event stream → agentui rendering: turns, plan/tool blocks, notices (PLAN §7).

The TranscriptController is the single consumer of the typed bus inside the
TUI (ruling M6.2). Fakes mirror exactly the agentui surface it touches —
Session.print / Session.assistant_turn and Turn's segment API — so the
lifecycle rules (ruling M6.4) are assertable without a terminal.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rich.console import Console

from workbench.core.events import Event
from workbench.tui.state import TuiState
from workbench.tui.transcript import TranscriptController

_WRITE_ARGS = '{"path": "hello.txt", "content": "hello world\\n"}'
_EDIT_ARGS = (
    '{"path": "a.txt", "old_string": "old text", "new_string": "new text", '
    '"replace_all": false}'
)


class FakeToolCtx:
    """Stands in for agentui's ToolCallContext inside a turn."""

    def __init__(self, name: str, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments
        self.running = False
        self.completed = False
        self.error_message: str | None = None

    async def __aenter__(self) -> FakeToolCtx:
        self.running = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def complete(self, result: str | None = None) -> None:
        self.completed = True

    async def error(self, message: str) -> None:
        self.error_message = message


class FakeTurn:
    """Records the segment operations the controller performs."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.markdown: list[str] = []
        self.blocks: list[FakeToolCtx] = []
        self.diffs: list[tuple[str, str]] = []

    async def __aenter__(self) -> FakeTurn:
        self.opened = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True
        return None

    def append_markdown_sync(self, text: str) -> None:
        self.markdown.append(text)

    def tool_call(self, name: str, arguments: Any) -> FakeToolCtx:
        ctx = FakeToolCtx(name, arguments)
        self.blocks.append(ctx)
        return ctx

    def diff(self, path: str, patch: str) -> object:
        self.diffs.append((path, patch))
        return object()


class FakeUI:
    """Stands in for agentui's Session: print + assistant_turn only."""

    def __init__(self) -> None:
        self.turns: list[FakeTurn] = []
        self.prints: list[SimpleNamespace] = []

    def print(self, *objects: object, **kwargs: object) -> None:
        # Snapshot which turns were already committed when this hit scrollback.
        self.prints.append(
            SimpleNamespace(objects=objects, closed=[t.closed for t in self.turns])
        )

    def assistant_turn(self) -> FakeTurn:
        turn = FakeTurn()
        self.turns.append(turn)
        return turn


def rendered(obj: object) -> str:
    console = Console(width=100, force_terminal=False, record=True)
    console.print(obj)
    return console.export_text()


def printed_text(ui: FakeUI, index: int = -1) -> str:
    out = []
    for obj in ui.prints[index].objects:
        out.append(obj if isinstance(obj, str) else rendered(obj))
    return "\n".join(out)


def make_controller() -> tuple[TranscriptController, FakeUI, TuiState]:
    ui = FakeUI()
    state = TuiState()
    return TranscriptController(ui, state), ui, state


def start_message(*, model: str = "small-model") -> Event:
    return Event("message_start", {"model": model, "role": "assistant"})


# --- transcript lifecycle -------------------------------------------------------


async def test_streaming_renders_one_turn_of_markdown() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_update", {"text": "Hel"}))
    await controller.handle(Event("message_update", {"text": "lo"}))
    await controller.handle(Event("message_end", {"stop_reason": "stop"}))
    await controller.handle(Event("turn_end", {"outcome": "completed"}))

    assert len(ui.turns) == 1
    assert ui.turns[0].markdown == ["Hel", "lo"]
    assert ui.turns[0].opened and ui.turns[0].closed


async def test_turn_stays_open_across_message_end_until_turn_end() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_end", {"stop_reason": "tool_calls"}))

    assert ui.turns[0].closed is False  # tools follow: the block joins this turn

    await controller.handle(Event("turn_end", {"outcome": "completed"}))

    assert ui.turns[0].closed is True


async def test_close_commits_the_turn_so_a_prompt_never_corrupts_it() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_update", {"text": "thinking"}))
    await controller.close()  # what the bridge does before any prompt

    assert ui.turns[0].closed is True

    # a later round opens a fresh turn instead of resuming the committed one
    await controller.handle(start_message())
    await controller.handle(Event("message_end", {"stop_reason": "stop"}))
    await controller.handle(Event("turn_end", {"outcome": "completed"}))

    assert len(ui.turns) == 2
    assert ui.turns[1].opened and ui.turns[1].closed


# --- tool blocks -----------------------------------------------------------------


async def test_write_tool_compacts_args_and_renders_an_inline_diff() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_end", {"stop_reason": "tool_calls"}))
    await controller.handle(
        Event(
            "tool_execution_start",
            {"call_id": "c1", "tool": "write", "arguments": _WRITE_ARGS},
        )
    )
    await controller.handle(
        Event(
            "tool_execution_end",
            {
                "call_id": "c1",
                "tool": "write",
                "status": "ok",
                "duration_s": 0.1,
                "output": "wrote 12 bytes to hello.txt",
            },
        )
    )

    block = ui.turns[0].blocks[0]
    assert block.name == "write"
    assert block.running is True  # badge: running while the tool executes
    assert block.completed is True
    # compact args: the payload shows up as a diff, not as raw JSON
    assert block.arguments == {"path": "hello.txt"}

    assert len(ui.turns[0].diffs) == 1
    path, patch = ui.turns[0].diffs[0]
    assert path == "hello.txt"
    assert "--- /dev/null" in patch
    assert "+++ b/hello.txt" in patch
    assert "+hello world" in patch


async def test_edit_tool_diff_shows_the_replacement() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_end", {"stop_reason": "tool_calls"}))
    await controller.handle(
        Event(
            "tool_execution_start",
            {"call_id": "c1", "tool": "edit", "arguments": _EDIT_ARGS},
        )
    )
    await controller.handle(
        Event(
            "tool_execution_end",
            {
                "call_id": "c1",
                "tool": "edit",
                "status": "ok",
                "duration_s": 0.1,
                "output": "replaced old text",
            },
        )
    )

    block = ui.turns[0].blocks[0]
    assert block.arguments == {"path": "a.txt", "replace_all": False}
    assert block.completed is True

    path, patch = ui.turns[0].diffs[0]
    assert path == "a.txt"
    assert "-old text" in patch
    assert "+new text" in patch


async def test_bash_ok_shows_output_as_a_code_block() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_end", {"stop_reason": "tool_calls"}))
    await controller.handle(
        Event(
            "tool_execution_start",
            {"call_id": "c1", "tool": "bash", "arguments": '{"command": "ls"}'},
        )
    )
    await controller.handle(
        Event(
            "tool_execution_end",
            {
                "call_id": "c1",
                "tool": "bash",
                "status": "ok",
                "duration_s": 0.2,
                "output": "hello.txt\n",
            },
        )
    )

    block = ui.turns[0].blocks[0]
    assert block.completed is True
    assert ui.turns[0].markdown == ["```text\nhello.txt\n\n```"]
    assert ui.turns[0].diffs == []  # bash never renders a diff


async def test_failed_tool_expands_with_the_error_output() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_end", {"stop_reason": "tool_calls"}))
    await controller.handle(
        Event(
            "tool_execution_start",
            {"call_id": "c1", "tool": "bash", "arguments": '{"command": "rm /"}'},
        )
    )
    await controller.handle(
        Event(
            "tool_execution_end",
            {
                "call_id": "c1",
                "tool": "bash",
                "status": "error",
                "duration_s": 0.0,
                "output": "blocked: command rejected by guardrails (deny:rm_root)",
            },
        )
    )

    block = ui.turns[0].blocks[0]
    assert block.error_message is not None
    assert "deny:rm_root" in block.error_message
    assert block.completed is False


async def test_tool_is_recreated_at_end_when_a_prompt_closed_the_turn() -> None:
    """Confirm prompt mid-tool: the start-block was committed, the end re-renders it."""
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_end", {"stop_reason": "tool_calls"}))
    await controller.handle(
        Event(
            "tool_execution_start",
            {"call_id": "c1", "tool": "bash", "arguments": '{"command": "make"}'},
        )
    )
    await controller.close()  # bridged confirmation prompt happened here
    await controller.handle(
        Event(
            "tool_execution_end",
            {
                "call_id": "c1",
                "tool": "bash",
                "status": "ok",
                "duration_s": 1.0,
                "output": "built ok",
            },
        )
    )

    assert len(ui.turns) == 2
    assert ui.turns[0].closed and ui.turns[1].opened
    block = ui.turns[1].blocks[0]
    assert block.name == "bash"
    assert block.arguments == {"command": "make"}
    assert block.completed is True


# --- plan, reflection, notices ----------------------------------------------------


async def test_plan_proposed_commits_the_turn_then_prints_the_block() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_update", {"text": "planning…"}))
    await controller.handle(Event("message_end", {"stop_reason": "stop"}))
    await controller.handle(
        Event("plan_proposed", {"attempt": 1, "steps": ["create file", "verify"]})
    )

    assert ui.turns[0].closed is True
    assert ui.prints[-1].closed == [True]  # block hit committed scrollback
    text = printed_text(ui)
    assert "○ 1. create file" in text
    assert "○ 2. verify" in text


async def test_plan_approval_and_reflection_update_the_rendered_plan() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(
        Event("plan_proposed", {"attempt": 1, "steps": ["create file", "verify"]})
    )
    await controller.handle(Event("plan_approved", {"attempt": 1}))
    await controller.handle(
        Event("reflection", {"verdict": "done", "completed": [1]})
    )

    assert "plan approved" in printed_text(ui, 1)
    final = printed_text(ui, 2)
    assert "✓ 1. create file" in final
    assert "● 2. verify" in final  # reflection marks the first pending step active
    assert "DONE" in final
    assert "1" in final


async def test_plan_rejection_prints_the_feedback_note() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(
        Event(
            "plan_rejected",
            {"decision": "rejected", "attempt": 1, "feedback": "add a test step"},
        )
    )

    text = printed_text(ui)
    assert "rejected" in text
    assert "add a test step" in text


async def test_route_escalation_prints_a_notice() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(start_message())
    await controller.handle(Event("message_end", {"stop_reason": "stop"}))
    await controller.handle(
        Event("route_escalated", {"from_tier": "small", "to_tier": "mid", "failures": 2})
    )

    assert ui.turns[0].closed is True
    assert "escalated small→mid" in printed_text(ui)
    assert "2 tool failures" in printed_text(ui)


async def test_error_event_prints_a_visible_line() -> None:
    controller, ui, _state = make_controller()

    await controller.handle(Event("error", {"stage": "llm", "message": "boom"}))

    assert "boom" in printed_text(ui)


# --- status state projection (feeding T6.2's status line) -------------------------


async def test_state_tracks_route_usage_and_turn_reset() -> None:
    controller, ui, state = make_controller()

    await controller.handle(Event("agent_start", {"session_id": "abc123", "mode": "tui"}))
    await controller.handle(Event("turn_start", {"turn_id": "t1", "message": "hi"}))
    await controller.handle(
        Event(
            "route_decided",
            {
                "tier": "mid",
                "reason": "needs_tools",
                "intent": "code_gen",
                "difficulty": 2.5,
                "confidence": 0.8,
                "backend": "heuristic",
                "model": "coder7b",
                "suggest_plan": True,
            },
        )
    )
    await controller.handle(start_message())
    await controller.handle(
        Event(
            "message_end",
            {
                "stop_reason": "tool_calls",
                "ttft_s": 0.05,
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
            },
        )
    )

    assert state.session_id == "abc123"
    assert state.tier == "mid"
    assert state.model == "coder7b"
    assert state.intent == "code_gen"
    assert state.rounds == 1
    assert state.tokens == 140
    assert state.prompt_tokens == 100

    await controller.handle(Event("turn_end", {"outcome": "completed", "rounds": 1}))
    await controller.handle(Event("turn_start", {"turn_id": "t2", "message": "again"}))

    assert state.rounds == 0 and state.tokens == 0  # per-turn counters reset
    assert state.last_turn["outcome"] == "completed"


# --- interrupt hook + plan accessors (rulings M6.15 / resume) ---------------------------


async def test_on_turn_opened_hook_fires_for_every_turn() -> None:
    """The app reinstalls ITS Ctrl-C handler after agentui's Turn claims SIGINT."""
    opened: list[int] = []
    ui = FakeUI()
    controller = TranscriptController(
        ui, TuiState(), on_turn_opened=lambda: opened.append(1)
    )

    await controller.handle(start_message())
    await controller.handle(Event("turn_end", {"outcome": "completed"}))
    await controller.handle(start_message())
    await controller.handle(Event("turn_end", {"outcome": "completed"}))

    assert opened == [1, 1]  # once per turn open, never for closed turns


async def test_plan_accessor_exposes_the_last_rendered_plan_and_clears_on_resume() -> None:
    controller, _ui, _state = make_controller()
    assert controller.plan is None

    await controller.handle(
        Event("plan_proposed", {"attempt": 1, "steps": ["write the file"]})
    )

    assert controller.plan is not None
    assert [step.text for step in controller.plan.steps] == ["write the file"]

    controller.clear_plan()  # /resume swaps sessions: the old plan must not leak
    assert controller.plan is None


async def test_thinking_indicator_appears_at_start_and_clears_on_update() -> None:
    class TurnWithSegments(FakeTurn):
        def __init__(self) -> None:
            super().__init__()
            self._segments: list[object] = []
            self.rerenders = 0

        def _rerender(self) -> None:
            self.rerenders += 1

    class UIWithSegments(FakeUI):
        def assistant_turn(self) -> TurnWithSegments:
            turn = TurnWithSegments()
            self.turns.append(turn)
            return turn

    ui = UIWithSegments()
    controller = TranscriptController(ui, TuiState())

    # message_start -> thinking indicator attached to segments
    await controller.handle(start_message())
    turn = ui.turns[0]
    assert len(turn._segments) == 1
    assert "thinking" in str(turn._segments[0])

    # message_update -> thinking indicator removed from segments
    await controller.handle(Event("message_update", {"text": "Hello"}))
    assert len(turn._segments) == 0
    assert turn.markdown == ["Hello"]

    # turn_end
    await controller.handle(Event("turn_end", {"outcome": "completed"}))


async def test_pii_redacted_event_renders_notice() -> None:
    ui = FakeUI()
    controller = TranscriptController(ui, TuiState())

    await controller.handle(
        Event(
            "pii_redacted",
            {
                "original": "email secret@example.com",
                "redacted": "email [EMAIL]",
                "counts": {"email": 1},
            },
        )
    )

    output = printed_text(ui)
    assert "PII redacted from prompt" in output
    assert "email: 1" in output
    assert "Sanitized prompt: email [EMAIL]" in output

