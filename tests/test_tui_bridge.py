"""Cross-thread prompt bridge + turn worker (rulings M6.2, M6.4, M6.10, M6.12, M6.16).

The worker thread runs its own event loop exactly like production does
(``asyncio.run`` inside ``TurnWorker._entry`` / ``asyncio.run`` around the
approver), while prompts are scheduled onto the test's main loop — so these
tests prove the two-loop plumbing for real rather than with mocks of it.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from workbench.core.session import Plan, PlanStep
from workbench.tui.bridge import PromptBridge, TurnWorker, tui_approver, tui_confirm


class FakeUI:
    """The Session surface the prompts touch; logs every interaction."""

    def __init__(
        self,
        log: list[str],
        *,
        yes: bool = True,
        feedback: str = "",
        interrupt: bool = False,
    ) -> None:
        self._log = log
        self._yes = yes
        self._feedback = feedback
        self._interrupt = interrupt

    async def confirm(self, message: str, *, default: bool = True) -> bool:
        self._log.append(f"confirm:{message}|default={default}")
        if self._interrupt:
            raise KeyboardInterrupt
        return self._yes

    async def input(self, message: str, *, default: str = "") -> str:
        self._log.append(f"input:{message}")
        return self._feedback


class FakeTranscript:
    """Just the commit-before-prompt surface (ruling M6.4)."""

    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def close(self) -> None:
        self._log.append("close")


def _plan() -> Plan:
    return Plan(
        steps=[PlanStep(index=1, text="step one"), PlanStep(index=2, text="step two")]
    )


# --- PromptBridge -----------------------------------------------------------------


async def test_ask_executes_on_the_main_loop_and_returns_the_answer() -> None:
    main_thread = threading.get_ident()
    bridge = PromptBridge(asyncio.get_running_loop())

    async def prompt() -> str:
        return f"tid={threading.get_ident()}"

    def worker() -> str:
        async def work() -> str:
            return await bridge.ask(prompt)

        return asyncio.run(work())

    answer = await asyncio.to_thread(worker)

    assert answer == f"tid={main_thread}"


async def test_ask_settles_before_running_the_prompt() -> None:
    """Ruling M6.16: prompts wait for the event consumer to drain first."""
    log: list[str] = []
    bridge = PromptBridge(asyncio.get_running_loop(), settle=_recording_settle(log))

    async def prompt() -> str:
        log.append("prompt")
        return "ok"

    def worker() -> str:
        async def work() -> str:
            return await bridge.ask(prompt)

        return asyncio.run(work())

    assert await asyncio.to_thread(worker) == "ok"
    assert log == ["settle", "prompt"]


def _recording_settle(log: list[str]) -> Any:
    async def settle() -> None:
        log.append("settle")

    return settle


# --- approver (ruling M6.10) ---------------------------------------------------------


async def test_approver_yes_approves_after_committing_the_turn() -> None:
    log: list[str] = []
    bridge = PromptBridge(asyncio.get_running_loop())
    approver = tui_approver(bridge, FakeUI(log, yes=True), FakeTranscript(log))

    approval = await asyncio.to_thread(lambda: asyncio.run(approver(_plan(), 1)))

    assert approval.decision == "approved"
    assert log[0] == "close"  # turn committed before the prompt (M6.4)
    assert log[1].startswith("confirm:Approve this plan (2 steps)?")
    assert "default=True" in log[1]  # plan approval defaults to yes


async def test_approver_no_with_feedback_rejects_and_forwards_it() -> None:
    log: list[str] = []
    bridge = PromptBridge(asyncio.get_running_loop())
    approver = tui_approver(
        bridge, FakeUI(log, yes=False, feedback="  add a test step  "), FakeTranscript(log)
    )

    approval = await asyncio.to_thread(lambda: asyncio.run(approver(_plan(), 2)))

    assert approval.decision == "rejected"
    assert approval.feedback == "add a test step"


async def test_approver_no_with_empty_feedback_cancels_the_turn() -> None:
    log: list[str] = []
    bridge = PromptBridge(asyncio.get_running_loop())
    approver = tui_approver(
        bridge, FakeUI(log, yes=False, feedback=""), FakeTranscript(log)
    )

    approval = await asyncio.to_thread(lambda: asyncio.run(approver(_plan(), 1)))

    assert approval.decision == "cancelled"
    assert approval.feedback == ""


async def test_approver_ctrl_c_at_the_prompt_cancels() -> None:
    log: list[str] = []
    bridge = PromptBridge(asyncio.get_running_loop())
    approver = tui_approver(
        bridge, FakeUI(log, interrupt=True), FakeTranscript(log)
    )

    approval = await asyncio.to_thread(lambda: asyncio.run(approver(_plan(), 1)))

    assert approval.decision == "cancelled"


# --- tool confirmation ---------------------------------------------------------------


async def test_tool_confirm_declines_by_default_and_passes_the_prompt_through() -> None:
    log: list[str] = []
    bridge = PromptBridge(asyncio.get_running_loop())
    confirm = tui_confirm(bridge, FakeUI(log, yes=False), FakeTranscript(log))

    accepted = await asyncio.to_thread(
        lambda: asyncio.run(confirm("Run command? [risk:privilege] sudo ls"))
    )

    assert accepted is False
    assert log[0] == "close"
    assert "sudo ls" in log[1]
    assert "default=False" in log[1]  # dangerous commands default to no


async def test_tool_confirm_accepts_an_explicit_yes() -> None:
    log: list[str] = []
    bridge = PromptBridge(asyncio.get_running_loop())
    confirm = tui_confirm(bridge, FakeUI(log, yes=True), FakeTranscript(log))

    accepted = await asyncio.to_thread(
        lambda: asyncio.run(confirm("Run command? [x] make"))
    )

    assert accepted is True


# --- TurnWorker (rulings M6.2, M6.12) ---------------------------------------------------


async def test_turn_worker_resolves_with_the_loops_outcome() -> None:
    class FakeAgentLoop:
        async def run_turn(self, message: str) -> str:
            return f"done:{message}"

    worker = TurnWorker(asyncio.get_running_loop())

    result = await worker.run(FakeAgentLoop(), "hello")  # type: ignore[arg-type]

    assert result == "done:hello"


async def test_turn_worker_surfaces_worker_exceptions() -> None:
    class BoomLoop:
        async def run_turn(self, message: str) -> str:
            raise RuntimeError("boom")

    worker = TurnWorker(asyncio.get_running_loop())

    with pytest.raises(RuntimeError, match="boom"):
        await worker.run(BoomLoop(), "x")  # type: ignore[arg-type]


async def test_cancel_aborts_the_in_flight_turn() -> None:
    started = threading.Event()
    cancelled = threading.Event()

    class SlowLoop:
        async def run_turn(self, message: str) -> str:
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "unreachable"

    worker = TurnWorker(asyncio.get_running_loop())
    turn = asyncio.create_task(worker.run(SlowLoop(), "x"))  # type: ignore[arg-type]
    assert await asyncio.to_thread(started.wait, 10)

    worker.cancel()

    with pytest.raises(asyncio.CancelledError):
        await turn
    assert await asyncio.to_thread(cancelled.wait, 10)
    await worker.drain()


async def test_cancel_without_a_running_turn_is_a_noop() -> None:
    worker = TurnWorker(asyncio.get_running_loop())
    worker.cancel()
    await worker.drain()
