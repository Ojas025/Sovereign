"""Two-loop plumbing: prompts on the main loop, turns on a worker thread.

Ruling M6.2 — the agent loop runs on a private thread + event loop because it
blocks for seconds at a time, while agentui/prompt_toolkit need the main
thread. Prompts flow the other way: the worker asks :class:`PromptBridge` to
schedule a coroutine on the main loop and awaits the wrapped future, so both
loops keep spinning. Every prompt settles behind the event consumer first
(ruling M6.16) and commits the open turn (M6.4) before touching the input.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Protocol, TypeVar

from agentui import Session

from workbench.agent.loop import AgentLoop, TurnOutcome
from workbench.agent.planning import Approval, Approver
from workbench.core.protocols import ConfirmFn
from workbench.core.session import Plan
from workbench.logging_setup import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# A cancelled turn is joined with a bound (M6.12): the TUI must return to the
# prompt promptly while the worker unwinds its own cleanup asynchronously.
_DRAIN_JOIN_TIMEOUT_S = 2.0


class PromptHost(Protocol):
    """The transcript surface a prompt needs: commit before asking (M6.4)."""

    async def close(self) -> None: ...


class PromptBridge:
    """Run a main-loop prompt coroutine on behalf of the worker loop."""

    def __init__(
        self,
        main_loop: asyncio.AbstractEventLoop,
        *,
        settle: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._main_loop = main_loop
        self._settle = settle

    def ask(self, factory: Callable[[], Coroutine[Any, Any, T]]) -> Awaitable[T]:
        """Schedule ``factory()`` on the main loop; await it from the worker."""
        return asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(self._run(factory), self._main_loop)
        )

    async def _run(self, factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
        if self._settle is not None:
            # Let the event consumer drain whatever the turn rendered so far,
            # so this prompt's turn-commit can't interleave with a render.
            await self._settle()
        return await factory()


async def _ask_for_plan(
    bridge: PromptBridge,
    ui: Session,
    host: PromptHost,
    plan: Plan,
) -> Approval:
    async def prompt() -> Approval:
        await host.close()
        message = f"Approve this plan ({len(plan.steps)} steps)?"
        try:
            if await ui.confirm(message, default=True):
                return Approval(decision="approved")
            feedback = (await ui.input("Feedback to revise (empty = cancel):")).strip()
        except KeyboardInterrupt:  # Ctrl-C at the prompt cancels the turn (M6.10)
            return Approval(decision="cancelled")
        if feedback:
            return Approval(decision="rejected", feedback=feedback)
        return Approval(decision="cancelled")

    return await bridge.ask(prompt)


def tui_approver(bridge: PromptBridge, ui: Session, host: PromptHost) -> Approver:
    """Interactive plan approval (ruling M6.10): y / n+feedback / n+empty=cancel."""

    async def approve(plan: Plan, attempt: int) -> Approval:
        _ = attempt  # attempt number is implicit in the printed plan block
        return await _ask_for_plan(bridge, ui, host, plan)

    return approve


async def _confirm_action(
    bridge: PromptBridge,
    ui: Session,
    host: PromptHost,
    message: str,
) -> bool:
    async def prompt() -> bool:
        await host.close()
        try:
            # Dangerous commands default to NO: an unattended Enter must decline.
            return await ui.confirm(message, default=False)
        except KeyboardInterrupt:  # Ctrl-C at the prompt = decline (M6.10)
            return False

    return await bridge.ask(prompt)


def tui_confirm(bridge: PromptBridge, ui: Session, host: PromptHost) -> ConfirmFn:
    """Risk-flag confirmation for tools; runs on the main loop (M6.4/M6.16)."""

    async def confirm(message: str) -> bool:
        return await _confirm_action(bridge, ui, host, message)

    return confirm


def _deliver(done: asyncio.Future[TurnOutcome], value: TurnOutcome | BaseException) -> None:
    if done.done():
        return  # the main loop already cancelled (or consumed) this turn
    if isinstance(value, BaseException):
        done.set_exception(value)
    else:
        done.set_result(value)


class TurnWorker:
    """Owns the worker thread/loop; one in-flight ``AgentLoop.run_turn``."""

    def __init__(self, main_loop: asyncio.AbstractEventLoop) -> None:
        self._main_loop = main_loop
        self._thread: threading.Thread | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[TurnOutcome] | None = None

    async def run(self, agent_loop: AgentLoop, message: str) -> TurnOutcome:
        """Run one turn off-thread; resolves on the main loop (M6.2)."""
        done: asyncio.Future[TurnOutcome] = self._main_loop.create_future()
        self._worker_loop = None
        self._task = None
        thread = threading.Thread(
            target=self._entry,
            args=(agent_loop, message, done),
            name="workbench-turn",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        return await done

    def cancel(self) -> None:
        """Abort the in-flight turn on the worker loop (M6.12)."""
        worker_loop, task = self._worker_loop, self._task
        if worker_loop is not None and task is not None and not task.done():
            worker_loop.call_soon_threadsafe(task.cancel)

    async def drain(self) -> None:
        """Join the worker with a bound so the prompt returns promptly (M6.12)."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, _DRAIN_JOIN_TIMEOUT_S)

    def _entry(
        self,
        agent_loop: AgentLoop,
        message: str,
        done: asyncio.Future[TurnOutcome],
    ) -> None:
        async def work() -> None:
            self._worker_loop = asyncio.get_running_loop()
            self._task = asyncio.create_task(agent_loop.run_turn(message))
            try:
                outcome = await self._task
            except BaseException as exc:  # noqa: BLE001 - delivered, not swallowed
                self._main_loop.call_soon_threadsafe(_deliver, done, exc)
            else:
                self._main_loop.call_soon_threadsafe(_deliver, done, outcome)

        try:
            asyncio.run(work())
        except KeyboardInterrupt:  # SIGINT is handled on the main thread
            self._main_loop.call_soon_threadsafe(
                _deliver, done, asyncio.CancelledError()
            )
        logger.debug("turn worker exited")
