"""Interactive session: the main loop owns the terminal, a worker runs turns.

Wiring (rulings M6.2 / M6.16): ``TurnWorker`` runs ``AgentLoop.run_turn`` on a
private thread + event loop; bus events hop in order onto the main loop into a
queue that a single consumer task renders through ``TranscriptController``;
prompts the worker needs (plan approval, risk confirmations) come back through
``PromptBridge``, which settles behind the consumer first so a prompt's
turn-commit can never interleave with a render. Ctrl-C is owned by the app:
agentui's per-Turn SIGINT handler is replaced right after each turn opens
(``on_turn_opened``, M6.15), and aborts cancel + drain the worker (M6.12).
"""

from __future__ import annotations

import asyncio
import signal
import threading
from pathlib import Path
from types import FrameType

from agentui import Session
from rich.markup import escape

from workbench.agent.loop import AgentLoop, TurnOutcome
from workbench.config import Config
from workbench.core.bootstrap import build_registry, build_tools, resolve_router
from workbench.core.events import Event, EventBus
from workbench.core.protocols import ToolContext
from workbench.core.session import Session as SessionState
from workbench.core.session import SessionStore
from workbench.llm.client import LLMClientHttp
from workbench.llm.server import ServerError, ServerManager
from workbench.logging_setup import get_logger
from workbench.routing.policy import TierPolicy
from workbench.tui.bridge import PromptBridge, TurnWorker, tui_approver, tui_confirm
from workbench.tui.commands import CommandContext, register_commands
from workbench.tui.state import TuiState
from workbench.tui.status import build_status_line
from workbench.tui.transcript import TranscriptController

logger = get_logger(__name__)

_PROMPT = "> "

# Ctrl-C while the loop is NOT running: conventional SIGINT exit status.
_INTERRUPT_STATUS = 130


def run_tui(config: Config) -> int:
    """Interactive entry (ruling M6.11): own the terminal until /quit or Ctrl-D."""
    try:
        return asyncio.run(_interactive(config))
    except SystemExit as exc:  # agentui's /quit raises SystemExit
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    except KeyboardInterrupt:
        return _INTERRUPT_STATUS


async def _interactive(config: Config) -> int:
    bus = EventBus()
    registry = build_registry(config)
    workspace = Path(config.runtime.workspace_root).expanduser()
    store = SessionStore(Path(config.agent.session_dir).expanduser())
    state = TuiState()
    manager = ServerManager(config, bus)
    router = resolve_router(registry, config)  # fail fast on a bad backend
    tools = build_tools(registry)
    session = SessionState.create(store=store, workspace=str(workspace))
    main_loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    main_ident = threading.get_ident()

    queue: asyncio.Queue[Event] = asyncio.Queue()
    idle = asyncio.Event()  # set when the consumer has rendered every queued event
    idle.set()

    def _put(event: Event) -> None:
        idle.clear()
        queue.put_nowait(event)

    def enqueue(event: Event) -> None:
        # The bus fires on any thread. Main-loop callers put inline so ``idle``
        # stays honest (a deferred put would let settle() pass before the put);
        # the worker thread hops via call_soon_threadsafe, which preserves the
        # stream's order on the main loop.
        if threading.get_ident() == main_ident:
            _put(event)
        else:
            main_loop.call_soon_threadsafe(_put, event)

    async def settle() -> None:
        # Prompts and post-turn prints wait until the stream is rendered (M6.16).
        await idle.wait()

    bridge = PromptBridge(main_loop, settle=settle)
    worker = TurnWorker(main_loop)

    def request_abort() -> None:
        """Cancel the main task; the abort path turns that into a turn cancel (M6.15)."""
        if main_task is not None and not main_task.done():
            main_task.cancel()

    def interrupt(_signum: int, _frame: FrameType | None) -> None:
        """SIGINT while NOT in the raw-mode prompt (which handles ^C itself)."""
        request_abort()

    def reinstall_interrupt() -> None:
        # agentui's Turn claims SIGINT in __aenter__; put ours straight back so
        # Ctrl-C keeps aborting the turn instead of the consumer task (M6.15).
        signal.signal(signal.SIGINT, interrupt)

    previous_sigint = signal.signal(signal.SIGINT, interrupt)

    async with Session() as ui:
        transcript = TranscriptController(ui, state, on_turn_opened=reinstall_interrupt)
        confirm = tui_confirm(bridge, ui, transcript)
        approver = tui_approver(bridge, ui, transcript)
        current_session = session

        def make_loop() -> AgentLoop:
            return AgentLoop(
                client=LLMClientHttp(manager.base_url),
                router=router,
                policy=TierPolicy(config.routing),
                tools=tools,
                session=current_session,
                bus=bus,
                config=config,
                context=ToolContext(workspace_root=workspace, confirm=confirm),
                approver=approver,
            )

        current_loop = make_loop()

        def on_resume(loaded: SessionState) -> None:
            nonlocal current_session, current_loop
            current_session = loaded
            current_loop = make_loop()
            state.session_id = loaded.id
            transcript.clear_plan()  # the old plan must not show under /plan

        register_commands(
            ui,
            CommandContext(
                config=config,
                state=state,
                store=store,
                registry=registry,
                session=lambda: current_session,
                plan=lambda: current_session.plan or transcript.plan,
                resume=on_resume,
            ),
        )

        async def consume() -> None:
            while True:
                event = await queue.get()
                try:
                    await transcript.handle(event)
                except Exception:  # a render bug must not wedge the whole TUI
                    logger.exception("failed to render %s", event.kind)
                if queue.empty():
                    idle.set()

        consumer = asyncio.create_task(consume())
        unsubscribe = bus.subscribe(None, enqueue)

        async def abort_turn() -> None:
            """Cancel the worker, drop its stragglers, commit the turn (M6.12)."""
            worker.cancel()
            await worker.drain()
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await settle()
            await transcript.close()

        try:
            bus.emit(Event("agent_start", {"session_id": session.id, "mode": "tui"}))
            await settle()  # first paint: status line sees the session id

            while True:
                toolbar = build_status_line(
                    state, config, pinned=current_session.pinned_tier is not None
                )
                try:
                    text = (await ui.prompt(_PROMPT, bottom_toolbar=toolbar)).strip()
                except KeyboardInterrupt:
                    ui.print("[dim]ctrl-c at the prompt is a no-op · /quit exits[/dim]")
                    continue
                except EOFError:  # Ctrl-D
                    break
                if text.startswith("/"):
                    await ui.handle_command(text)  # /quit raises SystemExit
                    continue
                if not text:
                    continue
                try:
                    await manager.ensure_running()
                    outcome = await worker.run(current_loop, text)
                    await settle()  # the turn's events are all rendered (M6.16)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        current.uncancel()  # the app survives its own Ctrl-C
                    await abort_turn()
                    ui.print("[dim]turn cancelled[/dim]")
                    continue
                except ServerError as error:
                    # The error event renders the red line; settle so it lands
                    # before the next prompt rather than under it.
                    bus.emit(Event("error", {"stage": "server", "message": str(error)}))
                    await settle()
                    continue
                except Exception as error:
                    logger.exception("turn crashed")
                    await abort_turn()
                    ui.print(f"[red]turn failed: {escape(str(error))}[/red]")
                    continue
                _print_outcome(ui, outcome)
        finally:
            bus.emit(Event("agent_end", {"outcome": "quit"}))
            unsubscribe()
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass
            await manager.stop(reason="tui exit")
            signal.signal(signal.SIGINT, previous_sigint)
    return 0


def _print_outcome(ui: Session, outcome: TurnOutcome) -> None:
    """Render the one outcome the stream never shows: budget (ruling M6.14).

    ``completed`` text arrived as streamed messages and ``error``/``cancelled``
    lines come from the error/plan_rejected events — printing their text again
    would duplicate them.
    """
    if outcome.outcome == "budget":
        ui.print(f"[dim]{escape(str(outcome.text))}[/dim]")
