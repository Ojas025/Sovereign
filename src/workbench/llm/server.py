"""Process manager for llama-serve in router mode.

Lifecycle: lazy spawn on first use → poll ``/health`` until ready → idle-stop
after ``server.idle_timeout_s`` without activity → SIGTERM (SIGKILL after a
grace period) so VRAM is freed. Unexpected exits restart with exponential
backoff up to ``server.restart_max``, then surface as an ``error`` event.
"""

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Literal

import httpx

from workbench.config import Config
from workbench.core.events import Event, EventBus
from workbench.logging_setup import get_logger

logger = get_logger("server")

ServerState = Literal["stopped", "starting", "ready", "failed"]

_HEALTH_TIMEOUT_S = 60.0
_HEALTH_POLL_INTERVAL_S = 0.05
_STOP_GRACE_S = 10.0
_RESTART_BACKOFF_BASE_S = 1.0
_IDLE_INTERVAL_RANGE_S = (0.01, 1.0)
_MODELS_TIMEOUT_S = 5.0


class ServerError(Exception):
    """llama-serve could not start, never became ready, or a call failed."""


class ServerManager:
    """Owns the llama-serve child process for a whole workbench run."""

    def __init__(
        self,
        config: Config,
        bus: EventBus,
        *,
        command: Sequence[str] | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._command = list(command) if command is not None else self._default_command()
        self._base_url = f"http://{config.server.host}:{config.server.port}"
        self._state: ServerState = "stopped"
        self._stopping = False
        self._restarts = 0
        self._last_used = monotonic()
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._drain_task: asyncio.Task[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None

    # --- public surface --------------------------------------------------

    @property
    def state(self) -> ServerState:
        return self._state

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    @property
    def command(self) -> tuple[str, ...]:
        """Effective spawn command (defaults built from config, or the test override)."""
        return tuple(self._command)

    def _default_command(self) -> list[str]:
        # Wrapper flags verified against ~/.local/bin/llama-serve in the M2 spike.
        models_dir = Path(self._config.models.search_paths[0]).expanduser()
        return [
            self._config.server.binary,
            "--router",
            "--models-dir",
            str(models_dir),
            "--models-max",
            str(self._config.server.models_max),
            "--port",
            str(self._config.server.port),
        ]

    async def ensure_running(self) -> None:
        """Spawn the server if needed and wait until ``/health`` is green."""
        async with self._lock:
            process = self._process
            if (
                self._state == "ready"
                and process is not None
                and process.returncode is None
            ):
                self._last_used = monotonic()
                return
            self._restarts = 0  # a manual start grants a fresh restart budget
            await self._start_locked()

    def touch(self) -> None:
        """Mark activity so the idle-stop timer restarts."""
        self._last_used = monotonic()

    async def list_models(self) -> list[str]:
        """Model ids llama-server has registered (profile ↔ router id mapping)."""
        try:
            async with httpx.AsyncClient(
            base_url=self._base_url, timeout=_MODELS_TIMEOUT_S
        ) as http:
                response = await http.get("/v1/models")
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise ServerError(f"cannot list models: {exc}") from exc
        return [str(entry["id"]) for entry in payload.get("data", [])]

    async def stop(self, *, reason: str = "explicit") -> None:
        """Terminate the child; idempotent and safe to call when not running."""
        async with self._lock:
            process = self._process
            was_running = process is not None and process.returncode is None
            self._stopping = True

            current = asyncio.current_task()
            if self._idle_task is not None and self._idle_task is not current:
                self._idle_task.cancel()
            self._idle_task = None

            if process is not None:
                await self._kill_child(process)
            self._process = None
            self._monitor_task = None
            self._state = "stopped"
            if was_running:
                self._bus.emit(Event("server_stopped", {"reason": reason}))
                logger.info("llama-serve stopped (%s)", reason)

    # --- lifecycle internals ---------------------------------------------

    async def _start_locked(self) -> None:
        """Spawn the child and wait for readiness. Caller holds the lock."""
        self._stopping = False
        self._state = "starting"
        try:
            process = await asyncio.create_subprocess_exec(
                *self._command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
        except OSError as exc:
            self._state = "failed"
            message = f"cannot start llama-server: {exc}"
            self._bus.emit(Event("error", {"message": message}))
            raise ServerError(message) from exc

        self._process = process
        logger.info("llama-serve spawned (pid %s): %s", process.pid, self._command)
        self._monitor_task = asyncio.create_task(self._watch(process))
        self._drain_task = asyncio.create_task(self._drain_stderr(process))

        try:
            await self._wait_ready(process)
        except ServerError:
            await self._kill_child(process)
            self._process = None
            self._state = "failed"
            raise
        except asyncio.CancelledError:
            await self._kill_child(process)
            self._process = None
            self._state = "stopped"
            raise

        self._state = "ready"
        self._last_used = monotonic()
        self._bus.emit(
            Event("server_started", {"port": self._config.server.port, "pid": process.pid})
        )
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def _wait_ready(self, process: asyncio.subprocess.Process) -> None:
        deadline = monotonic() + _HEALTH_TIMEOUT_S
        async with httpx.AsyncClient(base_url=self._base_url, timeout=2.0) as http:
            while monotonic() < deadline:
                if process.returncode is not None:
                    raise ServerError(
                        f"llama-server exited during startup (code {process.returncode})"
                    )
                try:
                    response = await http.get("/health")
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass  # not listening yet
                await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)
        raise ServerError(f"llama-server not ready within {_HEALTH_TIMEOUT_S:.0f}s")

    async def _watch(self, process: asyncio.subprocess.Process) -> None:
        """React to an unexpected child exit: restart with backoff, or give up."""
        code = await process.wait()
        if self._stopping:
            return
        logger.warning("llama-server exited unexpectedly (code %s)", code)

        async with self._lock:
            if self._stopping or self._state != "ready":
                return  # stop()/startup failure already handled this exit
            if self._process is process:
                self._process = None

            if self._restarts >= self._config.server.restart_max:
                self._state = "failed"
                message = (
                    f"llama-server crashed (exit {code}); restart budget exhausted"
                )
                self._bus.emit(Event("error", {"message": message}))
                logger.error("%s", message)
                return

            self._restarts += 1
            backoff = _RESTART_BACKOFF_BASE_S * 2 ** (self._restarts - 1)
            logger.info(
                "restarting llama-server in %.1fs (attempt %d/%d)",
                backoff,
                self._restarts,
                self._config.server.restart_max,
            )
            await asyncio.sleep(backoff)
            try:
                await self._start_locked()
            except ServerError:
                return  # already marked failed and reported

    async def _idle_loop(self) -> None:
        low, high = _IDLE_INTERVAL_RANGE_S
        interval = min(max(self._config.server.idle_timeout_s / 4, low), high)
        task = asyncio.current_task()
        while True:
            await asyncio.sleep(interval)
            if self._idle_task is not task:
                return  # superseded or cleared by stop()
            if self._state != "ready":
                continue
            if monotonic() - self._last_used >= self._config.server.idle_timeout_s:
                await self.stop(reason="idle")

    async def _kill_child(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), _STOP_GRACE_S)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        """Keep llama-server's stderr pipe drained (prevents child blocking)."""
        assert process.stderr is not None
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            logger.debug("llama-server: %s", line.decode(errors="replace").rstrip())
