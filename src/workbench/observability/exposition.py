"""Localhost /metrics exposition and the scheduler-slip sampler (PLAN §8.3/§8.4).

Ruling M7.4: the server binds 127.0.0.1 only — PLAN §1 keeps our entire HTTP
footprint to this one port — and it degrades (warn + disabled) instead of
ever failing a run, because observability must not take the workbench down.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from time import monotonic
from wsgiref.simple_server import WSGIServer

from prometheus_client import start_http_server

from workbench.logging_setup import get_logger
from workbench.observability.metrics import Metrics

logger = get_logger("observability")

# PLAN §1/§4: the only HTTP surface of our own is the *localhost* metrics port.
METRICS_HOST = "127.0.0.1"
_LAG_INTERVAL_S = 1.0
_STOP_JOIN_TIMEOUT_S = 2.0  # bounded joins: teardown must never hang a run


class MetricsServer:
    """Thread-backed /metrics endpoint; port 0 disables, busy port degrades."""

    def __init__(self, metrics: Metrics, *, port: int) -> None:
        self._metrics = metrics
        self._port = port
        self._server: WSGIServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Whether the exposition thread is currently serving."""
        return self._server is not None

    def start(self) -> bool:
        """Bring exposition up; False = disabled (port 0) or degraded (bind failed)."""
        if self._port == 0:
            return False
        try:
            self._server, self._thread = start_http_server(
                self._port, addr=METRICS_HOST, registry=self._metrics.registry
            )
        except OSError as exc:
            logger.warning(
                "metrics exposition disabled: cannot bind %s:%d (%s)",
                METRICS_HOST,
                self._port,
                exc,
            )
            return False
        logger.info("metrics exposition at http://%s:%d/metrics", METRICS_HOST, self._port)
        return True

    def stop(self) -> None:
        """Release the port promptly so restarts and tests never race it."""
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=_STOP_JOIN_TIMEOUT_S)


class LagSampler:
    """Wake every interval and record how late we are — PLAN's event-loop lag.

    The slip between *intended* and *actual* wake-up is exactly what a
    starved event loop feels; sampling it continuously makes a stall show up
    as one large observation instead of being averaged away.
    """

    def __init__(
        self, observe: Callable[[float], None], *, interval_s: float = _LAG_INTERVAL_S
    ) -> None:
        self._observe = observe
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the daemon sampler; idempotent while already running."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="workbench-lag-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal and join (bounded) so tests see a truly stopped sampler."""
        thread = self._thread
        self._thread = None
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=_STOP_JOIN_TIMEOUT_S)

    def _run(self) -> None:
        clock = monotonic
        deadline = clock() + self._interval_s
        while True:
            if self._stop.wait(max(0.0, deadline - clock())):
                return
            now = clock()
            self._observe(max(0.0, now - deadline))
            # After a long stall the next deadline is "now", not a backlog of
            # catch-up ticks that would flood the histogram.
            deadline = max(deadline + self._interval_s, now)
