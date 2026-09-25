"""Prometheus instrumentation, localhost exposition, and local stack glue (PLAN §8).

Ruling M7.4: runs call start_observability() exactly once (-p and TUI); the
stack lives only as long as the run, and a disabled or failing exposition
port never fails the workbench itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from workbench.config import Config
from workbench.core.events import EventBus
from workbench.observability.collector import MetricsCollector
from workbench.observability.exposition import LagSampler, MetricsServer
from workbench.observability.metrics import Metrics

__all__ = ["Observability", "start_observability"]


@dataclass
class Observability:
    """A run's live instrumentation; stop() tears down everything it started."""

    metrics: Metrics
    collector: MetricsCollector
    server: MetricsServer
    sampler: LagSampler | None

    def stop(self) -> None:
        """Detach from the bus and release the port; idempotent."""
        self.collector.stop()
        self.server.stop()
        if self.sampler is not None:
            self.sampler.stop()
            self.sampler = None


def start_observability(config: Config, bus: EventBus) -> Observability:
    """Wire bus → metrics and bring exposition up for this run (M7.2/M7.4).

    The lag sampler only runs when there is an exposition to feed — samples
    nobody can scrape are noise.
    """
    metrics = Metrics()
    collector = MetricsCollector(bus, config, metrics)
    server = MetricsServer(metrics, port=config.observability.metrics_port)
    sampler: LagSampler | None = None
    if server.start():
        sampler = LagSampler(metrics.observe_lag)
        sampler.start()
    return Observability(
        metrics=metrics, collector=collector, server=server, sampler=sampler
    )
