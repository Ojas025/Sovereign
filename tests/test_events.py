"""Event bus: typed pi-style event delivery, subscription lifecycle, isolation."""

import threading

import pytest

from workbench.core.events import Event, EventBus


def make_event(kind: str) -> Event:
    return Event(kind=kind, data={"turn": 1})


class TestDelivery:
    def test_subscriber_receives_only_its_kind(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("turn_start", received.append)

        bus.emit(make_event("turn_start"))
        bus.emit(make_event("turn_end"))

        assert [e.kind for e in received] == ["turn_start"]

    def test_wildcard_subscriber_receives_everything(self) -> None:
        bus = EventBus()
        received: list[str] = []
        bus.subscribe(None, lambda e: received.append(e.kind))

        bus.emit(make_event("turn_start"))
        bus.emit(make_event("route_decided"))

        assert received == ["turn_start", "route_decided"]

    def test_handlers_run_in_subscription_order(self) -> None:
        bus = EventBus()
        order: list[str] = []
        bus.subscribe("turn_start", lambda _: order.append("first"))
        bus.subscribe("turn_start", lambda _: order.append("second"))

        bus.emit(make_event("turn_start"))

        assert order == ["first", "second"]

    def test_unsubscribe_stops_delivery(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        unsubscribe = bus.subscribe("turn_start", received.append)

        unsubscribe()
        bus.emit(make_event("turn_start"))

        assert received == []


class TestIsolation:
    def test_handler_exception_is_logged_and_does_not_break_other_handlers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus = EventBus()
        received: list[Event] = []

        def boom(_: Event) -> None:
            raise RuntimeError("bad subscriber")

        bus.subscribe("turn_start", boom)
        bus.subscribe("turn_start", received.append)

        with caplog.at_level("ERROR", logger="workbench.events"):
            bus.emit(make_event("turn_start"))

        # The emitter must never crash; later subscribers still receive the event.
        assert len(received) == 1
        assert "bad subscriber" in caplog.text


class TestThreadSafety:
    def test_concurrent_emit_from_multiple_threads_delivers_all(self) -> None:
        bus = EventBus()
        count = 0
        lock = threading.Lock()
        total = 200

        def handler(_: Event) -> None:
            nonlocal count
            with lock:
                count += 1

        bus.subscribe(None, handler)

        def hammer() -> None:
            for _ in range(total // 4):
                bus.emit(make_event("turn_start"))

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert count == total
