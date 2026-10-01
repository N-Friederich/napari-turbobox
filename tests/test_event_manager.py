import os
import sys
import weakref

import pytest

from napari_turbobox.event_manager import BoundingBoxEventManager, BoundingBoxSubscriber


class MockSubscriber:
    def __init__(self):
        self.events = []

    def handle_event(self, event_type: str, payload) -> None:
        self.events.append((event_type, payload))


def test_subscribe_unsubscribe_and_emit():
    manager = BoundingBoxEventManager()
    sub = MockSubscriber()

    # Subscribe and emit
    manager.subscribe(sub)
    manager.emit("test_event", 42)
    assert sub.events == [("test_event", 42)]

    # Unsubscribe and emit
    manager.unsubscribe(sub)
    manager.emit("another_event", 100)
    # Events should not change
    assert sub.events == [("test_event", 42)]

    # Double unsubscribe (raises KeyError internally but handled gracefully)
    manager.unsubscribe(sub)


def test_batch_mode_deduplication():
    manager = BoundingBoxEventManager()
    sub = MockSubscriber()
    manager.subscribe(sub)

    with manager.batch():
        manager.emit("event_a", 1)
        manager.emit("event_b", 2)
        manager.emit("event_a", 3)  # Overwrites payload for event_a
        # No events should be received yet
        assert len(sub.events) == 0

    # After exiting batch block, unique events are emitted (event_a with latest payload first, then event_b)
    # Wait, unique_events = {}
    # for event in self._queued_events:
    #     unique_events[event.event_type] = event
    # unique_events holds {"event_a": event_a(3), "event_b": event_b(2)}
    # Order of values depends on dict ordering: event_a was first key inserted, so event_a will be emitted first.
    assert len(sub.events) == 2
    assert ("event_a", 3) in sub.events
    assert ("event_b", 2) in sub.events


def test_nested_batch_mode():
    manager = BoundingBoxEventManager()
    sub = MockSubscriber()
    manager.subscribe(sub)

    with manager.batch():
        manager.emit("event_a", 1)
        with manager.batch():
            manager.emit("event_b", 2)
        # Inner batch exited, but outer batch still active
        assert len(sub.events) == 0

    # Outer batch exited, now all emitted
    assert len(sub.events) == 2


def test_weak_subscriber_garbage_collection():
    manager = BoundingBoxEventManager()

    sub = MockSubscriber()
    manager.subscribe(sub)

    # The subscriber is stored in a WeakSet, so deleting it should remove it from subscribers
    ref = weakref.ref(sub)
    del sub

    # Force garbage collection
    import gc

    gc.collect()

    assert ref() is None
    # Emitting shouldn't raise any error
    manager.emit("event", 123)


if __name__ == "__main__":
    pytest.main([__file__])
