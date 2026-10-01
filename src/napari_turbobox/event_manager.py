"""Small publish/subscribe helper for bounding box change events, with batching."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol
from weakref import WeakSet


class BoundingBoxSubscriber(Protocol):
    """Anything with a ``handle_event(event_type, payload)`` method.

    TurboBoxLayer emits ``"added"``, ``"cleared"`` and ``"updated"`` with the
    current (N, 2, D) boxes as payload. The payload can be a read-only view
    of the shared store; copy it to keep it.
    """

    def handle_event(self, event_type: str, payload) -> None: ...


@dataclass
class _QueuedEvent:
    """An event held back by :meth:`BoundingBoxEventManager.batch`."""

    event_type: str
    payload: object


class BoundingBoxEventManager:
    """Calls subscribers synchronously, with optional batching.

    Subscribers are held in a ``WeakSet``: the manager does not keep them
    alive, so the caller has to keep a reference to each one. An exception
    raised by a subscriber is not caught; it reaches the caller of
    :meth:`emit`, and the remaining subscribers are not called.
    """

    def __init__(self) -> None:
        self._subscribers: WeakSet[BoundingBoxSubscriber] = WeakSet()
        self._batch_stack = 0
        self._queued_events: list[_QueuedEvent] = []

    def subscribe(self, subscriber: BoundingBoxSubscriber) -> None:
        """Add ``subscriber``; adding it twice has no effect."""
        self._subscribers.add(subscriber)

    def unsubscribe(self, subscriber: BoundingBoxSubscriber) -> None:
        """Remove ``subscriber``; does nothing if it is not subscribed."""
        try:
            self._subscribers.remove(subscriber)
        except KeyError:
            pass

    def emit(self, event_type: str, payload) -> None:
        """Call every subscriber now, or queue the event inside :meth:`batch`."""
        if self._batch_stack:
            self._queued_events.append(_QueuedEvent(event_type, payload))
            return
        for subscriber in list(self._subscribers):
            subscriber.handle_event(event_type, payload)

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Queue events inside the block and send them when it ends.

        Batches can be nested; the queue is sent when the outermost one
        exits, also if the block raised. Only the last event of each type is
        sent, with its payload, in the order in which the types first
        appeared; earlier payloads of the same type are dropped.
        """
        self._batch_stack += 1
        try:
            yield
        finally:
            self._batch_stack = max(0, self._batch_stack - 1)
            if self._batch_stack == 0 and self._queued_events:
                unique_events = {}
                for event in self._queued_events:
                    unique_events[event.event_type] = event
                self._queued_events.clear()
                for event in unique_events.values():
                    self.emit(event.event_type, event.payload)
