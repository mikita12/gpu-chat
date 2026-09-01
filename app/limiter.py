import asyncio
import contextlib
from dataclasses import dataclass


class QueueFullError(Exception):
    """Raised by GenerationLimiter.reserve() when the wait queue is already
    at capacity - the caller should reject the request immediately (e.g.
    HTTP 429) rather than enqueue it."""


@dataclass(eq=False)  # identity equality/hash - needed for list membership
class Ticket:
    id: int
    acquire_task: "asyncio.Task[bool]"


class GenerationLimiter:
    """Bounds how many generations run against Ollama at once, with a
    capped FIFO wait queue for the rest and visible queue-position
    reporting - one shared GPU cannot serve unlimited concurrent chats.

    Position tracking is kept as an explicit list separate from the
    semaphore itself (which doesn't expose waiter order/position safely).
    The semaphore acquire for each ticket runs as its own Task, started
    once and never cancelled-and-retried - repeatedly cancelling and
    re-acquiring would re-insert the waiter at the back of the semaphore's
    internal FIFO on every poll, breaking fairness against other waiters.
    """

    def __init__(self, max_concurrent: int, max_queue_size: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_queue_size = max_queue_size
        self._waiting: list[Ticket] = []
        self._next_id = 0

    def reserve(self) -> Ticket:
        """Synchronous, no awaiting - safe to call from a route handler
        before committing to a streaming response at all. Raises
        QueueFullError if the wait queue is already at capacity."""
        if len(self._waiting) >= self._max_queue_size:
            raise QueueFullError()
        self._next_id += 1
        ticket = Ticket(id=self._next_id, acquire_task=asyncio.create_task(self._semaphore.acquire()))
        self._waiting.append(ticket)
        return ticket

    def position(self, ticket: Ticket) -> int:
        """1-indexed position of `ticket` among tickets still waiting."""
        return self._waiting.index(ticket) + 1

    def mark_running(self, ticket: Ticket) -> None:
        """Call once ticket.acquire_task has completed and the permit is
        actually about to be used - removes it from wait-queue capacity
        accounting (reserve()'s cap and position()) without touching the
        semaphore itself. This keeps max_queue_size a count of requests
        actually *waiting*, independent of max_concurrent_generations - an
        active generation no longer occupies a wait-queue slot. release()
        must still be called later to free the permit."""
        if ticket in self._waiting:
            self._waiting.remove(ticket)

    async def release(self, ticket: Ticket, acquired: bool) -> None:
        """Must be called exactly once per ticket returned by reserve(),
        regardless of how the request ended (finished, errored, or was
        cancelled while queued or generating).

        `acquired` is whether the caller actually went on to use the
        permit (i.e. the queueing loop saw ticket.acquire_task complete).
        If so, this simply releases it. If not - the caller gave up while
        still waiting - there is a narrow race to close: acquire_task may
        have *just* won the semaphore at the exact moment we're abandoning
        it, since Task.cancel() only schedules a cancellation and does not
        synchronously fail an already-completed task. Left unhandled, that
        silently leaks a permit forever (with max_concurrent=1, one leak
        deadlocks the whole server). So: cancel, then *await* the task
        (suppressing the CancelledError) so the cancellation actually
        settles, then check whether it completed successfully anyway - if
        so, release the permit it won instead of leaking it.
        """
        if ticket in self._waiting:
            self._waiting.remove(ticket)
        if acquired:
            self._semaphore.release()
            return
        if not ticket.acquire_task.done():
            ticket.acquire_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticket.acquire_task
        if not ticket.acquire_task.cancelled() and ticket.acquire_task.exception() is None:
            self._semaphore.release()
