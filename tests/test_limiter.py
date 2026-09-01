import asyncio

import pytest

from app.limiter import GenerationLimiter, QueueFullError


async def test_reserve_beyond_capacity_raises_queue_full() -> None:
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=1)
    limiter.reserve()  # fills the one queue slot (even though it'll acquire instantly)
    with pytest.raises(QueueFullError):
        limiter.reserve()


async def test_free_slot_is_acquired_immediately() -> None:
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    ticket = limiter.reserve()
    await asyncio.sleep(0)  # let the acquire task actually run
    assert ticket.acquire_task.done() is True


async def test_second_reservation_waits_and_reports_position_one() -> None:
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    first = limiter.reserve()
    await asyncio.sleep(0)
    assert first.acquire_task.done() is True  # holds the only slot
    limiter.mark_running(first)  # as generate_events would, once it starts using the permit

    second = limiter.reserve()
    await asyncio.sleep(0.05)
    assert second.acquire_task.done() is False
    assert limiter.position(second) == 1

    await limiter.release(first, acquired=True)
    await asyncio.sleep(0.05)
    assert second.acquire_task.done() is True


async def test_release_while_still_waiting_does_not_touch_semaphore() -> None:
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    first = limiter.reserve()
    await asyncio.sleep(0)
    second = limiter.reserve()
    await asyncio.sleep(0.05)
    assert second.acquire_task.done() is False

    # "second" gives up while still queued - must not leak or steal the
    # permit "first" is holding.
    await limiter.release(second, acquired=False)
    with pytest.raises(ValueError):
        limiter.position(second)  # removed from the wait list

    # The slot is still held by "first" alone; releasing it should be the
    # only thing that frees it up (proving "second" didn't consume it).
    third = limiter.reserve()
    await asyncio.sleep(0.05)
    assert third.acquire_task.done() is False
    await limiter.release(first, acquired=True)
    await asyncio.sleep(0.05)
    assert third.acquire_task.done() is True
