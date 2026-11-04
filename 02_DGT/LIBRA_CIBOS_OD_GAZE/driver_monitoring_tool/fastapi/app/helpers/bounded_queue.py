"""Helpers for a fixed-size asyncio queue that favours the newest gaze samples."""

import asyncio

GAZE_QUEUE_MAXSIZE = 300

def put_drop_oldest(queue: asyncio.Queue, item) -> None:
    """Enqueue item without blocking, discarding the oldest entry if the queue is full."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(item)
