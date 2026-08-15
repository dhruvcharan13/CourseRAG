"""A one-worker background queue for syncs.

Indexing is slow — parse, chunk, then a forward pass per chunk through a real model —
so a browser cannot wait on it inside a request without looking hung. Uploads enqueue
a job and return immediately; the page polls for its log.

One worker, not a pool. Two syncs of the same course would race on the same LanceDB
table, and courses are usually synced one at a time anyway; serializing everything is
both the safe choice and, for a single-user local tool, an unnoticeable one. Jobs are
in-memory: restarting the server forgets the log, not the work, since everything a job
does is committed to disk as it goes.
"""

from __future__ import annotations

import queue
import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

#: Terminal states, for a poller deciding when to stop.
DONE_STATES = frozenset({"done", "error"})


@dataclass
class Job:
    """One queued unit of work and everything the UI shows about it."""

    id: str
    course_id: str
    label: str
    state: str = "queued"  # queued | running | done | error
    #: Human-readable progress, appended to as the job runs.
    lines: list[str] = field(default_factory=list)
    #: Set when ``state == "error"``.
    error: str | None = None
    #: 0.0-1.0 when the total is known up front, else None.
    progress: float | None = None

    def log(self, line: str) -> None:
        self.lines.append(line)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "course_id": self.course_id,
            "label": self.label,
            "state": self.state,
            "lines": list(self.lines),
            "error": self.error,
            "progress": self.progress,
            "done": self.state in DONE_STATES,
        }


class JobQueue:
    """Runs submitted callables one at a time on a daemon thread."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[tuple[Job, Callable[[Job], None]]] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="courserag-jobs", daemon=True)
        self._worker.start()

    def submit(self, course_id: str, label: str, fn: Callable[[Job], None]) -> Job:
        """Queue ``fn``, which is called with its own :class:`Job` to log into."""
        job = Job(id=uuid.uuid4().hex[:12], course_id=course_id, label=label)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job, fn))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active_for(self, course_id: str) -> Job | None:
        """The queued or running job for a course, if any.

        Lets the UI disable a course's controls while its index is being rewritten,
        rather than letting a second sync stack up behind the first.
        """
        with self._lock:
            for job in reversed(list(self._jobs.values())):
                if job.course_id == course_id and job.state not in DONE_STATES:
                    return job
        return None

    def _run(self) -> None:
        while True:
            job, fn = self._queue.get()
            job.state = "running"
            try:
                fn(job)
            except Exception as exc:  # noqa: BLE001 - a failed job must not kill the worker
                job.state = "error"
                job.error = str(exc) or exc.__class__.__name__
                job.log(f"error: {job.error}")
                traceback.print_exc()
            else:
                job.state = "done"
            finally:
                job.progress = 1.0
                self._queue.task_done()
