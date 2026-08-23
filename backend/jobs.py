"""In-process background job runner for video analysis.

Video processing takes minutes, so an HTTP request cannot wait for it. Each job
runs on a worker thread and the frontend polls for status. This is deliberately
simple: a dict plus a lock plus a single-worker queue, with no Redis/Celery.

Only ONE job runs at a time. The GTX 1650 Max-Q has 4 GB of VRAM and a single
run already holds the vehicle model plus the enabled analysis models, so
concurrent jobs would risk CUDA OOM.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from errors import ProcessingCancelled                              # noqa: E402

log = logging.getLogger("trafficintel.jobs")


def new_job_id() -> str:
    """Allocate a job id.

    Lives here so the API layer can mint the id *before* it writes the upload to
    disk and then hand the same id to submit(). Previously the API generated one
    id for the filenames and submit() generated a different one for the client,
    so a file on disk could not be traced back to the job that produced it.
    """
    return uuid.uuid4().hex[:12]


QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

# Coarse stage labels for the UI. No percentage is invented when the total frame
# count is unknown; the stage name is reported instead.
STAGE_QUEUED = "Queued"
STAGE_LOADING = "Loading models"
STAGE_ANALYZING = "Analyzing"
STAGE_FINALIZING = "Finalizing"
STAGE_DONE = "Completed"


@dataclass
class Job:
    """One video-processing request and everything the API needs to report it."""

    id: str
    input_path: Path
    output_path: Path
    original_filename: str
    options: dict

    status: str = QUEUED
    stage: str = STAGE_QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    frame: int = 0
    total_frames: int = 0
    live: dict = field(default_factory=dict)
    stats: Optional[dict] = None
    error: Optional[str] = None
    warnings: list = field(default_factory=list)

    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def progress(self) -> Optional[float]:
        """Fraction complete, or None when the frame count is unknown.

        Returning None is intentional: a fabricated percentage would be worse
        than an honest "unknown" for the UI.
        """
        if self.total_frames > 0:
            return min(1.0, self.frame / self.total_frames)
        return None

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 2)

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def disk_bytes(self) -> int:
        """Bytes this job currently occupies: upload, output video and stats."""
        total = 0
        for path in (self.input_path, self.output_path,
                     self.output_path.with_suffix(".json")):
            try:
                if path.exists():
                    total += path.stat().st_size
            except OSError:
                pass
        return total

    def public(self, result_url: Optional[str] = None) -> dict:
        """Serialisable view for the API. Never leaks absolute filesystem paths."""
        payload = {
            "job_id": self.id,
            "status": self.status,
            "stage": self.stage,
            "filename": self.original_filename,
            "options": self.options,
            "progress": self.progress,
            "frame": self.frame,
            "total_frames": self.total_frames,
            "live": self.live,
            "elapsed_seconds": self.elapsed,
            "queued_at": self.created_at,
            "warnings": self.warnings,
            "error": self.error,
        }
        if self.status == COMPLETED:
            payload["stats"] = self.stats
            payload["result_url"] = result_url or f"/api/result/{self.id}/video"
            payload["download_url"] = f"/api/result/{self.id}/download"
        return payload


class JobManager:
    """Thread-safe job registry with a single background worker."""

    def __init__(self, runner: Callable[[Job], dict], max_history: int = 50,
                 max_bytes: int = 8 * 1024 * 1024 * 1024):
        """
        `max_history` bounds how many finished jobs stay reachable; `max_bytes`
        bounds what they cost on disk. Both are needed: uploads may be up to 2 GB
        each, so 50 retained jobs is a 100 GB worst case on a count-only limit.
        Whichever ceiling is hit first evicts the oldest finished job.
        """
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._max_history = max_history
        self._max_bytes = max_bytes
        self._worker = threading.Thread(target=self._loop, name="trafficintel-worker", daemon=True)
        self._worker.start()

    def submit(self, input_path: Path, output_path: Path, original_filename: str,
               options: dict, job_id: Optional[str] = None) -> Job:
        """Register and queue one job.

        `job_id` lets the caller pre-allocate the id it already used to build the
        stored filenames, so a file on disk can be traced back to its job. When
        omitted an id is generated here.
        """
        job = Job(
            id=job_id or new_job_id(),
            input_path=input_path,
            output_path=output_path,
            original_filename=original_filename,
            options=options,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            evicted = self._trim_locked()
        # Deleted outside the lock: unlink touches the filesystem and can block,
        # and holding the registry lock during it would stall every status poll.
        for old in evicted:
            self._purge_files(old)
        self._queue.put(job.id)
        log.info("Job %s queued | file=%s | options=%s", job.id, original_filename, options)
        return job

    def pending(self) -> int:
        """How many jobs are waiting for the single worker.

        Exposed so the API can refuse new work instead of letting one client
        queue an unbounded backlog: only one job runs at a time, so a deep queue
        means everyone else waits behind it for hours.
        """
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == QUEUED)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return [self._jobs[j] for j in reversed(self._order) if j in self._jobs]

    def remove(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)
        return job

    def _trim_locked(self) -> list:
        """Forget the oldest finished jobs, returning them so their files can go.

        Returning the evicted jobs rather than just dropping them is the whole
        point: an uploaded video can be 2 GB, and once the job record is gone
        nothing can reach those files - not the API, not DELETE /api/result/{id},
        which needs the record to know the paths. They would sit on disk until
        someone cleared backend/uploads by hand. The caller unlinks them.
        """
        evicted = []
        while True:
            over_count = len(self._order) > self._max_history
            over_bytes = (sum(j.disk_bytes() for j in self._jobs.values())
                          > self._max_bytes)
            if not (over_count or over_bytes):
                break
            for jid in list(self._order):
                job = self._jobs.get(jid)
                if job and job.status in (COMPLETED, FAILED, CANCELLED):
                    self._order.remove(jid)
                    evicted.append(self._jobs.pop(jid, None))
                    break
            else:
                # Every remaining job is still queued or running; nothing may be
                # evicted yet. The registry is briefly over its limits, which is
                # correct - dropping a live job would orphan a running GPU task
                # and delete the video out from under it.
                break
        return [j for j in evicted if j is not None]

    def _purge_files(self, job: Job) -> None:
        """Delete a forgotten job's upload, output video and stats JSON."""
        for path in (job.input_path, job.output_path,
                     job.output_path.with_suffix(".json")):
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                # Never fatal: a locked or already-removed file must not take the
                # worker or an upload request down with it.
                log.warning("Could not delete %s for trimmed job %s: %s",
                            path.name, job.id, exc)
        log.info("Job %s trimmed from history; its files were deleted", job.id)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            if job.cancelled:
                job.status = CANCELLED
                job.stage = "Cancelled"
                continue
            self._execute(job)

    def _execute(self, job: Job) -> None:
        job.status = PROCESSING
        job.stage = STAGE_LOADING
        job.started_at = time.time()
        log.info("Job %s started", job.id)
        try:
            job.stats = self._runner(job)
            if job.cancelled:
                job.status = CANCELLED
                job.stage = "Cancelled"
                log.info("Job %s cancelled", job.id)
            else:
                job.status = COMPLETED
                job.stage = STAGE_DONE
                log.info(
                    "Job %s completed in %.1fs | output=%s",
                    job.id, job.elapsed, job.output_path.name,
                )
        except ProcessingCancelled:
            # A user-requested stop is not a failure. This must be caught before
            # the broad handler below: ProcessingCancelled is a RuntimeError, so
            # `except Exception` would otherwise report a deliberate cancel as a
            # crash and surface a scary error in the UI.
            job.status = CANCELLED
            job.stage = "Cancelled"
            log.info("Job %s cancelled during processing", job.id)
        except Exception as exc:                       # noqa: BLE001 - reported to the client
            job.status = FAILED
            job.stage = "Failed"
            job.error = str(exc)
            # Full traceback stays server-side; the client gets the message only.
            log.exception("Job %s failed: %s", job.id, exc)
        finally:
            job.finished_at = time.time()
