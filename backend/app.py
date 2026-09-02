"""TrafficIntel FastAPI backend.

Serves the React frontend and the analysis API from one process, so the user
only needs to open http://127.0.0.1:8000 - no second dev server.

Run with:
    python -m uvicorn backend.app:app --reload
or:
    python backend/app.py

To share it with other people while it runs on your GPU, see docs/SHARING.md.
Binding to anything but localhost requires either user accounts (the default) or
TRAFFICINTEL_ACCESS_TOKEN; the server refuses to start with neither, because an
open endpoint would let any visitor spend your GPU time and fill your disk.

Every endpoint that can spend GPU time or read a result requires a signed-in user
(see backend/auth.py) and costs credits (see backend/db.py). The analysis pipeline
itself is untouched by that: credits are reserved before jobs.submit() and the
outcome is recorded afterwards, so backend/jobs.py and the engine know nothing
about users.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import auth                                             # noqa: E402
from backend import db                                               # noqa: E402
from backend import engine_bridge                                    # noqa: E402
from backend import security                                         # noqa: E402
from errors import ProcessingCancelled                               # noqa: E402
from backend.jobs import (                                           # noqa: E402
    CANCELLED,
    COMPLETED,
    FAILED,
    JobManager,
    new_job_id,
)
from config import ACCIDENT_PRESETS, ACCIDENT_SENSITIVITY            # noqa: E402
from backend.storage import (                                        # noqa: E402
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    UploadError,
    resolve_inside,
    sanitize_filename,
    validate_extension,
    verify_readable_video,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trafficintel.api")

BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
RESULT_DIR = BACKEND_DIR / "results"
FRONTEND_DIR = ROOT / "frontend"

# Website uploads are kept out of test_videos/, which stays reserved for the
# developer CLI workflow.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TrafficIntel", description="AI-Powered Traffic Video Intelligence")

ACCESS_TOKEN = security.load_token()

# Cap the backlog. Only one job runs at a time (4 GB VRAM), so a deep queue means
# everyone waits behind it - and a shared URL makes it easy for one person to
# queue a dozen clips without realising nobody else can get through.
MAX_PENDING_JOBS = 3

# CORS is only needed when the frontend is served from a *different* origin, i.e.
# the Vite-on-:5173 dev workflow. Once a token is set the app is being shared, so
# the wildcard is dropped: "*" would let any website in the world drive this API
# from a visitor's browser. Same-origin requests need no CORS headers at all.
if not ACCESS_TOKEN:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _summary_of(stats) -> dict:
    """The scalar fields of a stats dict, for the history row.

    Only scalars are kept, deliberately. Mongo holds metadata; per-frame arrays
    and event lists belong in the JSON next to the video on disk, and copying
    them here would grow the database with every run.
    """
    if not isinstance(stats, dict):
        return {}
    return {k: v for k, v in stats.items()
            if isinstance(v, (int, float, bool, str)) or v is None}


def _finalise(job, status: str, error: str | None = None, stats=None) -> None:
    """Record how a job ended, and refund credits if it failed.

    Refunds happen on FAILED only, never on a cancel. Refunding a cancel would
    make the daily cap meaningless: anyone could let a job run to 99%, cancel it,
    get the credits back and go again indefinitely. A crash is the server's fault
    and should not cost the user anything; stopping your own job is a choice.

    Nothing in here may raise. A database hiccup while a job is finishing must not
    turn a completed analysis into a failure - the video is already on disk.
    """
    try:
        row = db.get_db().video_history.find_one(
            {"_id": job.id}, {"user_id": 1, "request_id": 1})
        if row is None:
            return
        if status == FAILED:
            db.refund_credits(row["user_id"], row["request_id"])
            db.finish_job(job.id, status, error=error, credits_used=0)
        else:
            db.finish_job(job.id, status, error=error, summary=_summary_of(stats))
    except Exception as exc:                                          # noqa: BLE001
        log.warning("Could not record the outcome of job %s: %s", job.id, exc)


def _runner(job):
    """Run the analysis, then record the outcome.

    Wrapped at this level rather than inside JobManager so that backend/jobs.py
    stays free of any knowledge of users, credits or MongoDB. The exception is
    re-raised in every case, so JobManager still decides the job's status exactly
    as it did before - this only observes.
    """
    try:
        stats = engine_bridge.run_job(job)
    except ProcessingCancelled:
        # Caught before the broad handler because it is a RuntimeError, and a
        # cancel must not be recorded as a failure or trigger a refund.
        _finalise(job, CANCELLED)
        raise
    except Exception as exc:                                          # noqa: BLE001
        _finalise(job, FAILED, error=str(exc))
        raise

    # A cancel requested during the final frames lands here: the runner returned
    # normally but JobManager is about to mark the job CANCELLED.
    _finalise(job, CANCELLED if job.cancelled else COMPLETED, stats=stats)
    return stats


jobs = JobManager(_runner)

# Registration, login, logout and /api/auth/me.
app.include_router(auth.router)

# Filenames and Mongo _id values are built from this, so it is restricted to
# characters that are safe in both.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _charge_id(user: dict, request_id: str) -> str:
    """The idempotency key for one upload attempt.

    Namespaced by user id, so one account cannot burn another account's request
    ids by guessing them - without the prefix, a stranger submitting id "abc"
    would make a legitimate user's "abc" look like a duplicate submit.
    """
    request_id = (request_id or "").strip()
    if not _REQUEST_ID_RE.match(request_id):
        # No usable id from the client means no duplicate protection is possible,
        # so fall back to a unique one rather than rejecting the upload.
        request_id = uuid.uuid4().hex
    return f"{user['_id']}:{request_id}"


def _owned_job(job_id: str, user: dict):
    """Fetch a job, but only if it belongs to this user.

    404 rather than 403 for someone else's job: a 403 would confirm that the id
    exists, which is exactly what an id-guessing attempt wants to learn.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    try:
        owner = db.job_owner(job_id)
    except db.DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if owner != str(user["_id"]):
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job


# --------------------------------------------------------------------------- #
# Health / capability
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health():
    """Report what the server can actually do right now."""
    models = engine_bridge.model_report()
    return {
        "status": "ok",
        "service": "TrafficIntel",
        "device": engine_bridge.device_report(),
        "models": models,
        "modules": {
            "accident_detection": models["accident"]["available"],
            "traffic_light": models["traffic_light"]["available"],
            "number_plate": engine_bridge.ocr_available(),
        },
        "limits": {
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        },
        # Advertised so the UI can state the daily allowance without hardcoding
        # numbers that would then disagree with what the backend enforces.
        "credits": {
            "daily": db.DAILY_CREDITS,
            "per_video": db.CREDITS_PER_VIDEO,
            "videos_per_day": db.VIDEOS_PER_DAY,
        },
        "database": db.status(),
        # Advertised so a client can offer the sensitivity control without
        # hardcoding preset names that only exist in config.py.
        "accident_sensitivity": {
            "default": ACCIDENT_SENSITIVITY,
            "options": sorted(ACCIDENT_PRESETS),
        },
    }


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #

@app.post("/api/process")
async def process(
    video: UploadFile = File(...),
    accident_detection: bool = Form(True),
    traffic_light: bool = Form(False),
    number_plate: bool = Form(False),
    stop_lines: str = Form(""),
    sensitivity: str = Form(""),
    strict_accidents: bool = Form(False),
    request_id: str = Form(""),
    user: dict = Depends(auth.current_user),
):
    """Accept an upload, validate it, charge credits, and queue one analysis job.

    The order of the checks below is the whole credit policy, so it matters:

    1. Cheap validation first. A rejected request must never cost credits.
    2. A read-only credit check *before* the upload is read, so a user with an
       empty balance is told immediately instead of transferring 2 GB first.
    3. The file is written and probed. Still no charge - a corrupt or undecodable
       video is the user's mistake to fix for free.
    4. Only once the job is genuinely about to be queued are credits reserved,
       atomically, in one MongoDB operation.

    Opening the site, opening this page and picking a file all cost nothing;
    nothing above happens until a real upload arrives.
    """
    if not (accident_detection or traffic_light):
        raise HTTPException(
            status_code=400,
            detail="Enable at least one analysis module (accident detection or traffic-light violation).",
        )

    # Validated here rather than in the worker: a typo should fail the request
    # immediately, not a job that has already been queued and uploaded.
    sensitivity = (sensitivity or "").strip().lower()
    if sensitivity and sensitivity not in ACCIDENT_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sensitivity {sensitivity!r}. "
                   f"Expected one of: {', '.join(sorted(ACCIDENT_PRESETS))}.",
        )

    # Refuse a deep backlog before accepting the upload, so a rejected request
    # does not first spend minutes transferring a 2 GB file. 503 with Retry-After
    # is the honest status: the request is fine, the server is just busy.
    if jobs.pending() >= MAX_PENDING_JOBS:
        raise HTTPException(
            status_code=503,
            detail=f"{jobs.pending()} jobs are already waiting and only one can run "
                   f"at a time. Try again once the queue clears.",
            headers={"Retry-After": "120"},
        )

    # Advisory only - the balance is re-checked atomically below, because between
    # this read and that write the upload takes minutes and another tab could
    # spend the same credits. This exists purely so an out-of-credits user is not
    # made to upload a large file before being told no.
    try:
        pre = db.credit_state(user["_id"])
    except db.DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not pre["can_process"]:
        raise HTTPException(
            status_code=402,
            detail=f"You have {pre['credits_remaining']} credits left today and each "
                   f"video costs {pre['credits_per_video']}. Your allowance of "
                   f"{pre['videos_per_day']} videos resets tomorrow.",
        )

    display_name = sanitize_filename(video.filename or "video.mp4")
    try:
        ext = validate_extension(video.filename or "")
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Minted here because the id is needed to name the files before the job can
    # be submitted; it is then passed to submit() so the id the client polls is
    # the same one embedded in the filenames on disk.
    job_id = new_job_id()
    input_path = UPLOAD_DIR / f"job_{job_id}_original{ext}"
    output_path = RESULT_DIR / f"job_{job_id}_annotated.mp4"

    # Stream to disk in chunks so a large upload never has to fit in memory.
    written = 0
    try:
        with input_path.open("wb") as fh:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise UploadError(
                        f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                    )
                fh.write(chunk)
    except UploadError as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:                                          # noqa: BLE001
        input_path.unlink(missing_ok=True)
        log.exception("Upload failed for %s", display_name)
        raise HTTPException(status_code=500, detail="The upload could not be saved.") from exc
    finally:
        await video.close()

    if written == 0:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Confirm it is genuinely decodable before committing a multi-minute job.
    try:
        probe = verify_readable_video(input_path)
    except UploadError as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log.info(
        "Upload received | %s | %.1f MB | %sx%s | user=%s",
        display_name, written / (1024 * 1024), probe["width"], probe["height"],
        user.get("email"),
    )

    parsed_lines = _parse_stop_lines(stop_lines)

    options = {
        "accident_detection": bool(accident_detection),
        "traffic_light": bool(traffic_light),
        "number_plate": bool(number_plate),
        "stop_lines": parsed_lines,
        "sensitivity": sensitivity or ACCIDENT_SENSITIVITY,
        "strict_accidents": bool(strict_accidents),
    }

    # The charge. Atomic, and at most once per request id - see
    # db.reserve_credits. Nothing has been queued yet, so a refusal here costs
    # only the uploaded file, which is deleted.
    charge_id = _charge_id(user, request_id)
    try:
        charge = db.reserve_credits(user["_id"], charge_id)
    except db.DatabaseUnavailable as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not charge["ok"]:
        input_path.unlink(missing_ok=True)
        if charge["reason"] == "duplicate":
            # The same upload arrived twice. Charging again would be wrong, and so
            # would starting a second job on the GPU.
            raise HTTPException(
                status_code=409,
                detail="That video was already submitted. Check your history "
                       "instead of uploading it again.",
            )
        raise HTTPException(
            status_code=402,
            detail=f"Not enough credits: {charge['remaining']} left and this video "
                   f"costs {db.CREDITS_PER_VIDEO}. Your allowance resets tomorrow.",
        )

    # Recorded before submit() so that ownership exists the instant the job can be
    # polled. If this insert failed the credits would already be gone, so the
    # charge is released rather than silently kept.
    try:
        db.record_job(job_id, user["_id"], charge_id, display_name,
                      output_path.name, options)
    except Exception as exc:                                          # noqa: BLE001
        db.refund_credits(user["_id"], charge_id)
        input_path.unlink(missing_ok=True)
        log.exception("Could not record job %s", job_id)
        raise HTTPException(
            status_code=503,
            detail="The job could not be recorded, so nothing was started and your "
                   "credits were not charged.",
        ) from exc

    job = jobs.submit(
        input_path=input_path,
        output_path=output_path,
        original_filename=display_name,
        job_id=job_id,
        options=options,
    )
    job.total_frames = probe["frames"] or 0

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.id,
            "status": job.status,
            "stage": job.stage,
            "filename": display_name,
            "size_bytes": written,
            "source": probe,
            "options": job.options,
            "status_url": f"/api/status/{job.id}",
            "credits_remaining": charge["remaining"],
            "credits_charged": db.CREDITS_PER_VIDEO,
        },
    )


@app.get("/api/history")
def history(user: dict = Depends(auth.current_user)):
    """This user's past jobs, plus their current credit state.

    One endpoint for both because the dashboard renders them together, and two
    round trips would let the credit figure and the job list disagree.
    """
    try:
        rows = db.history_for(user["_id"])
        credits = db.credit_state(user["_id"])
    except db.DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Mongo says a result was produced; only this layer knows whether the file is
    # still there. Retention trims old jobs and deletes their videos, so a history
    # row can outlive its output - and offering a download that 404s is worse than
    # saying up front that it is gone.
    for row in rows:
        if row["has_output"]:
            row["has_output"] = (RESULT_DIR / f"job_{row['job_id']}_annotated.mp4").exists()

    return {"history": rows, **credits}


@app.get("/api/status/{job_id}")
def status(job_id: str, user: dict = Depends(auth.current_user)):
    return _owned_job(job_id, user).public()


@app.get("/api/result/{job_id}")
def result(job_id: str, user: dict = Depends(auth.current_user)):
    """Full statistics for a finished job."""
    job = _owned_job(job_id, user)
    if job.status == FAILED:
        raise HTTPException(status_code=409, detail=job.error or "Processing failed.")
    if job.status == CANCELLED:
        raise HTTPException(status_code=409, detail="The job was cancelled.")
    if job.status != COMPLETED:
        raise HTTPException(status_code=409, detail="The job is still processing.")
    return job.public()


@app.get("/api/result/{job_id}/video")
def result_video(job_id: str, user: dict = Depends(auth.current_user)):
    """Serve the annotated video for in-browser playback.

    This is why the session lives in a cookie: <video src="..."> cannot send an
    Authorization header, but the browser attaches the cookie by itself.
    """
    path = _completed_output(job_id, user)
    # FileResponse sets accept-ranges, which the HTML5 player needs to seek.
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/result/{job_id}/download")
def result_download(job_id: str, user: dict = Depends(auth.current_user)):
    path = _completed_output(job_id, user)
    job = jobs.get(job_id)
    stem = Path(job.original_filename).stem if job else "video"
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{stem}_trafficintel.mp4",
        headers={"Content-Disposition": f'attachment; filename="{stem}_trafficintel.mp4"'},
    )


@app.post("/api/cancel/{job_id}")
def cancel(job_id: str, user: dict = Depends(auth.current_user)):
    job = _owned_job(job_id, user)
    if job.status in (COMPLETED, FAILED, CANCELLED):
        return {"job_id": job_id, "status": job.status, "cancelled": False}
    job.cancel()
    log.info("Job %s cancellation requested", job_id)
    return {"job_id": job_id, "status": job.status, "cancelled": True}


@app.delete("/api/result/{job_id}")
def delete_result(job_id: str, user: dict = Depends(auth.current_user)):
    """Remove a job and its files from disk.

    The history row stays. It is the record that credits were spent, and deleting
    it would make the daily total look wrong for the rest of the day.
    """
    _owned_job(job_id, user)
    job = jobs.remove(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    job.cancel()
    removed = []
    for path in (job.input_path, job.output_path, job.output_path.with_suffix(".json")):
        try:
            safe = resolve_inside(BACKEND_DIR, path)
        except UploadError:
            continue
        if safe.exists():
            safe.unlink()
            removed.append(safe.name)
    log.info("Job %s deleted (%s)", job_id, ", ".join(removed) or "no files")
    return {"job_id": job_id, "deleted": removed}


@app.get("/api/jobs")
def list_jobs(user: dict = Depends(auth.current_user)):
    """Only this user's jobs. The in-memory registry holds everyone's."""
    try:
        mine = {j["job_id"] for j in db.history_for(user["_id"])}
    except db.DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"jobs": [j.public() for j in jobs.list() if j.id in mine]}



# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_stop_lines(raw: str):
    """Parse the optional stop-line geometry sent as a JSON string."""
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="stop_lines is not valid JSON.") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="stop_lines must be a list of line objects.")
    return data


def _completed_output(job_id: str, user: dict) -> Path:
    job = _owned_job(job_id, user)
    if job.status != COMPLETED:
        raise HTTPException(status_code=409, detail="The result is not ready yet.")
    try:
        path = resolve_inside(RESULT_DIR, job.output_path)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="The result video is no longer on disk.")
    return path


# --------------------------------------------------------------------------- #
# Frontend (mounted last so /api/* wins)
# --------------------------------------------------------------------------- #

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def missing_frontend():
        return {"detail": f"Frontend not found at {FRONTEND_DIR}"}


# Installed at import time, not inside __main__, because `uvicorn backend.app:app`
# never executes __main__ - registering the gate there would leave the app wide
# open under the exact command the README suggests for development.
security.install(app, ACCESS_TOKEN)


def orphaned_bytes() -> tuple[int, int]:
    """Size and count of files in uploads/ and results/ at startup.

    Job records live only in memory, so anything already on disk when the process
    starts belongs to a job nobody can reach any more: /api/status/{id} returns
    404 for it and DELETE needs the record to find the paths. Reporting the total
    turns invisible disk creep into a number the operator can act on.
    """
    total = count = 0
    for directory in (UPLOAD_DIR, RESULT_DIR):
        for path in directory.glob("*"):
            if path.is_file():
                total += path.stat().st_size
                count += 1
    return total, count


def sweep_orphans() -> int:
    """Delete those unreachable files. Returns bytes reclaimed."""
    freed = 0
    for directory in (UPLOAD_DIR, RESULT_DIR):
        for path in directory.glob("*"):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
                path.unlink()
                freed += size
            except OSError as exc:
                log.warning("Could not delete %s: %s", path.name, exc)
    return freed


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Run the TrafficIntel web app.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address. 127.0.0.1 (default) is reachable only from "
                         "this machine; 0.0.0.0 exposes it to your network, where "
                         "user accounts are what protect it. Use --host 0.0.0.0 to "
                         "open the site on your phone or another PC.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sweep", action="store_true",
                    help="Delete leftover uploads and results from previous runs "
                         "before starting.")
    cli = ap.parse_args()

    # Fails closed. See backend/security.py for why this is SystemExit and not a
    # warning. User accounts satisfy the same requirement the shared token was
    # introduced for - every endpoint that can spend GPU time now needs a login -
    # so they count as a gate here. Only the call site changed; the policy
    # function itself still refuses a public bind with no gate at all.
    security.enforce_bind_policy(cli.host, ACCESS_TOKEN or "user-accounts")

    # Connect now rather than on the first request, so a missing mongod is a clear
    # line at startup instead of a 503 the first time someone tries to register.
    try:
        db.connect()
    except db.DatabaseUnavailable as exc:
        log.error("%s", exc)
        log.error("Accounts, credits and history will not work until MongoDB is "
                  "running. Start it, then restart this server.")

    if cli.sweep:
        freed = sweep_orphans()
        log.info("Swept %.1f MB of unreachable files from previous runs",
                 freed / (1024 * 1024))
    else:
        stale, n = orphaned_bytes()
        if stale:
            log.info("%d leftover file(s) from previous runs using %.1f MB "
                     "(unreachable - restart with --sweep to delete)",
                     n, stale / (1024 * 1024))

    if ACCESS_TOKEN:
        log.info("Shared access token required as well as a user account. Share "
                 "the URL and the token; visitors sign in at /login")
    elif not security.is_loopback(cli.host):
        # Reachable from the network, gated only by user accounts - and signing up
        # is open, so anyone who reaches the URL can create one and get a daily
        # allowance. Fine on a home network, worth knowing before a tunnel.
        log.warning("Bound to %s with open registration: anyone who can reach this "
                    "address can create an account and use your GPU. Set %s as "
                    "well to require a shared password. See docs/SHARING.md",
                    cli.host, security.ENV_VAR)

    log.info("Serving on http://%s:%d", cli.host, cli.port)

    # The app OBJECT is passed, not the import string "backend.app:app".
    # An import string makes uvicorn import this file a second time under a
    # different module name, so `jobs = JobManager(...)` runs twice and a second
    # worker thread is started against a registry no request handler can see.
    # Passing the object reuses the module already loaded. Reload mode is the one
    # case that genuinely needs the string form, and it is not used here:
    #     python -m uvicorn backend.app:app --reload
    uvicorn.run(app, host=cli.host, port=cli.port)
