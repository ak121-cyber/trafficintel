"""TrafficIntel FastAPI backend.

Serves the React frontend and the analysis API from one process, so the user
only needs to open http://127.0.0.1:8000 - no second dev server.

Run with:
    python -m uvicorn backend.app:app --reload
or:
    python backend/app.py

To share it with other people while it runs on your GPU, see docs/SHARING.md.
Binding to anything but localhost requires TRAFFICINTEL_ACCESS_TOKEN to be set;
the server refuses to start otherwise, because an open endpoint would let any
visitor spend your GPU time and fill your disk.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import engine_bridge                                    # noqa: E402
from backend import security                                         # noqa: E402
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


def _runner(job):
    return engine_bridge.run_job(job)


jobs = JobManager(_runner)


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
):
    """Accept an upload, validate it, and queue one analysis job."""
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
        "Upload received | %s | %.1f MB | %sx%s",
        display_name, written / (1024 * 1024), probe["width"], probe["height"],
    )

    parsed_lines = _parse_stop_lines(stop_lines)

    job = jobs.submit(
        input_path=input_path,
        output_path=output_path,
        original_filename=display_name,
        job_id=job_id,
        options={
            "accident_detection": bool(accident_detection),
            "traffic_light": bool(traffic_light),
            "number_plate": bool(number_plate),
            "stop_lines": parsed_lines,
            "sensitivity": sensitivity or ACCIDENT_SENSITIVITY,
            "strict_accidents": bool(strict_accidents),
        },
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
        },
    )


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job.public()


@app.get("/api/result/{job_id}")
def result(job_id: str):
    """Full statistics for a finished job."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    if job.status == FAILED:
        raise HTTPException(status_code=409, detail=job.error or "Processing failed.")
    if job.status == CANCELLED:
        raise HTTPException(status_code=409, detail="The job was cancelled.")
    if job.status != COMPLETED:
        raise HTTPException(status_code=409, detail="The job is still processing.")
    return job.public()


@app.get("/api/result/{job_id}/video")
def result_video(job_id: str):
    """Serve the annotated video for in-browser playback."""
    path = _completed_output(job_id)
    # FileResponse sets accept-ranges, which the HTML5 player needs to seek.
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/result/{job_id}/download")
def result_download(job_id: str):
    path = _completed_output(job_id)
    job = jobs.get(job_id)
    stem = Path(job.original_filename).stem if job else "video"
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{stem}_trafficintel.mp4",
        headers={"Content-Disposition": f'attachment; filename="{stem}_trafficintel.mp4"'},
    )


@app.post("/api/cancel/{job_id}")
def cancel(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    if job.status in (COMPLETED, FAILED, CANCELLED):
        return {"job_id": job_id, "status": job.status, "cancelled": False}
    job.cancel()
    log.info("Job %s cancellation requested", job_id)
    return {"job_id": job_id, "status": job.status, "cancelled": True}


@app.delete("/api/result/{job_id}")
def delete_result(job_id: str):
    """Remove a job and its files from disk."""
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
def list_jobs():
    return {"jobs": [j.public() for j in jobs.list()]}


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


def _completed_output(job_id: str) -> Path:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
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
                         "this machine; 0.0.0.0 exposes it to your network and "
                         "requires TRAFFICINTEL_ACCESS_TOKEN.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sweep", action="store_true",
                    help="Delete leftover uploads and results from previous runs "
                         "before starting.")
    cli = ap.parse_args()

    # Fails closed. See backend/security.py for why this is SystemExit and not a
    # warning.
    security.enforce_bind_policy(cli.host, ACCESS_TOKEN)

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
        log.info("Access token required. Share the URL and the token; visitors "
                 "sign in at /login")
    else:
        # The bind check above cannot catch this case: a tunnel (cloudflared,
        # ngrok) connects to 127.0.0.1 from this same machine, so the server is
        # publicly reachable while still bound to loopback. Only the operator
        # knows whether a tunnel is running, so all this can do is say so.
        log.warning("No %s set - anyone who can reach this address can upload "
                    "video and use your GPU. Set a token before starting a "
                    "tunnel. See docs/SHARING.md", security.ENV_VAR)

    log.info("Serving on http://%s:%d", cli.host, cli.port)

    # The app OBJECT is passed, not the import string "backend.app:app".
    # An import string makes uvicorn import this file a second time under a
    # different module name, so `jobs = JobManager(...)` runs twice and a second
    # worker thread is started against a registry no request handler can see.
    # Passing the object reuses the module already loaded. Reload mode is the one
    # case that genuinely needs the string form, and it is not used here:
    #     python -m uvicorn backend.app:app --reload
    uvicorn.run(app, host=cli.host, port=cli.port)
