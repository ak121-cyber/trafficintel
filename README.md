# TrafficIntel — AI-Powered Smart Traffic Surveillance & Accident Analysis System

## Overview

TrafficIntel is a web application that analyses traffic camera footage on a single
GPU. A user signs in, uploads a video, and receives one annotated video plus a
JSON report containing vehicle counts, confirmed accident events and red-light
violations. All detection, tracking and reporting happens in a single pass over
the footage.

The system is built as a complete application rather than a notebook: a React
frontend, a FastAPI backend, MongoDB for accounts and history, and a YOLO26M
based computer-vision pipeline behind a job queue.

## Problem statement

Traffic cameras produce far more footage than any operator can watch. Incidents
are therefore found late or not at all, and manual review of recorded video is
slow and inconsistent.

Automating this is not simply a matter of running an object detector on each
frame. An accident is an *event* that occupies a span of time, so a per-frame
detector reports the same collision repeatedly and inflates the count. A
red-light violation is not visible in a single frame either: it requires a red
signal, a tracked vehicle, a known stop-line position, and a crossing of that
line in the direction of travel. A system that ignores this reports numbers that
look plausible and are not trustworthy.

## Objectives

1. Detect and track vehicles across frames rather than per frame.
2. Report each accident as one event with a start time and a duration.
3. Detect red-light violations only when all required conditions are satisfied,
   and state clearly whether enforcement was possible at all.
4. Optionally read number plates from tracked vehicles.
5. Provide a usable web interface with accounts, a daily usage limit and history.
6. Never report a value the system cannot support with evidence.

## Key features

- Vehicle detection and multi-object tracking in one pass over the video.
- Accident events with identity and lifetime instead of per-frame alarms.
- Red-light violation detection with automatic stop-line calibration.
- Optional number-plate OCR over tracked vehicle crops.
- Annotated MP4 output that plays in the browser, plus a JSON statistics file.
- Email and password accounts with server-side session management.
- A daily credit allowance enforced by the backend, not the interface.
- Per-user video history with download links.
- Live progress reporting with named stages and a working cancel button.
- A command-line interface that runs the identical pipeline without the web layer.

## System architecture

The application runs as a single Python process. FastAPI serves both the REST API
and the frontend files, so there is one origin and no separate development server.

```
Browser (React)
      |  HTTP + HttpOnly session cookie
      v
FastAPI  (backend/app.py)
      |-- backend/auth.py        registration, login, sessions
      |-- backend/db.py          MongoDB: users, history, credit charges
      |-- backend/storage.py     upload validation
      |-- backend/security.py    optional shared access gate
      |-- backend/jobs.py        single-worker job queue
                |
                v
      backend/engine_bridge.py
                |
                v
      trafficintel.py            one frame loop, all modules
      |-- accidents.py           accident event confirmation
      |-- redlight.py            signal state, stop line, violations
                |
                v
      Filesystem: uploaded video, annotated MP4, JSON statistics
```

Only metadata is stored in MongoDB. Videos, model weights and datasets remain on
the filesystem. Because the target GPU has 4 GB of VRAM, the queue runs exactly
one job at a time and additional submissions wait.

## Technology stack

| Layer | Technology |
|---|---|
| Frontend | React 18 (UMD build), JSX transpiled in the browser by Babel |
| Backend | FastAPI, Uvicorn, Python 3.11 |
| Database | MongoDB (via PyMongo) |
| Computer vision | Ultralytics YOLO26M, ByteTrack, OpenCV |
| OCR (optional) | PaddleOCR |
| Deep learning | PyTorch with CUDA |
| Authentication | JSON Web Tokens, `hashlib.scrypt` password hashing |

### React frontend

The interface is a React application in `frontend/`, consisting of `index.html`,
`app.jsx` and `style.css`. React and Babel are loaded from a CDN and the JSX is
transpiled in the browser, so there is no bundler, no `node_modules` directory and
no build step to run before starting the application. Routing is hash-based across
six views: landing page, sign in, register, dashboard, analysis tool and history.

This is a deliberate choice for a single-machine deployment: the backend already
serves the files, and adding a build toolchain would add failure modes without
changing what the user sees. The consequence is that the frontend is not a
standalone npm package and has no `package.json`.

### FastAPI backend

`backend/app.py` exposes the HTTP surface. Every endpoint that reads user data or
can consume GPU time depends on an authenticated session; the only public routes
are the health check and the frontend itself.

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/auth/register` | Create an account |
| POST | `/api/auth/login` | Sign in, set the session cookie |
| POST | `/api/auth/logout` | Clear the session |
| GET | `/api/auth/me` | Current account and credit state |
| POST | `/api/process` | Upload a video and queue a job |
| GET | `/api/status/{job_id}` | Progress of a running job |
| GET | `/api/result/{job_id}` | Full statistics for a finished job |
| GET | `/api/result/{job_id}/video` | Stream the annotated video |
| GET | `/api/result/{job_id}/download` | Download the annotated video |
| POST | `/api/cancel/{job_id}` | Stop a running job |
| DELETE | `/api/result/{job_id}` | Delete a job and its files |
| GET | `/api/history` | This account's past jobs |
| GET | `/api/health` | Device, model and database status |

### MongoDB database

Three collections hold application metadata only:

- `users` — name, lowercased email, password hash, credit balance, date of the
  last credit reset.
- `video_history` — one record per submitted job: filename, timestamp, modules
  requested, status, credits charged, summary results.
- `credit_charges` — one record per charged submission, used as an idempotency
  key so the same submission cannot be billed twice.

A unique index on `users.email` prevents duplicate accounts, and a compound index
on `video_history` keyed by user and creation time serves the history view. No
video data, model weight or dataset is ever written to the database.

## Machine learning components

### Vehicle detection and tracking

YOLO26M performs detection on each frame and ByteTrack associates detections into
tracks, configured in `bytetrack_custom.yaml`. Tracking is always enabled because
every other module depends on stable vehicle identities. Lane occupancy counts are
derived from the resulting tracks.

### Accident detection

A separately trained accident model is combined with motion evidence taken from the
tracker. Rather than treating each positive frame as a new accident, `accidents.py`
maintains incidents with an identity and a lifetime: repeated evidence about the
same collision updates the existing event, and an event closes only after the
evidence has been absent for a defined interval. This is the difference between
reporting one collision once and reporting it dozens of times.

Sensitivity is configurable through presets in `config.ACCIDENT_PRESETS`, selected
with `--sensitivity strict|balanced|sensitive`. Under `strict`, the trained model
must agree before an event is raised. Under `balanced`, a sufficiently severe
collapse in tracked speed can confirm an impact the detector missed. When an event
rests on motion evidence alone, its confidence is reported as `null` rather than as
a numeric score, and the statistics include `accident_model_confirmed_count`
alongside `accident_count` so the two categories remain distinguishable.

### Traffic-light detection and red-light violations

The traffic-light model classifies the signal state and also detects the
`stop_line` class. The stop-line position is measured from those detections — a
median taken over at least twelve confident frames, with a rejection guard for
inconsistent geometry — so no manual region drawing is required.

A violation is recorded only when four conditions hold simultaneously: the signal
is red, a vehicle is being tracked, stop-line geometry has been established, and
the vehicle crosses that line in the direction of travel. If geometry was never
established, enforcement does not arm, and the report states this explicitly
through `red_light_enforcement_active`, `stop_line_source` and
`enforcement_active_from_frame`. This matters because a violation count of zero
from an unarmed system is indistinguishable from a genuinely clean intersection
unless the system says which one it is.

The class indices are fixed in `redlight.py` and must match the trained model:

```
0 car   1 green_light   2 motobike   3 red_light   4 stop_line   5 yellow_light
```

### Number-plate recognition (optional)

When enabled, PaddleOCR reads text from tracked vehicle crops. There is no
dedicated plate-detection dataset in this project, so the crop is the vehicle
rather than the plate. This module adds substantial processing time and is off by
default.

## Application features in detail

### Video upload and processing

The user selects a video and chooses which modules to run. The backend validates
the file before accepting it: extension, size limit (2 GB) and a probe confirming
the file is a readable video with a positive frame count. Accepted jobs enter a
single-worker queue; a fourth queued job is refused with HTTP 503 rather than
allowing an unbounded backlog. Progress is polled and reported with named stages,
the current frame number and live counts, and a cancel request stops the worker.

### Authentication and account management

Registration stores a salted `scrypt` hash of the password and nothing else. The
plaintext password is never logged, never returned by any endpoint and never
retained by the frontend.

The session is a signed JWT delivered in an HttpOnly cookie. JavaScript cannot read
it, and because the browser attaches cookies automatically, the HTML `<video>`
element can stream a protected result — which a header-based scheme could not do,
since a video element cannot send an `Authorization` header. Scripts may
alternatively authenticate with `Authorization: Bearer`. Failed login attempts
return an identical message whether the email is unknown or the password is wrong,
so accounts cannot be enumerated.

### Credit system

Each account receives 15 credits per day. Processing one video costs 5 credits,
giving 3 videos per day. Both values are configurable; the videos-per-day figure is
derived from them rather than configured separately, so the number shown in the
interface cannot disagree with what the backend enforces.

Enforcement is entirely server-side. Editing the React code or posting directly to
`/api/process` does not bypass it, because the deduction is a single atomic MongoDB
update whose filter requires a sufficient balance — concurrent uploads therefore
cannot overspend. Each submission carries a client-generated identifier that is
inserted as a unique key, so a duplicated request is rejected instead of charged
twice.

Nothing is charged for opening the site, selecting a file, a rejected upload, a
failed login or a validation error. A job that fails on the server is refunded
automatically, and the refund is applied at most once.

The daily reset requires no scheduler or background task: each user record stores
the date of its last reset, and the balance is topped up on that account's first
request on a new day.

### Video history and output

Every submission is recorded against the account that made it, with its filename,
timestamp, status, credit cost and summary results. Requesting another user's job
returns 404 rather than their data. The history view links to each annotated video
for download.

Disk usage is bounded: once history exceeds a configured number of jobs or total
size, the oldest finished jobs are evicted and their files deleted. A history row
can therefore outlive its video, and the interface reports the file as removed
rather than offering a download that would fail.

Output is written as H.264 MP4 so it plays directly in the browser, accompanied by
a JSON file containing the full statistics for the run.

## Project structure

```
config.py                  paths, thresholds, sensitivity presets
errors.py                  exceptions shared by the engine and web layer
trafficintel.py            the engine: one frame loop, all modules
accidents.py               accident incident confirmation
redlight.py                signal state, stop-line estimation, violations
run_all.py                 command-line entry point
train_accident.py          accident model training
train_traffic_light.py     traffic-light model training
bytetrack_custom.yaml      tracker configuration
requirements.txt           Python dependencies
.env.example               template for environment variables

backend/
    app.py                 FastAPI application and HTTP routes
    auth.py                password hashing, JWT sessions, auth routes
    db.py                  MongoDB layer: users, credits, history
    jobs.py                single-worker job queue
    engine_bridge.py       call into the vision pipeline
    storage.py             upload validation
    security.py            optional shared access gate

frontend/
    index.html             page shell, loads React and Babel
    app.jsx                the entire interface
    style.css              styling

tools/                     tests and checks
docs/                      baseline measurements, sharing guide
models/                    trained weights and training metrics
datasets/                  dataset configuration (data not tracked)
test_videos/               input clips (not tracked)
outputs/                   command-line results (not tracked)
```

Model weights (`*.pt`) and dataset contents are excluded from version control: the
traffic-light checkpoint is 167 MB, above GitHub's 100 MB per-file limit. Training
metrics and plots *are* included so results remain reviewable.

## Local setup

Requirements: Python 3.11, an NVIDIA GPU with CUDA for usable performance, and
MongoDB Community Server running locally.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
```

Confirm the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Create the configuration file and set a session signing key:

```bash
copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Start MongoDB, then start the application:

```bash
python backend/app.py
```

Open <http://127.0.0.1:8000> and create an account. The frontend is served by the
same process, so there is no second server to start.

To run a single video without the web layer:

```bash
python run_all.py --video test_videos/test1.mp4 --traffic-light
```

## Environment variables

Configuration is read from `.env`, which is excluded from version control.
`.env.example` documents every variable and contains no real values.

| Variable | Purpose |
|---|---|
| `MONGODB_URI` | MongoDB connection string |
| `MONGODB_DATABASE` | Database name |
| `JWT_SECRET` | Key used to sign session tokens |
| `DAILY_CREDITS` | Credits granted per account per day |
| `CREDITS_PER_VIDEO` | Credits charged per accepted video |
| `TRAFFICINTEL_ACCESS_TOKEN` | Optional shared password for the whole site |

No credential, key or password value is committed to this repository.

## How the system works

1. The user signs in; the server issues a session cookie.
2. The user uploads a video and selects modules. The backend validates the file.
3. If the balance is sufficient, 5 credits are deducted atomically and a history
   record is created. Otherwise the request is refused and nothing is charged.
4. The job is queued. The single worker loads only the models required by the
   selected modules.
5. The engine iterates the video once. Each frame is detected and tracked; enabled
   modules consume those tracks to confirm accident events, evaluate the signal
   state and stop-line crossings, and optionally read plates.
6. Annotated frames are written to an H.264 MP4 and statistics accumulate.
7. On completion the history record is updated and the result becomes available for
   playback and download. If the job fails, the credits are refunded.

## Model information

Both models are YOLO26M, trained with Ultralytics at 640 px on a GTX 1650 Max-Q
(4 GB VRAM), which constrains batch size.

| | Accident model | Traffic-light model |
|---|---|---|
| Classes | 5 | 6 |
| Class names | No Accident, Minor Accident, Moderate Accident, Severe Accident, Totaled Vehicle | car, green_light, motobike, red_light, stop_line, yellow_light |
| Weights | `models/accident/best.pt` | `models/traffic_light/yolo26m/weights/best.pt` |

### Recorded training metrics

Taken directly from the `results.csv` files in `models/`. These are validation
metrics from training, not results measured on independent footage.

| Model | Best epoch | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| Accident | 41 of 53 run | 0.507 | 0.266 | — | — |
| Traffic light | 17 | 0.840 | 0.537 | 0.717 | 0.805 |

The traffic-light run was scheduled for 40 epochs and stopped after 20; the
recorded history covers epochs 9 to 20. Per-class validation output was not saved,
so the average precision for the `stop_line` class specifically is not known. No
independent test-set evaluation has been performed.

## Tests

The logic that determines whether a report can be trusted is covered by tests that
need no GPU, PyTorch or MongoDB:

```bash
python tools/test_accidents.py     # accident event confirmation
python tools/test_redlight.py      # stop-line estimation and violations
python tools/test_sharing.py       # access gate and disk retention
python tools/test_accounts.py      # passwords, sessions, credits, history
python tools/check_jsx.py frontend/app.jsx
```

These verify, among other properties, that stop-line geometry is adopted only when
the evidence supports it, that a violation requires all four conditions, that one
collision produces one event, that a password is never stored or returned in
readable form, that six simultaneous uploads against a 15-credit balance charge
exactly three, and that a failed job is refunded once and only once.

With the server and MongoDB running, an end-to-end test drives the real pipeline
over HTTP:

```bash
python tools/test_api.py --traffic-light
```

## Limitations

- **Not validated against ground-truth footage.** Detection and event thresholds
  have not been calibrated against manually labelled video, so counts should be
  treated as indicative rather than authoritative.
- **Training was incomplete.** The traffic-light model stopped at epoch 20 of 40,
  and the accident model's validation mAP50 of 0.507 leaves clear room for
  improvement.
- **Accident labels are image-level.** The dataset is object-detection data, not
  event-annotated video, which is why temporal confirmation is implemented in
  application code rather than learned.
- **Number-plate OCR crops the vehicle, not the plate,** because no plate-detection
  dataset is included. Accuracy is correspondingly limited.
- **One job at a time.** 4 GB of VRAM does not allow concurrent inference, so
  simultaneous users queue.
- **Single machine.** Processing requires a local CUDA GPU; the application is not
  currently deployed to any hosted environment.
- **Results are stored on the local filesystem** and are subject to retention
  limits, so older annotated videos are eventually deleted.

## Future improvements

- Evaluate both models against a held-out, manually labelled test set and publish
  per-class metrics.
- Complete the traffic-light training schedule and retrain the accident model with
  more data.
- Add a dedicated licence-plate detector ahead of the OCR stage.
- Calibrate accident and violation thresholds against footage from the specific
  camera in use.
- Support multiple stop lines and per-lane signal states for complex intersections.
- Deploy to a GPU-backed host so the system does not depend on one local machine.
