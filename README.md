# TrafficIntel YOLO26

Traffic video analysis on a single GPU. One pass over a video detects and tracks
vehicles, confirms accidents, enforces red lights, and optionally reads number
plates, producing one annotated video plus a JSON of statistics.

There are two ways to run it: a **web app** (upload a video in the browser, watch
progress, review results) and a **CLI** for one-off runs and calibration.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
```

Confirm the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Start the web app:

```bash
python backend/app.py
```

Then open **http://127.0.0.1:8000**. The frontend is served by the same process,
so there is no second dev server to start.

For auto-reload while editing the backend:

```bash
python -m uvicorn backend.app:app --reload
```

To let other people use it while it runs on your GPU, see **[docs/SHARING.md](docs/SHARING.md)**.
Short version: set `TRAFFICINTEL_ACCESS_TOKEN`, start the server, and point a
tunnel at it. The server refuses to bind to a public address without a token.

Or run a single video from the command line:

```bash
python run_all.py --video test_videos/test1.mp4 --traffic-light
```

---

## What each module does

**Vehicle detection and tracking** — YOLO26M with ByteTrack
(`bytetrack_custom.yaml`). Always on; everything else builds on the tracks.

**Accident detection** — the trained accident model combined with motion
evidence from the tracker. An accident is an *incident* with identity and a
lifetime, not a per-frame detection: repeated evidence about the same crash
updates one event instead of creating a new one every couple of seconds. See
`accidents.py` and the note on false positives below.

**Red-light violation** — the trained traffic-light model reads the signal and
detects the `stop_line` class. The stop-line geometry is measured from those
detections (median over at least 12 confident frames), then a violation requires
red **and** a tracked vehicle **and** a directional crossing of that line. See
`redlight.py`.

**Number-plate OCR** — optional PaddleOCR pass over vehicle crops. There is no
plate-detection dataset in this project; it crops the tracked vehicle and reads
it. Adds significant processing time. A dedicated plate detector is the next
upgrade if accuracy is insufficient.

---

## Two things worth knowing before trusting the output

**A violation count of 0 is only meaningful if enforcement armed.** Red-light
checking cannot run without stop-line geometry. Every run recorded before
auto-calibration existed reported `red_light_enforcement_active: false` and 0
violations — which looks identical to a clean intersection but measured nothing
at all (see `docs/BASELINE.md`). Both the CLI and the web UI now state whether
enforcement armed, from which timestamp, and why it refused if it did not.

**Motion-only accident events carry no confidence score.** When the trained model
did not confirm an event, `confidence` is `null` and `evidence` is `["motion"]`,
rather than a plausible-looking number. The stats include
`accident_model_confirmed_count` next to `accident_count` so the split is
visible.

---

## Tuning accident sensitivity

Presets live in `config.ACCIDENT_PRESETS` and are selected with
`ACCIDENT_SENSITIVITY` or per-run:

```bash
python run_all.py --video clip.mp4 --sensitivity strict
python run_all.py --video clip.mp4 --sensitivity sensitive
```

`strict` requires the model to agree, so motion evidence alone never raises an
event. `balanced` (default) lets a genuine speed collapse confirm an impact the
detector missed. `sensitive` catches more and false-alarms more.

**These are starting points, not tuned values.** Run the same clip at all three
and compare against what you can actually see happening in the footage. The CLI
prints every event with its timestamp so this is quick to check.

---

## Layout

```
config.py                 paths, thresholds, presets
errors.py                 exceptions shared by engine and web layer
trafficintel.py           the engine: one frame loop, all modules
accidents.py              accident incident confirmation
redlight.py               signal state, stop-line estimation, violations
run_all.py                CLI entry point
train_accident.py         training
train_traffic_light.py    training

backend/                  FastAPI app, job queue, auth gate, storage validation
frontend/                 React UI (no build step; Babel in the browser)
tools/                    tests and checks
docs/                     baseline measurements, sharing guide

datasets/                 training data (not tracked in git)
models/                   trained weights (not tracked; too large)
test_videos/              input clips (not tracked)
outputs/                  CLI results (not tracked)
```

Weights and datasets are gitignored — the traffic-light checkpoints are 167 MB
each, over GitHub's 100 MB per-file limit. Training metrics and plots *are*
tracked so results stay reviewable.

---

## Tests

No GPU, torch, or ultralytics needed:

```bash
python tools/test_accidents.py     # accident confirmation logic
python tools/test_redlight.py      # stop-line estimation and violations
python tools/test_sharing.py       # access gate and disk retention
python tools/check_jsx.py frontend/app.jsx
```

These cover the logic that decides whether a report can be trusted: that
geometry is only adopted when the evidence supports it, that a violation needs
red plus tracking plus a real crossing, and that one crash produces one event.

With the server running, the end-to-end HTTP test drives the real pipeline:

```bash
python tools/test_api.py --traffic-light
```

It asserts the reporting contract as well as the status codes — that a
motion-only accident reports `confidence: null` rather than an invented number,
and that violations are never reported unless enforcement armed. Both of those
failures produce plausible-looking output, so nothing else catches them.

---

## Training

```bash
python train_accident.py
python train_traffic_light.py
```

Accident output is copied to `models/accident/best.pt`. Traffic-light output
stays at `models/traffic_light/yolo26m/weights/best.pt`; `config.py` accepts
either that path or `models/traffic_light/best.pt`, so a new checkpoint can be
dropped in either place.

**The traffic-light class order must match the dataset.** The code hardcodes
these indices (`redlight.py`), and a mismatch fails silently — enforcement would
read the wrong class and report zero violations forever:

```
0 car   1 green_light   2 motobike   3 red_light   4 stop_line   5 yellow_light
```

`tools/test_redlight.py` pins this against `config.TRAFFIC_LIGHT_NAMES`, so a
re-export with a different order fails the test rather than corrupting results.

### If YOLO26M runs out of VRAM

The GTX 1650 Max-Q has 4 GB, so training uses a small batch. On CUDA OOM, first
drop `IMG_SIZE` from 640 to 512 in `config.py`; if it still will not fit, set
`YOLO_MODEL = "yolo26s.pt"`. Do not start the project over.

The web app loads only the modules you enable, and runs one job at a time, for
the same reason.

---

## Accuracy limits

A better detector cannot fix labels or missing temporal logic. The accident
dataset is image-level object-detection data; accident *events* in video depend
on tracking and confirmation, which is why `accidents.py` exists.

Red-light violation is `red light + vehicle + stop line + crossing during red`.
All four now run, but the thresholds should be calibrated against footage from
your actual camera before violation counts are treated as authoritative.
