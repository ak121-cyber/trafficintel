"""End-to-end API test for the TrafficIntel web backend.

Exercises the real HTTP surface the React frontend uses: upload -> poll -> result.
Nothing is mocked; each run drives the actual ML pipeline, so it needs the server
running (`python backend/app.py`) and takes as long as the video does.

    python tools/test_api.py                       # accident only
    python tools/test_api.py --traffic-light       # both models, one pass
    python tools/test_api.py --traffic-light --sensitivity strict
    python tools/test_api.py --no-accident --traffic-light

Beyond "did it return 200", this asserts the reporting contract that the UI and
the CLI both depend on: that a motion-only accident reports confidence null
rather than an invented number, and that violations are never reported unless
red-light enforcement actually armed. Both are failures that produce
plausible-looking output, so nothing else would catch them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"

# When the server is started with TRAFFICINTEL_ACCESS_TOKEN (as it must be to be
# shared - see docs/SHARING.md), every request needs the token. Read from the same
# environment variable so no copy of it lives in this file.
ACCESS_TOKEN = (os.environ.get("TRAFFICINTEL_ACCESS_TOKEN") or "").strip()


def auth_headers(extra: dict | None = None) -> dict:
    headers = dict(extra or {})
    if ACCESS_TOKEN:
        headers["X-Access-Token"] = ACCESS_TOKEN
    return headers


def post_multipart(url: str, video: Path, fields: dict) -> tuple[int, dict]:
    """Minimal multipart/form-data POST using only the standard library."""
    boundary = f"----TrafficIntel{uuid.uuid4().hex}"
    parts = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="{video.name}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n".encode()
    )
    parts.append(video.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers=auth_headers({
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=auth_headers())
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def head_bytes(url: str, count: int = 64) -> tuple[int, bytes, str]:
    """Fetch the first bytes of a response, to confirm the video really serves."""
    req = urllib.request.Request(url, headers=auth_headers({"Range": f"bytes=0-{count - 1}"}))
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read(count), resp.headers.get("Content-Type", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test_videos/test1.mp4")
    ap.add_argument("--accident", dest="accident", action="store_true", default=True)
    ap.add_argument("--no-accident", dest="accident", action="store_false")
    ap.add_argument("--traffic-light", action="store_true")
    ap.add_argument("--plate", action="store_true")
    ap.add_argument("--sensitivity", default="",
                    help="Accident preset: strict, balanced or sensitive.")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    video = (ROOT / args.video).resolve()
    if not video.exists():
        print(f"FAIL  video not found: {video}")
        return 1

    print("=" * 66)
    print(f"  accident={args.accident}  traffic_light={args.traffic_light}  ocr={args.plate}")
    print("=" * 66)

    status, health = get(f"{BASE}/api/health")
    if status != 200:
        print(f"FAIL  /api/health returned {status}")
        return 1
    print(f"health      : {health['status']} | device={health['device']['device']}")

    # Bad input must be refused before anything is queued. Checked first because a
    # typo silently falling back to the default preset would mean a run reported
    # under a sensitivity nobody selected.
    status, body = post_multipart(
        f"{BASE}/api/process", video,
        {"accident_detection": "true", "sensitivity": "definitely-not-a-preset"},
    )
    if status != 400:
        print(f"FAIL  bogus sensitivity was accepted with HTTP {status} "
              f"(expected 400)")
        return 1
    print(f"validation  : 400 as expected | {body.get('detail')}")

    fields = {
        "accident_detection": str(args.accident).lower(),
        "traffic_light": str(args.traffic_light).lower(),
        "number_plate": str(args.plate).lower(),
    }
    # Only sent when asked for, so the default run also proves the field is
    # genuinely optional and the server falls back to config.ACCIDENT_SENSITIVITY.
    if args.sensitivity:
        fields["sensitivity"] = args.sensitivity

    t0 = time.time()
    status, body = post_multipart(f"{BASE}/api/process", video, fields)
    if status != 202:
        print(f"FAIL  /api/process returned {status}: {body.get('detail')}")
        return 1

    job_id = body["job_id"]
    src = body.get("source", {})
    print(f"upload      : {status} job={job_id} "
          f"{src.get('width')}x{src.get('height')} frames={src.get('frames')} "
          f"({time.time() - t0:.1f}s)")

    last = None
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        status, job = get(f"{BASE}/api/status/{job_id}")
        if status != 200:
            print(f"FAIL  /api/status returned {status}")
            return 1
        line = f"{job['status']:<11} {job['stage']:<14} frame {job['frame']}/{job['total_frames']}"
        if line != last:
            print(f"  {line}")
            last = line
        if job["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(2)
    else:
        print("FAIL  timed out")
        return 1

    if job["status"] != "completed":
        print(f"FAIL  job {job['status']}: {job.get('error')}")
        return 1

    s = job["stats"]
    print("-" * 66)
    print(f"elapsed     : {job['elapsed_seconds']}s")
    print(f"modules run : {s['modules']}")
    print(f"vehicles    : {s['unique_tracked_vehicles']} unique | lanes {s['lane_counts']}")
    # .get throughout this block: a missing key is a failure the assertions below
    # report properly, and crashing here with a KeyError would hide every other
    # result behind a traceback.
    print(f"accidents   : {s['accident_count']} "
          f"({s.get('accident_model_confirmed_count')} model-confirmed) "
          f"[sensitivity={s.get('accident_sensitivity')}]")
    for e in s.get("accident_events", []):
        conf = f"{e['confidence']:.0%} model" if e.get("confidence") is not None else "motion only"
        print(f"    {e['time_seconds']:7.2f}s  {e['type']:<34} {conf:<12} "
              f"{e['duration_seconds']}s")

    # A violation count on its own is not interpretable, so the enforcement state
    # is printed beside it. See docs/BASELINE.md: every pre-fix run reported 0
    # violations *because enforcement never armed*, which looks identical to a
    # clean intersection.
    print(f"violations  : {s['red_light_violation_count']} "
          f"(enforcement_active={s['red_light_enforcement_active']})")
    if args.traffic_light:
        if s["red_light_enforcement_active"]:
            fps = s.get("fps") or 1
            armed_at = s.get("enforcement_active_from_frame") or 0
            share = 100 * (s.get("enforced_frames") or 0) / max(1, s.get("frames") or 1)
            print(f"  armed via {s.get('stop_line_source')} geometry at "
                  f"{armed_at / fps:.2f}s ({share:.0f}% of clip enforced)")
        else:
            print("  NOT ARMED - the 0 above measures nothing")
            print(f"  calibration: {s.get('stop_line_calibration')}")
    print(f"plates      : {s['number_plate_count']}")
    print(f"codec       : {s['output_codec']} browser_playable={s['browser_playable']}")

    # Confirm the annotated video actually serves and is an MP4.
    vstatus, head, ctype = head_bytes(f"{BASE}{job['result_url']}")
    is_mp4 = b"ftyp" in head
    print(f"video serve : HTTP {vstatus} {ctype} ftyp={is_mp4}")

    dstatus, _, _ = head_bytes(f"{BASE}{job['download_url']}", 32)
    print(f"download    : HTTP {dstatus}")

    # Assertions for the toggle contract.
    ok = True
    if args.accident != s["modules"]["accident_detection"]:
        print(f"FAIL  accident module flag mismatch")
        ok = False
    if args.traffic_light != s["modules"]["traffic_light"]:
        print(f"FAIL  traffic-light module flag mismatch")
        ok = False
    if not args.accident and s["accident_count"] != 0:
        print(f"FAIL  accidents reported while module disabled")
        ok = False
    if vstatus not in (200, 206) or not is_mp4:
        print("FAIL  result video did not serve as MP4")
        ok = False

    # The reporting keys the UI and the CLI both read. If the engine ever stops
    # emitting one, the frontend renders "undefined" rather than crashing, so the
    # regression is invisible unless it is asserted here.
    for key in ("stop_line_source", "enforcement_active_from_frame", "enforced_frames",
                "stop_line_calibration", "accident_model_confirmed_count",
                "accident_sensitivity", "fps", "frames"):
        if key not in s:
            print(f"FAIL  stats is missing {key!r}")
            ok = False

    if args.sensitivity and s.get("accident_sensitivity") != args.sensitivity:
        print(f"FAIL  requested sensitivity {args.sensitivity!r} but ran "
              f"{s.get('accident_sensitivity')!r}")
        ok = False

    # An event the trained model did not confirm must carry confidence: null.
    # A plausible-looking number there would be fabricated, and every pre-fix run
    # reported exactly 0.82/0.86 for events with no model agreement at all.
    for e in s.get("accident_events", []):
        motion_only = e.get("evidence") == ["motion"]
        if motion_only and e.get("confidence") is not None:
            print(f"FAIL  motion-only event at {e['time_seconds']}s reports "
                  f"confidence {e['confidence']} instead of null")
            ok = False
    confirmed = sum(1 for e in s.get("accident_events", [])
                    if e.get("confidence") is not None)
    if confirmed != s.get("accident_model_confirmed_count"):
        print(f"FAIL  {confirmed} events carry a model confidence but "
              f"accident_model_confirmed_count is {s.get('accident_model_confirmed_count')}")
        ok = False

    # Violations cannot exist without geometry to cross. Read with .get so a
    # missing key is reported by the check above rather than raising here.
    armed = s.get("red_light_enforcement_active")
    if not armed and s.get("red_light_violation_count"):
        print("FAIL  violations reported while enforcement was never armed")
        ok = False
    if armed and s.get("stop_line_source", "none") == "none":
        print("FAIL  enforcement armed with no stop-line source")
        ok = False
    if not args.traffic_light and armed:
        print("FAIL  enforcement armed while the traffic-light module was off")
        ok = False

    print("PASS" if ok else "FAIL")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
