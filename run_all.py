"""CLI entry point for one-off analysis runs.

    python run_all.py --video test_videos/test1.mp4
    python run_all.py --video clip.mp4 --sensitivity strict
    python run_all.py --video clip.mp4 --traffic-light --plate

--sensitivity selects an accident preset from config.ACCIDENT_PRESETS. Use it to
calibrate against real footage: run the same clip at strict/balanced/sensitive
and compare the reported events against what you can see happening.

With --traffic-light the stop line is measured from the model's own stop_line
detections, and the run reports whether enforcement actually armed. That matters:
before this existed, every run reported 0 violations because enforcement never
started, which reads exactly like a clean intersection.
"""

import argparse
from pathlib import Path

from config import ACCIDENT_PRESETS, ACCIDENT_SENSITIVITY
from trafficintel import TrafficIntel

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--output", default="outputs/annotated.mp4")
parser.add_argument("--traffic-light", action="store_true")
parser.add_argument("--plate", action="store_true")
parser.add_argument(
    "--sensitivity",
    choices=sorted(ACCIDENT_PRESETS),
    default=ACCIDENT_SENSITIVITY,
    help="Accident confirmation preset (default: %(default)s).",
)
parser.add_argument(
    "--strict-accidents",
    action="store_true",
    help="Require the accident model to agree; motion evidence alone will not "
         "raise an event.",
)
parser.add_argument(
    "--no-auto-calibrate",
    action="store_true",
    help="Do not measure the stop line from the model's stop_line detections. "
         "Enforcement then stays off unless geometry is supplied in code.",
)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)

engine = TrafficIntel(
    traffic_light=args.traffic_light,
    plate=args.plate
)

stats = engine.run(
    args.video,
    args.output,
    traffic_light=args.traffic_light,
    plate=args.plate,
    sensitivity=args.sensitivity,
    strict_accidents=args.strict_accidents,
    auto_calibrate=not args.no_auto_calibrate,
)

print("DONE")
print("Output:", args.output)
print("Tracked vehicles:", stats["unique_tracked_vehicles"])
print(f"Accidents: {stats['accident_count']} "
      f"({stats['accident_model_confirmed_count']} model-confirmed) "
      f"[sensitivity={stats['accident_sensitivity']}]")

# Print each event so a run can be checked against the footage immediately.
for e in stats["accident_events"]:
    conf = f"{e['confidence']:.0%} model" if e["confidence"] is not None else "motion evidence"
    vehicles = ", ".join(f"#{v}" for v in e["vehicles"]) or "-"
    print(f"  {e['time_seconds']:7.2f}s  {e['type']:<40} {conf:<16} "
          f"vehicles {vehicles}  ({e['duration_seconds']}s)")

# Red-light enforcement is reported explicitly, because a violation count of 0
# means two very different things depending on whether enforcement ever armed,
# and reading only the count has previously been misleading.
if args.traffic_light:
    fps = stats["fps"] or 1
    if stats["red_light_enforcement_active"]:
        armed = stats["enforcement_active_from_frame"] or 0
        share = 100 * stats["enforced_frames"] / max(1, stats["frames"])
        print(f"Red-light: enforcement ACTIVE via {stats['stop_line_source']} geometry "
              f"from {armed / fps:.2f}s ({share:.0f}% of the clip)")
        print(f"  violations: {stats['red_light_violation_count']}")
        for v in stats["red_light_violations"]:
            print(f"  {v['time_seconds']:7.2f}s  vehicle #{v['vehicle_id']} "
                  f"crossed {v['stop_line']} on {v['light_state']}")
    else:
        cal = stats["stop_line_calibration"]
        print("Red-light: enforcement NEVER ARMED - the 0 below is not evidence of "
              "compliance")
        if cal is None:
            print("  auto-calibration was disabled and no stop line was supplied")
        else:
            print(f"  stop_line samples {cal['samples']}/{cal['min_samples']}"
                  + (f" - {cal['reason']}" if cal.get("reason") else ""))
