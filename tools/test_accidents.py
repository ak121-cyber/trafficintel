"""Unit tests for accident event confirmation.

These test the logic that turns per-frame detections into events, which is where
the false-positive bug lived. No torch, ultralytics or GPU needed, so this runs
anywhere and runs in under a second:

    python tools/test_accidents.py

Each test is a synthetic traffic scenario built from explicit per-frame speeds,
so the pass/fail condition is a real physical claim rather than a snapshot of
whatever the code currently happens to do. The scenario that matters most is
`test_long_aftermath_is_still_one_event`: the recorded failure was 27 events for
a single 76-second clip because a stationary aftermath kept re-triggering.
"""

from __future__ import annotations

import math
import sys
import unittest
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config                                                  # noqa: E402
from accidents import AccidentMonitor                          # noqa: E402

FPS = 30.0


# --------------------------------------------------------------------------- #
# Scenario building
# --------------------------------------------------------------------------- #

class FakeTrack:
    """Mirrors the normalised-speed bookkeeping of the real engine Track.

    Speed is stored in units of the vehicle's own box diagonal per frame, which
    is what makes the thresholds independent of resolution and of how far the
    vehicle is from the camera.
    """

    def __init__(self, tid):
        self.id = int(tid)
        self.box = None
        self.centers = deque(maxlen=20)
        self.norm_speeds = deque(maxlen=20)

    def update(self, box):
        b = tuple(float(v) for v in box)
        c = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        diag = math.hypot(b[2] - b[0], b[3] - b[1]) or 1.0
        self.norm_speeds.append(math.dist(c, self.centers[-1]) / diag if self.centers else 0.0)
        self.centers.append(c)
        self.box = b
        return self


def positions_from_speeds(x0, speeds, w, h):
    """Turn a per-frame normalised-speed profile into x positions.

    speeds[i] is the speed the tracker should observe on frame i, so the first
    entry is ignored (there is no previous frame to differentiate against).
    """
    diag = math.hypot(w, h)
    xs, x = [], float(x0)
    for i, s in enumerate(speeds):
        if i > 0:
            x += s * diag
        xs.append(x)
    return xs


def constant(value, n):
    return [value] * n


def ramp_down(v0, n):
    """Smooth deceleration to a standstill over n frames - ordinary braking."""
    return [v0 * (1.0 - i / float(n)) for i in range(n)]


def preset(name="balanced"):
    return config.ACCIDENT_PRESETS[name]


def make_monitor(name="balanced", **kw):
    return AccidentMonitor(
        FPS,
        preset=preset(name),
        same_place_radius=config.ACCIDENT_SAME_PLACE_RADIUS,
        contact_gap_factor=config.ACCIDENT_CONTACT_GAP_FACTOR,
        frame_size=(1920, 1080),
        **kw,
    )


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

def run_rear_end(monitor, scale=1.0, aftermath=300, model_at=None,
                 model_conf=0.7, model_type="Minor Accident"):
    """Car B travelling at speed strikes stationary car A, then both stay put.

    This is the canonical true positive: contact, an abrupt speed collapse, and
    a stationary aftermath. It must produce exactly one event no matter how long
    the aftermath runs.

    B closes at a steady 0.06 box-diagonals/frame and stops dead the moment its
    front edge reaches A's rear edge, which is what an impact looks like. Note
    that at the moment of contact the two boxes barely overlap at all - hence
    the edge-gap contact test.
    """
    w, h = 80.0 * scale, 60.0 * scale
    diag = math.hypot(w, h)
    y = 300.0 * scale
    a_x = 500.0 * scale
    stop_x = a_x - w                       # B's front edge meets A's rear edge

    a, b = FakeTrack(1), FakeTrack(2)
    a_box = (a_x, y, a_x + w, y + h)
    step = 0.06 * diag

    events = 0
    for f in range(int((stop_x - 200.0 * scale) / step) + 1 + aftermath):
        x = min(200.0 * scale + step * f, stop_x)
        a.update(a_box)
        b.update((x, y, x + w, y + h))
        dets = []
        if model_at is not None and model_at <= f < model_at + 5:
            dets = [((x, y, a_x + w, y + h), model_type, model_conf)]
        events += len(monitor.observe(f, [a, b], dets))
    return events


def run_queue_braking(monitor, frames_stopped=300):
    """Two cars in adjacent lanes brake smoothly to a stop at a red light.

    Their boxes overlap constantly because of the camera angle, and both end up
    stationary. This is the pattern that generated most of the old false
    positives, and it must produce no events: braking takes seconds, whereas an
    impact collapses the speed within a few frames.
    """
    w, h = 80.0, 60.0
    a, b = FakeTrack(11), FakeTrack(12)

    speeds = [0.0] + ramp_down(0.06, 60) + constant(0.0, frames_stopped)
    xs = positions_from_speeds(200.0, speeds, w, h)

    events = 0
    for f, x in enumerate(xs):
        a.update((x, 300.0, x + w, 360.0))
        b.update((x, 330.0, x + w, 390.0))       # overlapping under perspective
        events += len(monitor.observe(f, [a, b]))
    return events


def run_overlapping_cruise(monitor, frames=200):
    """Two vehicles with overlapping boxes both moving steadily - no incident."""
    w, h = 80.0, 60.0
    a, b = FakeTrack(21), FakeTrack(22)
    speeds = constant(0.06, frames)
    xs = positions_from_speeds(100.0, speeds, w, h)

    events = 0
    for f, x in enumerate(xs):
        a.update((x, 300.0, x + w, 360.0))
        b.update((x + 20.0, 330.0, x + 20.0 + w, 390.0))
        events += len(monitor.observe(f, [a, b]))
    return events


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestFalsePositives(unittest.TestCase):
    """Patterns that must NOT be reported."""

    def test_queue_braking_produces_no_events(self):
        m = make_monitor()
        self.assertEqual(run_queue_braking(m), 0)
        self.assertEqual(m.count, 0)

    def test_overlapping_boxes_while_moving_produce_no_events(self):
        m = make_monitor()
        self.assertEqual(run_overlapping_cruise(m), 0)

    def test_single_frame_model_blip_produces_no_event(self):
        """One confident frame is noise, not an accident."""
        m = make_monitor()
        box = (900.0, 400.0, 1000.0, 470.0)
        m.observe(0, [], [(box, "Severe Accident", 0.92)])
        for f in range(1, 120):
            m.observe(f, [])
        self.assertEqual(m.count, 0)

    def test_two_frames_below_threshold_count_produce_no_event(self):
        """balanced requires 3 frames; 2 must not be enough."""
        m = make_monitor()
        box = (900.0, 400.0, 1000.0, 470.0)
        for f in range(2):
            m.observe(f, [], [(box, "Minor Accident", 0.8)])
        self.assertEqual(m.count, 0)

    def test_low_confidence_detections_are_ignored(self):
        """Sustained but weak detections stay below model_min_conf."""
        m = make_monitor()
        box = (900.0, 400.0, 1000.0, 470.0)
        for f in range(40):
            m.observe(f, [], [(box, "Minor Accident", 0.20)])
        self.assertEqual(m.count, 0)

    def test_strict_preset_will_not_raise_on_motion_alone(self):
        m = make_monitor("strict")
        self.assertEqual(run_rear_end(m), 0)


class TestTruePositives(unittest.TestCase):
    """Patterns that must be reported, exactly once each."""

    def test_rear_end_impact_produces_exactly_one_event(self):
        m = make_monitor()
        self.assertEqual(run_rear_end(m), 1)
        self.assertEqual(m.count, 1)

    def test_long_aftermath_is_still_one_event(self):
        """The regression test for the 27-events-in-76s failure.

        The old code re-fired every 2 seconds for as long as the wreck sat in
        frame. 40 seconds of stationary aftermath must still be one incident.
        """
        m = make_monitor()
        self.assertEqual(run_rear_end(m, aftermath=1200), 1)
        self.assertEqual(m.count, 1)

    def test_sustained_model_detections_produce_one_event(self):
        m = make_monitor()
        box = (900.0, 400.0, 1000.0, 470.0)
        for f, conf in enumerate([0.60, 0.72, 0.55, 0.68, 0.51]):
            m.observe(f, [], [(box, "Moderate Accident", conf)])
        self.assertEqual(m.count, 1)

    def test_two_distinct_impacts_produce_two_events(self):
        """Separate crashes, separated in time and space, stay separate."""
        m = make_monitor()
        self.assertEqual(run_rear_end(m, aftermath=200), 1)

        # Nothing in frame long enough for the first incident to settle.
        for f in range(400, 700):
            m.observe(f, [])

        w, h = 80.0, 60.0
        c, d = FakeTrack(31), FakeTrack(32)
        step = 0.06 * math.hypot(w, h)
        stop_x = 500.0 - w
        d_box = (500.0, 800.0, 580.0, 860.0)          # different part of frame
        for i in range(int((stop_x - 200.0) / step) + 1 + 200):
            x = min(200.0 + step * i, stop_x)
            d.update(d_box)
            c.update((x, 800.0, x + w, 860.0))
            m.observe(700 + i, [c, d])

        self.assertEqual(m.count, 2)


class TestHonestConfidence(unittest.TestCase):
    """Confidence must be measured or absent - never invented."""

    def test_motion_only_event_reports_no_confidence(self):
        m = make_monitor()
        run_rear_end(m)
        ev = m.events[0]
        self.assertIsNone(ev["confidence"])
        self.assertEqual(ev["evidence"], ["motion"])
        self.assertFalse(ev["model_confirmed"])

    def test_motion_only_event_never_reports_the_old_fabricated_values(self):
        """0.82 and 0.86 were the only two values 0.50 + 0.08*score could give."""
        m = make_monitor()
        run_rear_end(m)
        self.assertNotIn(m.events[0]["confidence"], (0.82, 0.86))

    def test_model_event_reports_peak_model_confidence(self):
        m = make_monitor()
        box = (900.0, 400.0, 1000.0, 470.0)
        for conf in (0.60, 0.72, 0.55):
            m.observe(len(m.events), [], [(box, "Moderate Accident", conf)])
        ev = m.events[0]
        self.assertAlmostEqual(ev["confidence"], 0.72, places=4)
        self.assertTrue(ev["model_confirmed"])
        self.assertEqual(ev["type"], "Moderate Accident")

    def test_model_evidence_upgrades_a_motion_incident_in_place(self):
        """The detector agreeing later must not create a second event."""
        m = make_monitor()
        self.assertEqual(run_rear_end(m, model_at=120, model_conf=0.66,
                                      model_type="Severe Accident"), 1)
        ev = m.events[0]
        self.assertEqual(m.count, 1)
        self.assertEqual(ev["evidence"], ["model", "motion"])
        self.assertAlmostEqual(ev["confidence"], 0.66, places=4)
        self.assertEqual(ev["type"], "Severe Accident")


class TestScaleInvariance(unittest.TestCase):
    def test_same_verdict_at_different_resolutions(self):
        """Normalised speed means one set of thresholds works at any scale."""
        for scale in (0.5, 1.0, 3.0):
            m = make_monitor()
            self.assertEqual(run_rear_end(m, scale=scale), 1,
                             f"rear-end missed at scale {scale}")


class TestEventShape(unittest.TestCase):
    def test_event_dict_has_the_fields_the_frontend_reads(self):
        m = make_monitor()
        run_rear_end(m)
        ev = m.events[0]
        for key in ("frame", "time_seconds", "type", "confidence", "vehicles",
                    "evidence", "model_confirmed", "duration_seconds"):
            self.assertIn(key, ev)
        self.assertIsInstance(ev["vehicles"], list)
        self.assertGreater(ev["time_seconds"], 0.0)

    def test_incident_records_both_vehicles(self):
        m = make_monitor()
        run_rear_end(m)
        self.assertEqual(sorted(m.events[0]["vehicles"]), [1, 2])

    def test_duration_grows_with_the_aftermath(self):
        short, long_ = make_monitor(), make_monitor()
        run_rear_end(short, aftermath=120)
        run_rear_end(long_, aftermath=600)
        self.assertGreater(long_.events[0]["duration_seconds"],
                           short.events[0]["duration_seconds"])


class TestPresets(unittest.TestCase):
    def test_all_presets_define_the_same_keys(self):
        keys = [set(p) for p in config.ACCIDENT_PRESETS.values()]
        self.assertTrue(all(k == keys[0] for k in keys))

    def test_configured_sensitivity_exists(self):
        self.assertIn(config.ACCIDENT_SENSITIVITY, config.ACCIDENT_PRESETS)

    def test_strict_is_not_more_permissive_than_sensitive(self):
        s, v = preset("strict"), preset("sensitive")
        self.assertGreater(s["model_min_conf"], v["model_min_conf"])
        self.assertGreater(s["model_min_frames"], v["model_min_frames"])
        self.assertGreater(s["min_contact_frames"], v["min_contact_frames"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
