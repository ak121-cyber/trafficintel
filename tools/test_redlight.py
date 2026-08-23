"""Unit tests for red-light violation logic.

Runs without torch, ultralytics or a GPU:

    python tools/test_redlight.py

The focus is the two things that decide whether a violation report can be
trusted: that stop-line geometry is only adopted when the evidence genuinely
supports it, and that a confirmed violation requires red + tracking + a
directional crossing rather than mere proximity to a red light.
"""

from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config                                                        # noqa: E402
from redlight import (                                               # noqa: E402
    LIGHT_STATE_CLASSES,
    STOP_LINE_CLASS,
    LightState,
    RedLightMonitor,
    StopLine,
    StopLineEstimator,
    parse_stop_lines,
)

W, H, FPS = 1920, 1080, 30.0


class FakeTrack:
    """Minimal stand-in exposing the .id and .centers the monitor reads."""

    def __init__(self, tid, points=()):
        self.id = int(tid)
        self.centers = deque(maxlen=20)
        for p in points:
            self.centers.append((float(p[0]), float(p[1])))

    def move_to(self, x, y):
        self.centers.append((float(x), float(y)))
        return self


def estimator(**kw):
    opts = dict(
        min_samples=config.STOP_LINE_MIN_SAMPLES,
        min_conf=config.STOP_LINE_MIN_CONF,
        max_spread=config.STOP_LINE_MAX_SPREAD,
        margin=config.STOP_LINE_MARGIN,
    )
    opts.update(kw)
    return StopLineEstimator(**opts)


def stop_line_box(y_bottom_frac=0.60, x1_frac=0.20, x2_frac=0.80):
    """A stop_line detection in pixels at a given normalised position."""
    return (x1_frac * W, y_bottom_frac * H - 12, x2_frac * W, y_bottom_frac * H)


# --------------------------------------------------------------------------- #


class TestStopLineEstimator(unittest.TestCase):
    """Geometry must be earned, not assumed."""

    def test_does_not_lock_before_enough_samples(self):
        est = estimator()
        for f in range(config.STOP_LINE_MIN_SAMPLES - 1):
            self.assertIsNone(est.observe(stop_line_box(), 0.9, W, H, f))
        self.assertFalse(est.report()["locked"])

    def test_locks_after_consistent_samples(self):
        est = estimator()
        line = None
        for f in range(config.STOP_LINE_MIN_SAMPLES):
            line = est.observe(stop_line_box(0.60), 0.9, W, H, f)
        self.assertIsNotNone(line)
        # y is taken from the bottom edge of the detected box.
        self.assertAlmostEqual(line.p1[1], 0.60, places=2)
        self.assertTrue(est.report()["locked"])

    def test_low_confidence_detections_are_not_sampled(self):
        est = estimator()
        for f in range(40):
            est.observe(stop_line_box(), config.STOP_LINE_MIN_CONF - 0.05, W, H, f)
        self.assertEqual(est.report()["samples"], 0)
        self.assertFalse(est.report()["locked"])

    def test_refuses_to_lock_when_detections_disagree(self):
        """Scattered detections mean the model is unsure; guessing would accuse
        real vehicles on invented geometry."""
        est = estimator()
        for f in range(30):
            y = 0.30 if f % 2 else 0.85          # far apart, alternating
            est.observe(stop_line_box(y), 0.9, W, H, f)
        report = est.report()
        self.assertFalse(report["locked"])
        self.assertIn("disagreed", report["reason"])

    def test_outlier_does_not_move_the_locked_line(self):
        """Median, not mean: one bad frame must not drag the geometry."""
        est = estimator()
        line = None
        for f in range(config.STOP_LINE_MIN_SAMPLES - 1):
            line = est.observe(stop_line_box(0.60), 0.9, W, H, f)
        line = est.observe(stop_line_box(0.66), 0.9, W, H, 99)   # within spread
        self.assertIsNotNone(line)
        self.assertAlmostEqual(line.p1[1], 0.60, places=2)

    def test_locked_line_stops_consuming_samples(self):
        est = estimator()
        for f in range(config.STOP_LINE_MIN_SAMPLES):
            est.observe(stop_line_box(0.60), 0.9, W, H, f)
        before = est.report()["samples"]
        for f in range(50, 80):
            est.observe(stop_line_box(0.20), 0.95, W, H, f)
        self.assertEqual(est.report()["samples"], before)
        self.assertAlmostEqual(est.locked.p1[1], 0.60, places=2)

    def test_locked_line_spans_at_least_the_detection(self):
        est = estimator()
        line = None
        for f in range(config.STOP_LINE_MIN_SAMPLES):
            line = est.observe(stop_line_box(0.60, 0.30, 0.70), 0.9, W, H, f)
        self.assertLessEqual(line.p1[0], 0.30)
        self.assertGreaterEqual(line.p2[0], 0.70)


class TestEnforcementGating(unittest.TestCase):
    """Enforcement must not run on geometry it does not have."""

    def test_monitor_is_inert_without_geometry(self):
        m = RedLightMonitor([], FPS)
        self.assertFalse(m.enabled)
        self.assertEqual(m.source, "none")
        self.assertIsNone(m.active_from_frame)

    def test_adopting_geometry_arms_enforcement_from_that_frame(self):
        m = RedLightMonitor([], FPS)
        line = StopLine((0.1, 0.5), (0.9, 0.5), name="auto")
        self.assertTrue(m.adopt_stop_lines([line], 420, source="auto"))
        self.assertTrue(m.enabled)
        self.assertEqual(m.source, "auto")
        self.assertEqual(m.active_from_frame, 420)

    def test_manual_geometry_is_never_overridden(self):
        manual = StopLine((0.0, 0.4), (1.0, 0.4), name="manual")
        m = RedLightMonitor([manual], FPS)
        self.assertEqual(m.source, "manual")
        auto = StopLine((0.1, 0.9), (0.9, 0.9), name="auto")
        self.assertFalse(m.adopt_stop_lines([auto], 100, source="auto"))
        self.assertEqual(m.stop_lines[0].name, "manual")

    def test_adopting_empty_geometry_changes_nothing(self):
        m = RedLightMonitor([], FPS)
        self.assertFalse(m.adopt_stop_lines([], 10))
        self.assertFalse(m.adopt_stop_lines([None], 10))
        self.assertFalse(m.enabled)


class TestViolationConfirmation(unittest.TestCase):
    """A violation needs red + tracking + a real crossing."""

    def _monitor(self):
        line = StopLine((0.0, 0.50), (1.0, 0.50), name="auto")
        return RedLightMonitor([line], FPS, light_state=LightState(FPS))

    def _go_red(self, m, frame=1):
        # Two votes, matching LightState.min_votes.
        m.observe_lights([(3, 0.9)], frame)
        m.observe_lights([(3, 0.9)], frame + 1)
        self.assertTrue(m.light.is_red)

    def test_crossing_on_red_is_a_violation(self):
        m = self._monitor()
        self._go_red(m)
        t = FakeTrack(1, [(500, 0.45 * H)]).move_to(500, 0.55 * H)
        fresh = m.update([t], 10, W, H)
        self.assertEqual(len(fresh), 1)
        self.assertEqual(fresh[0]["vehicle_id"], 1)
        self.assertEqual(fresh[0]["light_state"], "red")

    def test_same_vehicle_is_latched_to_one_violation(self):
        """The redlight.py guarantee: one crossing, one violation."""
        m = self._monitor()
        self._go_red(m)
        t = FakeTrack(1, [(500, 0.45 * H)])
        for i, y in enumerate([0.55, 0.60, 0.65, 0.70]):
            t.move_to(500, y * H)
            m.update([t], 10 + i, W, H)
        self.assertEqual(m.count, 1)

    def test_no_violation_on_green(self):
        m = self._monitor()
        m.observe_lights([(1, 0.9)], 1)
        m.observe_lights([(1, 0.9)], 2)
        t = FakeTrack(1, [(500, 0.45 * H)]).move_to(500, 0.55 * H)
        self.assertEqual(m.update([t], 10, W, H), [])
        self.assertEqual(m.count, 0)

    def test_no_violation_when_light_state_unknown(self):
        m = self._monitor()
        t = FakeTrack(1, [(500, 0.45 * H)]).move_to(500, 0.55 * H)
        self.assertEqual(m.update([t], 10, W, H), [])

    def test_driving_up_to_the_line_but_stopping_is_not_a_violation(self):
        """The car that obeys the signal must never be flagged."""
        m = self._monitor()
        self._go_red(m)
        t = FakeTrack(1, [(500, 0.30 * H)])
        for i, y in enumerate([0.38, 0.44, 0.47, 0.485, 0.49]):
            t.move_to(500, y * H)
            m.update([t], 10 + i, W, H)
        self.assertEqual(m.count, 0)

    def test_lane_restricted_line_ignores_other_lanes(self):
        line = StopLine((0.0, 0.50), (1.0, 0.50), name="lane2", lane=2)
        m = RedLightMonitor([line], FPS, light_state=LightState(FPS))
        self._go_red(m)
        t = FakeTrack(7, [(500, 0.45 * H)]).move_to(500, 0.55 * H)
        self.assertEqual(m.update([t], 10, W, H, lane_of=lambda _t: 1), [])
        self.assertEqual(m.update([t], 11, W, H, lane_of=lambda _t: 2)[0]["lane"], 2)

    def test_directional_line_ignores_wrong_way_crossing(self):
        line = StopLine((0.0, 0.50), (1.0, 0.50), name="one_way", direction="positive")
        m = RedLightMonitor([line], FPS, light_state=LightState(FPS))
        self._go_red(m)
        up = FakeTrack(9, [(500, 0.55 * H)]).move_to(500, 0.45 * H)
        self.assertEqual(m.update([up], 10, W, H), [])


class TestLightState(unittest.TestCase):
    def test_single_weak_detection_does_not_flip_to_red(self):
        ls = LightState(FPS)
        ls.observe([(3, 0.30)], 1)
        self.assertFalse(ls.is_red)

    def test_state_is_held_across_detection_gaps(self):
        """The checkpoint reports a signal in a minority of frames, so holding
        the last confident observation is what makes enforcement usable."""
        ls = LightState(FPS, hold_seconds=2.0)
        ls.observe([(3, 0.9)], 1)
        ls.observe([(3, 0.9)], 2)
        self.assertTrue(ls.is_red)
        ls.observe([], 30)
        self.assertTrue(ls.is_red)

    def test_stale_hold_expires_to_unknown(self):
        ls = LightState(FPS, hold_seconds=1.0)
        ls.observe([(3, 0.9)], 1)
        ls.observe([(3, 0.9)], 2)
        ls.observe([], 200)
        self.assertEqual(ls.state, "unknown")


class TestClassIndices(unittest.TestCase):
    """The hardcoded class ids must match the trained checkpoint.

    These are bare integers in redlight.py, so if the dataset is ever re-exported
    with a different class order they would still be valid indices and nothing
    would raise - enforcement would just silently read the wrong class and report
    zero violations forever. That failure is invisible at runtime, which is
    exactly why it is pinned here.
    """

    def test_light_state_ids_match_the_class_names(self):
        names = config.TRAFFIC_LIGHT_NAMES
        for cls, state in LIGHT_STATE_CLASSES.items():
            self.assertEqual(
                names[cls], f"{state}_light",
                f"class id {cls} is {names[cls]!r}, not the {state} signal",
            )

    def test_stop_line_id_matches_the_class_names(self):
        self.assertEqual(config.TRAFFIC_LIGHT_NAMES[STOP_LINE_CLASS], "stop_line")

    def test_every_signal_class_is_accounted_for(self):
        """No signal colour may be left out of LIGHT_STATE_CLASSES."""
        signals = {i for i, n in enumerate(config.TRAFFIC_LIGHT_NAMES)
                   if n.endswith("_light")}
        self.assertEqual(signals, set(LIGHT_STATE_CLASSES))


class TestParsing(unittest.TestCase):
    def test_accepts_both_geometry_formats(self):
        self.assertEqual(len(parse_stop_lines([
            {"p1": [0.1, 0.5], "p2": [0.9, 0.5]},
            {"x1": 0.1, "y1": 0.6, "x2": 0.9, "y2": 0.6},
        ])), 2)

    def test_malformed_entries_are_skipped_not_fatal(self):
        self.assertEqual(parse_stop_lines([
            {"p1": [0.1, 0.5]},
            {"p1": [0.2, 0.2], "p2": [0.2, 0.2]},
            "nonsense",
        ]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
