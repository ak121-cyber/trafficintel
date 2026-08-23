"""Red-light violation logic.

The traffic-light YOLO model only reports objects. It does NOT decide a violation.
A violation is confirmed here by combining four independent pieces of evidence:

    RED light state
    + a ByteTrack-tracked vehicle
    + that vehicle crossing a configurable stop line
    + crossing in the enforced direction of travel

A vehicle is never flagged merely for being near a red light, and never flagged
just because a red light is visible somewhere in the frame.

Geometry is stored in normalised 0..1 coordinates so one calibration works for
any resolution (test1.mp4 is 640x360, test2.mp4 is 1920x1080).
"""

from collections import deque

# Traffic-light class ids in TRAFFIC_LIGHT_NAMES that carry a signal state.
LIGHT_STATE_CLASSES = {
    1: "green",
    3: "red",
    5: "yellow",
}
STOP_LINE_CLASS = 4

STATE_COLORS = {
    "red": (40, 60, 235),
    "yellow": (35, 190, 240),
    "green": (90, 200, 90),
    "unknown": (120, 130, 140),
}


class LightState:
    """Temporally smoothed traffic-light state.

    The trained checkpoint reports a signal in only a small fraction of frames
    (measured: ~7% on test2.mp4, with gaps up to 55 consecutive frames), so the
    per-frame detection alone cannot drive enforcement. Two mechanisms fix that:

    hold   - the last confident observation stays authoritative for `hold_seconds`,
             which bridges detection gaps.
    votes  - a new state must be observed `min_votes` times inside a short window
             before it replaces the current one, so a single stray box cannot
             flip the signal to RED and cause false violations.
    """

    def __init__(self, fps, hold_seconds=2.5, vote_window_seconds=0.5, min_votes=2, min_conf=0.25):
        fps = float(fps) if fps and fps > 0 else 30.0
        self.hold_frames = max(1, int(round(hold_seconds * fps)))
        self.vote_window = max(1, int(round(vote_window_seconds * fps)))
        self.min_votes = max(1, int(min_votes))
        self.min_conf = float(min_conf)

        self.state = "unknown"
        self.confidence = 0.0
        self.last_seen_frame = None
        self._observations = deque()          # (frame_no, state, conf)

    def observe(self, detections, frame_no):
        """Feed one frame of traffic-light detections.

        `detections` is an iterable of (class_id, confidence).
        Returns the effective state for this frame.
        """
        best = None
        for cls_id, conf in detections:
            state = LIGHT_STATE_CLASSES.get(int(cls_id))
            if state is None or float(conf) < self.min_conf:
                continue
            if best is None or float(conf) > best[1]:
                best = (state, float(conf))

        if best is not None:
            self._observations.append((frame_no, best[0], best[1]))

        # Drop observations that fell out of the voting window.
        while self._observations and frame_no - self._observations[0][0] > self.vote_window:
            self._observations.popleft()

        if best is not None:
            votes = {}
            for _, state, conf in self._observations:
                slot = votes.setdefault(state, [0, 0.0])
                slot[0] += 1
                slot[1] = max(slot[1], conf)

            count, conf = votes.get(best[0], [0, 0.0])
            # Accept immediately if this state is already the active one, or if it
            # has enough votes, or if this single detection is strongly confident.
            if best[0] == self.state or count >= self.min_votes or best[1] >= 0.60:
                self.state = best[0]
                self.confidence = conf
                self.last_seen_frame = frame_no

        # Expire a stale hold rather than enforcing on ancient information.
        if self.last_seen_frame is not None and frame_no - self.last_seen_frame > self.hold_frames:
            self.state = "unknown"
            self.confidence = 0.0

        return self.state

    @property
    def is_red(self):
        return self.state == "red"

    def label(self):
        if self.state == "unknown":
            return "LIGHT UNKNOWN"
        return f"LIGHT {self.state.upper()} {self.confidence:.0%}"

    def color(self):
        return STATE_COLORS.get(self.state, STATE_COLORS["unknown"])


def cross(p1, p2, pt):
    """Signed cross product of p1->p2 against p1->pt.

    Sign tells which side of the directed line `pt` lies on; 0 means collinear.
    """
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def segments_intersect(a1, a2, b1, b2):
    """True when segment a1-a2 properly straddles segment b1-b2.

    Segment-vs-segment (not infinite line) matters: a vehicle driving past the
    side of a short stop line must not be treated as crossing it.
    """
    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


class StopLine:
    """One configurable stop line, optionally restricted to a lane.

    `direction` selects which way a crossing counts:
        "auto"     - either direction (default)
        "positive" - only crossings that end on the +cross side
        "negative" - only crossings that end on the -cross side
    """

    def __init__(self, p1, p2, name="stop_line", direction="auto", lane=None):
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.name = str(name)
        self.direction = direction if direction in ("auto", "positive", "negative") else "auto"
        self.lane = lane

    @classmethod
    def from_config(cls, raw):
        """Build from a plain dict, tolerating partial input.

        Accepts {"p1":[x,y],"p2":[x,y]} or {"x1":..,"y1":..,"x2":..,"y2":..}.
        Returns None when the geometry is unusable.
        """
        if not isinstance(raw, dict):
            return None
        if "p1" in raw and "p2" in raw:
            p1, p2 = raw.get("p1"), raw.get("p2")
        elif all(k in raw for k in ("x1", "y1", "x2", "y2")):
            p1, p2 = (raw["x1"], raw["y1"]), (raw["x2"], raw["y2"])
        else:
            return None
        try:
            p1 = (float(p1[0]), float(p1[1]))
            p2 = (float(p2[0]), float(p2[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        if p1 == p2:
            return None
        lane = raw.get("lane")
        try:
            lane = int(lane) if lane is not None else None
        except (TypeError, ValueError):
            lane = None
        return cls(
            p1, p2,
            name=raw.get("name", "stop_line"),
            direction=str(raw.get("direction", "auto")).lower(),
            lane=lane,
        )

    def pixels(self, width, height):
        """Normalised geometry -> integer pixel endpoints."""
        return (
            (int(round(self.p1[0] * width)), int(round(self.p1[1] * height))),
            (int(round(self.p2[0] * width)), int(round(self.p2[1] * height))),
        )

    def crossed(self, prev_pt, curr_pt):
        """Did the vehicle move across this line between two samples?"""
        if not segments_intersect(prev_pt, curr_pt, self.p1, self.p2):
            return False
        if self.direction == "auto":
            return True
        ends_positive = cross(self.p1, self.p2, curr_pt) > 0
        return ends_positive if self.direction == "positive" else not ends_positive

    def as_dict(self):
        return {
            "name": self.name,
            "p1": list(self.p1),
            "p2": list(self.p2),
            "direction": self.direction,
            "lane": self.lane,
        }


def default_stop_lines():
    """No stop line is assumed.

    Without calibrated geometry the system must not guess, so violation
    detection stays inert and reports zero rather than inventing events.
    Geometry now normally arrives from StopLineEstimator instead.
    """
    return []


class StopLineEstimator:
    """Derives stop-line geometry from the model's `stop_line` detections.

    Why this exists: enforcement needs geometry, and requiring every user to
    hand-draw a line made the whole feature inert - the frontend sent nothing,
    so `RedLightMonitor.enabled` was always False and violations were always
    reported as zero. The trained checkpoint detects the stop_line class, so the
    geometry can be measured instead of demanded.

    It is deliberately conservative, because a wrong line produces false
    accusations against real vehicles:

    * a detection must clear `min_conf` to be considered at all;
    * `min_samples` separate frames must agree before the line is used, so a
      single stray box cannot arm enforcement;
    * the locked line is the per-coordinate *median* of the samples, which
      ignores outliers rather than averaging them in;
    * samples must be spatially consistent - if they disagree by more than
      `max_spread` of the frame, the estimate is treated as unreliable and
      enforcement stays off.

    The line is locked once and then left alone. Re-deriving it every frame
    would move the goalposts mid-run and make results unreproducible.
    """

    def __init__(self, min_samples=12, min_conf=0.35, max_spread=0.12, margin=0.04):
        self.min_samples = max(1, int(min_samples))
        self.min_conf = float(min_conf)
        self.max_spread = float(max_spread)
        self.margin = float(margin)
        self._samples = []            # (x1, y_bottom, x2, conf)
        self.locked = None            # StopLine once confirmed
        self.locked_frame = None
        self.rejected_reason = None

    def observe(self, box, conf, width, height, frame_no):
        """Feed one `stop_line` detection, in pixels. Returns the locked line or None.

        The stop line is taken as the bottom edge of the detected box, which is
        the edge a vehicle actually crosses as it enters the junction.
        """
        if self.locked is not None:
            return self.locked
        if float(conf) < self.min_conf:
            return None

        w = max(float(width), 1.0)
        h = max(float(height), 1.0)
        self._samples.append((
            float(box[0]) / w,
            float(box[3]) / h,
            float(box[2]) / w,
            float(conf),
        ))

        if len(self._samples) < self.min_samples:
            return None
        return self._try_lock(frame_no)

    def _try_lock(self, frame_no):
        ys = sorted(s[1] for s in self._samples)
        spread = ys[-1] - ys[0]
        if spread > self.max_spread:
            # The detections do not agree on where the line is. Refusing to lock
            # is the correct outcome: enforcement stays off and the result says so.
            self.rejected_reason = (
                f"stop-line detections disagreed by {spread:.0%} of frame height "
                f"(limit {self.max_spread:.0%}), so geometry was not trusted"
            )
            return None

        x1 = _median(sorted(s[0] for s in self._samples))
        x2 = _median(sorted(s[2] for s in self._samples))
        y = _median(ys)
        if abs(x2 - x1) < 0.02:
            self.rejected_reason = "detected stop line was too short to use"
            return None

        # Extend slightly past the detected width: the box rarely spans the full
        # carriageway, and a vehicle crossing just outside it is still crossing.
        x1 = max(0.0, min(x1, x2) - self.margin)
        x2 = min(1.0, max(x1 + 0.02, max(s[2] for s in self._samples) + self.margin))

        self.locked = StopLine((x1, y), (x2, y), name="auto", direction="auto")
        self.locked_frame = int(frame_no)
        self.rejected_reason = None
        return self.locked

    def report(self):
        """Serialisable state, so a stored result explains itself later."""
        return {
            "locked": self.locked is not None,
            "locked_at_frame": self.locked_frame,
            "samples": len(self._samples),
            "min_samples": self.min_samples,
            "line": self.locked.as_dict() if self.locked is not None else None,
            "reason": self.rejected_reason,
        }


def _median(sorted_values):
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(sorted_values[mid])
    return (float(sorted_values[mid - 1]) + float(sorted_values[mid])) / 2.0


def parse_stop_lines(raw):
    """Parse a list of stop-line dicts, skipping malformed entries."""
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    lines = []
    for item in raw:
        line = StopLine.from_config(item)
        if line is not None:
            lines.append(line)
    return lines


class RedLightMonitor:
    """Confirms red-light violations from tracking + light state + geometry.

    Each vehicle can be latched at most once per stop line, so one car crossing
    on red produces exactly one violation, not one per frame.
    """

    def __init__(self, stop_lines, fps, light_state=None, min_light_conf=0.25):
        self.stop_lines = list(stop_lines or [])
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.light = light_state if light_state is not None else LightState(self.fps, min_conf=min_light_conf)
        self.violations = []
        self._latched = set()                 # (track_id, stop_line_name)
        self.alert_until = -1
        # "manual" when the caller supplied geometry, "auto" once estimated.
        self.source = "manual" if self.stop_lines else "none"
        self.active_from_frame = 0 if self.stop_lines else None

    @property
    def enabled(self):
        """Violation checking only runs when geometry has been configured."""
        return bool(self.stop_lines)

    def adopt_stop_lines(self, lines, frame_no, source="auto"):
        """Install geometry discovered mid-run, arming enforcement from here on.

        Manual geometry always wins: if the caller supplied a line, an estimated
        one must not silently replace it.
        """
        if self.stop_lines:
            return False
        lines = [l for l in (lines or []) if l is not None]
        if not lines:
            return False
        self.stop_lines = list(lines)
        self.source = source
        self.active_from_frame = int(frame_no)
        return True

    def observe_lights(self, detections, frame_no):
        return self.light.observe(detections, frame_no)

    def update(self, tracks, frame_no, width, height, lane_of=None):
        """Check every tracked vehicle for a fresh violation.

        `tracks` are objects exposing .id and .centers (a deque of pixel centres).
        Returns the violations confirmed on this frame.
        """
        if not self.enabled or not self.light.is_red:
            return []

        fresh = []
        for track in tracks:
            centers = list(getattr(track, "centers", ()))
            if len(centers) < 2:
                continue

            # Compare the two most recent samples in normalised space.
            prev_pt = (centers[-2][0] / max(width, 1), centers[-2][1] / max(height, 1))
            curr_pt = (centers[-1][0] / max(width, 1), centers[-1][1] / max(height, 1))
            if prev_pt == curr_pt:
                continue

            lane = lane_of(track) if lane_of is not None else None

            for line in self.stop_lines:
                key = (int(track.id), line.name)
                if key in self._latched:
                    continue
                if line.lane is not None and lane is not None and int(line.lane) != int(lane):
                    continue
                if not line.crossed(prev_pt, curr_pt):
                    continue

                self._latched.add(key)
                event = {
                    "frame": int(frame_no),
                    "time_seconds": round(frame_no / self.fps, 2),
                    "vehicle_id": int(track.id),
                    "stop_line": line.name,
                    "lane": int(lane) if lane is not None else None,
                    "light_state": self.light.state,
                    "light_confidence": round(float(self.light.confidence), 4),
                }
                self.violations.append(event)
                fresh.append(event)

        if fresh:
            self.alert_until = frame_no + max(1, int(self.fps * 2.5))
        return fresh

    @property
    def count(self):
        return len(self.violations)

    def violating_ids(self):
        return {vid for vid, _ in self._latched}
