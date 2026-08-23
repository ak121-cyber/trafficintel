"""Accident event confirmation.

The accident YOLO model reports per-frame boxes. It does NOT decide that an
accident happened. Turning a stream of per-frame detections into a list of
*events* is this module's job, and it is where the previous implementation went
wrong.

What was wrong
--------------
The old logic emitted an event whenever a fixed 2-second cooldown expired while
evidence was still high. Evidence stayed high for most of the clip, so the
cooldown - not the physics - decided the count. The symptom is unmistakable in
the recorded output: 27 "accidents" in 76 seconds whose inter-event gaps were
2.66, 2.44, 2.23, 2.00, 2.00, 2.00, 2.20, 2.03, 2.07 ... i.e. the cooldown
itself. Vehicle pair [68, 76] was reported twice, [157, 150] twice, and vehicle
157 appeared in four separate "accidents". Motion-only events were also given a
fabricated confidence of 0.50 + 0.08 * score, which is why every one of them
came out as exactly 0.82 or 0.86.

The fix
-------
An incident has identity and a lifetime, exactly like a red-light violation in
redlight.py, where each vehicle is latched so "one car crossing on red produces
exactly one violation, not one per frame":

    open    enough sustained evidence accumulates to confirm an impact
    hold    further evidence updates that same incident, never creates a new one
    close   evidence stops for `settle_seconds`; only then may a genuinely new
            incident involving the same vehicles open

Two independent evidence sources feed it:

    model   the trained detector fired, at sufficient confidence, on enough
            frames inside a short window, in roughly the same place
    motion  two tracked vehicles made contact AND at least one lost nearly all
            of its speed AND stayed slow afterwards

The motion test is deliberately strict about the speed collapse. A car braking
into a queue at a red light decelerates smoothly and its neighbours' boxes
overlap constantly under perspective; that pattern produced most of the old
false positives. A real impact shows a near-instant collapse to a standstill
that persists. Speed is measured in units of the vehicle's own box diagonal per
frame, so the same thresholds work at 640x360 and 1920x1080 and for vehicles
near and far from the camera.

Confidence is never fabricated. A model-backed incident reports the peak model
confidence. A motion-only incident reports `confidence: None` and is labelled as
motion evidence, so no consumer can imply the detector agreed when it did not.
"""

from collections import deque
import math

# Evidence source labels used in the emitted events.
EVIDENCE_MODEL = "model"
EVIDENCE_MOTION = "motion"

MOTION_ONLY_TYPE = "Vehicle collision (motion evidence)"


def iou(a, b):
    """Intersection over union of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-6)


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def box_diagonal(b):
    return math.hypot(max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1]))


def center_distance(a, b):
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.hypot(ax - bx, ay - by)


def box_gap(a, b):
    """Shortest distance between the edges of two boxes; 0 when they overlap.

    This is the right measure for physical contact. Centre distance is not: two
    cars touching bumper to bumper have their centres a full car length apart
    and a box IoU near zero, so a centre-distance test misses the exact case it
    most needs to catch.
    """
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def union_box(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class _PairState:
    """Accumulated contact and speed evidence for one pair of tracked vehicles.

    Kept per pair rather than per frame so that a decision is made from a short
    history: a single frame of box overlap means very little, whereas several
    frames of contact plus a speed collapse that persists is an impact.
    """

    __slots__ = ("contact_frames", "first_contact_frame", "last_contact_frame",
                 "collapse_frame", "post_slow_frames", "box")

    def __init__(self):
        self.contact_frames = 0
        self.first_contact_frame = -1
        self.last_contact_frame = -1
        self.collapse_frame = None
        self.post_slow_frames = 0
        self.box = None

    def reset(self, frame_no):
        self.contact_frames = 0
        self.first_contact_frame = int(frame_no)
        self.collapse_frame = None
        self.post_slow_frames = 0


class Incident:
    """One confirmed accident, with a lifetime rather than a single frame.

    `confidence` is the peak confidence reported by the trained model, or None
    when the incident rests on motion evidence alone. It is never synthesised.
    """

    def __init__(self, key, frame_no, fps, box, vehicles):
        self.key = key
        self.fps = fps
        self.first_frame = int(frame_no)
        self.last_evidence_frame = int(frame_no)
        self.box = tuple(float(v) for v in box) if box is not None else None
        self.vehicles = list(vehicles or [])
        self.evidence = set()
        self.model_confidence = None
        self.model_type = None
        self.evidence_frames = 1
        self.closed = False

    # -- lifecycle ---------------------------------------------------------- #

    def touch(self, frame_no, box=None, vehicles=None):
        self.last_evidence_frame = max(self.last_evidence_frame, int(frame_no))
        self.evidence_frames += 1
        if box is not None:
            self.box = union_box(self.box, box) if self.box is not None else tuple(float(v) for v in box)
        for vid in vehicles or []:
            if vid not in self.vehicles:
                self.vehicles.append(vid)

    def add_model_evidence(self, conf, type_name):
        """Record a model hit, keeping the strongest confidence seen."""
        self.evidence.add(EVIDENCE_MODEL)
        conf = float(conf)
        if self.model_confidence is None or conf > self.model_confidence:
            self.model_confidence = conf
            self.model_type = type_name

    def add_motion_evidence(self):
        self.evidence.add(EVIDENCE_MOTION)

    # -- reporting ---------------------------------------------------------- #

    @property
    def type(self):
        return self.model_type if self.model_type else MOTION_ONLY_TYPE

    @property
    def duration_seconds(self):
        span = self.last_evidence_frame - self.first_frame
        return round(max(0, span) / self.fps, 2)

    def as_dict(self):
        """Serialisable event.

        `frame`/`time_seconds` mark the moment the incident was confirmed, so
        seeking to that timestamp in the annotated video lands on the impact.
        """
        return {
            "frame": self.first_frame,
            "time_seconds": round(self.first_frame / self.fps, 2),
            "type": self.type,
            # None, not a fabricated number, when the model did not fire.
            "confidence": round(self.model_confidence, 4) if self.model_confidence is not None else None,
            "vehicles": list(self.vehicles),
            "evidence": sorted(self.evidence),
            "model_confirmed": EVIDENCE_MODEL in self.evidence,
            "duration_seconds": self.duration_seconds,
            "evidence_frames": self.evidence_frames,
            "last_frame": self.last_evidence_frame,
        }


class AccidentMonitor:
    """Confirms accident events from model detections plus tracking evidence.

    Usage per frame:

        fresh = monitor.observe(frame_no, tracks, model_detections)

    `tracks` are objects exposing .id, .box and .norm_speeds (a deque of speeds
    in box-diagonals per frame). `model_detections` is an iterable of
    (box, type_name, confidence) already filtered to accident classes.

    Returns the incidents newly confirmed on this frame - usually empty.
    """

    def __init__(self, fps, preset=None, require_model=None,
                 same_place_radius=0.12, contact_gap_factor=0.25,
                 frame_size=None, **overrides):
        self.fps = float(fps) if fps and fps > 0 else 30.0

        cfg = dict(preset or {})
        cfg.update({k: v for k, v in overrides.items() if v is not None})

        self.model_min_conf = float(cfg.get("model_min_conf", 0.35))
        self.model_min_frames = max(1, int(cfg.get("model_min_frames", 3)))
        self.model_window = max(1, int(round(float(cfg.get("model_window_seconds", 1.0)) * self.fps)))
        self.contact_iou = float(cfg.get("contact_iou", 0.15))
        self.min_contact_frames = max(1, int(cfg.get("min_contact_frames", 3)))
        self.moving_speed = float(cfg.get("moving_speed", 0.018))
        self.stopped_speed = float(cfg.get("stopped_speed", 0.007))
        self.min_post_slow_frames = max(1, int(round(float(cfg.get("min_post_slow_seconds", 0.5)) * self.fps)))
        self.settle_frames = max(1, int(round(float(cfg.get("settle_seconds", 5.0)) * self.fps)))
        self.max_collapse_delay_frames = max(
            1, int(round(float(cfg.get("max_collapse_delay_seconds", 0.8)) * self.fps)))

        self.require_model = bool(cfg.get("require_model", False)) if require_model is None else bool(require_model)

        self.same_place_radius = float(same_place_radius)
        self.contact_gap_factor = float(contact_gap_factor)

        # Frame diagonal in pixels, used to turn same_place_radius into a
        # pixel distance. Falls back to a 640x360 reference until set.
        self.frame_diagonal = math.hypot(*frame_size) if frame_size else math.hypot(640, 360)

        # Contact lapses if the pair is not seen together for this long, which
        # discards a brush-past instead of letting it accumulate forever.
        self.contact_lapse_frames = max(2, int(round(0.4 * self.fps)))

        # Rolling window of qualifying model detections: (frame, box, type, conf).
        self._model_hits = deque()
        self._pairs = {}
        self._open = {}
        self._serial = 0
        self.incidents = []
        self.alert_until = -1

    # ------------------------------------------------------------------ setup #

    def set_frame_size(self, width, height):
        if width and height:
            self.frame_diagonal = math.hypot(float(width), float(height))

    # ------------------------------------------------------------- evidence -- #

    def _speed_collapsed(self, track):
        """True when this track was clearly moving and has just about stopped.

        Uses normalised speed (box diagonals per frame) so the test is
        independent of resolution and of how far the vehicle is from the camera.
        A smooth deceleration into a queue does not satisfy this: it requires
        the recent speed to be at or below `stopped_speed`, which is close to
        stationary, while the earlier speed was at or above `moving_speed`.
        """
        speeds = list(getattr(track, "norm_speeds", ()) or ())
        if len(speeds) < 6:
            return False
        before = speeds[-6:-3]
        after = speeds[-3:]
        was_moving = (sum(before) / len(before)) >= self.moving_speed
        now_stopped = max(after) <= self.stopped_speed
        return bool(was_moving and now_stopped)

    def _is_slow(self, track):
        speeds = list(getattr(track, "norm_speeds", ()) or ())
        if not speeds:
            return False
        recent = speeds[-3:]
        return (sum(recent) / len(recent)) <= self.stopped_speed

    def _in_contact(self, a, b):
        """Contact means overlapping boxes, or boxes almost touching.

        Contact is deliberately permissive: in dense traffic many pairs will
        satisfy it, and that is fine, because contact alone never raises an
        event. The discriminator is the speed collapse below. Being permissive
        here is what lets a bumper-to-bumper impact - box IoU near zero - be
        seen at all.
        """
        if iou(a.box, b.box) >= self.contact_iou:
            return True
        scale = min(box_diagonal(a.box), box_diagonal(b.box)) or 1.0
        return box_gap(a.box, b.box) <= scale * self.contact_gap_factor

    def _update_pairs(self, frame_no, tracks):
        """Accumulate contact/collapse evidence and return confirmed pairs."""
        confirmed = []
        seen = set()

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                if a.box is None or b.box is None:
                    continue
                if not self._in_contact(a, b):
                    continue

                key = (min(int(a.id), int(b.id)), max(int(a.id), int(b.id)))
                seen.add(key)
                ps = self._pairs.setdefault(key, _PairState())

                # A gap in contact resets the accumulation.
                if ps.last_contact_frame >= 0 and frame_no - ps.last_contact_frame > self.contact_lapse_frames:
                    ps.reset(frame_no)
                if ps.first_contact_frame < 0:
                    ps.first_contact_frame = frame_no

                ps.contact_frames += 1
                ps.last_contact_frame = frame_no
                ps.box = union_box(a.box, b.box)

                if ps.collapse_frame is None:
                    # The stop must be roughly simultaneous with contact: an
                    # impact is what caused it. A vehicle that has been beside
                    # another for seconds and only then brakes is traffic, not a
                    # crash, and this window is what rejects it.
                    within_window = (frame_no - ps.first_contact_frame) <= self.max_collapse_delay_frames
                    if within_window and (self._speed_collapsed(a) or self._speed_collapsed(b)):
                        ps.collapse_frame = frame_no
                else:
                    # Post-impact persistence: a real collision leaves the
                    # vehicles stationary. Traffic that merely slowed will
                    # start moving again and never accumulate enough frames.
                    if self._is_slow(a) or self._is_slow(b):
                        ps.post_slow_frames += 1

                if (ps.contact_frames >= self.min_contact_frames
                        and ps.collapse_frame is not None
                        and ps.post_slow_frames >= self.min_post_slow_frames):
                    confirmed.append((key, ps.box))

        # Forget pairs that have not been in contact recently.
        for key in [k for k, ps in self._pairs.items()
                    if k not in seen and frame_no - ps.last_contact_frame > self.contact_lapse_frames]:
            self._pairs.pop(key, None)

        return confirmed

    def _model_cluster(self, frame_no):
        """Sustained model evidence in one place, or None.

        Requires `model_min_frames` detections inside `model_window` frames
        whose centres sit within `same_place_radius` of the newest one. A single
        noisy frame therefore cannot raise an event, which the old code allowed
        by firing immediately at 0.25 confidence.
        """
        while self._model_hits and frame_no - self._model_hits[0][0] > self.model_window:
            self._model_hits.popleft()

        if len(self._model_hits) < self.model_min_frames:
            return None

        newest_center = box_center(self._model_hits[-1][1])
        radius = self.same_place_radius * self.frame_diagonal

        cluster = []
        for hit in self._model_hits:
            hx, hy = box_center(hit[1])
            if math.hypot(hx - newest_center[0], hy - newest_center[1]) <= radius:
                cluster.append(hit)
        if len(cluster) < self.model_min_frames:
            return None

        best = max(cluster, key=lambda h: h[3])
        box = cluster[0][1]
        for h in cluster[1:]:
            box = union_box(box, h[1])
        return box, best[2], best[3]

    # -------------------------------------------------------------- incidents #

    def _close_stale(self, frame_no):
        for key, inc in list(self._open.items()):
            if frame_no - inc.last_evidence_frame > self.settle_frames:
                inc.closed = True
                self._open.pop(key, None)

    def _find_open_near(self, box):
        """An open incident overlapping this region, so evidence merges into it."""
        if box is None:
            return None
        radius = self.same_place_radius * self.frame_diagonal
        cx, cy = box_center(box)
        best = None
        for inc in self._open.values():
            if inc.box is None:
                continue
            if iou(inc.box, box) > 0.0:
                return inc
            ix, iy = box_center(inc.box)
            d = math.hypot(ix - cx, iy - cy)
            if d <= radius and (best is None or d < best[0]):
                best = (d, inc)
        return best[1] if best else None

    # ------------------------------------------------------------------ main -- #

    def observe(self, frame_no, tracks, model_detections=()):
        """Feed one frame. Returns incidents newly confirmed on this frame."""
        frame_no = int(frame_no)
        self._close_stale(frame_no)

        # Record qualifying model detections into the rolling window.
        for det in model_detections or ():
            box, type_name, conf = det[0], det[1], float(det[2])
            if conf >= self.model_min_conf:
                self._model_hits.append((frame_no, box, type_name, conf))

        model_evidence = self._model_cluster(frame_no)
        motion_pairs = self._update_pairs(frame_no, list(tracks or ()))

        fresh = []

        # -- motion evidence -------------------------------------------------- #
        for key, box in motion_pairs:
            ikey = ("pair", key)
            inc = self._open.get(ikey) or self._find_open_near(box)
            if inc is not None:
                inc.touch(frame_no, box=box, vehicles=list(key))
                inc.add_motion_evidence()
                continue

            # In strict mode motion alone may not raise an event; it can only
            # reinforce one the model already confirmed.
            if self.require_model:
                continue

            inc = Incident(ikey, frame_no, self.fps, box, list(key))
            inc.add_motion_evidence()
            self._open[ikey] = inc
            self.incidents.append(inc)
            fresh.append(inc)

        # -- model evidence --------------------------------------------------- #
        if model_evidence is not None:
            box, type_name, conf = model_evidence
            inc = self._find_open_near(box)
            if inc is not None:
                # Upgrades an existing motion-only incident to model-confirmed
                # rather than logging a second event for the same crash.
                inc.touch(frame_no, box=box)
                inc.add_model_evidence(conf, type_name)
            else:
                # No open incident nearby, so this is a genuinely new one. The
                # key only needs to be unique; spatial de-duplication already
                # happened in _find_open_near above.
                self._serial += 1
                ikey = ("model", self._serial)
                inc = Incident(ikey, frame_no, self.fps, box, [])
                inc.add_model_evidence(conf, type_name)
                self._open[ikey] = inc
                self.incidents.append(inc)
                fresh.append(inc)

        if fresh:
            self.alert_until = frame_no + max(1, int(self.fps * 2.5))
        return fresh

    # ---------------------------------------------------------------- output -- #

    @property
    def count(self):
        return len(self.incidents)

    @property
    def events(self):
        return [inc.as_dict() for inc in self.incidents]

    def latest(self):
        return self.incidents[-1] if self.incidents else None

    def involved_ids(self):
        """Vehicle ids belonging to any confirmed incident, for highlighting."""
        out = set()
        for inc in self.incidents:
            out.update(inc.vehicles)
        return out
