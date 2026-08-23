from pathlib import Path

ROOT = Path(__file__).resolve().parent

ACCIDENT_DATASET = ROOT / "datasets" / "accident"
TRAFFIC_LIGHT_DATASET = ROOT / "datasets" / "traffic_light"

ACCIDENT_MODEL = ROOT / "models" / "accident" / "best.pt"


def _resolve_traffic_light_model():
    """Locate the trained traffic-light weights.

    train_traffic_light.py writes to models/traffic_light/yolo26m/weights/best.pt
    and, unlike train_accident.py, does not copy the file up to
    models/traffic_light/best.pt. Both layouts are accepted so a future
    checkpoint can simply be dropped in either place.
    """
    candidates = [
        ROOT / "models" / "traffic_light" / "best.pt",
        ROOT / "models" / "traffic_light" / "yolo26m" / "weights" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


TRAFFIC_LIGHT_MODEL = _resolve_traffic_light_model()

YOLO_MODEL = "yolo26m.pt"

ACCIDENT_NAMES = [
    "No Accident",
    "Minor Accident",
    "Moderate Accident",
    "Severe Accident",
    "Totaled Vehicle",
]

TRAFFIC_LIGHT_NAMES = [
    "car",
    "green_light",
    "motobike",
    "red_light",
    "stop_line",
    "yellow_light",
]

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

IMG_SIZE = 640
VEHICLE_CONF = 0.25
ACCIDENT_CONF = 0.30
TRAFFIC_LIGHT_CONF = 0.25
TRACKER = "bytetrack_custom.yaml"

# --------------------------------------------------------------------------- #
# Accident event confirmation
#
# These control how per-frame detections become *events*, which is a different
# question from how the detector is thresholded. The original implementation
# fired an event whenever a 2-second cooldown expired while evidence stayed
# high, so one ongoing situation was reported as dozens of accidents: a 76s
# clip produced 27 "accidents" whose inter-event gaps were almost all exactly
# 2.0-2.1s, and the same vehicle pairs reappeared repeatedly.
#
# Detection thresholds are only half the fix. The other half is that an
# incident must have identity and a lifetime, so repeated evidence about the
# same crash updates one event instead of creating new ones. See accidents.py.
#
# Every value below is a starting point that should be calibrated against real
# footage from the target camera. ACCIDENT_SENSITIVITY selects a preset.
# --------------------------------------------------------------------------- #

# Inference threshold for the accident model. Detections below this are not
# even considered as candidate evidence.
ACCIDENT_DETECT_CONF = 0.25

ACCIDENT_SENSITIVITY = "balanced"

ACCIDENT_PRESETS = {
    # Fewest false positives. Motion evidence alone can never raise an event;
    # the trained model must agree. Use this when a false alarm is expensive.
    "strict": {
        "require_model": True,
        "model_min_conf": 0.45,
        "model_min_frames": 4,
        "model_window_seconds": 1.0,
        "contact_iou": 0.20,
        "min_contact_frames": 4,
        "moving_speed": 0.022,
        "stopped_speed": 0.006,
        "min_post_slow_seconds": 0.60,
        "max_collapse_delay_seconds": 0.60,
        "settle_seconds": 6.0,
    },
    # Default. Motion evidence can confirm an impact the detector misses at the
    # exact contact frame, but only with a genuine speed collapse that persists.
    "balanced": {
        "require_model": False,
        "model_min_conf": 0.35,
        "model_min_frames": 3,
        "model_window_seconds": 1.0,
        "contact_iou": 0.15,
        "min_contact_frames": 3,
        "moving_speed": 0.018,
        "stopped_speed": 0.007,
        "min_post_slow_seconds": 0.50,
        "max_collapse_delay_seconds": 0.80,
        "settle_seconds": 5.0,
    },
    # Catches more, at the cost of more false alarms. Intended for reviewing
    # footage where a missed incident matters more than a wasted look.
    "sensitive": {
        "require_model": False,
        "model_min_conf": 0.28,
        "model_min_frames": 2,
        "model_window_seconds": 1.2,
        "contact_iou": 0.10,
        "min_contact_frames": 2,
        "moving_speed": 0.014,
        "stopped_speed": 0.009,
        "min_post_slow_seconds": 0.30,
        "max_collapse_delay_seconds": 1.20,
        "settle_seconds": 4.0,
    },
}

# Two detections count as the same incident when their centres are within this
# fraction of the frame diagonal. Prevents one crash in the middle of the frame
# from being logged as several separate events.
ACCIDENT_SAME_PLACE_RADIUS = 0.12

# Two vehicles count as "in contact" when the gap between their box edges is
# within this multiple of the smaller vehicle's box size. Edge gap is used
# rather than centre distance because cars touching bumper to bumper have their
# centres a full car length apart and a box IoU near zero - the exact case that
# most needs to be caught. Contact alone never raises an event; see accidents.py.
ACCIDENT_CONTACT_GAP_FACTOR = 0.25

# --------------------------------------------------------------------------- #
# Red-light stop-line auto-calibration
#
# Enforcement needs stop-line geometry. Requiring the operator to hand-draw it
# made the feature inert in practice: nothing supplied a line, so violation
# checking never armed and every run honestly reported zero violations. The
# trained traffic-light checkpoint detects the stop_line class, so the geometry
# is measured from the footage instead. See StopLineEstimator in redlight.py.
#
# These are deliberately cautious. A misplaced line accuses real vehicles, so
# the estimator refuses to lock rather than guess.
# --------------------------------------------------------------------------- #

# Frames that must independently agree before geometry is trusted.
STOP_LINE_MIN_SAMPLES = 12
# A stop_line detection below this confidence is not used as a sample.
STOP_LINE_MIN_CONF = 0.35
# If samples disagree vertically by more than this fraction of frame height, the
# estimate is treated as unreliable and enforcement stays off.
STOP_LINE_MAX_SPREAD = 0.12
# The detected box rarely spans the full carriageway, so the locked line is
# extended by this fraction of frame width at each end.
STOP_LINE_MARGIN = 0.04

# Output codec. 'avc1' is real H.264 (verified avcC, Baseline profile), which is
# what an HTML5 <video> element can play. 'mp4v' is MPEG-4 Part 2 and will not
# play in Chrome/Edge/Firefox, so it is only a last-resort fallback.
BROWSER_CODECS = ["avc1", "H264"]
FALLBACK_CODECS = ["mp4v"]

# Number-plate OCR. PaddleOCR's oneDNN path raises NotImplementedError on this
# machine, so MKL-DNN is disabled explicitly.
OCR_ENABLE_MKLDNN = False
# OCR costs ~0.55s per call, so it is attempted a limited number of times per
# tracked vehicle rather than every frame.
OCR_MAX_ATTEMPTS_PER_VEHICLE = 2
OCR_MIN_CROP_PIXELS = 40
OCR_MIN_TEXT_SCORE = 0.60
