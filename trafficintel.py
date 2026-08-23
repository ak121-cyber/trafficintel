from collections import deque
from pathlib import Path
import json
import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import (
    YOLO_MODEL,
    ACCIDENT_MODEL,
    TRAFFIC_LIGHT_MODEL,
    ACCIDENT_NAMES,
    TRAFFIC_LIGHT_NAMES,
    VEHICLE_CLASSES,
    IMG_SIZE,
    VEHICLE_CONF,
    TRAFFIC_LIGHT_CONF,
    ACCIDENT_DETECT_CONF,
    ACCIDENT_SENSITIVITY,
    ACCIDENT_PRESETS,
    ACCIDENT_SAME_PLACE_RADIUS,
    ACCIDENT_CONTACT_GAP_FACTOR,
    STOP_LINE_MIN_SAMPLES,
    STOP_LINE_MIN_CONF,
    STOP_LINE_MAX_SPREAD,
    STOP_LINE_MARGIN,
    BROWSER_CODECS,
    FALLBACK_CODECS,
    OCR_ENABLE_MKLDNN,
    OCR_MAX_ATTEMPTS_PER_VEHICLE,
    OCR_MIN_CROP_PIXELS,
    OCR_MIN_TEXT_SCORE,
)
from accidents import AccidentMonitor
from redlight import (
    LightState,
    RedLightMonitor,
    StopLineEstimator,
    STOP_LINE_CLASS,
    LIGHT_STATE_CLASSES,
    parse_stop_lines,
)

TRACKER_FILE = str(Path(__file__).resolve().parent / "bytetrack_custom.yaml")
LANES = 3


# Defined in errors.py so the web layer can catch them without importing torch.
from errors import ModelLoadError, ProcessingCancelled  # noqa: F401
def open_writer(output_path, fps, size):
    """Open a VideoWriter, preferring a browser-playable H.264 stream.

    cv2.VideoWriter.isOpened() returns True even for a fourcc the muxer rejects,
    so the codec actually used is confirmed afterwards by inspecting the written
    container. Returns (writer, fourcc_used, browser_playable).
    """
    for codec in list(BROWSER_CODECS) + list(FALLBACK_CODECS):
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec, codec in BROWSER_CODECS
        writer.release()
    raise RuntimeError(
        "Could not open a video writer for any supported codec "
        f"({', '.join(BROWSER_CODECS + FALLBACK_CODECS)})."
    )


def verify_browser_codec(path):
    """Inspect an MP4 for an H.264 sample-entry box.

    'avcC' means the stream is H.264 and an HTML5 <video> can decode it; 'esds'
    indicates MPEG-4 Part 2, which browsers cannot play.
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        return False
    return b"avcC" in data


class Track:
    """One tracked vehicle, with both pixel and scale-normalised speed history.

    `norm_speeds` is in units of the vehicle's own box diagonal per frame. That
    normalisation matters: a raw pixel speed of 6 px/frame is fast for a distant
    vehicle and slow for one filling the frame, so a single pixel threshold
    cannot work across a perspective view or across resolutions. Accident logic
    reads `norm_speeds`; `speeds` is kept for display and compatibility.
    """

    def __init__(self, tid):
        self.id = int(tid)
        self.box = None
        self.cls = 0
        self.conf = 0.0
        self.centers = deque(maxlen=20)
        self.speeds = deque(maxlen=20)
        self.norm_speeds = deque(maxlen=20)

    def update(self, box, cls, conf):
        b = np.asarray(box, dtype=float)
        c = ((b[0]+b[2])/2, (b[1]+b[3])/2)
        speed = math.dist(c, self.centers[-1]) if self.centers else 0.0
        diag = math.hypot(max(0.0, b[2]-b[0]), max(0.0, b[3]-b[1])) or 1.0
        self.box = b
        self.cls = int(cls)
        self.conf = float(conf)
        self.centers.append(c)
        self.speeds.append(speed)
        self.norm_speeds.append(speed / diag)

    def speed_drop(self):
        """Fractional drop in recent pixel speed, 0..1.

        Superseded for accident logic by the normalised-speed test in
        accidents.py, which is scale-invariant. Kept because it is a reasonable
        general-purpose signal and is cheap.
        """
        if len(self.speeds) < 6:
            return 0.0
        old = float(np.median(list(self.speeds)[-6:-3]))
        new = float(np.median(list(self.speeds)[-3:]))
        return max(0.0, min(1.0, (old-new)/max(old, 1e-6)))

# Geometry helpers now live in accidents.py, which is the only consumer. They are
# re-exported here so existing imports of trafficintel.iou keep working.
from accidents import box_center, box_diagonal, box_gap, center_distance, iou  # noqa: E402,F401

def draw_label(frame, text, x, y, fg=(245,250,250), bg=(10,17,23), scale=.44):
    font=cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),_=cv2.getTextSize(text,font,scale,1)
    y=max(th+8,y)
    cv2.rectangle(frame,(x,y-th-7),(x+tw+10,y+2),bg,-1)
    cv2.putText(frame,text,(x+5,y-4),font,scale,fg,1,cv2.LINE_AA)

def panel(frame,title,lines,x,y,w,accent):
    h=40+23*len(lines)
    ov=frame.copy()
    cv2.rectangle(ov,(x,y),(x+w,y+h),(10,17,23),-1)
    cv2.addWeighted(ov,.9,frame,.1,0,frame)
    cv2.rectangle(frame,(x,y),(x+4,y+h),accent,-1)
    cv2.putText(frame,title,(x+15,y+27),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),2,cv2.LINE_AA)
    yy=y+50
    for line in lines:
        cv2.putText(frame,line,(x+15,yy),cv2.FONT_HERSHEY_SIMPLEX,.42,(205,215,220),1,cv2.LINE_AA)
        yy+=23

class TrafficIntel:
    """Single-pass traffic video analysis engine.

    One read of the video drives every enabled module inside one frame loop:
    vehicle detection + ByteTrack, then the accident model, the traffic-light
    model and OCR as enabled, then event logic and annotation. Modules that are
    switched off are never loaded, which keeps VRAM free on small GPUs.
    """

    def __init__(self, traffic_light=False, plate=False, accident=True, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = 0 if torch.cuda.is_available() else "cpu"

        self.load_errors = []
        self.vehicle = YOLO(YOLO_MODEL)

        # Accident detection stays on by default so the existing CLI behaviour
        # is unchanged; the web UI can switch it off to skip the model entirely.
        self.accident = None
        if accident:
            if ACCIDENT_MODEL.exists():
                self.accident = YOLO(str(ACCIDENT_MODEL))
            else:
                self.load_errors.append(f"Accident model not found at {ACCIDENT_MODEL}")

        self.light = None
        if traffic_light:
            if TRAFFIC_LIGHT_MODEL.exists():
                self.light = YOLO(str(TRAFFIC_LIGHT_MODEL))
            else:
                self.load_errors.append(f"Traffic-light model not found at {TRAFFIC_LIGHT_MODEL}")

        self.ocr = None
        self.ocr_error = None
        if plate:
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=OCR_ENABLE_MKLDNN,
                )
            except Exception as exc:
                # OCR is optional; the rest of the pipeline still runs without it.
                self.ocr_error = f"{type(exc).__name__}: {exc}"
                self.load_errors.append(f"Number-plate OCR unavailable ({type(exc).__name__})")

    def _read_plate(self, crop):
        """Best-effort plate text from a vehicle crop.

        Returns (text, score) or None. PaddleOCR's result object differs between
        versions, so both the dict and the .json shapes are handled.
        """
        try:
            results = self.ocr.predict(crop)
        except Exception:
            return None

        best = None
        for res in results or []:
            if isinstance(res, dict):
                payload = res
            else:
                payload = getattr(res, "json", {}) or {}
                payload = payload.get("res", payload)
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            for text, score in zip(texts, scores):
                cleaned = "".join(ch for ch in str(text) if ch.isalnum()).upper()
                score = float(score)
                if len(cleaned) < 4 or score < OCR_MIN_TEXT_SCORE:
                    continue
                if best is None or score > best[1]:
                    best = (cleaned, score)
        return best

    def run(self, video_path, output_path, traffic_light=False, plate=False,
            stop_lines=None, strict_accidents=False, sensitivity=None,
            auto_calibrate=True, progress=None, should_cancel=None):
        """Process one video and write one annotated video.

        stop_lines        - normalised red-light stop-line geometry. When omitted,
                            it is measured from the model's stop_line detections
                            (see auto_calibrate). Supplied geometry always wins.
        strict_accidents  - require model confirmation, so motion evidence alone
                            cannot raise an event. Equivalent to forcing
                            require_model on whichever preset is in use.
        sensitivity       - accident preset name: "strict", "balanced" or
                            "sensitive". Defaults to config.ACCIDENT_SENSITIVITY.
        auto_calibrate    - allow the stop line to be derived from the footage
                            when no geometry was supplied. Set False to require
                            explicit calibration and leave enforcement inert.
        progress          - callback(frame_no, total_frames, stats_snapshot).
        should_cancel     - callable returning True to abort the run.
        """
        accident_enabled = self.accident is not None
        traffic_light = bool(traffic_light) and self.light is not None
        plate = bool(plate) and self.ocr is not None

        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            cap.release()
            raise RuntimeError(f"Video reports an invalid frame size ({w}x{h}): {video_path}")

        # CAP_PROP_FRAME_COUNT is unreliable for some containers; treat a
        # non-positive value as "unknown" rather than reporting a bogus percent.
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            total_frames = 0

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer, codec_used, browser_playable = open_writer(output_path, fps, (w, h))

        manual_lines = parse_stop_lines(stop_lines)
        monitor = RedLightMonitor(
            manual_lines,
            fps,
            light_state=LightState(fps, min_conf=TRAFFIC_LIGHT_CONF),
        ) if traffic_light else None

        # Without geometry, violation checking cannot run at all. Rather than
        # demand hand-calibration (which left the feature permanently inert), the
        # stop line is measured from the model's own stop_line detections.
        stop_line_estimator = None
        if traffic_light and not manual_lines and auto_calibrate:
            stop_line_estimator = StopLineEstimator(
                min_samples=STOP_LINE_MIN_SAMPLES,
                min_conf=STOP_LINE_MIN_CONF,
                max_spread=STOP_LINE_MAX_SPREAD,
                margin=STOP_LINE_MARGIN,
            )

        plates={}
        ocr_attempts={}

        tracks={}
        unique_ids=set()
        lane_ids=[set() for _ in range(LANES)]
        frame_no=0
        cancelled=False

        # Accident-event confirmation. See accidents.py: an incident has identity
        # and a lifetime, so a wreck sitting in frame updates one event instead
        # of emitting a new one every couple of seconds.
        preset_name = sensitivity or ACCIDENT_SENSITIVITY
        if preset_name not in ACCIDENT_PRESETS:
            raise ValueError(
                f"Unknown accident sensitivity {preset_name!r}. "
                f"Expected one of: {', '.join(sorted(ACCIDENT_PRESETS))}."
            )
        accident_monitor = AccidentMonitor(
            fps,
            preset=ACCIDENT_PRESETS[preset_name],
            require_model=True if strict_accidents else None,
            same_place_radius=ACCIDENT_SAME_PLACE_RADIUS,
            contact_gap_factor=ACCIDENT_CONTACT_GAP_FACTOR,
            frame_size=(w, h),
        ) if accident_enabled else None

        try:
            while True:
                ok,frame=cap.read()
                if not ok:
                    break
                frame_no+=1

                if should_cancel is not None and frame_no % 10 == 0 and should_cancel():
                    cancelled=True
                    break

                current=[]
                vr=self.vehicle.track(
                    frame,
                    persist=True,
                    tracker=TRACKER_FILE,
                    classes=list(VEHICLE_CLASSES.keys()),
                    imgsz=IMG_SIZE,
                    conf=VEHICLE_CONF,
                    device=self.device,
                    verbose=False
                )[0]

                if vr.boxes is not None and vr.boxes.id is not None:
                    boxes=vr.boxes.xyxy.cpu().numpy()
                    ids=vr.boxes.id.cpu().numpy().astype(int)
                    classes=vr.boxes.cls.cpu().numpy().astype(int)
                    confs=vr.boxes.conf.cpu().numpy()

                    for box,tid,cls,conf in zip(boxes,ids,classes,confs):
                        tid=int(tid)
                        t=tracks.setdefault(tid,Track(tid))
                        t.update(box,cls,conf)
                        current.append(t)
                        unique_ids.add(tid)

                        lane=max(1,min(LANES,int(((box[0]+box[2])/2)/max(w,1)*LANES)+1))
                        lane_ids[lane-1].add(tid)

                        x1,y1,x2,y2=map(int,box)
                        name=VEHICLE_CLASSES.get(int(cls),"vehicle").upper()
                        draw_label(frame,f"{name} #{tid} {float(conf):.0%}",x1,max(25,y1))
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(55,205,155),2)

                        pts=list(t.centers)
                        for a,b in zip(pts[:-1],pts[1:]):
                            cv2.line(frame,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),(55,205,155),2,cv2.LINE_AA)

                # Accident detector. Detections are gathered as *candidate
                # evidence*; whether any of them constitutes an event is decided
                # by AccidentMonitor, not by this frame in isolation.
                accident_dets=[]
                if accident_enabled:
                    ar=self.accident.predict(
                        frame,
                        imgsz=IMG_SIZE,
                        conf=ACCIDENT_DETECT_CONF,
                        device=self.device,
                        verbose=False
                    )[0]

                    if ar.boxes is not None:
                        for box,cls,conf in zip(
                            ar.boxes.xyxy.cpu().numpy(),
                            ar.boxes.cls.cpu().numpy().astype(int),
                            ar.boxes.conf.cpu().numpy()
                        ):
                            cls=int(cls); conf=float(conf)
                            if cls==0:                      # "No Accident"
                                continue
                            name=ACCIDENT_NAMES[cls] if cls<len(ACCIDENT_NAMES) else f"Accident {cls}"
                            accident_dets.append((box,name.upper(),conf))

                            x1,y1,x2,y2=map(int,box)
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(40,90,255),2)
                            draw_label(frame,f"{name.upper()} {conf:.0%}",x1,max(25,y1),
                                       fg=(255,245,245),bg=(35,10,16),scale=.48)

                if accident_monitor is not None:
                    accident_monitor.observe(frame_no, current, accident_dets)

                if (accident_monitor is not None
                        and frame_no <= accident_monitor.alert_until
                        and accident_monitor.incidents):
                    e=accident_monitor.latest().as_dict()
                    vids=", ".join(f"#{x}" for x in e["vehicles"]) if e["vehicles"] else "Model detected"
                    # Motion-only incidents have no model confidence. Saying so
                    # is the point: the old build printed a fabricated 82%.
                    conf_text=(f"{e['confidence']:.0%} model"
                               if e["confidence"] is not None else "motion evidence")
                    panel(
                        frame,
                        "ACCIDENT ALERT",
                        [
                            f"{e['type']}   {conf_text}",
                            f"Vehicles   {vids}",
                            f"Event      {e['time_seconds']:.1f}s",
                        ],
                        16,62,370,(40,90,255)
                    )

                if traffic_light and self.light is not None:
                    lr=self.light.predict(frame,imgsz=IMG_SIZE,conf=TRAFFIC_LIGHT_CONF,device=self.device,verbose=False)[0]
                    light_dets=[]
                    if lr.boxes is not None:
                        for box,cls,conf in zip(
                            lr.boxes.xyxy.cpu().numpy(),
                            lr.boxes.cls.cpu().numpy().astype(int),
                            lr.boxes.conf.cpu().numpy()
                        ):
                            cls=int(cls); conf=float(conf)
                            light_dets.append((cls,conf))

                            # The model also reports 'car'/'motobike'; vehicles are
                            # already drawn from the tracked detector, so only the
                            # signal and stop-line classes are annotated here.
                            if cls not in LIGHT_STATE_CLASSES and cls != STOP_LINE_CLASS:
                                continue

                            if cls == STOP_LINE_CLASS and stop_line_estimator is not None:
                                # Accumulate geometry evidence. The estimator locks
                                # only after enough frames agree, and enforcement
                                # arms from that frame onward - never retroactively,
                                # since crossings before the lock were not measured.
                                locked=stop_line_estimator.observe(box, conf, w, h, frame_no)
                                if locked is not None:
                                    monitor.adopt_stop_lines([locked], frame_no, source="auto")

                            x1,y1,x2,y2=map(int,box)
                            name=TRAFFIC_LIGHT_NAMES[cls] if cls<len(TRAFFIC_LIGHT_NAMES) else f"class_{cls}"
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,190,255),2)
                            draw_label(frame,f"{name.replace('_',' ').upper()} {conf:.0%}",x1,max(25,y1),
                                       fg=(255,245,220),bg=(35,25,8))

                    monitor.observe_lights(light_dets, frame_no)

                    # Violation = RED + tracked vehicle + stop-line crossing.
                    fresh=monitor.update(
                        current, frame_no, w, h,
                        lane_of=lambda t: max(1,min(LANES,int(((t.box[0]+t.box[2])/2)/max(w,1)*LANES)+1)),
                    )

                    # Stop-line geometry, drawn so the calibration is visible.
                    for line in monitor.stop_lines:
                        (lx1,ly1),(lx2,ly2)=line.pixels(w,h)
                        col=(40,60,235) if monitor.light.is_red else (150,160,170)
                        cv2.line(frame,(lx1,ly1),(lx2,ly2),col,2,cv2.LINE_AA)
                        draw_label(frame,f"STOP LINE {line.name.upper()}",lx1,max(25,ly1-6),
                                   fg=(235,240,245),bg=(28,30,40),scale=.40)

                    # Current signal state, held across detection gaps.
                    sx=max(16,w-225)
                    cv2.rectangle(frame,(sx,h-40),(sx+205,h-12),(10,17,23),-1)
                    cv2.circle(frame,(sx+16,h-26),7,monitor.light.color(),-1)
                    cv2.putText(frame,monitor.light.label(),(sx+30,h-21),
                                cv2.FONT_HERSHEY_SIMPLEX,.42,(225,232,238),1,cv2.LINE_AA)

                    if fresh:
                        violation_alert_until=frame_no+max(1,int(fps*2.5))
                    else:
                        violation_alert_until=monitor.alert_until

                    if frame_no<=violation_alert_until and monitor.violations:
                        v=monitor.violations[-1]
                        panel(
                            frame,
                            "RED-LIGHT VIOLATION",
                            [
                                f"Vehicle    #{v['vehicle_id']}",
                                f"Stop line  {v['stop_line']}",
                                f"Time       {v['time_seconds']:.1f}s",
                            ],
                            16,h-150,330,(40,60,235)
                        )

                    # Highlight vehicles that have been latched as violators.
                    flagged=monitor.violating_ids()
                    for t in current:
                        if t.id in flagged:
                            x1,y1,x2,y2=map(int,t.box)
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(40,60,235),2)

                if plate and self.ocr is not None:
                    # OCR costs ~0.55s per call, so each vehicle is attempted only a
                    # couple of times and only when the crop is large enough to read.
                    for t in current:
                        if t.id in plates or ocr_attempts.get(t.id,0) >= OCR_MAX_ATTEMPTS_PER_VEHICLE:
                            continue
                        x1,y1,x2,y2=map(int,t.box)
                        x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
                        if (x2-x1) < OCR_MIN_CROP_PIXELS or (y2-y1) < OCR_MIN_CROP_PIXELS:
                            continue
                        crop=frame[y1:y2,x1:x2]
                        if not crop.size:
                            continue
                        ocr_attempts[t.id]=ocr_attempts.get(t.id,0)+1
                        found=self._read_plate(crop)
                        if found:
                            plates[t.id]={"text":found[0],"score":round(found[1],4),"frame":frame_no}

                    # Show a recognised plate next to its vehicle.
                    for t in current:
                        hit=plates.get(t.id)
                        if hit:
                            x1,y1,x2,y2=map(int,t.box)
                            draw_label(frame,f"PLATE {hit['text']}",x1,min(h-6,y2+20),
                                       fg=(230,255,235),bg=(12,40,22),scale=.42)

                # Three equal visual zones for lane analytics.
                for i in range(1,LANES):
                    x=int(w*i/LANES)
                    cv2.line(frame,(x,48),(x,h),(80,90,105),1,cv2.LINE_AA)

                count=len(current)
                density="HEAVY" if count>=16 else "MEDIUM" if count>=8 else "LOW"
                cv2.rectangle(frame,(0,0),(w,48),(10,17,23),-1)
                cv2.putText(frame,"TRAFFICINTEL",(14,31),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(frame,f"VEHICLES {count}",(190,30),cv2.FONT_HERSHEY_SIMPLEX,.43,(200,210,215),1,cv2.LINE_AA)
                cv2.putText(frame,f"DENSITY {density}",(300,30),cv2.FONT_HERSHEY_SIMPLEX,.43,(200,210,215),1,cv2.LINE_AA)
                if accident_enabled:
                    cv2.putText(frame,f"ACCIDENTS {accident_monitor.count}",(430,30),cv2.FONT_HERSHEY_SIMPLEX,.43,(200,210,215),1,cv2.LINE_AA)
                if monitor is not None and monitor.enabled:
                    cv2.putText(frame,f"VIOLATIONS {monitor.count}",(560,30),cv2.FONT_HERSHEY_SIMPLEX,.43,(200,210,215),1,cv2.LINE_AA)

                lane_lines=[f"Lane {i+1}: {len(lane_ids[i])} unique" for i in range(LANES)]
                panel(frame,"LANE ANALYTICS",lane_lines,max(16,w-225),62,210,(90,150,255))

                accident_total = accident_monitor.count if accident_monitor is not None else 0

                if frame_no%max(1,int(fps))==0:
                    if progress is None:
                        print(f"\rProcessing {frame_no/fps:7.1f}s | vehicles {count:2d} | accidents {accident_total:2d}",end="",flush=True)
                    else:
                        # Driven by the backend: report structured progress instead of
                        # writing carriage-return lines into the server log.
                        progress({
                            "frame":frame_no,
                            "total_frames":total_frames,
                            "seconds":round(frame_no/fps,2),
                            "vehicles":count,
                            "accidents":accident_total,
                            "violations":monitor.count if monitor is not None else 0,
                        })

                writer.write(frame)

        except torch.cuda.OutOfMemoryError as exc:
            # Surface a readable message instead of a CUDA traceback. 4 GB cards
            # can run out when several models are enabled on a large video.
            torch.cuda.empty_cache()
            raise RuntimeError(
                "GPU ran out of memory. Disable a module, use a smaller video, "
                "or set TRAFFICINTEL_DEVICE=cpu to run on CPU."
            ) from exc
        finally:
            # Always release the capture and the writer, otherwise a failed run
            # leaves a locked, zero-byte MP4 behind on Windows.
            cap.release()
            writer.release()

        if progress is None:
            print()

        if cancelled:
            raise ProcessingCancelled("Processing was cancelled.")

        if frame_no == 0:
            raise RuntimeError(
                "No frames could be decoded from the video. The file may be corrupt "
                "or use an unsupported codec."
            )

        lane_counts=[len(x) for x in lane_ids]
        violations=monitor.violations if monitor is not None else []

        accident_events = accident_monitor.events if accident_monitor is not None else []

        stats={
            "frames":frame_no,
            "fps":fps,
            "duration_seconds":round(frame_no/fps,2),
            "unique_tracked_vehicles":len(unique_ids),
            "lane_unique_counts":lane_counts,
            "accident_count":len(accident_events),
            "accident_events":accident_events,
            "traffic_light_enabled":bool(traffic_light),
            "number_plate_ocr_enabled":bool(plate),
            "tracker":"ByteTrack",

            # Added for the web UI. Existing keys above are unchanged so the CLI
            # and any prior consumer keep working.
            "accident_detection_enabled":bool(accident_enabled),
            "resolution":{"width":w,"height":h},
            "lane_counts":{f"lane_{i+1}":lane_counts[i] for i in range(LANES)},
            "red_light_violation_count":len(violations),
            "red_light_violations":violations,
            "red_light_enforcement_active":bool(monitor is not None and monitor.enabled),
            "stop_lines":[l.as_dict() for l in monitor.stop_lines] if monitor is not None else [],
            # How the geometry was obtained, and from which frame enforcement was
            # actually armed. A zero violation count means something different
            # when enforcement never armed than when it ran over the whole clip,
            # and the UI needs to be able to tell the difference.
            "stop_line_source":monitor.source if monitor is not None else "none",
            "enforcement_active_from_frame":(
                monitor.active_from_frame if monitor is not None else None),
            "enforced_frames":(
                max(0, frame_no - monitor.active_from_frame)
                if monitor is not None and monitor.active_from_frame is not None else 0),
            "stop_line_calibration":(
                stop_line_estimator.report() if stop_line_estimator is not None else None),
            "number_plates":[{"vehicle_id":k,**v} for k,v in sorted(plates.items())],
            "number_plate_count":len(plates),
            "strict_accidents":bool(strict_accidents),
            # Recorded so a stored result can be re-read later and understood:
            # an event count is only meaningful alongside the thresholds used.
            "accident_sensitivity":preset_name,
            "accident_model_confirmed_count":sum(
                1 for e in accident_events if e.get("model_confirmed")),
            "output_codec":codec_used,
            "browser_playable":bool(browser_playable and verify_browser_codec(output_path)),
            "device":"cuda" if self.device == 0 else str(self.device),
        }

        Path(output_path).with_suffix(".json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
        return stats
