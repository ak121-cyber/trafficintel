"""Bridge between the web API and the existing TrafficIntel engine.

The engine is imported and called directly - no subprocess, no shelling out to
run_all.py. One TrafficIntel instance is built per job with only the enabled
models loaded, then the same instance processes every frame of the video in a
single pass.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ACCIDENT_MODEL, TRAFFIC_LIGHT_MODEL          # noqa: E402
from trafficintel import ProcessingCancelled, TrafficIntel      # noqa: E402

log = logging.getLogger("trafficintel.engine")


def device_report() -> dict:
    """Describe the compute device without loading any model."""
    available = torch.cuda.is_available()
    info = {
        "cuda_available": available,
        "device": "cuda" if available else "cpu",
        "torch_version": torch.__version__,
    }
    if available:
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_total_mb"] = round(props.total_memory / (1024 * 1024))
        except Exception:                                        # noqa: BLE001
            pass
    return info


def model_report() -> dict:
    """Report which trained checkpoints are actually present on disk.

    Paths come from config.py, so replacing a checkpoint later needs no code
    change here.
    """
    return {
        "accident": {
            "path": str(ACCIDENT_MODEL.relative_to(ROOT)) if ACCIDENT_MODEL.is_relative_to(ROOT) else str(ACCIDENT_MODEL),
            "available": ACCIDENT_MODEL.exists(),
        },
        "traffic_light": {
            "path": str(TRAFFIC_LIGHT_MODEL.relative_to(ROOT)) if TRAFFIC_LIGHT_MODEL.is_relative_to(ROOT) else str(TRAFFIC_LIGHT_MODEL),
            "available": TRAFFIC_LIGHT_MODEL.exists(),
        },
    }


def ocr_available() -> bool:
    """True when PaddleOCR can be imported. Never imported unless asked for."""
    try:
        import paddleocr                                          # noqa: F401
    except Exception:                                             # noqa: BLE001
        return False
    return True


def run_job(job, on_progress=None) -> dict:
    """Execute one analysis job against the existing ML pipeline.

    Only the modules the user enabled are loaded, which keeps VRAM free on a
    4 GB card. Every enabled model then runs inside the engine's single frame
    loop over the one uploaded video.
    """
    options = job.options
    accident = bool(options.get("accident_detection", True))
    traffic_light = bool(options.get("traffic_light", False))
    plate = bool(options.get("number_plate", False))
    stop_lines = options.get("stop_lines") or None

    log.info(
        "Job %s | accident=%s traffic_light=%s ocr=%s",
        job.id,
        "enabled" if accident else "disabled",
        "enabled" if traffic_light else "disabled",
        "enabled" if plate else "disabled",
    )

    try:
        engine = TrafficIntel(
            traffic_light=traffic_light,
            plate=plate,
            accident=accident,
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU ran out of memory while loading the models. Disable a module "
            "or set TRAFFICINTEL_DEVICE=cpu to run on the CPU."
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"A required model file is missing: {exc}") from exc
    except Exception as exc:                                      # noqa: BLE001
        raise RuntimeError(f"Failed to load the analysis models: {exc}") from exc

    # A module the user asked for but that could not load is a warning, not a
    # silent no-op: the UI must not imply the module ran.
    for message in engine.load_errors:
        job.warnings.append(message)
        log.warning("Job %s | %s", job.id, message)

    if accident and engine.accident is None:
        raise RuntimeError(
            "Accident detection was enabled but the trained model could not be "
            f"loaded from {ACCIDENT_MODEL.name}."
        )
    if traffic_light and engine.light is None:
        raise RuntimeError(
            "Traffic-light detection was enabled but the trained model could not "
            "be found. Train it with train_traffic_light.py or place a checkpoint "
            f"at {TRAFFIC_LIGHT_MODEL}."
        )
    if plate and engine.ocr is None:
        # OCR is genuinely optional; the run continues without it.
        job.warnings.append(
            "Number-plate OCR could not start, so the rest of the analysis ran without it."
        )

    def progress(snapshot: dict) -> None:
        job.frame = snapshot.get("frame", job.frame)
        job.total_frames = snapshot.get("total_frames", job.total_frames) or job.total_frames
        job.live = snapshot
        job.stage = "Analyzing"
        if on_progress is not None:
            on_progress(snapshot)

    try:
        stats = engine.run(
            job.input_path,
            job.output_path,
            traffic_light=traffic_light,
            plate=plate,
            stop_lines=stop_lines,
            sensitivity=options.get("sensitivity") or None,
            strict_accidents=bool(options.get("strict_accidents")),
            progress=progress,
            should_cancel=lambda: job.cancelled,
        )
    except ProcessingCancelled:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU ran out of memory during processing. Try a shorter or smaller "
            "video, disable a module, or set TRAFFICINTEL_DEVICE=cpu."
        ) from exc
    finally:
        # Release VRAM between jobs so the next run starts from a clean slate.
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    job.stage = "Finalizing"

    if not job.output_path.exists() or job.output_path.stat().st_size == 0:
        raise RuntimeError(
            "Processing finished but no output video was written. Check the "
            "server log for video-writer errors."
        )

    # Echo the toggles back so the UI never shows statistics for a module that
    # did not run.
    stats["modules"] = {
        "accident_detection": bool(accident and engine_ran(stats, "accident")),
        "traffic_light": bool(traffic_light),
        "number_plate": bool(plate and stats.get("number_plate_ocr_enabled")),
    }
    stats["output_file"] = job.output_path.name
    stats["source_file"] = job.original_filename
    return stats


def engine_ran(stats: dict, module: str) -> bool:
    """Whether the engine reports a module as actually executed."""
    if module == "accident":
        return bool(stats.get("accident_detection_enabled", True))
    return True
