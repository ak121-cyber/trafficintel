"""Upload validation and safe path handling.

Uploads are untrusted: the original filename is never used on disk. Each job
gets a generated id and a sanitised extension, so a hostile name like
"../../models/accident/best.pt" cannot escape the uploads directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2

# Container formats OpenCV/FFMPEG can decode on this setup.
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# 2 GB. test2.mp4 is ~207 MB, so this leaves generous headroom while still
# rejecting a runaway upload.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(ValueError):
    """Upload rejected. The message is safe to show to the user."""


def sanitize_filename(name: str) -> str:
    """Reduce an arbitrary client filename to a harmless display name."""
    name = Path(str(name or "")).name          # strip any directory component
    name = _SAFE_NAME.sub("_", name).strip("._")
    return name[:120] or "video"


def validate_extension(filename: str) -> str:
    """Return the lowercase extension, or raise if unsupported."""
    ext = Path(sanitize_filename(filename)).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".").upper() for e in ALLOWED_EXTENSIONS))
        raise UploadError(
            f"Unsupported file type '{ext or 'unknown'}'. Allowed formats: {allowed}."
        )
    return ext


def verify_readable_video(path: Path) -> dict:
    """Confirm the saved file actually decodes, and return its properties.

    A file can carry a valid extension and still be truncated or corrupt, so one
    frame is decoded before the job is accepted.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise UploadError(
            "The video could not be opened. It may be corrupt or use an unsupported codec."
        )
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise UploadError("The video contains no readable frames. It may be corrupt.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            raise UploadError(f"The video reports an invalid frame size ({width}x{height}).")
    finally:
        cap.release()

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3) if fps > 0 else None,
        "frames": frames if frames > 0 else None,
        "duration_seconds": round(frames / fps, 2) if fps > 0 and frames > 0 else None,
    }


def resolve_inside(base: Path, candidate: Path) -> Path:
    """Guard against path traversal when serving a stored file."""
    base = base.resolve()
    resolved = candidate.resolve()
    if base != resolved and base not in resolved.parents:
        raise UploadError("Requested file is outside the permitted directory.")
    return resolved
