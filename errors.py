"""Exceptions shared by the engine and the web layer.

These live in their own module so the backend can distinguish a cancellation
from a genuine failure without importing trafficintel.py, which pulls in torch
and ultralytics. Importing several hundred megabytes of CUDA runtime just to
name an exception class would be a poor trade.

trafficintel re-exports both names, so existing imports keep working.
"""


class ProcessingCancelled(RuntimeError):
    """Raised when a caller cancels an in-flight run.

    Deliberately distinct from a failure: the job did what it was told. The web
    layer must report CANCELLED, not FAILED - a broad `except Exception` that
    swallows this reports a user-requested stop as a crash.
    """


class ModelLoadError(RuntimeError):
    """A required model could not be loaded."""
