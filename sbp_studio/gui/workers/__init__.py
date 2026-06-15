"""GUI background-worker layer (QThread wrappers around the core task contract)."""
from .base import CoreWorker, Job

__all__ = ["CoreWorker", "Job"]
