"""Batch perspective correction for architectural photography."""
from .config import Settings          # noqa: F401
from .pipeline import process, analyse, Result, OK, SKIPPED, ERROR   # noqa: F401

__version__ = "0.1.0"
__all__ = ["Settings", "process", "analyse", "Result", "OK", "SKIPPED", "ERROR"]
