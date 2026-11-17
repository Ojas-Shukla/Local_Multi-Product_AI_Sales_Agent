"""Local AI inference and host-awareness utilities for EDXSO workflows."""

from sellbird.local_ai.hardware import HardwareReport, detect_hardware
from sellbird.local_ai.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModelNotFound,
    OllamaUnavailable,
)

__all__ = [
    "HardwareReport",
    "OllamaClient",
    "OllamaError",
    "OllamaModelNotFound",
    "OllamaUnavailable",
    "detect_hardware",
]
