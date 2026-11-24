"""Engine-agnostic ASR interface.

Every engine takes raw 16 kHz mono float32 PCM and returns text plus the
language it detected. Keeping this narrow is what lets the WebSocket layer
stay identical whether transcription happens in faster-whisper locally, in
Gemini Live, or in the browser.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class ASRSegment:
    text: str
    language: str = "unknown"
    start_ms: int = 0
    end_ms: int = 0
    confidence: float | None = None


@dataclass
class ASRResult:
    text: str
    language: str = "unknown"
    segments: list[ASRSegment] = field(default_factory=list)


class ASREngine(Protocol):
    name: str

    @property
    def ready(self) -> bool: ...

    def describe(self) -> dict[str, object]: ...

    def transcribe(
        self, audio: np.ndarray, language: str | None = None
    ) -> ASRResult: ...


class EngineUnavailable(RuntimeError):
    """Raised when an engine is selected but its dependencies are missing."""
