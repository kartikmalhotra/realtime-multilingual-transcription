"""Pass-through engine for browser-side recognition.

The Web Speech API already produces interim + final text in the browser, so
there is nothing to transcribe server-side; the WebSocket layer only persists
what the client sends. This exists so the hosted demo keeps working on free
tiers that cannot hold a Whisper model in memory.
"""
from __future__ import annotations

import numpy as np

from app.stt.base import ASRResult


class BrowserEngine:
    name = "browser"

    @property
    def ready(self) -> bool:
        return True

    def describe(self) -> dict[str, object]:
        return {"model": "Web Speech API (client-side)", "loaded": True}

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> ASRResult:
        # Audio never reaches the server in this mode.
        return ASRResult(text="", language=language or "unknown")
