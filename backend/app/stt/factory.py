"""Engine registry - one process holds at most one instance per engine."""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.stt.base import ASREngine
from app.stt.browser_engine import BrowserEngine
from app.stt.faster_whisper_engine import FasterWhisperEngine
from app.stt.gemini_engine import GeminiLiveEngine

AVAILABLE = ("whisper", "browser", "gemini")

_BUILDERS = {
    "whisper": FasterWhisperEngine,
    "browser": BrowserEngine,
    "gemini": GeminiLiveEngine,
}


@lru_cache(maxsize=len(AVAILABLE))
def get_engine(name: str | None = None) -> ASREngine:
    key = (name or settings.stt_engine).lower()
    if key not in _BUILDERS:
        key = settings.stt_engine
    return _BUILDERS[key]()
