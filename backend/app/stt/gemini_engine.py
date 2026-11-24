"""Gemini Live STT - the option named in the assignment brief.

Kept as an alternative to the default self-hosted Whisper engine. Audio is
sent as a single-turn realtime input per utterance rather than one long
bidirectional session, which maps cleanly onto the same chunk-and-finalise
loop the other engines use.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from app.config import settings
from app.stt.base import ASRResult, ASRSegment, EngineUnavailable

logger = logging.getLogger(__name__)


def _to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class GeminiLiveEngine:
    name = "gemini"

    def __init__(self) -> None:
        self._client = None
        self._error: str | None = None

    @property
    def ready(self) -> bool:
        return bool(settings.gemini_api_key)

    def describe(self) -> dict[str, object]:
        return {
            "model": settings.gemini_model,
            "loaded": self._client is not None,
            "error": self._error if self._error else None,
            "key_present": bool(settings.gemini_api_key),
        }

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not settings.gemini_api_key:
            raise EngineUnavailable(
                "GEMINI_API_KEY is not set; use STT_ENGINE=whisper instead."
            )
        try:
            from google import genai
        except ImportError as exc:
            self._error = str(exc)
            raise EngineUnavailable(
                "google-genai is not installed. Install with: pip install google-genai"
            ) from exc
        self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    async def _transcribe_async(
        self, audio: np.ndarray, language: str | None
    ) -> ASRResult:
        client = self._get_client()
        from google.genai import types

        hint = (
            "Transcribe the speech verbatim. It may mix Hindi and English; "
            "keep each word in the language it was spoken in. "
            "Reply with the transcript only."
        )
        if language and language not in ("auto", "mixed", "unknown"):
            hint += f" The primary language is '{language}'."

        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": hint,
        }

        chunks: list[str] = []
        async with client.aio.live.connect(
            model=settings.gemini_model, config=config
        ) as session:
            await session.send_realtime_input(
                audio=types.Blob(
                    data=_to_pcm16_bytes(audio),
                    mime_type=f"audio/pcm;rate={settings.sample_rate}",
                )
            )
            await session.send_realtime_input(audio_stream_end=True)
            async for response in session.receive():
                if response.text:
                    chunks.append(response.text)
                if getattr(response, "server_content", None) and getattr(
                    response.server_content, "turn_complete", False
                ):
                    break

        text = "".join(chunks).strip()
        duration_ms = int(len(audio) / settings.sample_rate * 1000)
        segments = (
            [
                ASRSegment(
                    text=text,
                    language=language or "mixed",
                    start_ms=0,
                    end_ms=duration_ms,
                )
            ]
            if text
            else []
        )
        return ASRResult(text=text, language=language or "mixed", segments=segments)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> ASRResult:
        # Called from a worker thread, so it owns its own event loop.
        try:
            return asyncio.run(self._transcribe_async(audio, language))
        except EngineUnavailable:
            raise
        except Exception as exc:  # network / quota / API shape changes
            logger.warning("gemini transcription failed: %s", exc)
            self._error = str(exc)
            return ASRResult(text="", language=language or "unknown")
