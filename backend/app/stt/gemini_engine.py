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
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"api_version": settings.gemini_api_version},
        )
        return self._client

    async def _transcribe_async(
        self, audio: np.ndarray, language: str | None
    ) -> ASRResult:
        client = self._get_client()
        from google.genai import types

        # Live models reply with audio only, so the transcript is not the
        # model's answer: it is the server-side transcription of the audio we
        # send in (input_audio_transcription). The spoken reply is ignored.
        config = {
            "response_modalities": ["AUDIO"],
            "input_audio_transcription": {},
        }

        chunks: list[str] = []

        async def _run() -> None:
            async with client.aio.live.connect(
                model=settings.gemini_model, config=config
            ) as session:
                # The Live endpoint expects a stream, not one large blob:
                # sending the whole utterance at once yields no transcription
                # at all. Feed it in realtime-sized chunks instead.
                data = _to_pcm16_bytes(audio)
                chunk = settings.sample_rate * 2 // 10  # 100 ms of int16 PCM
                mime = f"audio/pcm;rate={settings.sample_rate}"
                for offset in range(0, len(data), chunk):
                    await session.send_realtime_input(
                        audio=types.Blob(data=data[offset : offset + chunk], mime_type=mime)
                    )
                await session.send_realtime_input(audio_stream_end=True)
                # `turn_complete` does not arrive while the model is still
                # speaking its reply, so waiting for it hangs until the
                # timeout. Stop once the transcript has stopped growing.
                quiet = 0
                async for response in session.receive():
                    content = getattr(response, "server_content", None)
                    if content is None:
                        continue
                    transcription = getattr(content, "input_transcription", None)
                    if transcription is not None and transcription.text:
                        chunks.append(transcription.text)
                        quiet = 0
                    elif chunks:
                        quiet += 1
                        if quiet >= settings.gemini_quiet_events:
                            break
                    if getattr(content, "turn_complete", False):
                        break

        await asyncio.wait_for(_run(), timeout=settings.gemini_timeout_s)

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
