"""Real-time transcription over a WebSocket.

Design note
-----------
Receiving audio and transcribing it are deliberately decoupled. The receive
loop only ever buffers frames and flips flags, so it can always drain the
socket and react to a control message immediately. A single worker task per
connection does the decoding.

That split matters because Whisper transcribes a *window*, so an interim
pass re-decodes the whole current utterance and gets more expensive as the
utterance grows. Transcribing inline from the receive loop therefore builds
a backlog: frames queue behind a decode, the next decode is bigger, and a
`stop` sent by the client waits behind all of it. With one worker that
always reads the newest buffer state, intermediate interim passes are simply
skipped instead of queued, and the interim interval adapts to however long
the last decode actually took.

Protocol
--------
client -> server
    {"type": "config", "language": "auto", "title": "..."}
    <binary>                      16 kHz mono little-endian int16 PCM frames
    {"type": "client_transcript", "text": "...", "is_final": true}
                                  browser-engine mode: recognition already
                                  happened client-side, just persist it
    {"type": "flush"}             end the current utterance now
    {"type": "stop"}              end utterance, save the record, return it

server -> client
    {"type": "ready",   "engine": ..., "model": ..., "sample_rate": ...}
    {"type": "partial", "text": ...}                     interim, replaces prior
    {"type": "final",   "segment": {...}, "text": ...}   committed utterance
    {"type": "saved",   "transcription": {...}}
    {"type": "info" | "error", "message": ...}
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import crud, schemas
from app.audio import UtteranceBuffer, pcm16_to_float32
from app.config import settings
from app.database import SessionLocal
from app.stt.base import ASRResult, EngineUnavailable
from app.stt.factory import get_engine

logger = logging.getLogger(__name__)
router = APIRouter()

# How long the client is given to be told its recording was stored.
SAVE_DRAIN_TIMEOUT_S = 60.0
IDLE_POLL_S = 0.03


class TranscriptionSession:
    """Per-connection state: the audio buffer and the segments committed so far."""

    def __init__(self, engine_name: str | None, language: str, title: str) -> None:
        self.engine = get_engine(engine_name)
        self.language = language or "auto"
        self.title = title
        self.buffer = UtteranceBuffer()
        self.segments: list[schemas.SegmentCreate] = []
        self.total_duration_ms = 0

        # Worker coordination.
        self.flush_requested = False
        self.closing = False
        self.drained = asyncio.Event()
        # Measured cost of the last decode, used to back off interim passes
        # on a machine that cannot keep up.
        self.last_decode_ms = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    def add_segment(self, seg: schemas.SegmentCreate) -> None:
        self.segments.append(seg)

    def partial_interval_ms(self) -> int:
        """Never ask for interim passes faster than the engine can deliver."""
        return max(settings.partial_interval_ms, int(self.last_decode_ms))

    async def run_engine(self, audio) -> ASRResult:
        started = time.monotonic()
        try:
            return await asyncio.to_thread(self.engine.transcribe, audio, self.language)
        finally:
            self.last_decode_ms = (time.monotonic() - started) * 1000

    def save(self) -> schemas.TranscriptionOut:
        db = SessionLocal()
        try:
            record = crud.create_transcription(
                db,
                schemas.TranscriptionCreate(
                    title=self.title or _auto_title(self.text),
                    text=self.text,
                    language=self.language,
                    engine=self.engine.name,
                    duration_ms=self.total_duration_ms,
                    segments=self.segments,
                ),
            )
            return schemas.TranscriptionOut.model_validate(record)
        finally:
            db.close()


def _auto_title(text: str) -> str:
    snippet = " ".join(text.split()[:8])
    return (snippet[:80] or "Untitled") if snippet else "Untitled"


async def _send(ws: WebSocket, payload: dict) -> None:
    with contextlib.suppress(Exception):
        await ws.send_text(json.dumps(payload, default=str))


@router.websocket("/ws/transcribe")
async def transcribe_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    params = websocket.query_params
    session = TranscriptionSession(
        engine_name=params.get("engine"),
        language=params.get("language", "auto"),
        title=params.get("title", ""),
    )

    await _send(
        websocket,
        {
            "type": "ready",
            "engine": session.engine.name,
            "sample_rate": settings.sample_rate,
            **session.engine.describe(),
        },
    )

    async def finalize() -> None:
        """Transcribe the buffered utterance, emit it, and keep it for saving."""
        if session.buffer.is_empty:
            session.buffer.reset()
            return

        audio = session.buffer.audio()
        offset = session.buffer.session_offset_ms
        duration = session.buffer.duration_ms
        # Reset before decoding so incoming frames land in the next utterance.
        session.buffer.reset()
        session.total_duration_ms += duration

        try:
            result = await session.run_engine(audio)
        except EngineUnavailable as exc:
            await _send(websocket, {"type": "error", "message": str(exc)})
            return
        except Exception as exc:
            logger.exception("transcription failed")
            await _send(
                websocket, {"type": "error", "message": f"transcription failed: {exc}"}
            )
            return

        if not result.text:
            return

        for seg in result.segments or []:
            session.add_segment(
                schemas.SegmentCreate(
                    text=seg.text,
                    language=seg.language,
                    start_ms=offset + seg.start_ms,
                    end_ms=offset + seg.end_ms,
                    confidence=seg.confidence,
                )
            )

        await _send(
            websocket,
            {
                "type": "final",
                "segment": {
                    "text": result.text,
                    "language": result.language,
                    "start_ms": offset,
                    "end_ms": offset + duration,
                },
                "text": session.text,
            },
        )

    async def emit_partial() -> None:
        """Best-effort interim result over the utterance so far."""
        session.buffer.mark_partial_emitted()
        audio = session.buffer.audio()
        if audio.size == 0:
            return
        try:
            result = await session.run_engine(audio)
        except EngineUnavailable as exc:
            await _send(websocket, {"type": "error", "message": str(exc)})
            return
        except Exception as exc:
            logger.warning("partial transcription failed: %s", exc)
            return
        if result.text:
            await _send(
                websocket,
                {"type": "partial", "text": result.text, "language": result.language},
            )

    async def worker() -> None:
        """Single decoder. Always acts on the newest buffer state."""
        try:
            while True:
                if session.flush_requested or session.buffer.should_finalize():
                    session.flush_requested = False
                    await finalize()
                    continue

                if session.closing:
                    if session.buffer.has_speech:
                        await finalize()
                        continue
                    return

                if session.buffer.should_emit_partial(session.partial_interval_ms()):
                    await emit_partial()
                    continue

                await asyncio.sleep(IDLE_POLL_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("decoder worker failed")
        finally:
            session.drained.set()

    worker_task = asyncio.create_task(worker())

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # --- audio frames: buffer only, never decode here -----------
            if (raw := message.get("bytes")) is not None:
                if session.engine.name == "browser":
                    continue  # recognition happens client-side in this mode
                session.buffer.feed(pcm16_to_float32(raw))
                continue

            # --- control messages ---------------------------------------
            text = message.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await _send(websocket, {"type": "error", "message": "invalid JSON"})
                continue

            kind = payload.get("type")

            if kind == "config":
                session.language = payload.get("language", session.language)
                session.title = payload.get("title", session.title)
                await _send(websocket, {"type": "info", "message": "config applied"})

            elif kind == "client_transcript":
                # Browser-engine mode: trust the client's recognised text.
                body = (payload.get("text") or "").strip()
                if not body:
                    continue
                if payload.get("is_final"):
                    start = session.total_duration_ms
                    end = start + int(payload.get("duration_ms") or 0)
                    session.total_duration_ms = end
                    language = payload.get("language") or session.language
                    session.add_segment(
                        schemas.SegmentCreate(
                            text=body, language=language, start_ms=start, end_ms=end
                        )
                    )
                    await _send(
                        websocket,
                        {
                            "type": "final",
                            "segment": {
                                "text": body,
                                "language": language,
                                "start_ms": start,
                                "end_ms": end,
                            },
                            "text": session.text,
                        },
                    )
                else:
                    await _send(websocket, {"type": "partial", "text": body})

            elif kind == "flush":
                session.flush_requested = True

            elif kind == "stop":
                # Let the worker finish the current utterance, then persist.
                session.closing = True
                session.flush_requested = True
                try:
                    await asyncio.wait_for(
                        session.drained.wait(), timeout=SAVE_DRAIN_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    logger.warning("decoder did not drain before the save deadline")

                if session.segments:
                    try:
                        saved = await asyncio.to_thread(session.save)
                    except Exception as exc:
                        logger.exception("saving transcription failed")
                        await _send(
                            websocket,
                            {"type": "error", "message": f"could not save: {exc}"},
                        )
                        break
                    await _send(
                        websocket,
                        {
                            "type": "saved",
                            "transcription": saved.model_dump(mode="json"),
                        },
                    )
                else:
                    await _send(
                        websocket,
                        {"type": "info", "message": "nothing recognised, not saved"},
                    )
                break

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("websocket session failed")
    finally:
        session.closing = True
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await worker_task
        with contextlib.suppress(Exception):
            await websocket.close()
