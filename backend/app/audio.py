"""Audio buffering and utterance segmentation for the streaming endpoint.

The client sends small 16 kHz mono PCM frames. This module decides, purely
from frame energy, when there is enough new audio to refresh the interim
transcript and when the speaker has stopped so the utterance can be finalised
and written to the database.
"""
from __future__ import annotations

import numpy as np

from app.config import settings

INT16_MAX = 32768.0


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Decode little-endian int16 PCM into the float32 range Whisper expects."""
    if not raw:
        return np.zeros(0, dtype=np.float32)
    # An odd trailing byte would misalign the whole frame; drop it.
    if len(raw) % 2:
        raw = raw[:-1]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_MAX


def rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


class UtteranceBuffer:
    """Accumulates one utterance and reports when to transcribe it."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._samples_at_last_partial = 0
        self._trailing_silence = 0
        self._has_speech = False
        # Monotonic sample offset of this utterance within the whole session,
        # so stored segment timestamps stay meaningful across utterances.
        self.session_offset_ms = 0

    # --- state ----------------------------------------------------------
    @property
    def has_speech(self) -> bool:
        return self._has_speech

    @property
    def duration_ms(self) -> int:
        return int(self._samples / settings.sample_rate * 1000)

    @property
    def is_empty(self) -> bool:
        return self._samples == 0

    # --- ingest ---------------------------------------------------------
    def feed(self, frame: np.ndarray) -> None:
        if frame.size == 0:
            return

        level = rms(frame)
        is_speech = level >= settings.vad_energy_threshold

        if is_speech:
            self._has_speech = True
            self._trailing_silence = 0
        elif self._has_speech:
            self._trailing_silence += frame.size
        else:
            # Leading silence is dropped so Whisper never sees dead air.
            self.session_offset_ms += int(frame.size / settings.sample_rate * 1000)
            return

        self._chunks.append(frame)
        self._samples += frame.size

    # --- decisions ------------------------------------------------------
    def new_audio_ms(self) -> int:
        """Audio accumulated since the last interim pass."""
        new_samples = self._samples - self._samples_at_last_partial
        return int(new_samples / settings.sample_rate * 1000)

    def should_emit_partial(self, min_new_ms: int | None = None) -> bool:
        if not self._has_speech:
            return False
        needed = settings.partial_interval_ms if min_new_ms is None else min_new_ms
        return self.new_audio_ms() >= needed

    def mark_partial_emitted(self) -> None:
        self._samples_at_last_partial = self._samples

    def should_finalize(self) -> bool:
        if not self._has_speech:
            return False
        silence_needed = settings.silence_timeout_ms * settings.sample_rate // 1000
        if self._trailing_silence >= silence_needed:
            return True
        # Bound latency on someone who never pauses.
        return self.duration_ms >= settings.max_utterance_ms

    # --- output ---------------------------------------------------------
    def audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    def reset(self) -> None:
        self.session_offset_ms += self.duration_ms
        self._chunks.clear()
        self._samples = 0
        self._samples_at_last_partial = 0
        self._trailing_silence = 0
        self._has_speech = False
