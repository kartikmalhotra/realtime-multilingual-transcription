"""Self-hosted open-source STT via faster-whisper (CTranslate2 Whisper).

Whisper handles Hindi-English code-switching natively, which is why it is the
default here rather than a cloud API. The model is loaded lazily on first use
so the server starts instantly and the download only happens once.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from app.config import settings
from app.stt.base import ASRResult, ASRSegment, EngineUnavailable

logger = logging.getLogger(__name__)


_DEFAULT_COMPUTE = {"cuda": "float16", "mps": "float16", "cpu": "int8"}

# Only the large models keep Hindi and English in their own scripts, but they
# are only fast enough for streaming on an accelerator.
_MODEL_FOR_DEVICE = {"cuda": "large-v3-turbo", "mps": "large-v3-turbo", "cpu": "small"}


def _resolve_model(device: str) -> str:
    if settings.whisper_model != "auto":
        return settings.whisper_model
    return _MODEL_FOR_DEVICE.get(device, "small")


def _ct2_devices() -> set[str]:
    import ctranslate2

    devices = set()
    for device in ("cuda", "mps"):
        try:
            if ctranslate2.get_supported_compute_types(device):
                devices.add(device)
        except Exception:
            continue
    return devices


def _resolve_device() -> tuple[str, str]:
    """Pick the best device/compute-type pair available on this machine."""
    if settings.whisper_device != "auto":
        device = settings.whisper_device
        compute = (
            settings.whisper_compute_type
            if settings.whisper_compute_type != "default"
            else _DEFAULT_COMPUTE.get(device, "int8")
        )
        return device, compute

    # Metal is roughly 2x CPU on Apple Silicon; CUDA wins outright where present.
    available = _ct2_devices()
    for candidate in ("cuda", "mps"):
        if candidate in available:
            return candidate, _DEFAULT_COMPUTE[candidate]
    return "cpu", "int8"


class FasterWhisperEngine:
    name = "whisper"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._device, self._compute_type = _resolve_device()
        self._model_name = _resolve_model(self._device)
        self._load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def describe(self) -> dict[str, object]:
        return {
            "model": self._model_name,
            "device": self._device,
            "compute_type": self._compute_type,
            "loaded": self._model is not None,
            "error": self._load_error,
        }

    def load(self):
        """Idempotent, thread-safe model load."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover
                self._load_error = str(exc)
                raise EngineUnavailable(
                    "faster-whisper is not installed. "
                    "Run: pip install -r requirements.txt"
                ) from exc

            logger.info(
                "loading whisper model=%s device=%s compute=%s",
                self._model_name,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._load_error = None
            return self._model

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> ASRResult:
        model = self.load()

        # "auto"/None lets Whisper detect per utterance, which is what keeps a
        # Hinglish sentence from being forced into a single language.
        lang = None if language in (None, "", "auto", "mixed") else language

        segments, info = model.transcribe(
            audio,
            language=lang,
            task="transcribe",
            beam_size=settings.whisper_beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            initial_prompt=settings.whisper_initial_prompt or None,
        )

        detected = lang or getattr(info, "language", None) or "unknown"
        out: list[ASRSegment] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            out.append(
                ASRSegment(
                    text=text,
                    language=detected,
                    start_ms=int(seg.start * 1000),
                    end_ms=int(seg.end * 1000),
                    confidence=getattr(seg, "avg_logprob", None),
                )
            )

        return ASRResult(
            text=" ".join(s.text for s in out).strip(),
            language=detected,
            segments=out,
        )
