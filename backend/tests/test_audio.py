"""Utterance segmentation: the logic that decides when to transcribe."""
import numpy as np

from app.audio import UtteranceBuffer, pcm16_to_float32, rms
from app.config import settings

RATE = settings.sample_rate


def frame(ms: int, amplitude: float) -> np.ndarray:
    n = RATE * ms // 1000
    return np.full(n, amplitude, dtype=np.float32)


def test_pcm16_roundtrip():
    original = np.array([0, 16384, -16384, 32767], dtype="<i2")
    decoded = pcm16_to_float32(original.tobytes())
    assert decoded.shape == (4,)
    assert np.isclose(decoded[1], 0.5, atol=1e-3)
    assert np.isclose(decoded[2], -0.5, atol=1e-3)


def test_pcm16_drops_misaligned_trailing_byte():
    # A truncated frame must not raise or shift every sample.
    assert pcm16_to_float32(b"\x00\x40\x7f").shape == (1,)


def test_pcm16_handles_empty():
    assert pcm16_to_float32(b"").shape == (0,)


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0


def test_leading_silence_is_discarded():
    buf = UtteranceBuffer()
    buf.feed(frame(500, 0.0))
    assert buf.is_empty
    assert not buf.has_speech
    # The dropped silence still advances the session clock.
    assert buf.session_offset_ms == 500


def test_speech_marks_buffer_active():
    buf = UtteranceBuffer()
    buf.feed(frame(200, 0.2))
    assert buf.has_speech
    assert buf.duration_ms == 200


def test_partial_emitted_once_enough_new_audio_arrives():
    buf = UtteranceBuffer()
    buf.feed(frame(200, 0.2))
    assert not buf.should_emit_partial()

    buf.feed(frame(settings.partial_interval_ms, 0.2))
    assert buf.should_emit_partial()

    buf.mark_partial_emitted()
    assert not buf.should_emit_partial()


def test_trailing_silence_finalizes_utterance():
    buf = UtteranceBuffer()
    buf.feed(frame(400, 0.2))
    assert not buf.should_finalize()

    buf.feed(frame(settings.silence_timeout_ms, 0.0))
    assert buf.should_finalize()


def test_silence_before_any_speech_never_finalizes():
    buf = UtteranceBuffer()
    buf.feed(frame(5000, 0.0))
    assert not buf.should_finalize()


def test_long_utterance_is_force_finalized():
    buf = UtteranceBuffer()
    # Someone talking without pause must still get a result eventually.
    buf.feed(frame(settings.max_utterance_ms + 100, 0.2))
    assert buf.should_finalize()


def test_reset_advances_session_offset():
    buf = UtteranceBuffer()
    buf.feed(frame(1000, 0.2))
    buf.reset()
    assert buf.session_offset_ms == 1000
    assert buf.is_empty
    assert not buf.has_speech


def test_audio_concatenates_fed_frames():
    buf = UtteranceBuffer()
    buf.feed(frame(100, 0.2))
    buf.feed(frame(100, 0.2))
    assert buf.audio().shape[0] == RATE * 200 // 1000


# --- script detection (used to label code-switched records) ---
def test_has_mixed_scripts():
    from app.crud import has_mixed_scripts

    assert has_mixed_scripts("कल की meeting postpone हो गई")
    assert not has_mixed_scripts("कल की मीटिंग रद्द हो गई")
    assert not has_mixed_scripts("the meeting was postponed")
    # Digits and punctuation alongside Devanagari are not "Latin".
    assert not has_mixed_scripts("कल 10 बजे, ठीक है?")
    assert not has_mixed_scripts("")


def test_romanized_hindi_is_not_detectable_by_script():
    """A documented limit: Hinglish typed in Latin looks like English."""
    from app.crud import has_mixed_scripts

    assert not has_mixed_scripts("kal ki meeting postpone ho gayi")
