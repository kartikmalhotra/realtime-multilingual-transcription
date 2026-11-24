"""Gemini Live engine: the transcript comes from input_audio_transcription.

Live models answer with audio, so the engine must read the server-side
transcription of the audio it sent, not the model's reply. A fake session
stands in for the network so this runs without an API key.
"""
from types import SimpleNamespace

import numpy as np

from app.config import settings
from app.stt.gemini_engine import GeminiLiveEngine


def _event(text=None, turn_complete=False, model_audio=False):
    content = SimpleNamespace(
        input_transcription=SimpleNamespace(text=text) if text else None,
        turn_complete=turn_complete,
        model_turn=SimpleNamespace(parts=["audio"]) if model_audio else None,
    )
    return SimpleNamespace(server_content=content)


class _FakeSession:
    def __init__(self, events):
        self.events = events
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)

    async def receive(self):
        for event in self.events:
            yield event


class _FakeClient:
    def __init__(self, session):
        self.session = session
        self.config = None
        live = SimpleNamespace(connect=self._connect)
        self.aio = SimpleNamespace(live=live)

    def _connect(self, model, config):
        self.config = config
        return self.session


def _engine_with(events, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    session = _FakeSession(events)
    client = _FakeClient(session)
    engine = GeminiLiveEngine()
    engine._client = client
    return engine, client, session


def test_collects_input_transcription_and_ignores_spoken_reply(monkeypatch):
    events = [
        _event(text="mujhe kal "),
        _event(model_audio=True),
        _event(text="meeting schedule karni hai"),
        _event(turn_complete=True),
        _event(text="never read"),
    ]
    engine, client, session = _engine_with(events, monkeypatch)

    result = engine.transcribe(np.zeros(16_000, dtype=np.float32))

    assert result.text == "mujhe kal meeting schedule karni hai"
    assert result.segments and result.segments[0].end_ms == 1000
    assert client.config["response_modalities"] == ["AUDIO"]
    assert "input_audio_transcription" in client.config
    assert session.sent[-1] == {"audio_stream_end": True}


def test_api_failure_returns_empty_result_instead_of_raising(monkeypatch):
    engine, client, _ = _engine_with([], monkeypatch)

    def boom(model, config):
        raise RuntimeError("quota exceeded")

    client.aio.live.connect = boom
    result = engine.transcribe(np.zeros(1_600, dtype=np.float32))

    assert result.text == ""
    assert engine.describe()["error"] == "quota exceeded"


def test_client_pins_the_api_version_that_serves_live(monkeypatch):
    """Without an explicit api_version the Live socket 404s on handshake."""
    captured = {}

    class FakeGenaiModule:
        @staticmethod
        def Client(**kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setitem(
        __import__("sys").modules, "google.genai", FakeGenaiModule
    )
    import google

    monkeypatch.setattr(google, "genai", FakeGenaiModule, raising=False)

    GeminiLiveEngine()._get_client()
    assert captured["http_options"] == {"api_version": settings.gemini_api_version}


def test_audio_is_streamed_in_chunks_not_one_blob(monkeypatch):
    """One large blob yields no transcription at all from the Live endpoint."""
    events = [_event(text="theek hai"), _event(turn_complete=True)]
    engine, _, session = _engine_with(events, monkeypatch)

    # 1 second of audio -> ten 100 ms chunks, then the end-of-stream marker.
    engine.transcribe(np.zeros(16_000, dtype=np.float32))

    audio_sends = [s for s in session.sent if "audio" in s]
    assert len(audio_sends) == 10, len(audio_sends)
    assert session.sent[-1] == {"audio_stream_end": True}


def test_stops_once_the_transcript_stops_growing(monkeypatch):
    """`turn_complete` never arrives while the model speaks its reply."""
    events = [_event(text="haan")] + [_event() for _ in range(50)]
    engine, _, _ = _engine_with(events, monkeypatch)

    result = engine.transcribe(np.zeros(1_600, dtype=np.float32))

    # Returns on the quiet run instead of draining 50 events or timing out.
    assert result.text == "haan"
