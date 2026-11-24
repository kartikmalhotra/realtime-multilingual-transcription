"""End-to-end streaming test with a stubbed recogniser."""
import json

import numpy as np
import pytest

from app.stt.base import ASRResult, ASRSegment


class StubEngine:
    """Returns fixed Hinglish text so the transport can be tested on its own."""

    name = "whisper"

    def __init__(self) -> None:
        self.calls: list[int] = []

    @property
    def ready(self) -> bool:
        return True

    def describe(self) -> dict:
        return {"model": "stub"}

    def transcribe(self, audio: np.ndarray, language=None) -> ASRResult:
        self.calls.append(int(audio.size))
        return ASRResult(
            text="kal meeting hai at 10 am",
            language="mixed",
            segments=[
                ASRSegment(
                    text="kal meeting hai at 10 am",
                    language="mixed",
                    start_ms=0,
                    end_ms=1000,
                )
            ],
        )


@pytest.fixture
def stub_engine(monkeypatch):
    engine = StubEngine()
    monkeypatch.setattr("app.routers.ws.get_engine", lambda _name=None: engine)
    return engine


def pcm(ms: int, amplitude: int) -> bytes:
    from app.config import settings

    n = settings.sample_rate * ms // 1000
    return np.full(n, amplitude, dtype="<i2").tobytes()


def drain(ws, want: str, limit: int = 40) -> dict:
    """Read messages until one of type `want` shows up."""
    for _ in range(limit):
        msg = json.loads(ws.receive_text())
        if msg.get("type") == want:
            return msg
    raise AssertionError(f"never received a {want!r} message")


def test_stream_emits_final_and_persists(client, stub_engine):
    with client.websocket_connect("/ws/transcribe?engine=whisper&language=auto") as ws:
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready"
        assert ready["engine"] == "whisper"

        ws.send_bytes(pcm(600, 6000))          # speech
        ws.send_bytes(pcm(1000, 0))            # silence -> finalise

        final = drain(ws, "final")
        assert final["segment"]["text"] == "kal meeting hai at 10 am"
        assert final["text"] == "kal meeting hai at 10 am"

        ws.send_text(json.dumps({"type": "stop"}))
        saved = drain(ws, "saved")

    record = saved["transcription"]
    assert record["language"] == "auto" or record["language"] == "mixed"
    assert record["engine"] == "whisper"
    assert len(record["segments"]) == 1

    # The record really is in the database, not just echoed back.
    listed = client.get("/api/transcriptions").json()
    assert listed["total"] == 1
    assert listed["items"][0]["text"] == "kal meeting hai at 10 am"


def test_silence_only_stream_saves_nothing(client, stub_engine):
    with client.websocket_connect("/ws/transcribe?engine=whisper") as ws:
        json.loads(ws.receive_text())  # ready
        ws.send_bytes(pcm(1500, 0))
        ws.send_text(json.dumps({"type": "stop"}))
        info = drain(ws, "info")
        assert "not saved" in info["message"]

    assert stub_engine.calls == []
    assert client.get("/api/transcriptions").json()["total"] == 0


def test_flush_finalizes_without_closing(client, stub_engine):
    with client.websocket_connect("/ws/transcribe?engine=whisper") as ws:
        json.loads(ws.receive_text())
        ws.send_bytes(pcm(300, 6000))
        ws.send_text(json.dumps({"type": "flush"}))
        assert drain(ws, "final")["segment"]["text"]

        # Socket is still usable after a flush.
        ws.send_text(json.dumps({"type": "stop"}))
        drain(ws, "saved")


def test_browser_engine_persists_client_text(client):
    with client.websocket_connect("/ws/transcribe?engine=browser&language=hi") as ws:
        json.loads(ws.receive_text())

        ws.send_text(
            json.dumps({"type": "client_transcript", "text": "interim words"})
        )
        assert drain(ws, "partial")["text"] == "interim words"

        ws.send_text(
            json.dumps(
                {
                    "type": "client_transcript",
                    "text": "mujhe kal call karna hai",
                    "is_final": True,
                    "language": "hi",
                    "duration_ms": 2500,
                }
            )
        )
        assert drain(ws, "final")["segment"]["text"] == "mujhe kal call karna hai"

        ws.send_text(json.dumps({"type": "stop"}))
        saved = drain(ws, "saved")

    assert saved["transcription"]["text"] == "mujhe kal call karna hai"
    assert saved["transcription"]["duration_ms"] == 2500


def test_invalid_json_is_reported_not_fatal(client, stub_engine):
    with client.websocket_connect("/ws/transcribe?engine=whisper") as ws:
        json.loads(ws.receive_text())
        ws.send_text("{not json")
        assert "invalid JSON" in drain(ws, "error")["message"]

        ws.send_text(json.dumps({"type": "stop"}))
        drain(ws, "info")


def test_save_failure_is_reported_instead_of_hanging(client, stub_engine, monkeypatch):
    """The client waits on 'saved'; a DB error must come back as 'error'."""
    def boom(self):
        raise RuntimeError("relation \"transcriptions\" does not exist")

    monkeypatch.setattr(
        "app.routers.ws.TranscriptionSession.save", boom, raising=True
    )

    with client.websocket_connect("/ws/transcribe?engine=whisper") as ws:
        json.loads(ws.receive_text())
        ws.send_bytes(pcm(400, 6000))
        ws.send_text(json.dumps({"type": "flush"}))
        drain(ws, "final")
        ws.send_text(json.dumps({"type": "stop"}))
        err = drain(ws, "error")
        assert "could not save" in err["message"]


class SlowEngine(StubEngine):
    """Mimics a host where decoding is slower than the interim interval."""

    def __init__(self, delay_s: float = 0.25) -> None:
        super().__init__()
        self.delay_s = delay_s

    def transcribe(self, audio, language=None):
        import time

        time.sleep(self.delay_s)
        return super().transcribe(audio, language)


def test_slow_engine_still_saves_and_coalesces_partials(client, monkeypatch):
    """A decoder slower than real time must not strand the client.

    Before ingestion and decoding were split, every interim interval queued
    another decode behind the audio frames, so `stop` waited behind the whole
    backlog. The worker should instead skip stale interim passes.
    """
    import time

    engine = SlowEngine(delay_s=0.25)
    monkeypatch.setattr("app.routers.ws.get_engine", lambda _name=None: engine)

    with client.websocket_connect("/ws/transcribe?engine=whisper") as ws:
        json.loads(ws.receive_text())

        # 6 seconds of speech: at a 900 ms interim interval that would be
        # ~6 queued decodes if each frame triggered one.
        for _ in range(20):
            ws.send_bytes(pcm(300, 6000))

        started = time.monotonic()
        ws.send_text(json.dumps({"type": "stop"}))
        saved = drain(ws, "saved", limit=80)
        elapsed = time.monotonic() - started

    assert saved["transcription"]["text"]
    # Far fewer decodes than one per interim interval.
    assert len(engine.calls) <= 6, engine.calls
    # And the save is not stuck behind a backlog of them.
    assert elapsed < 10, elapsed


def test_disconnect_without_stop_does_not_leak_the_worker(client, stub_engine):
    """Closing the socket mid-utterance must cancel the decoder task."""
    with client.websocket_connect("/ws/transcribe?engine=whisper") as ws:
        json.loads(ws.receive_text())
        ws.send_bytes(pcm(400, 6000))
        # Leave the context manager without sending "stop".

    # A second connection still works, which it would not if the previous
    # worker were wedged holding the engine.
    with client.websocket_connect("/ws/transcribe?engine=whisper") as ws:
        assert json.loads(ws.receive_text())["type"] == "ready"
        ws.send_text(json.dumps({"type": "stop"}))
        drain(ws, "info")
