"""Device and model auto-selection.

These encode a measured trade-off: large-v3-turbo is the only model that
keeps Hindi and English in their own scripts, but it only runs faster than
real-time on an accelerator.
"""
import pytest

from app.stt import faster_whisper_engine as fwe


@pytest.fixture
def no_accelerator(monkeypatch):
    monkeypatch.setattr(fwe, "_ct2_devices", lambda: set())


@pytest.fixture(autouse=True)
def auto_settings(monkeypatch):
    monkeypatch.setattr(fwe.settings, "whisper_device", "auto")
    monkeypatch.setattr(fwe.settings, "whisper_model", "auto")
    monkeypatch.setattr(fwe.settings, "whisper_compute_type", "default")


def test_cpu_only_host_falls_back_to_cpu_int8(no_accelerator):
    assert fwe._resolve_device() == ("cpu", "int8")


def test_cuda_preferred_over_mps(monkeypatch):
    monkeypatch.setattr(fwe, "_ct2_devices", lambda: {"cuda", "mps"})
    assert fwe._resolve_device() == ("cuda", "float16")


def test_mps_used_when_no_cuda(monkeypatch):
    monkeypatch.setattr(fwe, "_ct2_devices", lambda: {"mps"})
    assert fwe._resolve_device() == ("mps", "float16")


def test_explicit_device_is_respected(monkeypatch):
    monkeypatch.setattr(fwe.settings, "whisper_device", "cuda")
    assert fwe._resolve_device() == ("cuda", "float16")


def test_explicit_device_with_explicit_compute_type(monkeypatch):
    monkeypatch.setattr(fwe.settings, "whisper_device", "cpu")
    monkeypatch.setattr(fwe.settings, "whisper_compute_type", "float32")
    assert fwe._resolve_device() == ("cpu", "float32")


def test_unknown_explicit_device_defaults_to_int8(monkeypatch):
    monkeypatch.setattr(fwe.settings, "whisper_device", "rocm")
    assert fwe._resolve_device() == ("rocm", "int8")


@pytest.mark.parametrize(
    "device,expected",
    [("cpu", "small"), ("mps", "large-v3-turbo"), ("cuda", "large-v3-turbo")],
)
def test_auto_model_tracks_device(device, expected):
    assert fwe._resolve_model(device) == expected


def test_explicit_model_overrides_auto(monkeypatch):
    monkeypatch.setattr(fwe.settings, "whisper_model", "medium")
    assert fwe._resolve_model("cpu") == "medium"
    assert fwe._resolve_model("mps") == "medium"


def test_describe_reports_resolved_model_before_load(no_accelerator):
    described = fwe.FasterWhisperEngine().describe()
    assert described["model"] == "small"
    assert described["device"] == "cpu"
    assert described["loaded"] is False


def test_initial_prompt_defaults_to_empty():
    """A Hinglish primer measurably drops whole utterances; see ARCHITECTURE.md."""
    from app.config import Settings

    assert Settings().whisper_initial_prompt == ""


def test_ct2_devices_tolerates_missing_backends(monkeypatch):
    class FakeCT2:
        @staticmethod
        def get_supported_compute_types(device):
            if device == "cuda":
                raise RuntimeError("no cuda")
            return {"float16"}

    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", FakeCT2)
    assert fwe._ct2_devices() == {"mps"}
