"""Application settings, loaded from environment / .env."""
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database -------------------------------------------------------
    # Postgres in docker-compose / production; SQLite keeps a bare `uvicorn`
    # run working with no external services.
    database_url: str = "sqlite:///./transcriptions.db"

    # --- speech to text -------------------------------------------------
    # whisper  -> self-hosted faster-whisper (open source, default)
    # browser  -> client-side Web Speech API, server only persists
    # gemini   -> Gemini Live API (needs gemini_api_key)
    stt_engine: Literal["whisper", "browser", "gemini"] = "whisper"

    # "auto" picks per device, measured on a Hindi/English code-switch clip:
    #   large-v3-turbo keeps each language in its own script and runs at
    #   ~0.4x real-time on Metal/CUDA, but ~2.2x (too slow) on CPU;
    #   small is ~0.2x real-time on CPU but transliterates English into
    #   Devanagari. So: accelerator -> turbo, CPU-only -> small.
    whisper_model: str = "auto"
    whisper_device: str = "auto"
    whisper_compute_type: str = "default"
    whisper_beam_size: int = 1
    # Deliberately empty. An English/Hinglish `initial_prompt` seems like it
    # should help code-switching, but measured on Hindi-then-English audio it
    # biases decoding toward the prompt's language and makes Whisper stop at
    # the switch point - a Hindi clip came back with zero segments instead of
    # eleven words. Left configurable for domain vocabulary (names, jargon),
    # which is what it is actually good for. See docs/ARCHITECTURE.md.
    whisper_initial_prompt: str = ""

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-live-2.5-flash-preview"

    # --- streaming tuning ----------------------------------------------
    sample_rate: int = 16_000
    partial_interval_ms: int = 900      # how often interim results are emitted
    silence_timeout_ms: int = 800       # trailing silence that ends an utterance
    max_utterance_ms: int = 25_000      # hard cap so latency stays bounded
    vad_energy_threshold: float = 0.006 # RMS gate for "is anyone talking"

    # --- http -----------------------------------------------------------
    cors_origins: str = "*"

    @field_validator("database_url")
    @classmethod
    def _normalise_database_url(cls, value: str) -> str:
        """Accept the URL shapes hosting providers actually hand out.

        Render, Heroku and Fly all inject `postgres://...`, a scheme
        SQLAlchemy 2.0 removed - it raises NoSuchModuleError on boot rather
        than falling back. Rewriting it here means the deployed app works
        with the provider's variable untouched.
        """
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value[len("postgres://") :]
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value[len("postgresql://") :]
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
