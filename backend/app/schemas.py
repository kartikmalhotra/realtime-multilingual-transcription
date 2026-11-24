"""Pydantic request/response models."""
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class SegmentBase(BaseModel):
    text: str
    language: str = "unknown"
    start_ms: int = 0
    end_ms: int = 0
    confidence: float | None = None


class SegmentCreate(SegmentBase):
    pass


class SegmentOut(SegmentBase):
    model_config = ConfigDict(from_attributes=True)

    id: str


class TranscriptionCreate(BaseModel):
    title: str = Field(default="Untitled", max_length=200)
    text: str = ""
    language: str = "unknown"
    engine: str = "whisper"
    duration_ms: int = 0
    segments: list[SegmentCreate] = Field(default_factory=list)


class TranscriptionUpdate(BaseModel):
    """PATCH body - every field optional."""

    title: str | None = Field(default=None, max_length=200)
    text: str | None = None
    language: str | None = None


class TranscriptionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    text: str
    language: str
    engine: str
    duration_ms: int
    word_count: int
    created_at: datetime
    updated_at: datetime
    segments: list[SegmentOut] = Field(default_factory=list)

    # SQLite drops the offset on the way out, so a UTC timestamp comes back
    # naive and `new Date(...)` in the browser would read it as local time.
    # Tag naive values as UTC, which is what they are.
    @field_serializer("created_at", "updated_at")
    def _as_utc(self, value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


class TranscriptionPage(BaseModel):
    items: list[TranscriptionOut]
    total: int
    limit: int
    offset: int


class Stats(BaseModel):
    total_transcriptions: int
    total_words: int
    total_duration_ms: int
    by_language: dict[str, int]


class EngineInfo(BaseModel):
    active: str
    available: list[str]
    model: str | None = None
    ready: bool
    detail: str | None = None
    # So the dashboard can state the database it is actually using rather
    # than hardcoding one.
    database: str = "unknown"
