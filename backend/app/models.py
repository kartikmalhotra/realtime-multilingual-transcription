"""ORM models: a transcription and its timed segments."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Transcription(Base):
    __tablename__ = "transcriptions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="Untitled")
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # "hi", "en", or "mixed" when more than one language was detected.
    language: Mapped[str] = mapped_column(String(20), nullable=False, default="unknown")
    engine: Mapped[str] = mapped_column(String(40), nullable=False, default="whisper")
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )

    segments: Mapped[list["Segment"]] = relationship(
        back_populates="transcription",
        cascade="all, delete-orphan",
        order_by="Segment.start_ms",
        lazy="selectin",
    )

    __table_args__ = (Index("ix_transcriptions_created_at", "created_at"),)


class Segment(Base):
    """One utterance chunk, as emitted by the streaming recogniser."""

    __tablename__ = "segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    transcription_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("transcriptions.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(20), nullable=False, default="unknown")
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # faster-whisper avg_logprob, kept as a rough confidence signal.
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    transcription: Mapped[Transcription] = relationship(back_populates="segments")

    __table_args__ = (Index("ix_segments_transcription_id", "transcription_id"),)
