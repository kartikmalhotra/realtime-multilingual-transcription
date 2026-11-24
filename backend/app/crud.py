"""Database operations for transcriptions."""
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app import models, schemas


def _word_count(text: str) -> int:
    return len(text.split())


# Whisper reports one language per decode window, so a genuinely bilingual
# utterance comes back labelled with whichever language won. Checking which
# scripts are actually present recovers the "mixed" case for free.
_DEVANAGARI = range(0x0900, 0x0980)


def has_mixed_scripts(text: str) -> bool:
    """True when the text contains both Devanagari and Latin letters."""
    devanagari = latin = False
    for ch in text:
        if ord(ch) in _DEVANAGARI:
            devanagari = True
        elif ch.isascii() and ch.isalpha():
            latin = True
        if devanagari and latin:
            return True
    return False


def summarise_language(languages: list[str], text: str = "") -> str:
    """Collapse per-segment languages into one label for the parent record."""
    # Script evidence beats the model's single-language guess.
    if has_mixed_scripts(text):
        return "mixed"
    distinct = {lang for lang in languages if lang and lang != "unknown"}
    if not distinct:
        return "unknown"
    if len(distinct) == 1:
        return distinct.pop()
    return "mixed"


def create_transcription(
    db: Session, payload: schemas.TranscriptionCreate
) -> models.Transcription:
    text = payload.text.strip()
    if not text and payload.segments:
        text = " ".join(s.text.strip() for s in payload.segments if s.text.strip())

    language = payload.language
    if language in ("", "unknown", "auto", "mixed"):
        language = summarise_language(
            [s.language for s in payload.segments], text
        )

    record = models.Transcription(
        title=payload.title.strip() or "Untitled",
        text=text,
        language=language or "unknown",
        engine=payload.engine,
        duration_ms=payload.duration_ms,
        word_count=_word_count(text),
        segments=[
            models.Segment(
                text=s.text,
                language=s.language,
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                confidence=s.confidence,
            )
            for s in payload.segments
        ],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_transcription(db: Session, transcription_id: str) -> models.Transcription | None:
    return db.get(models.Transcription, transcription_id)


def list_transcriptions(
    db: Session,
    *,
    search: str | None = None,
    language: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[models.Transcription], int]:
    stmt = select(models.Transcription)
    count_stmt = select(func.count()).select_from(models.Transcription)

    if search:
        pattern = f"%{search.lower()}%"
        clause = or_(
            func.lower(models.Transcription.text).like(pattern),
            func.lower(models.Transcription.title).like(pattern),
        )
        stmt = stmt.where(clause)
        count_stmt = count_stmt.where(clause)

    if language and language != "all":
        stmt = stmt.where(models.Transcription.language == language)
        count_stmt = count_stmt.where(models.Transcription.language == language)

    total = db.execute(count_stmt).scalar_one()
    stmt = (
        stmt.order_by(models.Transcription.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.execute(stmt).scalars()), total


def update_transcription(
    db: Session, record: models.Transcription, payload: schemas.TranscriptionUpdate
) -> models.Transcription:
    data = payload.model_dump(exclude_unset=True)
    if "title" in data and data["title"] is not None:
        record.title = data["title"].strip() or "Untitled"
    if "text" in data and data["text"] is not None:
        record.text = data["text"]
        record.word_count = _word_count(data["text"])
    if "language" in data and data["language"] is not None:
        record.language = data["language"]
    db.commit()
    db.refresh(record)
    return record


def delete_transcription(db: Session, record: models.Transcription) -> None:
    db.delete(record)
    db.commit()


def stats(db: Session) -> schemas.Stats:
    totals = db.execute(
        select(
            func.count(models.Transcription.id),
            func.coalesce(func.sum(models.Transcription.word_count), 0),
            func.coalesce(func.sum(models.Transcription.duration_ms), 0),
        )
    ).one()

    by_language = {
        lang: count
        for lang, count in db.execute(
            select(models.Transcription.language, func.count(models.Transcription.id))
            .group_by(models.Transcription.language)
            .order_by(func.count(models.Transcription.id).desc())
        ).all()
    }

    return schemas.Stats(
        total_transcriptions=totals[0],
        total_words=int(totals[1]),
        total_duration_ms=int(totals[2]),
        by_language=by_language,
    )
