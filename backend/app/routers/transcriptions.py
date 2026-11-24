"""REST CRUD for transcription history (the dashboard's data source)."""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import engine as db_engine, get_db
from app.stt.factory import AVAILABLE, get_engine

router = APIRouter(prefix="/api", tags=["transcriptions"])


def _get_or_404(db: Session, transcription_id: str):
    record = crud.get_transcription(db, transcription_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcription not found"
        )
    return record


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/engine", response_model=schemas.EngineInfo)
def engine_info() -> schemas.EngineInfo:
    engine = get_engine()
    described = engine.describe()
    return schemas.EngineInfo(
        active=engine.name,
        available=list(AVAILABLE),
        model=str(described.get("model")) if described.get("model") else None,
        ready=bool(engine.ready),
        detail=str(described.get("error")) if described.get("error") else None,
        database=db_engine.dialect.name,
    )


@router.get("/stats", response_model=schemas.Stats)
def get_stats(db: Session = Depends(get_db)) -> schemas.Stats:
    return crud.stats(db)


# --- CREATE ------------------------------------------------------------
@router.post(
    "/transcriptions",
    response_model=schemas.TranscriptionOut,
    status_code=status.HTTP_201_CREATED,
)
def create(
    payload: schemas.TranscriptionCreate, db: Session = Depends(get_db)
) -> schemas.TranscriptionOut:
    return crud.create_transcription(db, payload)


# --- READ --------------------------------------------------------------
@router.get("/transcriptions", response_model=schemas.TranscriptionPage)
def list_all(
    search: str | None = Query(default=None, description="match title or body text"),
    language: str | None = Query(default=None, description="hi | en | mixed | all"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> schemas.TranscriptionPage:
    items, total = crud.list_transcriptions(
        db, search=search, language=language, limit=limit, offset=offset
    )
    return schemas.TranscriptionPage(
        items=[schemas.TranscriptionOut.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/transcriptions/export")
def export_csv(db: Session = Depends(get_db)) -> Response:
    items, _ = crud.list_transcriptions(db, limit=100, offset=0)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["id", "title", "language", "engine", "duration_ms", "words", "created_at", "text"]
    )
    for item in items:
        writer.writerow(
            [
                item.id,
                item.title,
                item.language,
                item.engine,
                item.duration_ms,
                item.word_count,
                item.created_at.isoformat(),
                item.text,
            ]
        )
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="transcriptions.csv"'},
    )


@router.get("/transcriptions/{transcription_id}", response_model=schemas.TranscriptionOut)
def get_one(
    transcription_id: str, db: Session = Depends(get_db)
) -> schemas.TranscriptionOut:
    return _get_or_404(db, transcription_id)


# --- UPDATE ------------------------------------------------------------
@router.patch("/transcriptions/{transcription_id}", response_model=schemas.TranscriptionOut)
def update(
    transcription_id: str,
    payload: schemas.TranscriptionUpdate,
    db: Session = Depends(get_db),
) -> schemas.TranscriptionOut:
    record = _get_or_404(db, transcription_id)
    return crud.update_transcription(db, record, payload)


# --- DELETE ------------------------------------------------------------
@router.delete("/transcriptions/{transcription_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete(transcription_id: str, db: Session = Depends(get_db)) -> Response:
    record = _get_or_404(db, transcription_id)
    crud.delete_transcription(db, record)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
