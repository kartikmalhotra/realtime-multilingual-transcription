import os
import tempfile

import pytest

# Point the app at a throwaway SQLite file before any app module is imported.
# An externally supplied DATABASE_URL wins, so the same suite can be run
# against PostgreSQL:
#   DATABASE_URL=postgresql+psycopg://postgres@localhost:5432/transcriptions pytest
if not os.environ.get("DATABASE_URL"):
    _tmp = tempfile.mkdtemp(prefix="transcription-tests-")
    os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_tmp, 'test.db')}"
os.environ["STT_ENGINE"] = "browser"

from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_schema():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
