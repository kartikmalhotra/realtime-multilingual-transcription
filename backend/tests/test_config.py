"""Database URL handling.

Hosting providers inject `postgres://`, which SQLAlchemy 2.0 refuses to
load. Getting this wrong fails at import time on the deployed instance
only, so it is worth pinning.
"""
import pytest
from sqlalchemy import create_engine

from app.config import Settings


@pytest.mark.parametrize(
    "given",
    [
        "postgres://user:pw@host:5432/db",      # Render / Heroku / Fly
        "postgresql://user:pw@host:5432/db",    # psycopg2 default, not installed
        "postgresql+psycopg://user:pw@host:5432/db",
    ],
)
def test_postgres_urls_normalise_to_psycopg(given):
    url = Settings(database_url=given).database_url
    assert url.startswith("postgresql+psycopg://")
    # The credentials and target must survive the rewrite.
    assert url.endswith("user:pw@host:5432/db")


def test_normalised_url_actually_builds_an_engine():
    url = Settings(database_url="postgres://user:pw@host:5432/db").database_url
    assert create_engine(url).dialect.driver == "psycopg"


def test_raw_render_url_would_have_failed():
    """Guards the assumption that motivated the normaliser."""
    from sqlalchemy.exc import NoSuchModuleError

    with pytest.raises(NoSuchModuleError):
        create_engine("postgres://user:pw@host:5432/db")


def test_sqlite_url_is_left_alone():
    assert Settings(database_url="sqlite:///./x.db").database_url == "sqlite:///./x.db"


def test_mysql_url_is_left_alone():
    given = "mysql+pymysql://user:pw@host/db"
    assert Settings(database_url=given).database_url == given
