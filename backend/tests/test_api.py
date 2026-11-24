"""CRUD coverage for the dashboard's API."""


def _payload(**overrides):
    body = {
        "title": "Standup notes",
        "text": "Aaj ka standup short tha, we covered the deployment plan.",
        "language": "mixed",
        "engine": "whisper",
        "duration_ms": 4200,
        "segments": [
            {"text": "Aaj ka standup short tha,", "language": "hi", "start_ms": 0, "end_ms": 2000},
            {"text": "we covered the deployment plan.", "language": "en", "start_ms": 2000, "end_ms": 4200},
        ],
    }
    body.update(overrides)
    return body


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_create_returns_record_with_segments(client):
    res = client.post("/api/transcriptions", json=_payload())
    assert res.status_code == 201
    body = res.json()
    assert body["title"] == "Standup notes"
    assert body["language"] == "mixed"
    assert body["word_count"] == 10
    assert len(body["segments"]) == 2
    # Segments come back in timeline order.
    assert body["segments"][0]["start_ms"] == 0


def test_create_derives_text_and_language_from_segments(client):
    res = client.post(
        "/api/transcriptions",
        json=_payload(text="", language="auto"),
    )
    body = res.json()
    assert body["text"].startswith("Aaj ka standup")
    # Two different segment languages collapse to "mixed".
    assert body["language"] == "mixed"


def test_list_is_newest_first_and_paginated(client):
    for i in range(3):
        client.post("/api/transcriptions", json=_payload(title=f"rec-{i}"))

    page = client.get("/api/transcriptions", params={"limit": 2}).json()
    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert page["items"][0]["title"] == "rec-2"

    second = client.get("/api/transcriptions", params={"limit": 2, "offset": 2}).json()
    assert len(second["items"]) == 1


def test_search_matches_title_and_body(client):
    client.post("/api/transcriptions", json=_payload(title="Deployment"))
    client.post("/api/transcriptions", json=_payload(title="Groceries", text="doodh aur bread"))

    by_title = client.get("/api/transcriptions", params={"search": "deploy"}).json()
    assert by_title["total"] == 1

    by_body = client.get("/api/transcriptions", params={"search": "doodh"}).json()
    assert by_body["total"] == 1
    assert by_body["items"][0]["title"] == "Groceries"


def test_filter_by_language(client):
    client.post("/api/transcriptions", json=_payload(language="hi", segments=[]))
    client.post("/api/transcriptions", json=_payload(language="en", segments=[]))

    assert client.get("/api/transcriptions", params={"language": "hi"}).json()["total"] == 1
    assert client.get("/api/transcriptions", params={"language": "all"}).json()["total"] == 2


def test_get_one_and_404(client):
    created = client.post("/api/transcriptions", json=_payload()).json()
    assert client.get(f"/api/transcriptions/{created['id']}").status_code == 200
    assert client.get("/api/transcriptions/does-not-exist").status_code == 404


def test_patch_updates_text_and_recounts_words(client):
    created = client.post("/api/transcriptions", json=_payload()).json()
    res = client.patch(
        f"/api/transcriptions/{created['id']}",
        json={"text": "two words", "title": "Renamed"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["title"] == "Renamed"
    assert body["word_count"] == 2
    # Untouched fields survive a partial update.
    assert body["language"] == "mixed"


def test_patch_404(client):
    assert client.patch("/api/transcriptions/nope", json={"title": "x"}).status_code == 404


def test_delete_removes_record_and_cascades_segments(client):
    created = client.post("/api/transcriptions", json=_payload()).json()
    assert client.delete(f"/api/transcriptions/{created['id']}").status_code == 204
    assert client.get(f"/api/transcriptions/{created['id']}").status_code == 404
    assert client.get("/api/transcriptions").json()["total"] == 0

    from app import models
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        assert db.query(models.Segment).count() == 0
    finally:
        db.close()


def test_delete_404(client):
    assert client.delete("/api/transcriptions/nope").status_code == 404


def test_stats_aggregates(client):
    client.post("/api/transcriptions", json=_payload(language="hi", segments=[]))
    client.post("/api/transcriptions", json=_payload(language="hi", segments=[]))
    client.post("/api/transcriptions", json=_payload(language="en", segments=[]))

    stats = client.get("/api/stats").json()
    assert stats["total_transcriptions"] == 3
    assert stats["by_language"]["hi"] == 2
    assert stats["total_duration_ms"] == 12600


def test_export_csv(client):
    client.post("/api/transcriptions", json=_payload(title="Exported"))
    res = client.get("/api/transcriptions/export")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "Exported" in res.text


def test_engine_endpoint(client):
    body = client.get("/api/engine").json()
    assert body["active"] == "browser"
    assert "whisper" in body["available"]


def test_timestamps_are_serialized_with_a_timezone(client):
    """A naive value would be read as local time by the browser.

    The offset itself is not pinned: SQLite loses it and we re-tag as UTC,
    while PostgreSQL returns it in the session timezone (e.g. +05:30). Both
    are unambiguous, which is the only thing that matters here.
    """
    from datetime import datetime

    created = client.post("/api/transcriptions", json=_payload()).json()
    for field in ("created_at", "updated_at"):
        value = created[field]
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None, value
        assert parsed.utcoffset() is not None, value


def test_devanagari_plus_latin_is_labelled_mixed(client):
    """Whisper reports one language per window; script evidence corrects it."""
    body = client.post(
        "/api/transcriptions",
        json=_payload(
            language="auto",
            text="",
            segments=[
                {"text": "कल की meeting postpone हो गई है,", "language": "hi",
                 "start_ms": 0, "end_ms": 1800},
                {"text": "so please reschedule it", "language": "hi",
                 "start_ms": 2100, "end_ms": 4700},
            ],
        ),
    ).json()
    # Both segments were detected as "hi", but the text is plainly bilingual.
    assert body["language"] == "mixed"


def test_single_script_keeps_detected_language(client):
    body = client.post(
        "/api/transcriptions",
        json=_payload(
            language="auto",
            text="",
            segments=[
                {"text": "कल की मीटिंग रद्द हो गई", "language": "hi",
                 "start_ms": 0, "end_ms": 1500},
            ],
        ),
    ).json()
    assert body["language"] == "hi"


def test_explicit_language_is_not_overridden(client):
    body = client.post("/api/transcriptions", json=_payload(language="en")).json()
    assert body["language"] == "en"
