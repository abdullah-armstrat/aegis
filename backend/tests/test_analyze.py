"""Tests for the /analyze endpoint (image + caption -> Scorecard JSON).

Exercises the full HTTP path with the real adapter and rules (LLM off by default), using
small PIL-generated images so no binary fixtures are committed. Validates the contract the
frontend depends on, plus input guards.
"""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def _png(size=(80, 40), colour="white") -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_returns_scorecard(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "false")
    get_settings.cache_clear()

    resp = client.post(
        "/analyze",
        files={"image": ("photo.png", _png(), "image/png")},
        data={"caption": "A calm day in the park"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "flags" in body and isinstance(body["flags"], list)
    assert body["modality"] == "image"
    assert body["source_ref"] == "photo.png"
    assert body["summary"]
    # Every flag carries the explain-don't-verdict fields and a status (ADR-002, ADR-009).
    for flag in body["flags"]:
        assert {"type", "status", "plain_explanation", "what_to_check"} <= flag.keys()
    assert "verdict" not in body
    get_settings.cache_clear()


def test_analyze_rejects_non_image():
    resp = client.post(
        "/analyze",
        files={"image": ("notes.txt", b"hello", "text/plain")},
        data={"caption": "x"},
    )
    assert resp.status_code == 415


def test_analyze_rejects_empty_upload():
    resp = client.post(
        "/analyze",
        files={"image": ("empty.png", b"", "image/png")},
        data={"caption": "x"},
    )
    assert resp.status_code == 400


def test_analyze_works_without_caption():
    resp = client.post("/analyze", files={"image": ("p.png", _png(), "image/png")})
    assert resp.status_code == 200
    assert resp.json()["modality"] == "image"
