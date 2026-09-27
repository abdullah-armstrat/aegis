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
    # Every flag carries the explain-don't-verdict fields and a status.
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


# --- Posting date and content matching through the real HTTP path ---

from pathlib import Path  # noqa: E402

_FLOOD = Path(__file__).resolve().parents[2] / "data" / "illustrative" / "flood_illustrative.png"


def _recycled(body: dict) -> dict:
    return next(f for f in body["flags"] if f["type"] == "recycled_context")


@pytest.mark.parametrize("bad", ["2026-9-1", "01/09/2026", "yesterday", "2026-02-30"])
def test_analyze_rejects_malformed_posted_date(bad):
    resp = client.post(
        "/analyze",
        files={"image": ("p.png", _png(), "image/png")},
        data={"caption": "x", "posted_date": bad},
    )
    assert resp.status_code == 400
    assert "YYYY-MM-DD" in resp.json()["detail"]


def test_analyze_rejects_future_posted_date():
    resp = client.post(
        "/analyze",
        files={"image": ("p.png", _png(), "image/png")},
        data={"caption": "x", "posted_date": "2999-01-01"},
    )
    assert resp.status_code == 400
    assert "future" in resp.json()["detail"]


def test_renamed_copy_is_matched_by_content_and_gated_by_date():
    """The illustrative flood image is in the shipped history index (earliest 2019-03-04). Upload
    it under an unrelated filename: the old filename lookup would have missed it entirely."""
    get_settings.cache_clear()
    image = ("holiday_snap_final_v2.png", _FLOOD.read_bytes(), "image/png")

    after = _recycled(client.post("/analyze", files={"image": image},
                                  data={"posted_date": "2026-09-01"}).json())
    assert after["status"] == "fired"
    assert "2019-03-04" in after["evidence"]

    before = _recycled(client.post("/analyze", files={"image": image},
                                   data={"posted_date": "2018-12-31"}).json())
    assert before["status"] == "clear"

    undated = _recycled(client.post("/analyze", files={"image": image}).json())
    assert undated["status"] == "fired"
    assert "posting date" in undated["plain_explanation"]


def test_unregistered_image_is_clear_not_not_assessed():
    """A readable image that is not in the index: the lookup ran and found nothing."""
    resp = client.post("/analyze", files={"image": ("p.png", _png(colour="navy"), "image/png")})
    assert _recycled(resp.json())["status"] == "clear"


def test_screenshot_of_a_known_image_is_found_through_the_api():
    """A screenshot defeats the hash; the keypoint stage still finds it, and says so."""
    from tests.eval.image_transforms import screenshot

    get_settings.cache_clear()
    buf = BytesIO()
    screenshot(Image.open(_FLOOD)).save(buf, format="PNG")
    body = client.post("/analyze", files={"image": ("screenshot_2026.png", buf.getvalue(), "image/png")},
                       data={"posted_date": "2026-09-01"}).json()
    flag = _recycled(body)
    assert flag["status"] == "fired"
    assert "2019-03-04" in flag["evidence"]
    assert "image details that line up" in flag["evidence"]
