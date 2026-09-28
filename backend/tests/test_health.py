"""Smoke test: the app starts and /health responds."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "aegis"
    assert "version" in body
    # The main settings are included so you can see how the server is configured.
    assert "reverse_image_mode" in body["config"]
    assert "use_llm" in body["config"]
    assert body["config"]["caption_match_method"] in ("meaning", "overlap")
