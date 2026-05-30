"""Smoke test for the Week-0 scaffold: the app boots and /health responds."""

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
    # Feature flags are surfaced so the running configuration is observable.
    assert "reverse_image_mode" in body["config"]
    assert "use_llm" in body["config"]
