"""API tests. Ollama is mocked so these run in CI with no GPU and no model."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from model import inference

client = TestClient(app)


def test_health_ok_when_ollama_reachable():
    with patch.object(inference, "ping", return_value=True):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ollama"] is True
    assert body["model"]


def test_health_degraded_when_ollama_down():
    with patch.object(inference, "ping", return_value=False):
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_review_returns_review():
    fake = inference.ReviewResult(review="Use a dict lookup - O(1).", model="test", latency_ms=12)
    with patch.object(inference, "review_code", return_value=fake) as m:
        resp = client.post(
            "/review",
            json={"code": "def f(): return [i for i in range(10**6)]", "language": "python"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "review" in body and len(body["review"]) > 0
    assert body["model"] == "test"
    assert body["latency_ms"] == 12
    m.assert_called_once()


def test_review_rejects_empty_code():
    resp = client.post("/review", json={"code": "", "language": "python"})
    assert resp.status_code == 422  # pydantic min_length


def test_review_503_when_inference_fails():
    with patch.object(inference, "review_code", side_effect=RuntimeError("ollama down")):
        resp = client.post("/review", json={"code": "x = 1", "language": "python"})
    assert resp.status_code == 503


def test_models_lists_models():
    with patch.object(inference, "list_models", return_value=["qwen2.5-coder:3b"]):
        resp = client.get("/models")
    assert resp.status_code == 200
    assert resp.json()["models"] == ["qwen2.5-coder:3b"]
