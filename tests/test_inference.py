"""Inference wrapper tests. The ollama client is mocked, so these run with no GPU/model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from model import inference


def test_review_code_returns_stripped_result():
    fake = MagicMock()
    fake.chat.return_value = {"message": {"content": "  Bug: off-by-one in the loop.  "}}
    with patch.object(inference, "_client", return_value=fake):
        result = inference.review_code("for i in range(n): a[i+1]", language="python")
    assert result.review == "Bug: off-by-one in the loop."
    assert result.model == inference.DEFAULT_MODEL
    assert result.latency_ms >= 0
    fake.chat.assert_called_once()


def test_review_code_respects_model_override():
    fake = MagicMock()
    fake.chat.return_value = {"message": {"content": "ok"}}
    with patch.object(inference, "_client", return_value=fake):
        result = inference.review_code("x = 1", model="custom:latest")
    assert result.model == "custom:latest"
    assert fake.chat.call_args.kwargs["model"] == "custom:latest"


def test_list_models_parses_dict_shapes():
    fake = MagicMock()
    fake.list.return_value = {"models": [{"model": "a:1"}, {"name": "b:2"}, {}]}
    with patch.object(inference, "_client", return_value=fake):
        assert inference.list_models() == ["a:1", "b:2"]


def test_ping_reflects_client_health():
    ok = MagicMock()
    ok.list.return_value = {"models": []}
    with patch.object(inference, "_client", return_value=ok):
        assert inference.ping() is True

    down = MagicMock()
    down.list.side_effect = RuntimeError("daemon down")
    with patch.object(inference, "_client", return_value=down):
        assert inference.ping() is False
