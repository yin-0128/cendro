"""Ollama-backed inference wrapper.

Keeps all model/transport logic out of the FastAPI layer so the API stays a thin HTTP shell.
The same ``review_code`` function is what the CLI's ``cendro review`` calls.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from api.prompts import build_review_messages

DEFAULT_MODEL = os.environ.get("CENDRO_MODEL", "qwen2.5-coder:7b")


@dataclass
class ReviewResult:
    review: str
    model: str
    latency_ms: int


def _client():
    """Import the ollama client lazily so importing this module never requires it.

    This keeps unit tests (which mock the client) and the API import path cheap, and gives a
    clear error only when inference is actually attempted without ollama installed.
    """
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "The 'ollama' package is required for inference. Install it with `pip install ollama` "
            "and make sure the Ollama daemon is running (https://ollama.ai)."
        ) from exc
    return ollama


def review_code(
    code: str,
    language: str = "python",
    focus: str | None = None,
    model: str | None = None,
) -> ReviewResult:
    """Generate a code review for ``code`` using the configured Ollama model."""
    model = model or DEFAULT_MODEL
    messages = build_review_messages(code, language=language, focus=focus)

    start = time.perf_counter()
    response = _client().chat(model=model, messages=messages)
    latency_ms = int((time.perf_counter() - start) * 1000)

    review = response["message"]["content"].strip()
    return ReviewResult(review=review, model=model, latency_ms=latency_ms)


def list_models() -> list[str]:
    """Return the names of models available locally in Ollama."""
    data = _client().list()
    models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
    names: list[str] = []
    for m in models:
        name = m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
        if name:
            names.append(name)
    return names


def ping() -> bool:
    """Return True if the Ollama daemon is reachable."""
    try:
        _client().list()
        return True
    except Exception:
        return False
