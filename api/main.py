"""Cendro FastAPI inference server.

Transport only -- all model logic lives in ``model.inference``. Endpoints match
docs/ARCHITECTURE.md: POST /review, GET /health, GET /models.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import inference

app = FastAPI(
    title="Cendro",
    description="Local AI code reviewer - your code never leaves your machine.",
    version="0.1.0",
)


class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=1, description="The source code to review.")
    language: str = Field("python", description="Programming language of the snippet.")
    focus: str | None = Field(None, description="Optional aspect to focus the review on.")
    model: str | None = Field(None, description="Override the Ollama model to use.")


class ReviewResponse(BaseModel):
    review: str
    model: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    model: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    reachable = inference.ping()
    return HealthResponse(
        status="ok" if reachable else "degraded",
        ollama=reachable,
        model=inference.DEFAULT_MODEL,
    )


@app.get("/models")
def models() -> dict[str, list[str]]:
    try:
        return {"models": inference.list_models()}
    except Exception as exc:  # surface a clean 503 instead of a stack trace
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {exc}") from exc


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest) -> ReviewResponse:
    try:
        result = inference.review_code(
            code=req.code,
            language=req.language,
            focus=req.focus,
            model=req.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc
    return ReviewResponse(
        review=result.review,
        model=result.model,
        latency_ms=result.latency_ms,
    )
