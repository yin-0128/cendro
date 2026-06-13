"""Tests for the shared prompt builder (no external deps)."""

from __future__ import annotations

from api.prompts import build_review_messages


def test_messages_have_system_and_user():
    msgs = build_review_messages("x = 1", language="python")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "x = 1" in msgs[1]["content"]
    assert "python" in msgs[1]["content"]


def test_focus_is_included():
    msgs = build_review_messages("x = 1", language="python", focus="security")
    assert "security" in msgs[1]["content"]


def test_no_focus_clause_when_absent():
    msgs = build_review_messages("x = 1", language="python")
    assert "Focus especially" not in msgs[1]["content"]
