"""Shared code-review prompt.

This is the single source of truth for how Cendro is asked to review code. The MVP (Ollama)
and the DPO training pipeline both import from here so that serving and fine-tuning stay aligned
-- if you change the persona, change it once.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are Cendro, a senior software engineer doing a focused code review.

Be opinionated and specific. For every issue you raise:
- Name the concrete problem (not "consider improving X").
- Explain the impact (correctness bug, performance at scale, security, readability).
- Give a fix -- a short code snippet when it helps.

Rules:
- Prioritize correctness and security bugs over style.
- If something is genuinely fine, say so briefly instead of inventing problems.
- Be concise. No filler, no restating the code back.
- Reference line/identifier names when possible.

Format your answer as a short bulleted list of findings, most important first. If there are no
real issues, say so in one line.
"""

_USER_TEMPLATE = """Review the following {language} code.{focus_clause}

```{language}
{code}
```"""


def build_review_messages(
    code: str,
    language: str = "python",
    focus: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for a code review request.

    Returns a list of ``{"role", "content"}`` dicts usable by Ollama's chat API and by the
    training data pipeline (the ``user`` content is what we store as the DPO ``prompt``).
    """
    focus_clause = f" Focus especially on: {focus}." if focus else ""
    user = _USER_TEMPLATE.format(
        language=language or "text",
        code=code.rstrip(),
        focus_clause=focus_clause,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
