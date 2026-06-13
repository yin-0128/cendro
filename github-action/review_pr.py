"""Cendro PR reviewer — runs inside the Cendro GitHub Action.

Reads the pull request diff, asks a reviewer backend for a focused review of each changed
file, and posts a single summary comment on the PR. Self-contained on purpose: this runs in
the consumer's repository checkout, so it does not import the `cendro` package. The review
prompt below mirrors api/prompts.py — keep them in sync.

Configured entirely via environment variables (set by action.yml):
    GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_EVENT_PATH   (provided by Actions)
    CENDRO_BACKEND        anthropic | ollama
    CENDRO_MODEL          backend-specific model id (optional)
    CENDRO_SERVER_URL     ollama server URL (ollama backend)
    ANTHROPIC_API_KEY     (anthropic backend)
    CENDRO_MAX_FILES      cap on reviewed files
"""

from __future__ import annotations

import json
import os
import sys

import requests

# --- Mirror of api/prompts.py (kept in sync intentionally; this file ships standalone) ---
SYSTEM_PROMPT = """You are Cendro, a senior software engineer doing a focused code review.

Be opinionated and specific. For every issue you raise:
- Name the concrete problem (not "consider improving X").
- Explain the impact (correctness bug, performance at scale, security, readability).
- Give a fix -- a short code snippet when it helps.

Rules:
- Prioritize correctness and security bugs over style.
- If something is genuinely fine, say so briefly instead of inventing problems.
- Be concise. No filler, no restating the code back.

Format your answer as a short bulleted list of findings, most important first. If there are
no real issues, say so in one line."""

GITHUB_API = "https://api.github.com"
SKIP_SUFFIXES = (".lock", ".min.js", ".map", ".svg", ".png", ".jpg", ".gif", ".pdf")


def _gh_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _user_prompt(filename: str, patch: str) -> str:
    return (
        f"Review the changes to `{filename}` shown in this unified diff:"
        f"\n\n```diff\n{patch}\n```"
    )


def pr_number_from_event() -> int | None:
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        event = json.load(fh)
    pr = event.get("pull_request") or {}
    return pr.get("number") or event.get("number")


def list_changed_files(repo: str, pr: int, token: str) -> list[dict]:
    files, page = [], 1
    while True:
        resp = requests.get(
            f"{GITHUB_API}/repos/{repo}/pulls/{pr}/files",
            headers=_gh_headers(token),
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        files.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return files


def review_with_anthropic(filename: str, patch: str, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    user = _user_prompt(filename, patch)
    resp = client.messages.create(
        model=model or "claude-opus-4-8",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def review_with_ollama(filename: str, patch: str, model: str, server_url: str) -> str:
    user = _user_prompt(filename, patch)
    resp = requests.post(
        f"{server_url.rstrip('/')}/api/chat",
        json={
            "model": model or "qwen2.5-coder:3b",
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def post_comment(repo: str, pr: int, token: str, body: str) -> None:
    resp = requests.post(
        f"{GITHUB_API}/repos/{repo}/issues/{pr}/comments",
        headers=_gh_headers(token),
        json={"body": body},
        timeout=30,
    )
    resp.raise_for_status()


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    backend = os.environ.get("CENDRO_BACKEND", "anthropic")
    model = os.environ.get("CENDRO_MODEL", "")
    server_url = os.environ.get("CENDRO_SERVER_URL", "http://localhost:11434")
    max_files = int(os.environ.get("CENDRO_MAX_FILES", "25"))

    if not token or not repo:
        print("GITHUB_TOKEN and GITHUB_REPOSITORY are required.", file=sys.stderr)
        return 1

    pr = pr_number_from_event()
    if pr is None:
        print("Not a pull_request event — nothing to review.")
        return 0

    files = list_changed_files(repo, pr, token)
    reviewable = [
        f for f in files
        if f.get("patch") and not f["filename"].endswith(SKIP_SUFFIXES) and f["status"] != "removed"
    ][:max_files]

    if not reviewable:
        post_comment(repo, pr, token, "🔍 **Cendro**: no reviewable code changes in this PR.")
        return 0

    sections = []
    for f in reviewable:
        filename, patch = f["filename"], f["patch"]
        try:
            if backend == "ollama":
                review = review_with_ollama(filename, patch, model, server_url)
            else:
                review = review_with_anthropic(filename, patch, model)
        except Exception as exc:
            review = f"_Review failed for this file: {exc}_"
        sections.append(f"### `{filename}`\n\n{review}")

    body = "## 🔍 Cendro Code Review\n\n" + "\n\n---\n\n".join(sections)
    body += (
        "\n\n<sub>Cendro reviews changed files individually. "
        "Not a substitute for human review.</sub>"
    )
    post_comment(repo, pr, token, body)
    print(f"Posted review covering {len(reviewable)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
