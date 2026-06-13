# Cendro GitHub Action

Reviews the changed files in a pull request and posts a summary comment.

## Backends

| Backend | Where the model runs | Use when |
|---------|----------------------|----------|
| `anthropic` (default) | Claude API (`claude-opus-4-8`) | GitHub-hosted runners; you accept sending diffs to the Claude API. |
| `ollama` | Your self-hosted Cendro/Ollama server | Privacy-first: code stays on your infra. Requires a **self-hosted runner** that can reach `server-url`. |

> The `anthropic` backend sends changed-file diffs to the Claude API. If your reason for using
> Cendro is that code must not leave your machines, use the `ollama` backend on a self-hosted runner.

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `github-token` | yes | — | Token to read the PR diff and post the comment. `${{ secrets.GITHUB_TOKEN }}` is enough. |
| `backend` | no | `anthropic` | `anthropic` or `ollama`. |
| `model` | no | backend default | `claude-opus-4-8` / `qwen2.5-coder:7b`. |
| `server-url` | no | `http://localhost:11434` | Ollama server URL (ollama backend). |
| `anthropic-api-key` | no | — | Anthropic API key (anthropic backend). |
| `max-files` | no | `25` | Max changed files reviewed per PR. |

## Usage (Claude API)

```yaml
# .github/workflows/code-review.yml
name: Cendro Code Review
on: [pull_request]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: yin-0128/cendro/github-action@main
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          backend: anthropic
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Usage (self-hosted Ollama)

```yaml
jobs:
  review:
    runs-on: [self-hosted]      # a runner that can reach your Cendro server
    steps:
      - uses: yin-0128/cendro/github-action@main
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          backend: ollama
          server-url: http://localhost:11434
          model: qwen2.5-coder:7b
```

## Notes

- Needs `pull-requests: write` permission to post the comment.
- Binary/lock/minified files are skipped; removed files are ignored.
- The review prompt mirrors [`api/prompts.py`](../api/prompts.py) so CI reviews match local ones.
- This action is **not a substitute for human review** — it surfaces likely issues per file.
