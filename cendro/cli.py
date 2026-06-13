"""``cendro`` command-line interface.

Subcommands:
    cendro pull   [--model NAME]              Pull the model into Ollama.
    cendro serve  [--host H] [--port P] [--model NAME]
                                              Run the FastAPI server.
    cendro review PATH [--language L] [--focus F] [--model NAME]
                                              Review a file from the terminal.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from cendro import __version__

DEFAULT_MODEL = "qwen2.5-coder:3b"


def _resolve_model(arg_model: str | None) -> str:
    return arg_model or os.environ.get("CENDRO_MODEL", DEFAULT_MODEL)


def cmd_pull(args: argparse.Namespace) -> int:
    model = _resolve_model(args.model)
    print(f"Pulling '{model}' via Ollama...")
    try:
        return subprocess.call(["ollama", "pull", model])
    except FileNotFoundError:
        print("Ollama not found. Install it from https://ollama.ai", file=sys.stderr)
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    if args.model:
        os.environ["CENDRO_MODEL"] = args.model
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required to serve. Run `pip install -e .`", file=sys.stderr)
        return 1
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    # Imported here so `pull`/`serve` don't require the inference deps at import time.
    from model import inference

    try:
        with open(args.path, encoding="utf-8") as fh:
            code = fh.read()
    except OSError as exc:
        print(f"Could not read {args.path}: {exc}", file=sys.stderr)
        return 1

    language = args.language or _guess_language(args.path)
    try:
        result = inference.review_code(
            code=code, language=language, focus=args.focus, model=_resolve_model(args.model)
        )
    except Exception as exc:
        print(f"Review failed: {exc}", file=sys.stderr)
        return 1

    print(result.review)
    print(f"\n— {result.model} · {result.latency_ms} ms", file=sys.stderr)
    return 0


_EXT_LANG = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript",
    ".jsx": "javascript", ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp", ".sh": "bash", ".sql": "sql",
}


def _guess_language(path: str) -> str:
    _, ext = os.path.splitext(path)
    return _EXT_LANG.get(ext.lower(), "text")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cendro", description="Local AI code reviewer.")
    parser.add_argument("--version", action="version", version=f"cendro {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help="Pull the model into Ollama.")
    p_pull.add_argument("--model", help=f"Model to pull (default: {DEFAULT_MODEL}).")
    p_pull.set_defaults(func=cmd_pull)

    p_serve = sub.add_parser("serve", help="Run the FastAPI server.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--model", help="Model to serve.")
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload (dev).")
    p_serve.set_defaults(func=cmd_serve)

    p_review = sub.add_parser("review", help="Review a file.")
    p_review.add_argument("path", help="Path to the file to review.")
    p_review.add_argument("--language", help="Override detected language.")
    p_review.add_argument("--focus", help="Aspect to focus on (e.g. 'security').")
    p_review.add_argument("--model", help="Model to use.")
    p_review.set_defaults(func=cmd_review)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
