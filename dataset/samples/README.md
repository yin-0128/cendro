# Sample code corpus

A committed set of ~32 deliberately flawed code snippets across 8 languages (Python, JS, TS,
Go, Rust, Java, Bash). Each file contains realistic, reviewable issues. These are the **input**
to DPO data generation:

```bash
python scripts/generate_preferences.py \
  --input dataset/samples/ \
  --provider ollama --model qwen2.5-coder:14b \
  --output dataset/dpo_pairs.jsonl
```

## Coverage by category

| Category | Examples |
|----------|----------|
| **Injection** | SQL injection, OS command injection, `eval`/`Function` of user input |
| **Unsafe deserialization** | `pickle.loads` of untrusted data |
| **Auth / access control** | missing ownership/permission checks before delete & read |
| **Crypto / secrets** | MD5 password hashing, `random` for tokens/session ids |
| **Path / traversal** | unsanitized filenames into file paths |
| **Correctness** | off-by-one, null/None deref, `is` vs `==`, loose `==`, dict mutated while iterating, naive datetimes, division by zero |
| **Concurrency** | unsynchronized shared counter, Go loop-var capture in goroutines |
| **Performance** | string `+=` in loops, `re.compile` inside loops, N+1 queries |
| **Error handling** | bare/`except Exception` swallowing, infinite blind retry, discarded Go errors |
| **Resource leaks** | unclosed files (Python/Java), missing request timeouts |
| **Async** | sequential awaits, floating/unawaited promises (JS/TS) |
| **Money** | `float` for currency |
| **Shell** | unquoted variables, `rm -rf $VAR` |

Add more files here (any recognized extension: `.py .js .ts .go .rs .java .rb .c .cpp .sh .sql
.cs .kt .php`) and rerun the generator to grow the dataset.

> `dataset/raw/` (git-ignored) is for your *own* private code. This `samples/` directory is the
> shareable, reproducible starter corpus.
