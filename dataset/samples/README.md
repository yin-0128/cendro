# Sample code corpus

A small, committed set of deliberately flawed code snippets across several languages. Each file
contains realistic, reviewable issues (security, correctness, performance, error handling).

These are the **input** to DPO data generation — feed them to the judge to produce
`{prompt, chosen, rejected}` pairs:

```bash
python scripts/generate_preferences.py \
  --input dataset/samples/ \
  --provider ollama --model qwen2.5-coder:14b \
  --output dataset/dpo_pairs.jsonl
```

| File | Primary issue |
|------|---------------|
| `py_sql_injection.py` | SQL injection via string building / f-strings |
| `py_cache_unbounded.py` | Unbounded cache + mutable default argument |
| `py_file_resource.py` | File handles never closed; no encoding |
| `py_path_traversal.py` | Path traversal on user-supplied filenames |
| `py_password_handling.py` | MD5 password hashing; non-constant-time compare |
| `py_division_stats.py` | Division by zero (empty list / zero base) |
| `py_race_condition.py` | Unsynchronized shared counter across threads |
| `py_n_plus_one.py` | N+1 query pattern in a loop |
| `py_input_validation.py` | Unvalidated int parsing of request params |
| `js_async_sequential.js` | Off-by-one loop + sequential awaits |
| `ts_null_deref.ts` | Optional deref without a guard; unsafe `!` |
| `go_ignored_error.go` | Discarded errors from I/O and unmarshal |

> `dataset/raw/` (git-ignored) is for your *own* private code samples. This `samples/` directory
> is the shareable, reproducible starter corpus.
