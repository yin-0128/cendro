"""Build a high-quality, hand-authored DPO dataset from dataset/samples/.

Unlike scripts/generate_preferences.py (which asks a local LLM judge to write the reviews and
can miss subtle bugs), the `chosen`/`rejected` reviews here are written by hand and grounded in
authoritative best practices (OWASP, language docs). This is the project's "gold" dataset.

Each entry is keyed by the sample filename; the script reads the actual code from
dataset/samples/<file>, builds the exact Cendro review prompt (api/prompts.py), and emits
{prompt, chosen, rejected} to dataset/gold_pairs.jsonl.

    python scripts/build_gold_dataset.py

`chosen`  = a specific review naming the real bug + impact + fix.
`rejected` = a deliberately generic, low-signal review (what we train AWAY from).
"""

from __future__ import annotations

import json
from pathlib import Path

from api.prompts import build_review_messages

SAMPLES = Path(__file__).resolve().parents[1] / "dataset" / "samples"
OUT = Path(__file__).resolve().parents[1] / "dataset" / "gold_pairs.jsonl"

REVIEWS: dict[str, dict[str, str]] = {
    "py_sql_injection.py": {
        "chosen": "- **SQL injection (critical).** Both functions build SQL by concatenating/interpolating untrusted input (`'... = '\\'' + name`, `f\"... id = {user_id}\"`). Input like `' OR '1'='1` dumps every row and `'; DROP TABLE users;--` is catastrophic.\n- **Fix:** use parameterized queries and let the driver bind values:\n```python\ncursor.execute(\"SELECT id, email FROM users WHERE name = ?\", (name,))\nconn.execute(\"DELETE FROM users WHERE id = ?\", (user_id,))\n```\nNever build SQL with string formatting on user input.",
        "rejected": "The queries could be cleaner. Consider using parameterized queries and adding some error handling.",
    },
    "py_cache_unbounded.py": {
        "chosen": "- **Unbounded memory growth.** `_cache` never evicts: every distinct `symbol` stays forever, and stale entries past 60s are kept (just not returned). Over time this leaks memory. Use a bounded LRU (`functools.lru_cache` with `maxsize`, or `cachetools.TTLCache`).\n- **Mutable default argument.** `add_listener(event, listeners=[])` shares one list across all calls — every call without an explicit list mutates the same object. Use `listeners=None` then `listeners = [] if listeners is None else listeners`.\n- **Thread safety.** `_cache` is a module global mutated without a lock; concurrent callers can race.",
        "rejected": "Caching looks fine overall. You might tweak the expiry logic and add a docstring.",
    },
    "py_file_resource.py": {
        "chosen": "- **Leaked file handle.** `load_settings` opens the file but never closes it; on an exception or under non-refcounting runtimes the descriptor leaks. Use a context manager.\n- **No encoding specified**, so it depends on the platform default (breaks on Windows cp1252 for UTF-8 files). Pass `encoding=\"utf-8\"`.\n```python\ndef load_settings(path):\n    with open(path, encoding=\"utf-8\") as f:\n        return json.load(f)\n```\n- `write_lines` closes explicitly but still leaks if `write` raises — prefer `with open(...)` there too.",
        "rejected": "Reading and writing files here is straightforward. Maybe add error handling in case the file is missing.",
    },
    "py_path_traversal.py": {
        "chosen": "- **Path traversal (critical).** `filename` is joined onto `UPLOAD_DIR` unchecked, so `../../etc/passwd` or an absolute path escapes the upload directory for both read and write. `os.path.join(base, \"/etc/passwd\")` returns `/etc/passwd`.\n- **Fix:** strip directories and confirm the resolved path stays inside the base:\n```python\nname = os.path.basename(filename)\npath = os.path.realpath(os.path.join(UPLOAD_DIR, name))\nif not path.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):\n    raise ValueError(\"invalid filename\")\n```",
        "rejected": "Consider validating the filename and handling the case where the file does not exist.",
    },
    "js_async_sequential.js": {
        "chosen": "- **Off-by-one bug.** The loop condition `i <= userIds.length` reads `userIds[userIds.length]` (== `undefined`) on the last iteration, so `fetchUser(undefined)` runs and a bad element is pushed. Use `i < userIds.length`, or better `for (const id of userIds)`.\n- **Needlessly sequential.** `await fetchUser(...)` inside the loop waits for each request before starting the next — N requests take N× the latency. Run them concurrently: `return Promise.all(userIds.map(fetchUser))`.\n- `totalCents` multiplies floats (`price * 100`); for money compute in integer cents to avoid rounding.",
        "rejected": "The async logic works. You could use Promise.all and maybe add error handling for failed fetches.",
    },
    "ts_null_deref.ts": {
        "chosen": "- **Unguarded optional access.** `user` is `User | undefined` and `profile` is optional, but `greeting` does `user.profile.displayName` — `greeting()` throws `Cannot read properties of undefined`. Under `strictNullChecks` it won't even compile. Guard it: `if (!user?.profile) return \"Hello, guest\";`.\n- **Unsafe non-null assertion.** `firstAdmin` uses `find(...)!`; if no admin exists, `find` returns `undefined` and the `!` hides a runtime crash from the type system. Return `User | undefined` and handle the empty case instead of asserting.",
        "rejected": "TypeScript should help here. Consider adding a null check to be safe.",
    },
    "go_ignored_error.go": {
        "chosen": "- **Discarded errors.** `os.ReadFile`'s error is dropped with `_`, and `json.Unmarshal`'s return is ignored entirely. A missing/unreadable file or malformed JSON silently yields a zero-value `Config{Port: 0}`, and the failure surfaces far away. This is exactly what `errcheck`/`go vet` flag.\n- **Fix:** return the error idiomatically:\n```go\nfunc Load(path string) (Config, error) {\n    data, err := os.ReadFile(path)\n    if err != nil { return Config{}, err }\n    var c Config\n    if err := json.Unmarshal(data, &c); err != nil { return Config{}, err }\n    return c, nil\n}\n```",
        "rejected": "In Go it's good practice to handle errors. You may want to look into that.",
    },
    "py_password_handling.py": {
        "chosen": "- **Broken password hashing (critical).** MD5 is fast and cryptographically broken — GPUs brute-force billions/sec — and there's no salt, so identical passwords share a hash and rainbow tables apply. Per the [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html), use **Argon2id** (e.g. `argon2-cffi`), or bcrypt (work factor >= 10) if Argon2 isn't available — both salt automatically.\n- **Non-constant-time compare.** `==` on the digests is timing-variable; use the library's `verify`, or `hmac.compare_digest` for raw comparisons.",
        "rejected": "MD5 is a bit old. Consider a more secure hashing algorithm for better security.",
    },
    "py_division_stats.py": {
        "chosen": "- **Division by zero.** `average([])` raises `ZeroDivisionError` because `len(numbers)` is 0; `percent_change(0, new)` divides by `old == 0`. Both crash on realistic inputs.\n- **Fix:** decide the contract and guard explicitly, e.g. `if not numbers: return 0.0` (or raise a clear `ValueError`), and special-case `old == 0` in `percent_change` (return `inf`/`None`/raise, but don't let it blow up).",
        "rejected": "The math is simple and correct. A docstring would help readability.",
    },
    "py_race_condition.py": {
        "chosen": "- **Data race / lost updates.** `counter += 1` is read-modify-write, which is *not* atomic in Python (the GIL can switch threads between the read and the write). With many threads the final `counter` is non-deterministic and lower than expected.\n- **Fix:** guard the shared state with a lock (or use `itertools.count`/an atomic):\n```python\nlock = threading.Lock()\nwith lock:\n    counter += 1\n```",
        "rejected": "Threading is tricky. You might want to synchronize access to the counter.",
    },
    "py_n_plus_one.py": {
        "chosen": "- **N+1 query pattern.** For each customer id this runs two queries inside the loop, so 500 customers = ~1000 round-trips — latency and DB load scale linearly with the input.\n- **Fix:** batch with a single query per table using `IN (...)` (or a `JOIN`) and group in memory:\n```python\ncustomers = db.query(\"SELECT id, name FROM customers WHERE id IN (...)\", customer_ids)\norders = db.query(\"SELECT customer_id, total FROM orders WHERE customer_id IN (...)\", customer_ids)\n```\nThen assemble the summaries from those two result sets.",
        "rejected": "The loop builds summaries correctly. Consider optimizing the queries if performance matters.",
    },
    "py_input_validation.py": {
        "chosen": "- **Unvalidated input crashes / abuses the DB.** `int(params.get(\"page\", \"1\"))` raises `ValueError` (HTTP 500) on `page=abc`; nothing bounds `page`/`size`, so `page=0` yields a negative offset and `size=100000000` lets a client request the whole table. `parse_ids` likewise throws on any non-int token.\n- **Fix:** parse defensively and clamp: catch the `ValueError` (return 400), force `page = max(1, page)`, and cap `size` (e.g. `min(size, 100)`). For `parse_ids`, skip/validate bad tokens explicitly.",
        "rejected": "Parsing query params here is fine. Maybe add validation for unexpected values.",
    },
    "py_command_injection.py": {
        "chosen": "- **OS command injection (critical).** `os.system(\"ping -c 1 \" + host)` and `subprocess.call(f\"... {folder}\", shell=True)` pass user input through a shell, so `host=\"x; rm -rf /\"` executes arbitrary commands.\n- **Fix:** never use `shell=True` with untrusted input — pass an argument list so there's no shell to inject into:\n```python\nsubprocess.run([\"ping\", \"-c\", \"1\", host], check=True)\nsubprocess.run([\"tar\", \"-czf\", \"backup.tgz\", folder], check=True)\n```",
        "rejected": "Running shell commands works. You could add error handling around the subprocess calls.",
    },
    "py_pickle_deser.py": {
        "chosen": "- **Insecure deserialization (critical, RCE).** `pickle.loads` on an attacker-controlled cookie payload executes arbitrary code during unpickling — a remote code execution hole, not just a parsing risk. Pickle is never safe on untrusted data.\n- **Fix:** use a data-only format (`json`) for anything crossing a trust boundary, and if you must round-trip objects in a cookie, sign them (e.g. `itsdangerous`) and verify the signature before deserializing.",
        "rejected": "Pickle is convenient here. Consider validating the input before loading it.",
    },
    "py_weak_random_token.py": {
        "chosen": "- **Predictable security tokens.** `random` is a Mersenne-Twister PRNG, not cryptographically secure — its output is predictable from observed values, so reset tokens and session ids can be guessed/forged. A 6-digit `randint` session id is also tiny (1M space, brute-forceable).\n- **Fix:** use the `secrets` module for anything security-sensitive:\n```python\nimport secrets\ndef make_reset_token(n=32): return secrets.token_urlsafe(n)\ndef make_session_id(): return secrets.token_hex(16)\n```",
        "rejected": "The token generation works. You might increase the length for more entropy.",
    },
    "py_string_concat_loop.py": {
        "chosen": "- **HTML/XSS injection.** `render_rows` interpolates cell values straight into HTML with no escaping, so a cell containing `<script>` is injected into the page. Escape every value (`html.escape(str(cell))`) or use a templating engine with autoescaping.\n- **Quadratic string building.** Repeated `html += ...` rebuilds the string each iteration (O(n²) for large inputs). Accumulate into a list and `\"\".join(...)` once. Same for `join_paths` — prefer `\"/\".join(parts)`.",
        "rejected": "Building the HTML string works. Using join could be a bit faster.",
    },
    "py_regex_in_loop.py": {
        "chosen": "- **Recompiling the regex every iteration.** `re.compile(...)` inside the loop discards Python's compile cache benefit and wastes work proportional to the number of documents. Compile once at module scope and reuse the pattern.\n- **`!= None` is a style/correctness smell.** Compare identity with `is not None`. Also note `re.match` only anchors at the start; if you mean the whole string, use `re.fullmatch` (here the trailing `$` makes it work, but `fullmatch` is clearer).",
        "rejected": "The regex usage is okay. Moving the compile outside the loop could help performance.",
    },
    "py_float_money.py": {
        "chosen": "- **Floating-point money.** Using `float` for currency accumulates representation errors (`0.1 + 0.2 != 0.3`), so totals, tax, and split bills drift by cents over time — unacceptable for financial code.\n- **Fix:** represent money as integer cents, or use `decimal.Decimal` with explicit rounding:\n```python\nfrom decimal import Decimal, ROUND_HALF_UP\ntax = (price * Decimal(\"1.07\")).quantize(Decimal(\"0.01\"), ROUND_HALF_UP)\n```\nAlso, `split_bill` returns equal shares that may not sum back to `amount` after rounding — allocate the remainder cents.",
        "rejected": "The arithmetic looks correct. Rounding to 2 decimals is a nice touch.",
    },
    "py_dict_mutation.py": {
        "chosen": "- **Mutating a dict while iterating it.** `drop_inactive` calls `del users[uid]` during `for ... in users.items()`, which raises `RuntimeError: dictionary changed size during iteration`. Iterate over a snapshot: `for uid in list(users): ...`, or build a new dict comprehension.\n- **`KeyError` on first count.** `counts[item] += 1` throws when `item` isn't already a key. Use `counts.get(item, 0) + 1` or a `collections.defaultdict(int)` / `Counter`.",
        "rejected": "The functions update the dictionaries as intended. Consider edge cases with missing keys.",
    },
    "py_no_timeout.py": {
        "chosen": "- **No request timeout.** `requests.get`/`post` with no `timeout` will block *forever* if the server stalls — one hung dependency can exhaust your worker pool and take the service down. `requests` has no default timeout.\n- **Fix:** always pass an explicit timeout (and ideally handle `requests.Timeout`):\n```python\nresp = requests.get(url, timeout=10)\nresp.raise_for_status()\n```",
        "rejected": "The HTTP calls work. You may want to add a timeout and check the status code.",
    },
    "py_identity_compare.py": {
        "chosen": "- **`is` used for value comparison.** `user_id is 0` and `code is 200` test object identity, not equality. They only ever appear to work because CPython caches small ints — it's unreliable across values/implementations and CPython even warns (`SyntaxWarning: \"is\" with a literal`). A larger code like `is 500` can be `False` even when equal.\n- **Fix:** use `==` for value checks: `user_id == 0`, `code == 200`.",
        "rejected": "The comparisons mostly work. Using == might be slightly clearer.",
    },
    "py_datetime_naive.py": {
        "chosen": "- **Naive datetimes / timezone bug.** `datetime.now()` returns a naive *local* time and `strptime` produces naive values, so comparisons silently assume the server timezone. If `issued_at` is UTC (typical for tokens), expiry is wrong by the UTC offset — tokens expire early/late.\n- **Fix:** work in timezone-aware UTC end to end:\n```python\nfrom datetime import datetime, timezone\nnow = datetime.now(timezone.utc)\nissued = datetime.fromisoformat(issued_at_iso)  # ensure it carries tzinfo\n```\nAlso prefer `fromisoformat` over a hard-coded `strptime` format that breaks on `Z`/offsets.",
        "rejected": "The date handling works for the expected format. Consider timezone handling.",
    },
    "js_eval_input.js": {
        "chosen": "- **Code injection (critical).** `eval(expr)` and `new Function(userCode)` execute attacker-controlled strings with full app privileges — arbitrary code execution from a query string. There is no safe way to `eval` untrusted input.\n- **Fix:** don't evaluate user input. For arithmetic use a real expression parser (e.g. a small math-expression library with no function access); for data use `JSON.parse`. If you truly need sandboxed execution, run it out-of-process with strict limits.",
        "rejected": "eval is flexible here. Be careful with the input and validate it first.",
    },
    "js_loose_equality.js": {
        "chosen": "- **Loose equality coercion bugs.** `==`/`!=` coerce types, producing surprising matches: `findById` with `id == i.id` treats `\"5\" == 5` as equal (and returns the wrong row if ids mix string/number), and `cart.count != 0` is true for `\"0\"`/`false`. Use strict `===`/`!==` everywhere.\n- Note `cart.count == null` is the one intentional loose check (matches both `null` and `undefined`); keep it but add a comment, or write `cart.count == null` explicitly as the documented idiom.",
        "rejected": "The comparisons work in most cases. Using === is generally recommended.",
    },
    "js_promise_no_await.js": {
        "chosen": "- **Unawaited async work.** `saveUser` returns `{ok: true}` immediately without awaiting `insertOne`, so it reports success before the write happens — and a rejected promise becomes an unhandled rejection. `chargeAndEmail` fires `charge` and `sendReceipt` without awaiting, so failures are silently lost and the receipt may send before the charge settles.\n- **Fix:** make them `async`, `await` each call, and wrap in `try/catch` so errors propagate:\n```js\nasync function saveUser(db, user) {\n  await db.collection(\"users\").insertOne(user);\n  return { ok: true };\n}\n```",
        "rejected": "The functions perform the operations. You could add await and error handling.",
    },
    "java_resource_leak.java": {
        "chosen": "- **Leaked reader.** `firstLine` opens a `BufferedReader`/`FileReader` and never closes it — on every call (and especially on exception) a file handle leaks. Use try-with-resources so it always closes:\n```java\ntry (BufferedReader r = new BufferedReader(new FileReader(path))) {\n    return r.readLine();\n}\n```\n- **String compared with `==`.** `answer == \"yes\"` compares references, not contents, so it's usually `false` for runtime strings. Use `\"yes\".equals(answer)` (constant first, null-safe).",
        "rejected": "The file reading works. In Java you might consider closing resources.",
    },
    "rust_unwrap.rs": {
        "chosen": "- **Panics on the error path.** `fs::read_to_string(path).unwrap()` panics if the file is missing/unreadable, and `.parse::<i32>().unwrap()` panics on non-numeric content — a malformed file crashes the program. `first_word(\"\")` also panics because `next().unwrap()` hits `None`.\n- **Fix:** propagate errors with `?` and return `Result`, or handle the empty case:\n```rust\nfn read_count(path: &str) -> Result<i32, Box<dyn std::error::Error>> {\n    Ok(fs::read_to_string(path)?.trim().parse()?)\n}\n```\nReserve `unwrap()` for cases that are truly impossible.",
        "rejected": "Using unwrap is concise. You may want to handle errors more gracefully.",
    },
    "go_loopvar_goroutine.go": {
        "chosen": "- **Loop-variable capture in goroutines.** Each goroutine closes over `item`; before Go 1.22 the loop variable is shared across iterations, so the goroutines often all print the *last* item (a classic data race). On Go 1.22+ the variable is per-iteration and this is fixed ([Go 1.22 loopvar](https://go.dev/blog/loopvar-preview)) — but code that must build on older toolchains is buggy. Make it explicit and version-independent:\n```go\nfor _, item := range items {\n    item := item // pin per iteration (no-op on 1.22+, required before)\n    go func() { defer wg.Done(); fmt.Println(\"processing\", item) }()\n}\n```\n- Confirm the module's `go` directive in `go.mod` is `1.22`+ to get the safe semantics.",
        "rejected": "The goroutines run concurrently. Be careful with the loop variable inside the closure.",
    },
    "bash_unquoted.sh": {
        "chosen": "- **Unquoted variables + dangerous `rm`.** `rm -rf $DIR/*` is unquoted, so if `$DIR` is empty/unset it becomes `rm -rf /*`, and a value with spaces/globs word-splits into unintended paths — potential catastrophic deletion. Quote and fail fast: add `set -euo pipefail`, then `rm -rf -- \"${DIR:?dir required}\"/*`.\n- **Parsing `ls`.** `for f in $(ls $1)` breaks on filenames with spaces and is unquoted; iterate with a glob instead: `for f in \"$1\"/*; do cp -- \"$f\" /backup/; done`.",
        "rejected": "The script removes and copies files. Quoting variables is usually a good idea.",
    },
    "py_broad_retry.py": {
        "chosen": "- **Infinite retry, no backoff, over-broad catch.** `call_with_retry` loops forever with a flat 1s sleep and retries on *any* `Exception` — so a permanent error (bad arg, auth failure) retries endlessly and hangs the caller, while `KeyboardInterrupt`-style control flow is swallowed.\n- **Fix:** bound the attempts, back off exponentially, and only retry errors that are actually transient:\n```python\nfor attempt in range(max_attempts):\n    try: return fn(*args)\n    except TransientError:\n        time.sleep(2 ** attempt)\nraise\n```\n- `safe_int` similarly hides real bugs by catching `Exception`; catch `(ValueError, TypeError)`.",
        "rejected": "The retry logic works. You might add a maximum number of attempts.",
    },
    "py_missing_auth_check.py": {
        "chosen": "- **Broken access control / IDOR (critical).** Both handlers act on a client-supplied `id` with no authorization check, so any authenticated user can delete *anyone's* account or read *anyone's* invoice just by changing the id. Parameterized SQL prevents injection but does nothing for authorization.\n- **Fix:** verify the current user owns (or may access) the resource before acting — scope the query to the session user, e.g. `DELETE FROM accounts WHERE id = ? AND owner_id = ?` with `(user_id, current_user.id)`, and return 403/404 if no row matches.",
        "rejected": "The handlers use parameterized queries, which is good. Consider adding authentication.",
    },
    "ts_floating_promise.ts": {
        "chosen": "- **Floating promises in `forEach`.** `this.queue.forEach((file) => { handler(file); })` ignores the returned promises, so `process` returns before any upload finishes, rejections become unhandled, and `this.queue = []` clears the queue while work is still in flight. `Array.forEach` is not async-aware.\n- **Fix:** await the work explicitly:\n```ts\nasync process(handler: (f: string) => Promise<void>): Promise<void> {\n  const files = this.queue;\n  this.queue = [];\n  await Promise.all(files.map(handler)); // or a for...of with await for sequential\n}\n```",
        "rejected": "The process method iterates the queue. You may want to handle the promises.",
    },
}


def main() -> None:
    written, missing = 0, []
    with open(OUT, "w", encoding="utf-8") as out:
        for filename, review in REVIEWS.items():
            path = SAMPLES / filename
            if not path.exists():
                missing.append(filename)
                continue
            code = path.read_text(encoding="utf-8")
            language = _language_for(filename)
            prompt = build_review_messages(code, language=language)[1]["content"]
            row = {"prompt": prompt, "chosen": review["chosen"], "rejected": review["rejected"]}
            out.write(json.dumps(row) + "\n")
            written += 1
    print(f"Wrote {written} gold pairs to {OUT}")
    if missing:
        print(f"WARNING: {len(missing)} reviews had no matching sample file: {missing}")


_EXT_LANG = {
    ".py": "python", ".ts": "typescript", ".js": "javascript", ".go": "go",
    ".rs": "rust", ".java": "java", ".sh": "bash",
}


def _language_for(filename: str) -> str:
    return _EXT_LANG.get(Path(filename).suffix.lower(), "text")


if __name__ == "__main__":
    main()
