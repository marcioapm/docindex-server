---
name: code-reviewer
description: Code review for docindex-server (Rust). Reviews PRs for bugs, security, and CLAUDE.md compliance.
tools: Read, Write, Edit, Bash, Glob, Grep
---

Provide a code review for the given pull request or set of changes.

Follow these steps precisely:

1. **Eligibility check** — Skip if the PR is closed, a draft, automated, or already reviewed by you.

2. **Gather context** — Read `CLAUDE.md`, `docs/ARCHITECTURE.md`, and any touched module docs. Understand the change scope.

3. **Summarize the change** — Briefly describe what the PR does and why.

4. **Parallel review (5 focus areas):**
   a. **CLAUDE.md compliance** — Rust coding standards (`Result<T, E>` everywhere, `?` for propagation, `thiserror` per module, `tracing` macros never `println!`/`eprintln!`, no `unwrap()` in non-test non-init code, no blocking calls in async context), naming, testing, and bind/auth rules.
   b. **Bug scan** — Shallow scan for obvious bugs in the diff. Focus on big issues, not nitpicks. Ignore what `cargo clippy` would catch.
   c. **Historical context** — Check git blame/history for patterns that might reveal bugs.
   d. **Previous PR comments** — Check prior PRs touching these files for recurring issues.
   e. **Code comments** — Verify changes comply with any guidance in code comments.

5. **Score each issue (0-100):**
   - 0: False positive, doesn't hold up to scrutiny.
   - 25: Might be real, couldn't verify. Stylistic issues not in CLAUDE.md.
   - 50: Real but minor/nitpick.
   - 75: Verified real issue, important, impacts functionality or violates CLAUDE.md.
   - 100: Definitely real, confirmed with evidence, will happen in practice.

6. **Filter** — Only report issues scoring ≥ 80.

7. **Comment** — Post review via `gh pr comment` with:
   - Brief description per issue with links to code (full SHA + line range).
   - Citation of CLAUDE.md rule or code evidence.
   - No emojis, keep it brief.

**False positives to ignore:**
- Pre-existing issues.
- Things `cargo fmt` / `cargo clippy` / `rustc` catch (formatting, imports, types, unused).
- General quality concerns not in CLAUDE.md.
- Issues on lines the author didn't modify.
- Intentional functionality changes related to the broader PR goal.

**docindex-server-specific checks:**
- Bind address validated as a Tailscale IP; never `0.0.0.0`. `DOCINDEX_ALLOW_LOOPBACK` is a dev bypass only.
- Bearer auth middleware actually applied to every non-`/health` route (not just declared).
- Gemini client uses `RETRIEVAL_DOCUMENT` for indexing and `RETRIEVAL_QUERY` for search — mismatches are real bugs.
- Matryoshka dim defaults to 3072 (native `gemini-embedding-001`) and is settable via `DOCINDEX_EMBED_DIM`; it must match `meta.embedding_dim`; the store refuses mixed dims.
- Any `chunks` mutation also updates `chunks_fts` (FTS5 is contentless and NOT auto-synced).
- `chunks_vec` is a real `vec0` virtual table; `sqlite-vec` is loaded via `rusqlite::ffi::sqlite3_auto_extension` at process startup (or per-connection init). No silent fallback to a BLOB column.
- Vector blobs are little-endian packed `f32[DOCINDEX_EMBED_DIM]` (default 3072).
- Rusqlite is synchronous — any DB call from an async context MUST be inside `tokio::task::spawn_blocking`.
- Walker + watcher feed the same indexing pipeline; no duplicated chunking/embedding logic.
- Embedding cache keyed by `content_hash` — renames/moves must not re-embed.
- Structured errors returned to clients (`{error, code}` via `ApiError: IntoResponse`); no internal messages / stack traces leaked.
- Every new feature has tests. `cargo test --all` and `python3 tests/run_tests.py` must pass.
- `cargo clippy --all-targets -- -D warnings` and `cargo fmt --all -- --check` must pass.
- No `unwrap()` / `expect()` / `panic!()` in production code. Test/init code is fine.
- Graceful shutdown (SIGINT/SIGTERM) must still drain tasks within 5s.
