"""Phase 1 smoke tests for the docindex binary.

Validates:
* Binary starts cleanly with a valid env and exits 0 (Phase 1 has no long-running HTTP).
* The sqlite DB is created with the expected schema (`chunks`, `chunks_fts`, `chunks_vec` `vec0` virtual table, `embedding_cache`, `meta`).
* `meta.schema_version` is written.
* Bind validation rejects `0.0.0.0`.
* Missing required env vars fail at startup.
"""
from __future__ import annotations

import pathlib
import shutil
import sqlite3

import pytest


pytestmark = pytest.mark.smoke


def _require_sqlite3():
    if shutil.which("sqlite3") is None:
        # stdlib sqlite3 is fine; we use Python's module for assertions.
        pass


def test_binary_starts_and_exits_zero(docindex_bin, base_env, run_bin, db_path):
    res = run_bin(docindex_bin, base_env)
    assert res.returncode == 0, (
        f"non-zero exit: {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    assert db_path.exists(), "expected sqlite db to be created"


def test_schema_created_with_vec0_and_fts5(docindex_bin, base_env, run_bin, db_path):
    res = run_bin(docindex_bin, base_env)
    assert res.returncode == 0, res.stderr
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','virtual table')"
            )
        }
        # `chunks_fts` and `chunks_vec` are virtual tables exposed as 'table'
        # in sqlite_master; the underlying shadow tables also appear.
        assert "chunks" in tables
        assert "embedding_cache" in tables
        assert "meta" in tables
        assert "chunks_fts" in tables
        assert "chunks_vec" in tables

        # meta.schema_version written
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None and row[0] == "1", f"schema_version={row}"

        # vec_version() resolves — proves sqlite-vec is loaded by the binary's
        # process. We can't call it from this connection (extension isn't
        # loaded here), but the vec0 virtual table's shadow tables are a
        # persistent artifact of its creation.
        shadow = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'chunks_vec%'"
            )
        }
        # vec0 creates several shadow tables (chunks_vec_rowids, etc.)
        assert any(s.startswith("chunks_vec") and s != "chunks_vec" for s in shadow), (
            f"expected vec0 shadow tables, got {shadow}"
        )
    finally:
        conn.close()


def test_rejects_zero_zero_bind(docindex_bin, base_env, run_bin):
    env = dict(base_env)
    env["DOCINDEX_LISTEN"] = "0.0.0.0:7777"
    res = run_bin(docindex_bin, env)
    assert res.returncode != 0
    combined = (res.stdout + res.stderr).lower()
    assert "0.0.0.0" in combined or "all interfaces" in combined


@pytest.mark.parametrize(
    "missing",
    [
        "DOCINDEX_VAULT_DIR",
        "DOCINDEX_DB_PATH",
        "DOCINDEX_LISTEN",
        "DOCINDEX_BEARER",
        "GEMINI_API_KEY",
    ],
)
def test_missing_required_env_fails(docindex_bin, base_env, run_bin, missing):
    env = dict(base_env)
    env.pop(missing, None)
    res = run_bin(docindex_bin, env)
    assert res.returncode != 0, f"expected failure when {missing} missing"


def test_fake_markdown_is_not_indexed_in_phase1(docindex_bin, base_env, run_bin, db_path):
    """Phase 1 only opens the store — it does not run the walker/indexer yet.

    This test documents the current surface: after a clean run there should
    be zero chunks. When Phase 2 wires the walker on startup, invert this
    assertion.
    """
    res = run_bin(docindex_bin, base_env)
    assert res.returncode == 0, res.stderr
    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == 0
    finally:
        conn.close()
