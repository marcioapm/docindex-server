"""Phase 1/2 smoke tests for the docindex binary.

Validates:
* Binary starts cleanly and serves /health with a valid env (via spawn_server).
* The sqlite DB is created with the expected schema (`chunks`, `chunks_fts`,
  `chunks_vec` `vec0` virtual table, `embedding_cache`, `meta`, `files`).
* `meta.schema_version` is written.
* Bind validation rejects `0.0.0.0` and bare loopback (startup error, quick exit).
* Missing required env vars fail at startup.
"""
from __future__ import annotations

import pathlib
import shutil
import sqlite3
import subprocess

import pytest


pytestmark = pytest.mark.smoke


def _require_sqlite3():
    if shutil.which("sqlite3") is None:
        # stdlib sqlite3 is fine; we use Python's module for assertions.
        pass


def test_binary_serves_health(spawn_server, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha body\n")
    server = spawn_server(vault)
    r = server.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_schema_created_with_vec0_and_fts5(spawn_server, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha\n")
    db = tmp_path / "schema.db"
    server = spawn_server(vault, db=db)
    server.wait_for_chunks(1)
    # While the server is up the file is locked by SQLite in WAL mode,
    # but concurrent reads are fine.
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','virtual table')"
            )
        }
        assert "chunks" in tables
        assert "embedding_cache" in tables
        assert "meta" in tables
        assert "chunks_fts" in tables
        assert "chunks_vec" in tables
        assert "files" in tables

        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None and row[0] == "2", f"schema_version={row}"

        shadow = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'chunks_vec%'"
            )
        }
        assert any(s.startswith("chunks_vec") and s != "chunks_vec" for s in shadow), (
            f"expected vec0 shadow tables, got {shadow}"
        )
    finally:
        conn.close()


def test_rejects_zero_zero_bind(docindex_bin, base_env, run_bin):
    env = dict(base_env)
    env["DOCINDEX_LISTEN"] = "0.0.0.0:7777"
    # Drop loopback bypass so config validation is all we test here.
    env.pop("DOCINDEX_ALLOW_LOOPBACK", None)
    res = run_bin(docindex_bin, env, timeout=5.0)
    assert res.returncode != 0
    combined = (res.stdout + res.stderr).lower()
    assert "0.0.0.0" in combined or "all interfaces" in combined


def test_rejects_loopback_without_bypass(docindex_bin, base_env, run_bin):
    env = dict(base_env)
    env["DOCINDEX_LISTEN"] = "127.0.0.1:7777"
    env.pop("DOCINDEX_ALLOW_LOOPBACK", None)
    res = run_bin(docindex_bin, env, timeout=5.0)
    assert res.returncode != 0
    combined = (res.stdout + res.stderr).lower()
    assert "allow_loopback" in combined or "loopback" in combined


@pytest.mark.parametrize(
    "missing",
    [
        "DOCINDEX_VAULT_DIR",
        "DOCINDEX_DB_PATH",
        "DOCINDEX_LISTEN",
        "DOCINDEX_BEARER",
    ],
)
def test_missing_required_env_fails(docindex_bin, base_env, run_bin, missing):
    env = dict(base_env)
    env.pop(missing, None)
    # With GEMINI_API_KEY set (from base_env), embed_backend defaults to
    # gemini — so these four are genuinely required. Use a short timeout
    # because the binary should error before binding.
    try:
        res = subprocess.run(
            [str(docindex_bin)],
            env=env,
            timeout=5.0,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"binary did not exit when {missing} missing")
    assert res.returncode != 0, f"expected failure when {missing} missing"
