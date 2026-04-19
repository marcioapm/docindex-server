"""E2E: /health endpoint returns 200 with expected fields."""
from __future__ import annotations

import pathlib

import pytest

from conftest import DEFAULT_E2E_EMBED_DIM


pytestmark = pytest.mark.e2e


def test_health_ok(spawn_server, tmp_path: pathlib.Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha\n")

    server = spawn_server(vault)
    r = server.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["embedding_model"] == "gemini-embedding-001"
    assert body["dim"] == DEFAULT_E2E_EMBED_DIM
    assert "indexed_chunks" in body
    assert "last_reindex_ms" in body


def test_health_is_public(spawn_server, tmp_path: pathlib.Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    server = spawn_server(vault, bearer="topsecret")
    # no auth header — should still succeed
    r = server.get("/health")
    assert r.status_code == 200


def test_health_reports_indexed_chunks_after_scan(spawn_server, tmp_path: pathlib.Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    for i in range(3):
        (vault / f"note_{i}.md").write_text(f"# Note {i}\n\nbody for note {i}\n")
    server = spawn_server(vault)
    got = server.wait_for_chunks(3, timeout=15.0)
    assert got >= 3


def test_health_dim_tracks_env(spawn_server, tmp_path: pathlib.Path):
    """Explicit override: /health.dim must reflect DOCINDEX_EMBED_DIM."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha\n")
    server = spawn_server(vault, env_overrides={"DOCINDEX_EMBED_DIM": "64"})
    r = server.get("/health")
    assert r.status_code == 200
    assert r.json()["dim"] == 64
