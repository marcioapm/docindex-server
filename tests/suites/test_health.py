"""E2E: /health serves public liveness and authenticated operational detail."""
from __future__ import annotations

import pathlib

import pytest

from conftest import DEFAULT_E2E_EMBED_DIM


pytestmark = pytest.mark.e2e

DETAIL_KEYS = {"indexed_chunks", "last_reindex_ms", "embedding_model", "dim"}


def test_health_without_bearer_is_minimal_liveness(spawn_server, tmp_path: pathlib.Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha\n")
    server = spawn_server(vault)

    r = server.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body == {"ok": True, "authenticated": False}
    assert not (DETAIL_KEYS & body.keys())


def test_health_with_wrong_bearer_matches_missing_bearer(spawn_server, tmp_path: pathlib.Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    server = spawn_server(vault, bearer="topsecret")

    missing = server.get("/health")
    wrong = server.get("/health", headers={"Authorization": "Bearer wrong"})

    assert missing.status_code == 200
    assert wrong.status_code == 200
    assert missing.content == wrong.content


def test_health_with_bearer_returns_operational_detail(spawn_server, tmp_path: pathlib.Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha\n")
    server = spawn_server(vault)

    r = server.get("/health", headers={"Authorization": f"Bearer {server.bearer}"})

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["authenticated"] is True
    assert body["embedding_model"] == "gemini-embedding-001"
    assert body["dim"] == DEFAULT_E2E_EMBED_DIM
    assert DETAIL_KEYS <= body.keys()


def test_health_reports_indexed_chunks_after_scan(spawn_server, tmp_path: pathlib.Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    for i in range(3):
        (vault / f"note_{i}.md").write_text(f"# Note {i}\n\nbody for note {i}\n")
    server = spawn_server(vault)
    got = server.wait_for_chunks(3, timeout=15.0)
    assert got >= 3


def test_health_dim_tracks_env(spawn_server, tmp_path: pathlib.Path):
    """Explicit override: authenticated /health.dim follows DOCINDEX_EMBED_DIM."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha\n")
    server = spawn_server(vault, env_overrides={"DOCINDEX_EMBED_DIM": "64"})
    r = server.get("/health", headers={"Authorization": f"Bearer {server.bearer}"})
    assert r.status_code == 200
    assert r.json()["dim"] == 64
