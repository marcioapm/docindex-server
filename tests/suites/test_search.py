"""E2E: /search returns relevant hits for distinctive phrases."""
from __future__ import annotations

import pathlib

import pytest


pytestmark = pytest.mark.e2e


VAULT_FILES = {
    "rust.md": "# Rust\n\nRust is a systems programming language with ownership and borrow checking.\n",
    "python.md": "# Python\n\nPython is a dynamic scripting language popular for data science.\n",
    "sqlite.md": "# SQLite\n\nSQLite is a self-contained zero-configuration embedded database.\n",
    "tailscale.md": "# Tailscale\n\nTailscale is a mesh VPN built on WireGuard for private networks.\n",
    "obsidian.md": "# Obsidian\n\nObsidian is a markdown note-taking application with backlinks.\n",
}


def _write_vault(tmp_path: pathlib.Path) -> pathlib.Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, content in VAULT_FILES.items():
        (vault / name).write_text(content)
    return vault


@pytest.mark.parametrize(
    "query,expected_file",
    [
        ("rust ownership borrow", "rust.md"),
        ("python scripting dynamic", "python.md"),
        ("sqlite embedded database", "sqlite.md"),
        ("tailscale wireguard vpn", "tailscale.md"),
        ("obsidian markdown notes", "obsidian.md"),
    ],
)
def test_search_top_hit_matches(spawn_server, tmp_path, query, expected_file):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))
    r = server.post("/search", {"query": query, "limit": 5})
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert hits, "expected at least one hit"
    top = hits[0]
    assert top["path"].endswith(expected_file), f"top hit {top['path']} for query {query!r}"
    assert "snippet" in top and top["snippet"]
    assert "score" in top
    assert "chunk_id" in top


def test_search_empty_query_is_400(spawn_server, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))
    r = server.post("/search", {"query": "   "})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_request"


def test_search_respects_limit(spawn_server, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))
    r = server.post("/search", {"query": "language database vpn markdown", "limit": 2})
    assert r.status_code == 200
    assert len(r.json()["hits"]) <= 2
