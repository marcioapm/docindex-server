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
    # Vault is flat in this suite, so the relative path is just the filename.
    assert top["path"] == expected_file, f"top hit {top['path']} for query {query!r}"
    assert "snippet" in top and top["snippet"]
    assert "score" in top
    assert "chunk_id" in top


def test_search_returns_relative_paths(spawn_server, tmp_path):
    """Every hit.path must be vault-relative — no absolute filesystem paths.

    The plugin relies on this (Obsidian's TFile.path is always vault-relative);
    leaking absolute paths would also leak host environment to the client.
    """
    vault = _write_vault(tmp_path)
    # Move one file into a subdir so we also exercise a non-flat path shape.
    nested_dir = vault / "nested"
    nested_dir.mkdir()
    (vault / "rust.md").rename(nested_dir / "rust.md")

    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))
    r = server.post(
        "/search", {"query": "rust python sqlite tailscale obsidian", "limit": 50}
    )
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert hits, "expected hits"
    for h in hits:
        p = h["path"]
        assert not p.startswith("/"), f"hit.path must not be absolute, got {p!r}"
        assert ".." not in p.split("/"), f"hit.path must not contain '..', got {p!r}"
    paths = {h["path"] for h in hits}
    assert "nested/rust.md" in paths, (
        f"expected nested/rust.md in hits, got {sorted(paths)}"
    )


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


def test_search_hits_carry_normalized_and_rrf_scores(spawn_server, tmp_path):
    """Every hit must expose score_rrf and score_normalized alongside score.

    - `score` stays as the RRF value for back-compat.
    - `score_rrf` equals `score` (duplicate for clarity).
    - `score_normalized` is in [0.0, 1.0].
    - Results are ordered by `score_rrf` descending (RRF is the ranker).
    """
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))
    r = server.post(
        "/search", {"query": "rust python sqlite tailscale obsidian", "limit": 50}
    )
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert hits, "expected hits"
    for h in hits:
        assert "score" in h and "score_rrf" in h and "score_normalized" in h, h
        assert h["score_rrf"] == h["score"], h
        assert 0.0 <= h["score_normalized"] <= 1.0, h
    scores = [h["score_rrf"] for h in hits]
    assert scores == sorted(scores, reverse=True), scores
    # The top hit in a clean query like this should be decisively normalized
    # above the default 0.40 threshold — rank-1 in at least one branch gets
    # at least max(W_VEC, W_BM25) = 0.55.
    assert hits[0]["score_normalized"] >= 0.40, hits[0]


# Mixed .md + .txt vault: the walker accepts both (case-insensitive), and the
# chunker falls back to the 500-word split for heading-less .txt files. Both
# should be searchable without any per-extension tweaks in the server.
MIXED_VAULT_FILES = {
    "rust.md": "# Rust\n\nRust is a systems programming language with ownership and borrow checking.\n",
    "plaintext.txt": "Kafka is a distributed commit log with partitioned topics and consumer groups.\n",
    "UPPER.TXT": "Zebra stripes help confuse predators with optical patterns.\n",
}


def _write_mixed_vault(tmp_path: pathlib.Path) -> pathlib.Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, content in MIXED_VAULT_FILES.items():
        (vault / name).write_text(content)
    return vault


@pytest.mark.parametrize(
    "query,expected_file",
    [
        ("rust ownership borrow", "rust.md"),
        ("kafka commit log partitioned", "plaintext.txt"),
        ("zebra stripes predators", "UPPER.TXT"),
    ],
)
def test_search_indexes_md_and_txt(spawn_server, tmp_path, query, expected_file):
    vault = _write_mixed_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(MIXED_VAULT_FILES))
    r = server.post("/search", {"query": query, "limit": 5})
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert hits, f"expected at least one hit for {query!r}"
    # Paths are vault-relative; the mixed vault is flat, so rel == filename.
    assert hits[0]["path"] == expected_file, (
        f"top hit {hits[0]['path']} for query {query!r}"
    )
