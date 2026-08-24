"""E2E: /similar finds related documents by path."""
from __future__ import annotations

import pathlib

import pytest


pytestmark = pytest.mark.e2e


VAULT = {
    "rust_lang.md": "# Rust\n\nRust is a systems programming language with ownership and borrow checking.\n",
    "rust_traits.md": "# Rust traits\n\nTraits in Rust describe shared behavior — similar to interfaces but with monomorphization.\n",
    "python.md": "# Python\n\nPython is a dynamic scripting language popular for data science.\n",
    "cooking.md": "# Cooking\n\nRoasted vegetables with olive oil and sea salt.\n",
}


def _write_vault(tmp_path: pathlib.Path) -> pathlib.Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, content in VAULT.items():
        (vault / name).write_text(content)
    return vault


def test_similar_returns_related_doc(spawn_server, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT))
    # Fake embedder is deterministic content-hash based; we rely on the FTS
    # side of RRF to return rust_traits.md as the closest peer to rust_lang.md.
    # Paths are vault-relative — clients (like Obsidian) don't know the server's
    # filesystem layout.
    target = "rust_lang.md"
    r = server.post("/similar", {"path": target, "limit": 5})
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert hits, "expected at least one similar hit"
    paths = [h["path"] for h in hits]
    assert target not in paths, "similar must not include the source path"
    assert any(p == "rust_traits.md" for p in paths), (
        f"expected rust_traits.md in top hits, got {paths}"
    )
    # Defense in depth: no hit should leak an absolute host path.
    for p in paths:
        assert not p.startswith("/"), f"hit.path must be vault-relative, got {p!r}"


def test_similar_known_empty_file_returns_empty_hits(spawn_server, tmp_path):
    vault = _write_vault(tmp_path)
    (vault / "empty.md").touch()
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT))

    r = server.post("/similar", {"path": "empty.md"})

    assert r.status_code == 200, r.text
    assert r.json() == {"hits": []}


def test_similar_unknown_path_is_404(spawn_server, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT))
    r = server.post("/similar", {"path": "nope.md"})
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == "not_found"
    assert body["error"] == "path not indexed: nope.md"
