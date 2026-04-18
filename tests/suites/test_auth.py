"""E2E: bearer auth on protected endpoints."""
from __future__ import annotations

import pathlib

import httpx
import pytest


pytestmark = pytest.mark.e2e


def _make_vault(tmp_path: pathlib.Path) -> pathlib.Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha body\n")
    return vault


def test_search_without_bearer_is_401(spawn_server, tmp_path):
    vault = _make_vault(tmp_path)
    server = spawn_server(vault, bearer="s3cret")
    server.wait_for_chunks(1)
    r = httpx.post(f"{server.base_url}/search", json={"query": "alpha"}, timeout=5.0)
    assert r.status_code == 401
    body = r.json()
    assert body["code"] == "unauthorized"


def test_search_with_wrong_bearer_is_401(spawn_server, tmp_path):
    vault = _make_vault(tmp_path)
    server = spawn_server(vault, bearer="right")
    server.wait_for_chunks(1)
    r = httpx.post(
        f"{server.base_url}/search",
        json={"query": "alpha"},
        headers={"Authorization": "Bearer wrong"},
        timeout=5.0,
    )
    assert r.status_code == 401


def test_search_with_right_bearer_is_200(spawn_server, tmp_path):
    vault = _make_vault(tmp_path)
    server = spawn_server(vault, bearer="right")
    server.wait_for_chunks(1)
    r = server.post("/search", {"query": "alpha"})
    assert r.status_code == 200, r.text


def test_similar_also_requires_bearer(spawn_server, tmp_path):
    vault = _make_vault(tmp_path)
    server = spawn_server(vault, bearer="s3cret")
    server.wait_for_chunks(1)
    r = httpx.post(
        f"{server.base_url}/similar",
        json={"path": str(vault / "a.md")},
        timeout=5.0,
    )
    assert r.status_code == 401
