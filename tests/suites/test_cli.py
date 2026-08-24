"""E2E: docindex-search CLI against the real server binary."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest


pytestmark = pytest.mark.e2e


VAULT_FILES = {
    "rust.md": "# Rust\n\nRust is a systems programming language with ownership and borrow checking.\n",
    "python.md": "# Python\n\nPython is a dynamic scripting language popular for data science.\n",
}


def _write_vault(tmp_path: pathlib.Path) -> pathlib.Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, content in VAULT_FILES.items():
        (vault / name).write_text(content)
    return vault


def _run_cli(search_bin: pathlib.Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(search_bin), *args], capture_output=True, text=True, timeout=15.0
    )


def test_cli_search_finds_expected_hit(spawn_server, docindex_search_bin, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))

    r = _run_cli(
        docindex_search_bin,
        [
            "search",
            "rust ownership borrow",
            "--server",
            server.base_url,
            "--token",
            server.bearer,
        ],
    )
    assert r.returncode == 0, r.stderr
    assert "rust.md" in r.stdout
    assert r.stderr == ""


def test_cli_bare_query_shorthand(spawn_server, docindex_search_bin, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))

    r = _run_cli(
        docindex_search_bin,
        ["rust ownership borrow", "--server", server.base_url, "--token", server.bearer],
    )
    assert r.returncode == 0, r.stderr
    assert "rust.md" in r.stdout


def test_cli_similar(spawn_server, docindex_search_bin, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))

    r = _run_cli(
        docindex_search_bin,
        ["similar", "rust.md", "--server", server.base_url, "--token", server.bearer],
    )
    assert r.returncode in (0, 4), r.stderr  # small vault: python.md may or may not clear a threshold


def test_cli_health(spawn_server, docindex_search_bin, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))

    r = _run_cli(
        docindex_search_bin,
        ["health", "--server", server.base_url, "--token", server.bearer],
    )
    assert r.returncode == 0, r.stderr
    assert "ok=true" in r.stdout


def test_cli_json_shape(spawn_server, docindex_search_bin, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))

    r = _run_cli(
        docindex_search_bin,
        [
            "search",
            "rust ownership borrow",
            "--server",
            server.base_url,
            "--token",
            server.bearer,
            "--json",
        ],
    )
    assert r.returncode == 0, r.stderr
    body = json.loads(r.stdout)
    assert "hits" in body
    assert body["hits"], body
    hit = body["hits"][0]
    for key in ("path", "title", "heading_path", "snippet", "score", "score_rrf", "score_normalized", "chunk_id"):
        assert key in hit, hit


def test_cli_path_filter(spawn_server, docindex_search_bin, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))

    r = _run_cli(
        docindex_search_bin,
        [
            "search",
            "rust python language",
            "--server",
            server.base_url,
            "--token",
            server.bearer,
            "--path-filter",
            "python",
            "--json",
        ],
    )
    assert r.returncode == 0, r.stderr
    body = json.loads(r.stdout)
    for h in body["hits"]:
        assert h["path"].startswith("python"), h


def test_cli_path_filter_no_match_is_exit_4(spawn_server, docindex_search_bin, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))

    r = _run_cli(
        docindex_search_bin,
        [
            "search",
            "rust python language",
            "--server",
            server.base_url,
            "--token",
            server.bearer,
            "--path-filter",
            "no-such-prefix",
        ],
    )
    assert r.returncode == 4, r.stdout


def test_cli_no_results_is_exit_4(spawn_server, docindex_search_bin, tmp_path):
    # Semantic search always returns its top-K nearest neighbors regardless
    # of query relevance, so an arbitrary query string against a non-empty
    # vault never yields zero hits. An empty vault is the reliable way to
    # exercise the "no results" exit path.
    vault = tmp_path / "vault"
    vault.mkdir()
    server = spawn_server(vault)

    r = _run_cli(
        docindex_search_bin,
        [
            "search",
            "anything at all",
            "--server",
            server.base_url,
            "--token",
            server.bearer,
        ],
    )
    assert r.returncode == 4


def test_cli_wrong_token_is_exit_3(spawn_server, docindex_search_bin, tmp_path):
    vault = _write_vault(tmp_path)
    server = spawn_server(vault)
    server.wait_for_chunks(len(VAULT_FILES))

    r = _run_cli(
        docindex_search_bin,
        ["search", "rust", "--server", server.base_url, "--token", "wrong-token"],
    )
    assert r.returncode == 3
    assert r.stdout == ""
    assert r.stderr != ""


def test_cli_unreachable_server_is_exit_2(docindex_search_bin):
    r = _run_cli(
        docindex_search_bin,
        ["search", "rust", "--server", "http://127.0.0.1:1", "--token", "x"],
    )
    assert r.returncode == 2
    assert r.stdout == ""
    assert r.stderr != ""


def test_cli_missing_config_is_exit_1(docindex_search_bin):
    # Strip host-machine env/config influence: a real ~/.config/docindex/cli.toml
    # or DOCINDEX_CLI_SERVER on the dev machine must not leak into this test.
    env = {k: v for k, v in os.environ.items() if not k.startswith("DOCINDEX_")}
    env["XDG_CONFIG_HOME"] = "/nonexistent-xdg-config-dir"
    r = subprocess.run(
        [str(docindex_search_bin), "search", "rust"],
        capture_output=True,
        text=True,
        timeout=15.0,
        env=env,
    )
    assert r.returncode == 1
    assert r.stdout == ""
    assert r.stderr != ""
