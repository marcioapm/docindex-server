"""E2E: the filesystem watcher picks up a new file after startup."""
from __future__ import annotations

import pathlib
import time

import pytest


pytestmark = pytest.mark.e2e


def test_watcher_indexes_new_file(spawn_server, tmp_path: pathlib.Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "seed.md").write_text("# Seed\n\nseed body\n")

    server = spawn_server(vault)
    baseline = server.wait_for_chunks(1)

    # Drop in a new file with a distinctive phrase and wait for the watcher.
    (vault / "fresh.md").write_text(
        "# Fresh\n\nquokka pineapple xylophone — distinctive phrase.\n"
    )

    # Debounce defaults to 500ms in tests; allow up to 10s.
    deadline = time.monotonic() + 10.0
    found = False
    while time.monotonic() < deadline:
        r = server.post(
            "/search", {"query": "quokka pineapple xylophone", "limit": 5}
        )
        if r.status_code == 200:
            hits = r.json()["hits"]
            if any(h["path"].endswith("fresh.md") for h in hits):
                found = True
                break
        time.sleep(0.25)

    assert found, (
        f"watcher did not pick up fresh.md (baseline={baseline})\n"
        f"server log:\n{server.log_path.read_text()}"
    )


def test_watcher_picks_up_deletions(spawn_server, tmp_path: pathlib.Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "keep.md").write_text("# Keep\n\nstays\n")
    doomed = vault / "doomed.md"
    doomed.write_text("# Doomed\n\nunique-doomed-phrase-abcdef\n")

    server = spawn_server(vault)
    server.wait_for_chunks(2)

    # Confirm it's there.
    r = server.post("/search", {"query": "unique-doomed-phrase-abcdef", "limit": 5})
    assert r.status_code == 200
    assert any(h["path"].endswith("doomed.md") for h in r.json()["hits"])

    doomed.unlink()

    deadline = time.monotonic() + 10.0
    gone = False
    while time.monotonic() < deadline:
        r = server.post(
            "/search", {"query": "unique-doomed-phrase-abcdef", "limit": 5}
        )
        if r.status_code == 200:
            hits = r.json()["hits"]
            if not any(h["path"].endswith("doomed.md") for h in hits):
                gone = True
                break
        time.sleep(0.25)

    assert gone, "expected doomed.md to be removed from the index"
