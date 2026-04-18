"""Shared fixtures for the docindex pytest harness."""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def docindex_bin() -> pathlib.Path:
    """Absolute path to the docindex binary (release build)."""
    env_bin = os.environ.get("DOCINDEX_BIN")
    if env_bin:
        p = pathlib.Path(env_bin)
        if not p.exists():
            pytest.skip(f"DOCINDEX_BIN={p} not found")
        return p
    # Fallback: try target/release.
    candidate = REPO_ROOT / "target" / "release" / "docindex"
    if not candidate.exists():
        pytest.skip("docindex binary not built; run tests/run_tests.py")
    return candidate


@pytest.fixture
def vault_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "vault"
    d.mkdir()
    (d / "README.md").write_text("# Hello\n\nworld\n")
    (d / "nested").mkdir()
    (d / "nested" / "note.md").write_text("## Nested\n\nbody text\n")
    return d


@pytest.fixture
def db_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "index.db"


@pytest.fixture
def base_env(vault_dir: pathlib.Path, db_path: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "DOCINDEX_VAULT_DIR": str(vault_dir),
            "DOCINDEX_DB_PATH": str(db_path),
            "DOCINDEX_LISTEN": "100.83.46.59:7777",
            "DOCINDEX_BEARER": "test-bearer",
            "GEMINI_API_KEY": "test-key",
            "DOCINDEX_EMBED_MODEL": "gemini-embedding-001",
            "DOCINDEX_EMBED_DIM": "768",
            "DOCINDEX_LOG_FORMAT": "text",
        }
    )
    return env


def run_docindex(
    bin_path: pathlib.Path, env: dict[str, str], timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    """Run docindex, wait for exit, and return the completed process."""
    return subprocess.run(
        [str(bin_path)],
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def run_bin():
    return run_docindex


def have_sqlite3() -> bool:
    return shutil.which("sqlite3") is not None
