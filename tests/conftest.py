"""Shared fixtures for the docindex pytest harness."""
from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import socket
import subprocess
import time
from typing import Iterator

import httpx
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# E2E tests run the fake embedder — a small dim keeps index.db tiny and
# vector ops fast. The fake embedder is deterministic at any size.
# Production default is 3072. Tests that pin an explicit /health.dim
# expectation should reference this constant.
DEFAULT_E2E_EMBED_DIM = 128


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


@pytest.fixture(scope="session")
def docindex_search_bin() -> pathlib.Path:
    """Absolute path to the docindex-search binary (release build)."""
    env_bin = os.environ.get("DOCINDEX_SEARCH_BIN")
    if env_bin:
        p = pathlib.Path(env_bin)
        if not p.exists():
            pytest.skip(f"DOCINDEX_SEARCH_BIN={p} not found")
        return p
    candidate = REPO_ROOT / "target" / "release" / "docindex-search"
    if not candidate.exists():
        pytest.skip("docindex-search binary not built; run tests/run_tests.py")
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
            "DOCINDEX_LISTEN": "100.64.0.1:7777",
            "DOCINDEX_BEARER": "test-bearer",
            "GEMINI_API_KEY": "test-key",
            "DOCINDEX_EMBED_MODEL": "gemini-embedding-001",
            # E2E tests run the fake embedder — a small dim keeps index.db
            # tiny and vector ops fast. Production default is 3072.
            "DOCINDEX_EMBED_DIM": str(DEFAULT_E2E_EMBED_DIM),
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SpawnedServer:
    def __init__(self, proc: subprocess.Popen[str], base_url: str, bearer: str, log_path: pathlib.Path):
        self.proc = proc
        self.base_url = base_url
        self.bearer = bearer
        self.log_path = log_path

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.bearer}"}

    def get(self, path: str, **kwargs) -> httpx.Response:
        return httpx.get(f"{self.base_url}{path}", timeout=10.0, **kwargs)

    def post(self, path: str, json: dict, auth: bool = True) -> httpx.Response:
        headers = self.headers() if auth else {}
        return httpx.post(
            f"{self.base_url}{path}", json=json, headers=headers, timeout=10.0
        )

    def wait_for_chunks(self, n: int, timeout: float = 15.0) -> int:
        """Poll /health.indexed_chunks until it reaches `n`. Return observed count."""
        deadline = time.monotonic() + timeout
        last = -1
        while time.monotonic() < deadline:
            try:
                r = self.get("/health", headers=self.headers())
                if r.status_code == 200:
                    last = r.json().get("indexed_chunks", -1)
                    if last >= n:
                        return last
            except httpx.RequestError:
                pass
            time.sleep(0.2)
        raise AssertionError(
            f"indexed_chunks did not reach {n} within {timeout}s (last={last})"
        )


@contextlib.contextmanager
def _spawn_server(
    docindex_bin: pathlib.Path,
    vault: pathlib.Path,
    db_path: pathlib.Path,
    bearer: str,
    log_path: pathlib.Path,
    env_overrides: dict[str, str] | None = None,
) -> Iterator[SpawnedServer]:
    port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "DOCINDEX_VAULT_DIR": str(vault),
            "DOCINDEX_DB_PATH": str(db_path),
            "DOCINDEX_LISTEN": f"127.0.0.1:{port}",
            "DOCINDEX_ALLOW_LOOPBACK": "true",
            "DOCINDEX_BEARER": bearer,
            "DOCINDEX_EMBED": "fake",
            "DOCINDEX_EMBED_MODEL": "gemini-embedding-001",
            # Small dim for E2E — the fake embedder is deterministic at any
            # size; keeping it small makes index.db and vec math cheap.
            "DOCINDEX_EMBED_DIM": str(DEFAULT_E2E_EMBED_DIM),
            "DOCINDEX_LOG_FORMAT": "text",
            "DOCINDEX_DEBOUNCE_MS": "500",
        }
    )
    if env_overrides:
        env.update(env_overrides)

    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [str(docindex_bin)],
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    server = SpawnedServer(proc, base_url, bearer, log_path)
    try:
        _wait_for_ready(server, timeout=10.0)
        yield server
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        log_f.close()


def _wait_for_ready(server: SpawnedServer, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if server.proc.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {server.proc.returncode}\n"
                f"log:\n{server.log_path.read_text()}"
            )
        try:
            r = httpx.get(f"{server.base_url}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("ok") is True:
                return
        except httpx.RequestError as e:
            last_err = e
        time.sleep(0.1)
    raise RuntimeError(
        f"server did not become ready within {timeout}s: {last_err}\n"
        f"log:\n{server.log_path.read_text()}"
    )


@pytest.fixture
def spawn_server(docindex_bin, tmp_path):
    """Yield a callable that spawns the docindex binary and returns a SpawnedServer."""
    active: list[contextlib.AbstractContextManager] = []

    def _factory(
        vault: pathlib.Path,
        bearer: str = "test-bearer",
        env_overrides: dict[str, str] | None = None,
        db: pathlib.Path | None = None,
    ) -> SpawnedServer:
        idx = len(active)
        db_p = db or (tmp_path / f"index_{idx}.db")
        log_p = tmp_path / f"server_{idx}.log"
        cm = _spawn_server(
            docindex_bin=docindex_bin,
            vault=vault,
            db_path=db_p,
            bearer=bearer,
            log_path=log_p,
            env_overrides=env_overrides,
        )
        server = cm.__enter__()
        active.append(cm)
        return server

    try:
        yield _factory
    finally:
        for cm in reversed(active):
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
