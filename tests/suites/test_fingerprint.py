"""E2E: index fingerprint guard — provider/model/dim mismatch detection and
`--reembed` recovery."""
from __future__ import annotations

import pathlib
import socket
import subprocess
import time

import httpx
import pytest

from conftest import DEFAULT_E2E_EMBED_DIM


pytestmark = pytest.mark.e2e


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(vault: pathlib.Path, db: pathlib.Path, port: int, extra: dict[str, str]) -> dict[str, str]:
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("DOCINDEX_")}
    env.update(
        {
            "DOCINDEX_VAULT_DIR": str(vault),
            "DOCINDEX_DB_PATH": str(db),
            "DOCINDEX_LISTEN": f"127.0.0.1:{port}",
            "DOCINDEX_ALLOW_LOOPBACK": "true",
            "DOCINDEX_BEARER": "fp-bearer",
            "DOCINDEX_LOG_FORMAT": "text",
        }
    )
    env.update(extra)
    return env


def _run_until_ready_or_exit(
    docindex_bin: pathlib.Path,
    env: dict[str, str],
    log_path: pathlib.Path,
    port: int,
    extra_args: list[str] | None = None,
    timeout: float = 10.0,
) -> tuple[subprocess.Popen, bool]:
    """Spawn the server and wait for either readiness or early exit.

    Returns (proc, became_ready). Caller is responsible for cleanup if
    became_ready is True.
    """
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [str(docindex_bin), *(extra_args or [])],
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_f.close()
            return proc, False
        try:
            r = httpx.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200 and r.json().get("ok") is True:
                return proc, True
        except httpx.RequestError:
            pass
        time.sleep(0.15)
    log_f.close()
    raise AssertionError(f"server neither became ready nor exited within {timeout}s")


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


def test_dim_mismatch_exits_nonzero_naming_field(docindex_bin, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha body\n")
    db = tmp_path / "index.db"
    port = _free_port()

    # First boot: provider=fake (dim=DEFAULT_E2E_EMBED_DIM), establishes the
    # fingerprint.
    env_a = _env(
        vault,
        db,
        port,
        {"DOCINDEX_EMBED": "fake", "DOCINDEX_EMBED_DIM": str(DEFAULT_E2E_EMBED_DIM)},
    )
    log_a = tmp_path / "server_a.log"
    proc_a, ready_a = _run_until_ready_or_exit(docindex_bin, env_a, log_a, port)
    assert ready_a, log_a.read_text()
    _stop(proc_a)

    # Second boot: same provider, different dim — this is a dim mismatch.
    # The server must refuse to start and name both the stored and new dim
    # in its error output, along with the --reembed hint.
    port2 = _free_port()
    env_b = _env(
        vault,
        db,
        port2,
        {"DOCINDEX_EMBED": "fake", "DOCINDEX_EMBED_DIM": str(DEFAULT_E2E_EMBED_DIM + 8)},
    )
    log_b = tmp_path / "server_b.log"
    proc_b, ready_b = _run_until_ready_or_exit(docindex_bin, env_b, log_b, port2)
    assert not ready_b, "server should have refused to start on dim mismatch"
    rc = proc_b.wait(timeout=5.0)
    assert rc != 0
    log_text = log_b.read_text()
    assert f"dim={DEFAULT_E2E_EMBED_DIM}" in log_text, log_text
    assert f"dim={DEFAULT_E2E_EMBED_DIM + 8}" in log_text, log_text
    assert "--reembed" in log_text, log_text


def test_reembed_recovers_from_mismatch(docindex_bin, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha body\n")
    db = tmp_path / "index.db"
    port = _free_port()

    env_a = _env(
        vault,
        db,
        port,
        {"DOCINDEX_EMBED": "fake", "DOCINDEX_EMBED_DIM": str(DEFAULT_E2E_EMBED_DIM)},
    )
    log_a = tmp_path / "server_a.log"
    proc_a, ready_a = _run_until_ready_or_exit(docindex_bin, env_a, log_a, port)
    assert ready_a, log_a.read_text()
    health_headers = {"Authorization": "Bearer fp-bearer"}
    r = httpx.get(
        f"http://127.0.0.1:{port}/health", headers=health_headers, timeout=5.0
    )
    baseline_chunks = r.json()["indexed_chunks"]
    assert baseline_chunks >= 1
    _stop(proc_a)

    # Reopen at a different dim WITHOUT --reembed: must refuse.
    port2 = _free_port()
    env_b = _env(
        vault,
        db,
        port2,
        {"DOCINDEX_EMBED": "fake", "DOCINDEX_EMBED_DIM": str(DEFAULT_E2E_EMBED_DIM + 8)},
    )
    log_b = tmp_path / "server_b.log"
    proc_b, ready_b = _run_until_ready_or_exit(docindex_bin, env_b, log_b, port2)
    assert not ready_b
    assert proc_b.wait(timeout=5.0) != 0

    # Reopen at the new dim WITH --reembed: must succeed, wipe, and rebuild.
    port3 = _free_port()
    env_c = _env(
        vault,
        db,
        port3,
        {"DOCINDEX_EMBED": "fake", "DOCINDEX_EMBED_DIM": str(DEFAULT_E2E_EMBED_DIM + 8)},
    )
    log_c = tmp_path / "server_c.log"
    proc_c, ready_c = _run_until_ready_or_exit(
        docindex_bin, env_c, log_c, port3, extra_args=["--reembed"]
    )
    assert ready_c, log_c.read_text()
    try:
        base_url = f"http://127.0.0.1:{port3}"
        deadline = time.monotonic() + 10.0
        chunks = 0
        while time.monotonic() < deadline:
            chunks = httpx.get(
                f"{base_url}/health", headers=health_headers, timeout=2.0
            ).json()["indexed_chunks"]
            if chunks >= 1:
                break
            time.sleep(0.2)
        assert chunks >= 1, log_c.read_text()
        r = httpx.get(f"{base_url}/health", headers=health_headers, timeout=5.0)
        assert r.json()["dim"] == DEFAULT_E2E_EMBED_DIM + 8

        r = httpx.post(
            f"{base_url}/search",
            json={"query": "alpha"},
            headers={"Authorization": "Bearer fp-bearer"},
            timeout=5.0,
        )
        assert r.status_code == 200, r.text
        assert r.json()["hits"], r.text
    finally:
        _stop(proc_c)
