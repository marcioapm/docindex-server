"""E2E: TOML config file layering — file-only boot, env overrides file,
flag overrides env.

Uses a hand-rolled spawn (not the `spawn_server` fixture, which hardcodes a
full env-var config) so each test controls exactly which layer is present.
"""
from __future__ import annotations

import os
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


def _server_toml(vault: pathlib.Path, db: pathlib.Path, port: int, bearer: str) -> str:
    return f"""
vault_dir = "{vault}"
db_path = "{db}"
listen = "127.0.0.1:{port}"
allow_loopback = true
bearer = "{bearer}"
log_format = "text"

[embed]
provider = "fake"
dim = {DEFAULT_E2E_EMBED_DIM}
"""


def _spawn_with_argv(
    argv: list[str], env: dict[str, str], log_path: pathlib.Path, port: int
):
    log_f = open(log_path, "w")
    proc = subprocess.Popen(argv, env=env, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    base_url = f"http://127.0.0.1:{port}"
    return proc, base_url, log_f


def _wait_ready(proc: subprocess.Popen, base_url: str, log_path: pathlib.Path, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {proc.returncode}\nlog:\n{log_path.read_text()}"
            )
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("ok") is True:
                return
        except httpx.RequestError as e:
            last_err = e
        time.sleep(0.1)
    raise RuntimeError(f"server did not become ready within {timeout}s: {last_err}\nlog:\n{log_path.read_text()}")


def _stop(proc: subprocess.Popen, log_f) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
    log_f.close()


def _minimal_env() -> dict[str, str]:
    # Strip every DOCINDEX_* / GEMINI_*/VOYAGE_* var so the file layer is
    # the only thing driving config, aside from PATH (needed to exec).
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DOCINDEX_", "GEMINI_", "VOYAGE_"))}
    return env


def test_boots_from_toml_file_with_no_env_vars(docindex_bin, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha body\n")
    db = tmp_path / "index.db"
    port = _free_port()
    cfg_path = tmp_path / "server.toml"
    cfg_path.write_text(_server_toml(vault, db, port, "file-bearer"))

    env = _minimal_env()
    argv = [str(docindex_bin), "--config", str(cfg_path)]
    log_path = tmp_path / "server.log"
    proc, base_url, log_f = _spawn_with_argv(argv, env, log_path, port)
    try:
        _wait_ready(proc, base_url, log_path)
        r = httpx.get(f"{base_url}/health", timeout=5.0)
        assert r.status_code == 200
        r = httpx.post(
            f"{base_url}/search",
            json={"query": "alpha"},
            headers={"Authorization": "Bearer file-bearer"},
            timeout=5.0,
        )
        assert r.status_code == 200, r.text
    finally:
        _stop(proc, log_f)


def test_env_overrides_file(docindex_bin, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha body\n")
    db = tmp_path / "index.db"
    port = _free_port()
    cfg_path = tmp_path / "server.toml"
    cfg_path.write_text(_server_toml(vault, db, port, "file-bearer"))

    env = _minimal_env()
    env["DOCINDEX_BEARER"] = "env-bearer"
    argv = [str(docindex_bin), "--config", str(cfg_path)]
    log_path = tmp_path / "server.log"
    proc, base_url, log_f = _spawn_with_argv(argv, env, log_path, port)
    try:
        _wait_ready(proc, base_url, log_path)
        # The file's bearer must NOT work.
        r = httpx.post(
            f"{base_url}/search",
            json={"query": "alpha"},
            headers={"Authorization": "Bearer file-bearer"},
            timeout=5.0,
        )
        assert r.status_code == 401
        # The env bearer must work.
        r = httpx.post(
            f"{base_url}/search",
            json={"query": "alpha"},
            headers={"Authorization": "Bearer env-bearer"},
            timeout=5.0,
        )
        assert r.status_code == 200, r.text
    finally:
        _stop(proc, log_f)


def test_flag_overrides_env_config_path(docindex_bin, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha body\n")
    db = tmp_path / "index.db"
    port = _free_port()

    flag_cfg = tmp_path / "flag.toml"
    flag_cfg.write_text(_server_toml(vault, db, port, "flag-bearer"))
    other_db = tmp_path / "other.db"
    other_port = _free_port()
    env_cfg = tmp_path / "env.toml"
    env_cfg.write_text(_server_toml(vault, other_db, other_port, "env-config-bearer"))

    env = _minimal_env()
    env["DOCINDEX_CONFIG"] = str(env_cfg)
    argv = [str(docindex_bin), "--config", str(flag_cfg)]
    log_path = tmp_path / "server.log"
    proc, base_url, log_f = _spawn_with_argv(argv, env, log_path, port)
    try:
        _wait_ready(proc, base_url, log_path)
        # The flag-selected file's bearer must work (flag beats $DOCINDEX_CONFIG).
        r = httpx.post(
            f"{base_url}/search",
            json={"query": "alpha"},
            headers={"Authorization": "Bearer flag-bearer"},
            timeout=5.0,
        )
        assert r.status_code == 200, r.text
    finally:
        _stop(proc, log_f)
