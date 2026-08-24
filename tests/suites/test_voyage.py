"""E2E: server configured for Voyage against a local mock HTTP server.

Runs a tiny stdlib HTTP server that mimics Voyage's `/v1/embeddings`
endpoint with deterministic, content-hash-seeded vectors (same recipe as
the in-process `fake` embedder, reimplemented here in Python so the test
doesn't depend on Rust internals). Verifies indexing populates the index
and that `/search` produces the expected top hit — i.e. the whole
provider=voyage code path (config, client, task_type wiring) works
end-to-end, not just via the Rust unit tests' `wiremock`.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import pathlib
import socket
import struct
import subprocess
import threading
import time

import httpx
import pytest


pytestmark = pytest.mark.e2e

VOYAGE_DIM = 256  # smallest allowed voyage-4 output_dimension


def _seed_vector(text: str, input_type: str, dim: int) -> list[float]:
    """Deterministic unit vector from sha256(text|input_type), mirroring the
    Rust `Fake` embedder's block-hash recipe closely enough that same input
    -> same vector across runs (exact bit-parity with Rust isn't required —
    only determinism and doc/query task distinctness are asserted)."""
    seed = f"{text}|{input_type}"
    out = [0.0] * dim
    i = 0
    counter = 0
    while i < dim:
        h = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        for j in range(0, min(8, dim - i)):
            u = struct.unpack(">h", h[2 * j : 2 * j + 2])[0]
            out[i + j] = u / 32768.0
        i += 8
        counter += 1
    norm = sum(x * x for x in out) ** 0.5
    if norm > 0:
        out = [x / norm for x in out]
    return out


class _VoyageMockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default stderr logging
        pass

    def do_POST(self):
        if self.path != "/v1/embeddings":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        inputs = body["input"]
        input_type = body["input_type"]
        dim = body["output_dimension"]
        data = [
            {"embedding": _seed_vector(text, input_type, dim), "index": i}
            for i, text in enumerate(inputs)
        ]
        payload = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def voyage_mock():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _VoyageMockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_voyage_index_and_search_roundtrip(docindex_bin, voyage_mock, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "rust.md").write_text(
        "# Rust\n\nRust is a systems programming language with ownership.\n"
    )
    (vault / "python.md").write_text(
        "# Python\n\nPython is a dynamic scripting language for data science.\n"
    )
    db = tmp_path / "index.db"
    port = _free_port()
    cfg = tmp_path / "server.toml"
    cfg.write_text(
        f"""
vault_dir = "{vault}"
db_path = "{db}"
listen = "127.0.0.1:{port}"
allow_loopback = true
bearer = "voyage-bearer"
log_format = "text"

[embed]
provider = "voyage"
model = "voyage-4"
dim = {VOYAGE_DIM}
base_url = "{voyage_mock}"
"""
    )

    env = {k: v for k, v in os.environ.items() if not k.startswith("DOCINDEX_")}
    env["VOYAGE_API_KEY"] = "test-voyage-key"
    log_path = tmp_path / "server.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [str(docindex_bin), "--config", str(cfg)],
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    health_headers = {"Authorization": "Bearer voyage-bearer"}
    try:
        deadline = time.monotonic() + 15.0
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early with code {proc.returncode}\nlog:\n{log_path.read_text()}"
                )
            try:
                r = httpx.get(
                    f"{base_url}/health", headers=health_headers, timeout=2.0
                )
                if r.status_code == 200 and r.json().get("ok") is True:
                    ready = True
                    break
            except httpx.RequestError:
                pass
            time.sleep(0.2)
        assert ready, f"server never became ready\nlog:\n{log_path.read_text()}"

        deadline = time.monotonic() + 15.0
        chunks = 0
        while time.monotonic() < deadline:
            r = httpx.get(f"{base_url}/health", headers=health_headers, timeout=2.0)
            chunks = r.json().get("indexed_chunks", 0)
            if chunks >= 2:
                break
            time.sleep(0.2)
        assert chunks >= 2, f"expected 2 chunks, got {chunks}\nlog:\n{log_path.read_text()}"

        health = httpx.get(
            f"{base_url}/health", headers=health_headers, timeout=5.0
        ).json()
        assert health["embedding_model"] == "voyage-4"
        assert health["dim"] == VOYAGE_DIM

        r = httpx.post(
            f"{base_url}/search",
            json={"query": "rust ownership systems", "limit": 5},
            headers={"Authorization": "Bearer voyage-bearer"},
            timeout=5.0,
        )
        assert r.status_code == 200, r.text
        hits = r.json()["hits"]
        assert hits, "expected hits"
        assert hits[0]["path"] == "rust.md", hits
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        log_f.close()
