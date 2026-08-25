"""E2E: media indexing with the fake provider.

Fixtures are generated programmatically from minimal byte sequences — no
binary blobs are committed. The suite uses a TOML config file to enable the
[media] section (env-only config cannot set it).

Coverage:
  - startup scan indexes a generated PNG and a generated multi-page PDF
  - both land in chunks_vec (vector-only)
  - neither lands in chunks_fts (no FTS entries)
  - media metadata and file state are recorded
  - /search returns media hits in a mixed query
  - media_only=true excludes all text hits
  - watcher picks up a modified media file
  - watcher removes a deleted media file
"""
from __future__ import annotations

import os
import pathlib
import socket
import struct
import subprocess
import time
import zlib

import httpx
import pytest

from conftest import DEFAULT_E2E_EMBED_DIM


pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Minimal fixture generators (pure Python, no PIL/Pillow required)
# ---------------------------------------------------------------------------

def _make_minimal_png(width: int = 4, height: int = 4) -> bytes:
    """Return bytes for a minimal valid RGBA PNG."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        body = tag + data
        crc = struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        return length + body + crc

    # IHDR: width, height, bit-depth=8, colour-type=2 (RGB), compress=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # IDAT: one scanline per row, filter byte 0 + raw RGB pixels
    raw = b""
    for _ in range(height):
        raw += b"\x00" + b"\xff\x80\x40" * width
    compressed = zlib.compress(raw, level=1)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr_data)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def _make_minimal_pdf(page_count: int = 1) -> bytes:
    """Return bytes for a minimal structurally valid PDF with `page_count` pages."""
    if page_count == 1:
        return (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n197\n%%EOF\n"
        )
    # Two-page variant
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000196 00000 n \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n277\n%%EOF\n"
    )


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server_toml(
    vault: pathlib.Path,
    db: pathlib.Path,
    port: int,
    bearer: str,
    pdf_pages_per_chunk: int = 1,
) -> str:
    return f"""
vault_dir = "{vault}"
db_path = "{db}"
listen = "127.0.0.1:{port}"
allow_loopback = true
bearer = "{bearer}"
log_format = "text"

[embed]
provider = "fake"
model = "gemini-embedding-2"
dim = {DEFAULT_E2E_EMBED_DIM}

[media]
enabled = true
pdf_pages_per_chunk = {pdf_pages_per_chunk}
pdf_dpi = 72
max_file_mb = 20
"""


def _spawn_toml_server(
    docindex_bin: pathlib.Path,
    vault: pathlib.Path,
    db: pathlib.Path,
    port: int,
    cfg_path: pathlib.Path,
    log_path: pathlib.Path,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("DOCINDEX_", "GEMINI_", "VOYAGE_"))
    }
    if extra_env:
        env.update(extra_env)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [str(docindex_bin), "--config", str(cfg_path)],
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, f"http://127.0.0.1:{port}", log_f


def _wait_ready(
    proc: subprocess.Popen,
    base_url: str,
    log_path: pathlib.Path,
    timeout: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early (code {proc.returncode})\n"
                f"log:\n{log_path.read_text()}"
            )
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("ok"):
                return
        except httpx.RequestError as e:
            last_err = e
        time.sleep(0.1)
    raise RuntimeError(
        f"server did not become ready: {last_err}\nlog:\n{log_path.read_text()}"
    )


def _stop(proc: subprocess.Popen, log_f) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
    log_f.close()


def _wait_for_chunks(
    base_url: str,
    bearer: str,
    n: int,
    timeout: float = 20.0,
) -> int:
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        try:
            r = httpx.get(
                f"{base_url}/health",
                headers={"Authorization": f"Bearer {bearer}"},
                timeout=5.0,
            )
            if r.status_code == 200:
                last = r.json().get("indexed_chunks", -1)
                if last >= n:
                    return last
        except httpx.RequestError:
            pass
        time.sleep(0.3)
    raise AssertionError(
        f"indexed_chunks did not reach {n} within {timeout}s (last={last})"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def media_vault(tmp_path: pathlib.Path):
    """Vault with one PNG, one two-page PDF, and a text note."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note\n\ndistinctive text sphinx quokka\n")
    (vault / "photo.png").write_bytes(_make_minimal_png())
    (vault / "report.pdf").write_bytes(_make_minimal_pdf(page_count=2))
    return vault


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_startup_scan_indexes_image_and_pdf_into_vectors(
    docindex_bin, tmp_path: pathlib.Path, media_vault: pathlib.Path
):
    """Image and PDF chunks are stored in chunks_vec, not in chunks_fts."""
    port = _free_port()
    bearer = "test-media"
    db = tmp_path / "index.db"
    cfg = tmp_path / "server.toml"
    log = tmp_path / "server.log"
    # Two-page PDF with pdf_pages_per_chunk=1 → 2 PDF chunks + 1 PNG + 1 text = 4 total.
    cfg.write_text(_server_toml(media_vault, db, port, bearer, pdf_pages_per_chunk=1))

    proc, base_url, log_f = _spawn_toml_server(
        docindex_bin, media_vault, db, port, cfg, log
    )
    try:
        _wait_ready(proc, base_url, log, timeout=15.0)
        # Wait for all 4 chunks: 1 text + 1 image + 2 PDF pages.
        total = _wait_for_chunks(base_url, bearer, 4, timeout=30.0)
        assert total == 4, f"expected 4 chunks, got {total}"

        # Verify via SQLite that media chunks are vector-only.
        import sqlite3
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT path, media_type FROM chunks ORDER BY path, chunk_idx"
        ).fetchall()
        conn.close()

        paths_types = [(r[0], r[1]) for r in rows]
        # Identify what's image and pdf
        image_rows = [(p, t) for p, t in paths_types if t == "image"]
        pdf_rows = [(p, t) for p, t in paths_types if t == "pdf"]
        text_rows = [(p, t) for p, t in paths_types if t == "text"]

        assert len(image_rows) == 1, f"expected 1 image chunk, got {image_rows}"
        assert len(pdf_rows) == 2, f"expected 2 PDF chunks, got {pdf_rows}"
        assert len(text_rows) >= 1, f"expected at least 1 text chunk, got {text_rows}"

        # Verify FTS index membership using the fts5 docsize shadow table,
        # which holds exactly one row per document actually indexed in FTS.
        # Querying via MATCH 'sphinx' would be vacuous because media chunks
        # store empty content and could never match any term even if they were
        # erroneously inserted; docsize membership is the only reliable check.
        #
        # Precondition: verify the shadow table is present and readable.  If
        # fts5 internals ever change and the table vanishes or is renamed, this
        # assertion will fail loudly rather than silently turning into a no-op.
        conn2 = sqlite3.connect(str(db))
        try:
            shadow_accessible = conn2.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type='table' AND name='chunks_fts_docsize'"
            ).fetchone()[0]
            assert shadow_accessible == 1, (
                "chunks_fts_docsize shadow table is missing; the FTS membership "
                "check below would be a no-op.  Either fts5 internals changed or "
                "the FTS table was not created correctly."
            )
            # Smoke-read the shadow table to confirm it is actually readable.
            conn2.execute("SELECT COUNT(*) FROM chunks_fts_docsize").fetchone()
        finally:
            conn2.close()

        conn2 = sqlite3.connect(str(db))
        text_ids = set(
            r[0]
            for r in conn2.execute(
                "SELECT id FROM chunks WHERE media_type = 'text'"
            ).fetchall()
        )
        all_chunk_ids = set(
            r[0]
            for r in conn2.execute("SELECT id FROM chunks").fetchall()
        )
        fts_indexed_ids = set(
            r[0]
            for r in conn2.execute("SELECT id FROM chunks_fts_docsize").fetchall()
        )
        conn2.close()
        non_text_in_fts = fts_indexed_ids - text_ids
        missing_text_from_fts = text_ids - fts_indexed_ids
        assert not non_text_in_fts, (
            f"FTS index contains non-text chunk rowids: {non_text_in_fts}. "
            f"All chunk ids: {all_chunk_ids}, text ids: {text_ids}"
        )
        assert not missing_text_from_fts, (
            f"Text chunks missing from FTS index: {missing_text_from_fts}"
        )

        # Verify file state is recorded for all three files.
        conn3 = sqlite3.connect(str(db))
        file_paths = {
            r[0]
            for r in conn3.execute("SELECT path FROM files").fetchall()
        }
        conn3.close()
        assert "photo.png" in file_paths, f"photo.png missing from files: {file_paths}"
        assert "report.pdf" in file_paths, f"report.pdf missing from files: {file_paths}"
        assert "note.md" in file_paths, f"note.md missing from files: {file_paths}"

        # Verify media metadata is recorded.
        conn4 = sqlite3.connect(str(db))
        pdf_meta = conn4.execute(
            "SELECT media_type, mime_type, media_start, media_end, media_unit "
            "FROM chunks WHERE path = 'report.pdf' ORDER BY chunk_idx"
        ).fetchall()
        conn4.close()
        assert len(pdf_meta) == 2
        for i, (mt, mime, ms, me, mu) in enumerate(pdf_meta):
            assert mt == "pdf"
            assert mime == "application/pdf"
            assert ms == i, f"page {i}: media_start={ms}"
            assert me == i + 1, f"page {i}: media_end={me}"
            assert mu == "page"

    finally:
        _stop(proc, log_f)


def test_search_returns_media_hits_and_media_only_excludes_text(
    docindex_bin, tmp_path: pathlib.Path, media_vault: pathlib.Path
):
    """/search returns text+media; media_only=true returns only non-text."""
    port = _free_port()
    bearer = "test-media"
    db = tmp_path / "index.db"
    cfg = tmp_path / "server.toml"
    log = tmp_path / "server.log"
    cfg.write_text(_server_toml(media_vault, db, port, bearer, pdf_pages_per_chunk=1))

    proc, base_url, log_f = _spawn_toml_server(
        docindex_bin, media_vault, db, port, cfg, log
    )
    try:
        _wait_ready(proc, base_url, log, timeout=15.0)
        _wait_for_chunks(base_url, bearer, 4, timeout=30.0)

        hdrs = {"Authorization": f"Bearer {bearer}"}

        # Default /search must be able to return media hits.
        r = httpx.post(
            f"{base_url}/search",
            json={"query": "sphinx quokka photo", "limit": 10},
            headers=hdrs,
            timeout=10.0,
        )
        assert r.status_code == 200, r.text
        hits = r.json()["hits"]
        media_hit_paths = {h["path"] for h in hits if h["media_type"] != "text"}
        assert media_hit_paths, (
            "default /search should include media hits; got none\n"
            f"all hits: {[h['path'] for h in hits]}"
        )

        # media_only=true must return NO text hits.
        r2 = httpx.post(
            f"{base_url}/search",
            json={"query": "sphinx quokka photo", "limit": 10, "media_only": True},
            headers=hdrs,
            timeout=10.0,
        )
        assert r2.status_code == 200, r2.text
        hits2 = r2.json()["hits"]
        assert hits2, (
            "media_only search must return media hits; fixture has one image "
            "and one PDF"
        )
        text_leaks = [h for h in hits2 if h["media_type"] == "text"]
        assert not text_leaks, (
            f"media_only returned text hits: {text_leaks}"
        )
        media_only_paths = {h["path"] for h in hits2}
        assert media_only_paths == {"photo.png", "report.pdf"}, (
            f"media_only must return exactly the fixture media paths, got {media_only_paths}"
        )
        # The media_only rank-1 hit must have score_normalized == 1.0 when
        # there is at least one media chunk.
        top_norm = hits2[0]["score_normalized"]
        assert abs(top_norm - 1.0) < 1e-9, (
            f"media_only rank-1 score_normalized should be 1.0, got {top_norm}"
        )

        r3 = httpx.post(
            f"{base_url}/search",
            json={
                "query": "sphinx quokka photo",
                "limit": 10,
                "media_only": True,
                "media_types": ["pdf"],
            },
            headers=hdrs,
            timeout=10.0,
        )
        assert r3.status_code == 200, r3.text
        pdf_hits = r3.json()["hits"]
        assert pdf_hits, "PDF filter fixture must return at least one PDF hit"
        assert {hit["media_type"] for hit in pdf_hits} == {"pdf"}, pdf_hits
        assert "report.pdf" in {hit["path"] for hit in pdf_hits}, pdf_hits
        assert "photo.png" not in {hit["path"] for hit in pdf_hits}, pdf_hits

    finally:
        _stop(proc, log_f)


def test_watcher_handles_media_modify_and_delete(
    docindex_bin, tmp_path: pathlib.Path
):
    """Watcher picks up a modified image and then its deletion."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "anchor.md").write_text("# Anchor\n\nanchor text\n")
    img_path = vault / "watch_img.png"
    img_path.write_bytes(_make_minimal_png(4, 4))

    port = _free_port()
    bearer = "test-media"
    db = tmp_path / "index.db"
    cfg = tmp_path / "server.toml"
    log = tmp_path / "server.log"
    cfg.write_text(
        f"""
vault_dir = "{vault}"
db_path = "{db}"
listen = "127.0.0.1:{port}"
allow_loopback = true
bearer = "{bearer}"
log_format = "text"
[embed]
provider = "fake"
model = "gemini-embedding-2"
dim = {DEFAULT_E2E_EMBED_DIM}
[media]
enabled = true
pdf_pages_per_chunk = 1
pdf_dpi = 72
max_file_mb = 20
"""
    )

    proc, base_url, log_f = _spawn_toml_server(
        docindex_bin, vault, db, port, cfg, log
    )
    try:
        _wait_ready(proc, base_url, log, timeout=15.0)
        # 1 text + 1 image = 2 chunks.
        _wait_for_chunks(base_url, bearer, 2, timeout=30.0)

        # Capture the content_hash and the cached embedding blob for the image
        # BEFORE overwriting the file. The file-state row already exists at this
        # point (written during the startup scan), so polling for row existence
        # would be vacuous. Capturing the blob before modification lets us assert
        # later that the vector actually changed, not merely that some blob exists.
        import sqlite3
        import struct
        conn_pre = sqlite3.connect(str(db))
        pre_row = conn_pre.execute(
            "SELECT content_hash FROM files WHERE path = 'watch_img.png'"
        ).fetchone()
        conn_pre.close()
        assert pre_row is not None, (
            "watch_img.png file-state row must exist after initial scan"
        )
        hash_before = pre_row[0]

        # Read the cached embedding blob for the original image.
        conn_blob_pre = sqlite3.connect(str(db))
        blob_pre_row = conn_blob_pre.execute(
            "SELECT ec.embedding "
            "FROM chunks c "
            "JOIN embedding_cache ec ON ec.content_hash = c.content_hash "
            "WHERE c.path = 'watch_img.png' "
            "LIMIT 1"
        ).fetchone()
        conn_blob_pre.close()
        assert blob_pre_row is not None, (
            "embedding_cache must have an entry for watch_img.png before modification"
        )
        blob_before = blob_pre_row[0]

        # Modify the image and wait for the stored hash to change.
        img_path.write_bytes(_make_minimal_png(8, 8))

        modified_ok = False
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            conn = sqlite3.connect(str(db))
            row = conn.execute(
                "SELECT content_hash FROM files WHERE path = 'watch_img.png'"
            ).fetchone()
            conn.close()
            if row is not None and row[0] != hash_before:
                modified_ok = True
                break
            time.sleep(0.3)
        assert modified_ok, (
            f"watcher did not update content_hash for watch_img.png "
            f"(hash_before={hash_before!r})"
        )

        # Verify the cached embedding was updated. The Fake embedder seeds its
        # output from the media bytes, so 4×4 and 8×8 PNGs produce distinct
        # vectors. Read the new blob and assert it is present, decodes to floats,
        # is not all-zero, and differs from the pre-modification blob.
        conn_cache = sqlite3.connect(str(db))
        cache_row = conn_cache.execute(
            "SELECT ec.embedding "
            "FROM chunks c "
            "JOIN embedding_cache ec ON ec.content_hash = c.content_hash "
            "WHERE c.path = 'watch_img.png' "
            "LIMIT 1"
        ).fetchone()
        conn_cache.close()
        assert cache_row is not None, (
            "embedding_cache must have an entry for watch_img.png after re-index"
        )
        blob_after = cache_row[0]
        assert len(blob_after) > 0, (
            "cached embedding for modified image must not be empty"
        )
        n_floats = len(blob_after) // 4
        assert n_floats > 0, "embedding blob must contain at least one float"
        floats = struct.unpack(f"{n_floats}f", blob_after[:n_floats * 4])
        assert any(f != 0.0 for f in floats), (
            "embedding must not be all-zero"
        )
        assert blob_after != blob_before, (
            "embedding blob for the modified image must differ from the original; "
            "the watcher must have re-embedded the new file contents"
        )

        # Delete the image; chunk count should drop to 1.
        img_path.unlink()
        deadline = time.monotonic() + 15.0
        removed = False
        while time.monotonic() < deadline:
            try:
                r = httpx.get(
                    f"{base_url}/health",
                    headers={"Authorization": f"Bearer {bearer}"},
                    timeout=5.0,
                )
                if r.status_code == 200 and r.json().get("indexed_chunks", 99) == 1:
                    removed = True
                    break
            except httpx.RequestError:
                pass
            time.sleep(0.3)
        assert removed, (
            "image deletion should reduce chunk count to 1\n"
            f"log:\n{log.read_text()}"
        )

        # File state for the deleted image must be gone.
        import sqlite3
        conn = sqlite3.connect(str(db))
        row = conn.execute(
            "SELECT path FROM files WHERE path = 'watch_img.png'"
        ).fetchone()
        conn.close()
        assert row is None, "file state for deleted image must be removed"

    finally:
        _stop(proc, log_f)
