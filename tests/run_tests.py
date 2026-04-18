#!/usr/bin/env python3
"""Run the docindex end-to-end pytest suites.

Uses `uv` if available to manage an isolated dependency environment;
otherwise falls back to the system `pytest`. Builds the release binary
first and passes its path to the suites via the DOCINDEX_BIN env var.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
SUITES_DIR = TESTS_DIR / "suites"


def run(cmd: list[str], **kwargs) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, **kwargs)


def main() -> int:
    print(f"repo root: {REPO_ROOT}", flush=True)
    bin_path = REPO_ROOT / "target" / "release" / "docindex"
    if run(["cargo", "build", "--release", "--manifest-path", str(REPO_ROOT / "Cargo.toml")]) != 0:
        return 1
    if not bin_path.exists():
        print(f"binary not found at {bin_path}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["DOCINDEX_BIN"] = str(bin_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    if shutil.which("uv"):
        return run(
            ["uv", "run", "--project", str(TESTS_DIR), "pytest", str(SUITES_DIR), "-v"],
            env=env,
        )
    # Fallback: system pytest.
    return run(["pytest", str(SUITES_DIR), "-v"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
