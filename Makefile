.PHONY: help fmt fmt-check clippy test pytest check build build-release run clean

CARGO ?= cargo
PYTHON ?= python3

help:
	@echo "Targets:"
	@echo "  fmt           cargo fmt --all"
	@echo "  fmt-check     cargo fmt --all -- --check"
	@echo "  clippy        cargo clippy --all-targets --all-features -- -D warnings"
	@echo "  test          cargo test --all"
	@echo "  pytest        python3 tests/run_tests.py"
	@echo "  check         fmt-check + clippy + test + pytest (pre-push)"
	@echo "  build         cargo build"
	@echo "  build-release cargo build --release"
	@echo "  run           cargo run (requires .env sourced)"
	@echo "  clean         cargo clean"

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

test:
	$(CARGO) test --all

pytest:
	$(PYTHON) tests/run_tests.py

check: fmt-check clippy test pytest

build:
	$(CARGO) build

build-release:
	$(CARGO) build --release

run:
	$(CARGO) run

clean:
	$(CARGO) clean
