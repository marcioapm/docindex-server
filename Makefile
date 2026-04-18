.PHONY: fmt vet lint test build run tidy all

GO ?= go
BIN_DIR := bin
BIN := $(BIN_DIR)/docindex

all: fmt vet test build

fmt:
	$(GO) fmt ./...

vet:
	$(GO) vet ./...

lint:
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run; \
	else \
		echo "golangci-lint not installed; skipping (install from https://golangci-lint.run)"; \
	fi

test:
	$(GO) test ./... -race -cover

build: $(BIN_DIR)
	$(GO) build -o $(BIN) ./cmd/docindex

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run:
	$(GO) run ./cmd/docindex

tidy:
	$(GO) mod tidy
