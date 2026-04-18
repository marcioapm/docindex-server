// Command docindex is the binary entry point for the docindex-server.
//
// Phase 1 scope: parse configuration, open the store, log status, exit 0.
// HTTP, the watcher, and the search package are Phase 2.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/marcioapm/docindex-server/internal/config"
	"github.com/marcioapm/docindex-server/internal/store"
)

func main() {
	if err := run(); err != nil {
		// We deliberately avoid slog here because logger construction
		// depends on config.Load(). Emit a last-ditch message to stderr
		// and exit nonzero.
		fmt.Fprintf(os.Stderr, "docindex: startup failed: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	logger := newLogger(cfg.LogFormat)
	slog.SetDefault(logger)

	ctx := context.Background()
	st, err := store.Open(ctx, cfg.DBPath)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}
	defer func() {
		if err := st.Close(); err != nil {
			logger.Error("store close failed", "err", err)
		}
	}()

	schemaVer, _, _ := st.GetMeta(ctx, "schema_version")

	logger.Info("docindex-server ready (phase 1: no http/watcher yet)",
		"vault_dir", cfg.VaultDir,
		"db_path", cfg.DBPath,
		"listen", cfg.Listen,
		"embed_model", cfg.EmbedModel,
		"embed_dim", cfg.EmbedDim,
		"schema_version", schemaVer,
	)
	return nil
}

func newLogger(format string) *slog.Logger {
	opts := &slog.HandlerOptions{Level: slog.LevelInfo}
	if format == "text" {
		return slog.New(slog.NewTextHandler(os.Stderr, opts))
	}
	return slog.New(slog.NewJSONHandler(os.Stderr, opts))
}
