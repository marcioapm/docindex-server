package config

import (
	"path/filepath"
	"strings"
	"testing"
)

func mapLookup(m map[string]string) Lookup {
	return func(k string) (string, bool) {
		v, ok := m[k]
		return v, ok
	}
}

func baseEnv(t *testing.T) map[string]string {
	t.Helper()
	dir := t.TempDir()
	return map[string]string{
		"DOCINDEX_VAULT_DIR": dir,
		"DOCINDEX_DB_PATH":   filepath.Join(dir, "index.db"),
		"DOCINDEX_LISTEN":    "100.83.46.59:7777",
		"DOCINDEX_BEARER":    "secret",
		"GEMINI_API_KEY":     "key",
	}
}

func TestLoad_Valid(t *testing.T) {
	env := baseEnv(t)
	c, err := LoadFrom(mapLookup(env))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.VaultDir != env["DOCINDEX_VAULT_DIR"] {
		t.Errorf("VaultDir = %q, want %q", c.VaultDir, env["DOCINDEX_VAULT_DIR"])
	}
	if c.EmbedModel != "gemini-embedding-001" {
		t.Errorf("EmbedModel default = %q", c.EmbedModel)
	}
	if c.EmbedDim != 768 {
		t.Errorf("EmbedDim default = %d, want 768", c.EmbedDim)
	}
	if c.LogFormat != "json" {
		t.Errorf("LogFormat default = %q, want json", c.LogFormat)
	}
	if c.DebounceDur.Milliseconds() != 5000 {
		t.Errorf("DebounceDur default = %v", c.DebounceDur)
	}
}

func TestLoad_MissingRequired(t *testing.T) {
	tests := []struct {
		name   string
		delete string
	}{
		{"vault", "DOCINDEX_VAULT_DIR"},
		{"db", "DOCINDEX_DB_PATH"},
		{"listen", "DOCINDEX_LISTEN"},
		{"bearer", "DOCINDEX_BEARER"},
		{"gemini", "GEMINI_API_KEY"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			env := baseEnv(t)
			delete(env, tc.delete)
			_, err := LoadFrom(mapLookup(env))
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.delete) {
				t.Errorf("error %q does not mention %q", err, tc.delete)
			}
		})
	}
}

func TestLoad_Rejects0000(t *testing.T) {
	env := baseEnv(t)
	env["DOCINDEX_LISTEN"] = "0.0.0.0:7777"
	_, err := LoadFrom(mapLookup(env))
	if err == nil {
		t.Fatalf("expected error for 0.0.0.0 bind")
	}
	if !strings.Contains(err.Error(), "0.0.0.0") {
		t.Errorf("error should mention 0.0.0.0: %v", err)
	}
}

func TestLoad_RejectsIPv6Unspecified(t *testing.T) {
	env := baseEnv(t)
	env["DOCINDEX_LISTEN"] = "[::]:7777"
	_, err := LoadFrom(mapLookup(env))
	if err == nil {
		t.Fatalf("expected error for [::] bind")
	}
}

func TestLoad_VaultMustExist(t *testing.T) {
	env := baseEnv(t)
	env["DOCINDEX_VAULT_DIR"] = filepath.Join(t.TempDir(), "does-not-exist")
	_, err := LoadFrom(mapLookup(env))
	if err == nil {
		t.Fatalf("expected error for missing vault")
	}
}

func TestLoad_DBParentMustExist(t *testing.T) {
	env := baseEnv(t)
	env["DOCINDEX_DB_PATH"] = "/definitely/not/a/real/path/index.db"
	_, err := LoadFrom(mapLookup(env))
	if err == nil {
		t.Fatalf("expected error for missing DB parent")
	}
}

func TestLoad_RelativePathsExpanded(t *testing.T) {
	env := baseEnv(t)
	// Use "." which should always exist and resolve to an absolute path.
	env["DOCINDEX_VAULT_DIR"] = "."
	c, err := LoadFrom(mapLookup(env))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !filepath.IsAbs(c.VaultDir) {
		t.Errorf("VaultDir %q should be absolute", c.VaultDir)
	}
}

func TestLoad_InvalidIntField(t *testing.T) {
	env := baseEnv(t)
	env["DOCINDEX_EMBED_DIM"] = "not-a-number"
	_, err := LoadFrom(mapLookup(env))
	if err == nil {
		t.Fatalf("expected error for invalid int")
	}
}

func TestLoad_InvalidLogFormat(t *testing.T) {
	env := baseEnv(t)
	env["DOCINDEX_LOG_FORMAT"] = "yaml"
	_, err := LoadFrom(mapLookup(env))
	if err == nil {
		t.Fatalf("expected error for invalid log format")
	}
}

func TestLoad_ListenMissingPort(t *testing.T) {
	env := baseEnv(t)
	env["DOCINDEX_LISTEN"] = "100.83.46.59"
	_, err := LoadFrom(mapLookup(env))
	if err == nil {
		t.Fatalf("expected error for missing port")
	}
}
