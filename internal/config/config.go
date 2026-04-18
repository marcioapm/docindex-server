// Package config parses and validates environment-based configuration.
//
// All runtime configuration comes from environment variables (12-factor);
// there are no config files. See .env.example for the full list.
package config

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Config is the typed, validated runtime configuration.
type Config struct {
	VaultDir    string        // DOCINDEX_VAULT_DIR (must exist)
	DBPath      string        // DOCINDEX_DB_PATH (parent dir must exist)
	Listen      string        // DOCINDEX_LISTEN (host:port; host must not be 0.0.0.0)
	Bearer      string        // DOCINDEX_BEARER (required)
	GeminiKey   string        // GEMINI_API_KEY (required)
	EmbedModel  string        // DOCINDEX_EMBED_MODEL (default gemini-embedding-001)
	EmbedDim    int           // DOCINDEX_EMBED_DIM (default 768)
	DebounceDur time.Duration // DOCINDEX_DEBOUNCE_MS (default 5000ms)
	LogFormat   string        // DOCINDEX_LOG_FORMAT: "json" (default) or "text"
	HTTPTimeout time.Duration // DOCINDEX_HTTP_TIMEOUT_MS (default 30000ms)
}

// Lookup is the env-access interface used by Load. Defaults to os.LookupEnv.
type Lookup func(key string) (string, bool)

// Load reads configuration from the process environment and validates it.
func Load() (*Config, error) {
	return LoadFrom(os.LookupEnv)
}

// LoadFrom reads configuration using the given env lookup function.
// Exposed for testability.
func LoadFrom(getenv Lookup) (*Config, error) {
	c := &Config{
		VaultDir:   getOrDefault(getenv, "DOCINDEX_VAULT_DIR", ""),
		DBPath:     getOrDefault(getenv, "DOCINDEX_DB_PATH", ""),
		Listen:     getOrDefault(getenv, "DOCINDEX_LISTEN", ""),
		Bearer:     getOrDefault(getenv, "DOCINDEX_BEARER", ""),
		GeminiKey:  getOrDefault(getenv, "GEMINI_API_KEY", ""),
		EmbedModel: getOrDefault(getenv, "DOCINDEX_EMBED_MODEL", "gemini-embedding-001"),
		LogFormat:  strings.ToLower(getOrDefault(getenv, "DOCINDEX_LOG_FORMAT", "json")),
	}

	dim, err := parseIntDefault(getenv, "DOCINDEX_EMBED_DIM", 768)
	if err != nil {
		return nil, err
	}
	c.EmbedDim = dim

	debounceMs, err := parseIntDefault(getenv, "DOCINDEX_DEBOUNCE_MS", 5000)
	if err != nil {
		return nil, err
	}
	c.DebounceDur = time.Duration(debounceMs) * time.Millisecond

	httpMs, err := parseIntDefault(getenv, "DOCINDEX_HTTP_TIMEOUT_MS", 30000)
	if err != nil {
		return nil, err
	}
	c.HTTPTimeout = time.Duration(httpMs) * time.Millisecond

	if err := c.validate(); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *Config) validate() error {
	var errs []string

	if c.VaultDir == "" {
		errs = append(errs, "DOCINDEX_VAULT_DIR is required")
	} else {
		abs, err := filepath.Abs(c.VaultDir)
		if err != nil {
			errs = append(errs, fmt.Sprintf("DOCINDEX_VAULT_DIR: %v", err))
		} else {
			c.VaultDir = abs
			info, err := os.Stat(abs)
			if err != nil {
				errs = append(errs, fmt.Sprintf("DOCINDEX_VAULT_DIR %q: %v", abs, err))
			} else if !info.IsDir() {
				errs = append(errs, fmt.Sprintf("DOCINDEX_VAULT_DIR %q is not a directory", abs))
			}
		}
	}

	if c.DBPath == "" {
		errs = append(errs, "DOCINDEX_DB_PATH is required")
	} else {
		abs, err := filepath.Abs(c.DBPath)
		if err != nil {
			errs = append(errs, fmt.Sprintf("DOCINDEX_DB_PATH: %v", err))
		} else {
			c.DBPath = abs
			parent := filepath.Dir(abs)
			info, err := os.Stat(parent)
			if err != nil {
				errs = append(errs, fmt.Sprintf("DOCINDEX_DB_PATH parent %q: %v", parent, err))
			} else if !info.IsDir() {
				errs = append(errs, fmt.Sprintf("DOCINDEX_DB_PATH parent %q is not a directory", parent))
			}
		}
	}

	if c.Listen == "" {
		errs = append(errs, "DOCINDEX_LISTEN is required")
	} else if err := validateListen(c.Listen); err != nil {
		errs = append(errs, err.Error())
	}

	if c.Bearer == "" {
		errs = append(errs, "DOCINDEX_BEARER is required")
	}

	if c.GeminiKey == "" {
		errs = append(errs, "GEMINI_API_KEY is required")
	}

	if c.EmbedDim <= 0 {
		errs = append(errs, "DOCINDEX_EMBED_DIM must be > 0")
	}

	if c.LogFormat != "json" && c.LogFormat != "text" {
		errs = append(errs, fmt.Sprintf("DOCINDEX_LOG_FORMAT %q: must be 'json' or 'text'", c.LogFormat))
	}

	if len(errs) > 0 {
		return errors.New("config: " + strings.Join(errs, "; "))
	}
	return nil
}

// validateListen checks that addr parses as host:port and the host is
// not 0.0.0.0 (or the v6 equivalent). Tailscale IPs are in 100.64.0.0/10
// (CGNAT) but we don't hard-require that here — we just reject the obvious
// footguns. Operators wanting a stricter bind can set a specific IP.
func validateListen(addr string) error {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("DOCINDEX_LISTEN %q: %w", addr, err)
	}
	if port == "" {
		return fmt.Errorf("DOCINDEX_LISTEN %q: empty port", addr)
	}
	if _, err := strconv.Atoi(port); err != nil {
		return fmt.Errorf("DOCINDEX_LISTEN %q: port not numeric: %w", addr, err)
	}
	if host == "" {
		return fmt.Errorf("DOCINDEX_LISTEN %q: empty host (refusing to bind to all interfaces)", addr)
	}
	if host == "0.0.0.0" || host == "::" || host == "[::]" {
		return fmt.Errorf("DOCINDEX_LISTEN %q: binding to all interfaces is not allowed; use a Tailscale IP", addr)
	}
	ip := net.ParseIP(host)
	if ip != nil && ip.IsUnspecified() {
		return fmt.Errorf("DOCINDEX_LISTEN %q: unspecified IP is not allowed", addr)
	}
	return nil
}

func getOrDefault(getenv Lookup, key, def string) string {
	if v, ok := getenv(key); ok && v != "" {
		return v
	}
	return def
}

func parseIntDefault(getenv Lookup, key string, def int) (int, error) {
	v, ok := getenv(key)
	if !ok || v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("%s %q: must be an integer: %w", key, v, err)
	}
	return n, nil
}
