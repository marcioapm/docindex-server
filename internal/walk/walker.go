// Package walk performs the initial full-tree scan of a markdown vault.
//
// It emits a deterministic slice of FileState entries (sorted by path) that
// downstream code diffs against stored content_hash values to decide which
// files need to be re-chunked and re-embedded.
package walk

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// FileState is a snapshot of one markdown file in the vault.
type FileState struct {
	Path        string // absolute path
	MTimeNs     int64
	ContentHash string // hex sha256 of the file's bytes
}

// skippedDirs are directory names we never descend into, regardless of depth.
var skippedDirs = map[string]struct{}{
	".git":         {},
	".obsidian":    {},
	"node_modules": {},
}

// Scan walks root recursively and returns one FileState per markdown file.
//
// Rules:
//   - Only files ending in ".md" (case-insensitive) are returned.
//   - Directories whose names start with "." are skipped, plus the explicit
//     deny list in skippedDirs.
//   - Files whose names start with "." are skipped.
//   - Results are sorted by path for determinism.
//   - Respects ctx cancellation.
func Scan(ctx context.Context, root string) ([]FileState, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("walk: abs(%q): %w", root, err)
	}
	info, err := os.Stat(absRoot)
	if err != nil {
		return nil, fmt.Errorf("walk: stat(%q): %w", absRoot, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("walk: %q is not a directory", absRoot)
	}

	var out []FileState
	err = filepath.WalkDir(absRoot, func(path string, d os.DirEntry, werr error) error {
		if werr != nil {
			return werr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if path == absRoot {
				return nil
			}
			if _, skip := skippedDirs[name]; skip || strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasPrefix(name, ".") {
			return nil
		}
		if !strings.EqualFold(filepath.Ext(name), ".md") {
			return nil
		}
		fs, err := hashFile(path, d)
		if err != nil {
			return err
		}
		out = append(out, fs)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk: %w", err)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

func hashFile(path string, d os.DirEntry) (FileState, error) {
	info, err := d.Info()
	if err != nil {
		return FileState{}, fmt.Errorf("walk: info(%q): %w", path, err)
	}
	f, err := os.Open(path)
	if err != nil {
		return FileState{}, fmt.Errorf("walk: open(%q): %w", path, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return FileState{}, fmt.Errorf("walk: hash(%q): %w", path, err)
	}
	return FileState{
		Path:        path,
		MTimeNs:     info.ModTime().UnixNano(),
		ContentHash: hex.EncodeToString(h.Sum(nil)),
	}, nil
}
