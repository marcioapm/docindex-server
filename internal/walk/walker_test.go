package walk

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestScan_FiltersAndSorts(t *testing.T) {
	root := t.TempDir()

	write(t, filepath.Join(root, "b.md"), "beta")
	write(t, filepath.Join(root, "a.md"), "alpha")
	write(t, filepath.Join(root, "notes", "c.md"), "gamma")

	// Non-markdown, should be skipped.
	write(t, filepath.Join(root, "README.txt"), "nope")
	write(t, filepath.Join(root, "image.png"), "binary")

	// Dotfile, should be skipped.
	write(t, filepath.Join(root, ".secret.md"), "nope")

	// Dotdir, should be skipped entirely.
	write(t, filepath.Join(root, ".git", "config.md"), "nope")
	write(t, filepath.Join(root, ".obsidian", "plugins.md"), "nope")
	write(t, filepath.Join(root, "node_modules", "lib.md"), "nope")

	got, err := Scan(context.Background(), root)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d files, want 3: %+v", len(got), got)
	}

	wantPaths := []string{
		filepath.Join(root, "a.md"),
		filepath.Join(root, "b.md"),
		filepath.Join(root, "notes", "c.md"),
	}
	gotPaths := make([]string, len(got))
	for i, fs := range got {
		gotPaths[i] = fs.Path
	}
	if !sort.StringsAreSorted(gotPaths) {
		t.Errorf("paths not sorted: %v", gotPaths)
	}
	for i, want := range wantPaths {
		if gotPaths[i] != want {
			t.Errorf("path[%d] = %q, want %q", i, gotPaths[i], want)
		}
	}

	// Hashes match sha256 of content.
	wantHashes := map[string]string{
		filepath.Join(root, "a.md"):          sha("alpha"),
		filepath.Join(root, "b.md"):          sha("beta"),
		filepath.Join(root, "notes", "c.md"): sha("gamma"),
	}
	for _, fs := range got {
		if fs.ContentHash != wantHashes[fs.Path] {
			t.Errorf("hash(%q) = %q, want %q", fs.Path, fs.ContentHash, wantHashes[fs.Path])
		}
		if fs.MTimeNs == 0 {
			t.Errorf("mtime is zero for %q", fs.Path)
		}
	}
}

func sha(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func TestScan_NonDirectory(t *testing.T) {
	f := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Scan(context.Background(), f)
	if err == nil {
		t.Fatalf("expected error for non-directory root")
	}
}

func TestScan_MissingRoot(t *testing.T) {
	_, err := Scan(context.Background(), filepath.Join(t.TempDir(), "does-not-exist"))
	if err == nil {
		t.Fatalf("expected error for missing root")
	}
}

func TestScan_ContextCancelled(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "a.md"), "x")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Scan(ctx, root)
	if err == nil {
		t.Fatalf("expected context error")
	}
}

func TestScan_CaseInsensitiveExt(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "UPPER.MD"), "x")
	got, err := Scan(context.Background(), root)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 file, got %d", len(got))
	}
}
