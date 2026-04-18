package store

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/marcioapm/docindex-server/internal/chunk"
)

func openTemp(t *testing.T) *Store {
	t.Helper()
	dir := t.TempDir()
	s, err := Open(context.Background(), filepath.Join(dir, "x.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestOpen_AppliesSchema(t *testing.T) {
	s := openTemp(t)
	v, ok, err := s.GetMeta(context.Background(), "schema_version")
	if err != nil {
		t.Fatalf("GetMeta: %v", err)
	}
	if !ok {
		t.Fatalf("schema_version not set")
	}
	if v != SchemaVersion {
		t.Errorf("schema_version = %q, want %q", v, SchemaVersion)
	}
}

func TestOpen_FTSAvailable(t *testing.T) {
	// Smoke-test that FTS5 is compiled into the driver by creating a
	// contentless index and querying it. (sqlite-vec is deferred to
	// Phase 2 — see package doc comment.)
	s := openTemp(t)
	var n int
	if err := s.DB().QueryRow(`SELECT COUNT(*) FROM chunks_fts`).Scan(&n); err != nil {
		t.Fatalf("fts count: %v", err)
	}
}

func TestUpsertChunk_InsertAndFTS(t *testing.T) {
	s := openTemp(t)
	c := chunk.Chunk{
		Idx: 0, Heading: "T", HeadingPath: "T",
		Content: "# T\nhello world", ContentHash: "hash1", Tokens: 3,
	}
	id, err := s.UpsertChunk(context.Background(), c, "/vault/a.md", 42)
	if err != nil {
		t.Fatalf("UpsertChunk: %v", err)
	}
	if id <= 0 {
		t.Fatalf("bad id: %d", id)
	}
	// chunks has the row.
	var content string
	if err := s.DB().QueryRow(`SELECT content FROM chunks WHERE id=?`, id).Scan(&content); err != nil {
		t.Fatalf("select chunks: %v", err)
	}
	if content != c.Content {
		t.Errorf("content = %q, want %q", content, c.Content)
	}
	// FTS has the row (search for "hello" via MATCH).
	var hits int
	if err := s.DB().QueryRow(`SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'hello'`).Scan(&hits); err != nil {
		t.Fatalf("fts count: %v", err)
	}
	if hits != 1 {
		t.Errorf("fts hits = %d, want 1", hits)
	}
}

func TestUpsertChunk_Updates(t *testing.T) {
	s := openTemp(t)
	ctx := context.Background()
	c1 := chunk.Chunk{Idx: 0, Content: "alpha", ContentHash: "h1"}
	id1, err := s.UpsertChunk(ctx, c1, "/vault/a.md", 1)
	if err != nil {
		t.Fatal(err)
	}
	c2 := chunk.Chunk{Idx: 0, Content: "beta changed", ContentHash: "h2", HeadingPath: "X"}
	id2, err := s.UpsertChunk(ctx, c2, "/vault/a.md", 2)
	if err != nil {
		t.Fatal(err)
	}
	if id1 != id2 {
		t.Errorf("rowid should be stable on update: %d -> %d", id1, id2)
	}
	var hits int
	if err := s.DB().QueryRow(`SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'alpha'`).Scan(&hits); err != nil {
		t.Fatal(err)
	}
	if hits != 0 {
		t.Errorf("old fts row should be gone, got %d", hits)
	}
	if err := s.DB().QueryRow(`SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'beta'`).Scan(&hits); err != nil {
		t.Fatal(err)
	}
	if hits != 1 {
		t.Errorf("new fts row missing: %d", hits)
	}
}

func TestDeleteChunksForPath(t *testing.T) {
	s := openTemp(t)
	ctx := context.Background()
	c := chunk.Chunk{Idx: 0, Content: "searchable token", ContentHash: "h"}
	id, err := s.UpsertChunk(ctx, c, "/vault/a.md", 1)
	if err != nil {
		t.Fatal(err)
	}
	// Also write a vector so we can verify it gets cleaned up.
	if err := s.SetVectorForChunk(ctx, id, make([]float32, 768)); err != nil {
		t.Fatalf("SetVectorForChunk: %v", err)
	}
	if err := s.DeleteChunksForPath(ctx, "/vault/a.md"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	var n int
	if err := s.DB().QueryRow(`SELECT COUNT(*) FROM chunks`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("chunks not deleted: %d", n)
	}
	if err := s.DB().QueryRow(`SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'searchable'`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("fts not deleted: %d", n)
	}
	if err := s.DB().QueryRow(`SELECT COUNT(*) FROM chunks_vec WHERE rowid=?`, id).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("vec not deleted: %d", n)
	}
}

func TestEmbeddingCache_Roundtrip(t *testing.T) {
	s := openTemp(t)
	ctx := context.Background()
	vec := []float32{0.1, -0.2, 0.3, 0.4}
	if err := s.PutEmbeddingCache(ctx, "h", "m", "RETRIEVAL_DOCUMENT", 4, vec); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, ok, err := s.GetEmbeddingCache(ctx, "h")
	if err != nil || !ok {
		t.Fatalf("Get: err=%v ok=%v", err, ok)
	}
	if len(got) != len(vec) {
		t.Fatalf("len mismatch: %d vs %d", len(got), len(vec))
	}
	for i := range vec {
		if got[i] != vec[i] {
			t.Errorf("vec[%d] = %v, want %v", i, got[i], vec[i])
		}
	}
	_, ok, err = s.GetEmbeddingCache(ctx, "missing")
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Errorf("missing key should return ok=false")
	}
}

func TestEmbeddingCache_DimMismatch(t *testing.T) {
	s := openTemp(t)
	err := s.PutEmbeddingCache(context.Background(), "h", "m", "t", 4, []float32{1, 2, 3})
	if err == nil {
		t.Errorf("expected dim mismatch error")
	}
}

func TestSetVectorForChunk(t *testing.T) {
	s := openTemp(t)
	ctx := context.Background()
	c := chunk.Chunk{Idx: 0, Content: "x", ContentHash: "h"}
	id, err := s.UpsertChunk(ctx, c, "/v/a.md", 1)
	if err != nil {
		t.Fatal(err)
	}
	vec := make([]float32, 768)
	for i := range vec {
		vec[i] = float32(i) / 768.0
	}
	if err := s.SetVectorForChunk(ctx, id, vec); err != nil {
		t.Fatalf("SetVectorForChunk: %v", err)
	}
	// Replace with zero vector.
	zero := make([]float32, 768)
	if err := s.SetVectorForChunk(ctx, id, zero); err != nil {
		t.Fatalf("replace: %v", err)
	}
	var n int
	if err := s.DB().QueryRow(`SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?`, id).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("vec rows = %d, want 1", n)
	}
}

func TestMeta_Roundtrip(t *testing.T) {
	s := openTemp(t)
	ctx := context.Background()
	if err := s.SetMeta(ctx, "k", "v1"); err != nil {
		t.Fatal(err)
	}
	if err := s.SetMeta(ctx, "k", "v2"); err != nil {
		t.Fatal(err)
	}
	v, ok, err := s.GetMeta(ctx, "k")
	if err != nil || !ok {
		t.Fatalf("Get: err=%v ok=%v", err, ok)
	}
	if v != "v2" {
		t.Errorf("meta = %q, want v2", v)
	}
}

func TestReopenPersists(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.db")
	s1, err := Open(context.Background(), path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	ctx := context.Background()
	c := chunk.Chunk{Idx: 0, Content: "persist me", ContentHash: "h"}
	id, err := s1.UpsertChunk(ctx, c, "/v/a.md", 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := s1.SetVectorForChunk(ctx, id, make([]float32, 768)); err != nil {
		t.Fatal(err)
	}
	if err := s1.PutEmbeddingCache(ctx, "h", "m", "RETRIEVAL_DOCUMENT", 4, []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := s1.Close(); err != nil {
		t.Fatal(err)
	}

	s2, err := Open(context.Background(), path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer s2.Close()

	var content string
	if err := s2.DB().QueryRow(`SELECT content FROM chunks WHERE id=?`, id).Scan(&content); err != nil {
		t.Fatal(err)
	}
	if content != c.Content {
		t.Errorf("content = %q, want %q", content, c.Content)
	}
	cached, ok, err := s2.GetEmbeddingCache(context.Background(), "h")
	if err != nil || !ok || len(cached) != 4 {
		t.Errorf("cache not persisted: err=%v ok=%v cached=%v", err, ok, cached)
	}
}
