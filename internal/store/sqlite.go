// Package store owns the SQLite database: schema management, chunk/FTS
// upserts, embedding cache, and vector storage.
//
// Phase 1 deviation from the original CLAUDE.md spec:
//
//	We use modernc.org/sqlite (pure-Go, transpiled C) as originally named,
//	but do NOT load the sqlite-vec extension in Phase 1. modernc cannot
//	load compiled C extensions (ABI mismatch), and the alternative
//	(ncruces/go-sqlite3 + sqlite-vec-go-bindings) has a pinned-old
//	wazero that prevents the shipped WASM from running on current
//	toolchains. Phase 1 stores embeddings in a plain BLOB-valued
//	`chunks_vec(rowid INTEGER PRIMARY KEY, embedding BLOB)` table so that
//	the store API (UpsertChunk / SetVectorForChunk / embedding cache /
//	persistence) is fully exercisable end-to-end. Phase 2 (when search
//	and RRF are implemented) will replace this with either (a) the
//	sqlite-vec `vec0` virtual table via a fixed ncruces binding, or
//	(b) an in-Go cosine scan for small vaults. The schema.sql column
//	layout already matches what sqlite-vec will consume.
//
// FTS5 is available in modernc.org/sqlite and is used normally.
package store

import (
	"bytes"
	"context"
	"database/sql"
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/marcioapm/docindex-server/internal/chunk"

	_ "modernc.org/sqlite" // database/sql driver name "sqlite"
)

//go:embed schema.sql
var schemaSQL string

// SchemaVersion is written to meta.schema_version on open.
const SchemaVersion = "1"

// Store wraps a *sql.DB with the docindex schema applied.
type Store struct {
	db *sql.DB
}

// Open returns a Store backed by the SQLite DB at path.
func Open(ctx context.Context, path string) (*Store, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("store: open %q: %w", path, err)
	}
	// Single-writer SQLite app. Keep the pool tiny so PRAGMAs and writes
	// are consistent across callers.
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)

	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("store: ping: %w", err)
	}

	// Apply schema. CREATE IF NOT EXISTS makes this idempotent. Split on
	// ";" because modernc's Exec does not handle multi-statement strings
	// in every version.
	for _, stmt := range splitStatements(schemaSQL) {
		if _, err := db.ExecContext(ctx, stmt); err != nil {
			_ = db.Close()
			return nil, fmt.Errorf("store: apply schema (%.60q): %w", stmt, err)
		}
	}

	s := &Store{db: db}
	if err := s.SetMeta(ctx, "schema_version", SchemaVersion); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("store: write schema_version: %w", err)
	}
	return s, nil
}

// splitStatements is a minimal SQL splitter that splits on top-level ";"
// and drops lines starting with "--". It is deliberately dumb — schema.sql
// contains no strings with embedded semicolons.
func splitStatements(sqlText string) []string {
	var out []string
	var cur bytes.Buffer
	for _, line := range splitLines(sqlText) {
		trim := trimSpace(line)
		if trim == "" || hasPrefix(trim, "--") {
			continue
		}
		cur.WriteString(line)
		cur.WriteByte('\n')
		if hasSuffix(trim, ";") {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

// Close closes the underlying DB.
func (s *Store) Close() error { return s.db.Close() }

// DB exposes the underlying *sql.DB for tests and advanced callers.
func (s *Store) DB() *sql.DB { return s.db }

// UpsertChunk inserts or replaces a chunk for (path, chunk.Idx) and keeps
// the contentless FTS5 index in sync. It returns the chunks.id rowid.
func (s *Store) UpsertChunk(ctx context.Context, c chunk.Chunk, path string, mtimeNs int64) (int64, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// Look up existing row to keep rowid stable across upserts.
	var id int64
	err = tx.QueryRowContext(ctx,
		`SELECT id FROM chunks WHERE path = ? AND chunk_idx = ?`,
		path, c.Idx,
	).Scan(&id)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		res, err := tx.ExecContext(ctx,
			`INSERT INTO chunks(path, chunk_idx, heading, heading_path, content, content_hash, mtime_ns, tokens)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			path, c.Idx, nullIfEmpty(c.Heading), nullIfEmpty(c.HeadingPath),
			c.Content, c.ContentHash, mtimeNs, c.Tokens,
		)
		if err != nil {
			return 0, fmt.Errorf("insert chunk: %w", err)
		}
		id, err = res.LastInsertId()
		if err != nil {
			return 0, fmt.Errorf("last insert id: %w", err)
		}
	case err != nil:
		return 0, fmt.Errorf("lookup chunk: %w", err)
	default:
		// Fetch the current FTS-visible content/heading_path BEFORE the
		// update so we can issue the "delete" command (FTS5 contentless
		// tables require the old row's contents to remove it correctly).
		var oldContent string
		var oldPath sql.NullString
		if err := tx.QueryRowContext(ctx,
			`SELECT content, heading_path FROM chunks WHERE id=?`, id,
		).Scan(&oldContent, &oldPath); err != nil {
			return 0, fmt.Errorf("fetch old: %w", err)
		}
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path) VALUES('delete', ?, ?, ?)`,
			id, oldContent, oldPath.String,
		); err != nil {
			return 0, fmt.Errorf("fts delete: %w", err)
		}
		if _, err := tx.ExecContext(ctx,
			`UPDATE chunks SET heading=?, heading_path=?, content=?, content_hash=?, mtime_ns=?, tokens=?
			 WHERE id=?`,
			nullIfEmpty(c.Heading), nullIfEmpty(c.HeadingPath),
			c.Content, c.ContentHash, mtimeNs, c.Tokens, id,
		); err != nil {
			return 0, fmt.Errorf("update chunk: %w", err)
		}
	}

	// Insert into FTS5 contentless table. content_rowid = chunks.id.
	if _, err := tx.ExecContext(ctx,
		`INSERT INTO chunks_fts(rowid, content, heading_path) VALUES (?, ?, ?)`,
		id, c.Content, c.HeadingPath,
	); err != nil {
		return 0, fmt.Errorf("fts insert: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("commit: %w", err)
	}
	return id, nil
}

// DeleteChunksForPath removes all chunks (and their FTS + vec rows) for a path.
func (s *Store) DeleteChunksForPath(ctx context.Context, path string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	rows, err := tx.QueryContext(ctx, `SELECT id, content, heading_path FROM chunks WHERE path = ?`, path)
	if err != nil {
		return fmt.Errorf("select: %w", err)
	}
	type ftsRow struct {
		id          int64
		content     string
		headingPath sql.NullString
	}
	var ftsRows []ftsRow
	for rows.Next() {
		var r ftsRow
		if err := rows.Scan(&r.id, &r.content, &r.headingPath); err != nil {
			rows.Close()
			return fmt.Errorf("scan: %w", err)
		}
		ftsRows = append(ftsRows, r)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return fmt.Errorf("rows: %w", err)
	}
	for _, r := range ftsRows {
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path) VALUES('delete', ?, ?, ?)`,
			r.id, r.content, r.headingPath.String,
		); err != nil {
			return fmt.Errorf("fts delete: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM chunks_vec WHERE rowid = ?`, r.id); err != nil {
			return fmt.Errorf("vec delete: %w", err)
		}
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM chunks WHERE path = ?`, path); err != nil {
		return fmt.Errorf("delete chunks: %w", err)
	}
	return tx.Commit()
}

// GetEmbeddingCache returns the cached embedding for contentHash, if any.
func (s *Store) GetEmbeddingCache(ctx context.Context, contentHash string) ([]float32, bool, error) {
	var blob []byte
	err := s.db.QueryRowContext(ctx,
		`SELECT embedding FROM embedding_cache WHERE content_hash = ?`, contentHash,
	).Scan(&blob)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("cache get: %w", err)
	}
	vec, err := decodeFloat32(blob)
	if err != nil {
		return nil, false, fmt.Errorf("decode cached vec: %w", err)
	}
	return vec, true, nil
}

// PutEmbeddingCache stores an embedding keyed by contentHash.
func (s *Store) PutEmbeddingCache(ctx context.Context, contentHash, model, taskType string, dim int, embedding []float32) error {
	if len(embedding) != dim {
		return fmt.Errorf("cache put: dim mismatch: got %d, want %d", len(embedding), dim)
	}
	blob, err := encodeFloat32(embedding)
	if err != nil {
		return fmt.Errorf("encode vec: %w", err)
	}
	_, err = s.db.ExecContext(ctx,
		`INSERT INTO embedding_cache(content_hash, model, task_type, dim, embedding, created_at)
		 VALUES (?, ?, ?, ?, ?, strftime('%s','now'))
		 ON CONFLICT(content_hash) DO UPDATE SET
		   model=excluded.model, task_type=excluded.task_type,
		   dim=excluded.dim, embedding=excluded.embedding, created_at=excluded.created_at`,
		contentHash, model, taskType, dim, blob,
	)
	if err != nil {
		return fmt.Errorf("cache put: %w", err)
	}
	return nil
}

// SetVectorForChunk writes (or replaces) the embedding for chunkID. Phase 1
// stores raw bytes; Phase 2 will migrate this to a sqlite-vec vec0 table.
func (s *Store) SetVectorForChunk(ctx context.Context, chunkID int64, embedding []float32) error {
	blob, err := encodeFloat32(embedding)
	if err != nil {
		return fmt.Errorf("encode vec: %w", err)
	}
	_, err = s.db.ExecContext(ctx,
		`INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)
		 ON CONFLICT(rowid) DO UPDATE SET embedding = excluded.embedding`,
		chunkID, blob,
	)
	if err != nil {
		return fmt.Errorf("vec write: %w", err)
	}
	return nil
}

// GetMeta reads meta[key].
func (s *Store) GetMeta(ctx context.Context, key string) (string, bool, error) {
	var v string
	err := s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key = ?`, key).Scan(&v)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, fmt.Errorf("meta get: %w", err)
	}
	return v, true, nil
}

// SetMeta upserts meta[key] = value.
func (s *Store) SetMeta(ctx context.Context, key, value string) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO meta(key, value) VALUES (?, ?)
		 ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
		key, value,
	)
	if err != nil {
		return fmt.Errorf("meta set: %w", err)
	}
	return nil
}

// encodeFloat32 serializes a float32 slice as little-endian bytes. This is
// the same layout sqlite-vec will consume once Phase 2 switches to vec0.
func encodeFloat32(v []float32) ([]byte, error) {
	buf := bytes.NewBuffer(make([]byte, 0, len(v)*4))
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeFloat32(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("float32 blob length %d not divisible by 4", len(b))
	}
	out := make([]float32, len(b)/4)
	if err := binary.Read(bytes.NewReader(b), binary.LittleEndian, out); err != nil {
		return nil, err
	}
	return out, nil
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// Tiny local string helpers — avoid importing "strings" just for these,
// and keep splitStatements allocation-free enough for schema.sql.

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func trimSpace(s string) string {
	i, j := 0, len(s)
	for i < j && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r') {
		i++
	}
	for j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\r') {
		j--
	}
	return s[i:j]
}

func hasPrefix(s, p string) bool { return len(s) >= len(p) && s[:len(p)] == p }
func hasSuffix(s, p string) bool { return len(s) >= len(p) && s[len(s)-len(p):] == p }
