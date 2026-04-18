package chunk

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
	"testing"
)

func sha(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func TestChunk_Empty(t *testing.T) {
	if got := Split(nil); len(got) != 0 {
		t.Errorf("nil input should yield 0 chunks, got %d", len(got))
	}
	if got := Split([]byte("")); len(got) != 0 {
		t.Errorf("empty input should yield 0 chunks, got %d", len(got))
	}
}

func TestChunk_WhitespaceOnly(t *testing.T) {
	if got := Split([]byte("   \n\n\t\n")); len(got) != 0 {
		t.Errorf("whitespace-only input should yield 0 chunks, got %d", len(got))
	}
}

func TestChunk_NoHeading(t *testing.T) {
	doc := "just some text\nacross two lines"
	got := Split([]byte(doc))
	if len(got) != 1 {
		t.Fatalf("want 1 chunk, got %d", len(got))
	}
	c := got[0]
	if c.Heading != "" || c.HeadingPath != "" {
		t.Errorf("no-heading chunk should have empty heading; got %+v", c)
	}
	if c.Content != doc {
		t.Errorf("content mismatch: got %q", c.Content)
	}
	if c.ContentHash != sha(c.Content) {
		t.Errorf("hash mismatch")
	}
	if c.Idx != 0 {
		t.Errorf("Idx = %d, want 0", c.Idx)
	}
	if c.Tokens != len(strings.Fields(doc)) {
		t.Errorf("Tokens = %d, want %d", c.Tokens, len(strings.Fields(doc)))
	}
}

func TestChunk_SingleH1(t *testing.T) {
	doc := "# Title\n\nbody text here"
	got := Split([]byte(doc))
	if len(got) != 1 {
		t.Fatalf("want 1 chunk, got %d: %+v", len(got), got)
	}
	c := got[0]
	if c.Heading != "Title" {
		t.Errorf("Heading = %q, want Title", c.Heading)
	}
	if c.HeadingPath != "Title" {
		t.Errorf("HeadingPath = %q, want Title", c.HeadingPath)
	}
	if !strings.HasPrefix(c.Content, "# Title\n") {
		t.Errorf("chunk should start with heading line; got %q", c.Content)
	}
}

func TestChunk_NestedH1H2H3(t *testing.T) {
	doc := "# A\nintro\n## B\nmid\n### C\nleaf\n## D\nsecond"
	got := Split([]byte(doc))
	if len(got) != 4 {
		t.Fatalf("want 4 chunks, got %d: %+v", len(got), got)
	}
	wantPaths := []string{"A", "A > B", "A > B > C", "A > D"}
	for i, want := range wantPaths {
		if got[i].HeadingPath != want {
			t.Errorf("HeadingPath[%d] = %q, want %q", i, got[i].HeadingPath, want)
		}
	}
	// H3 -> H2 "D" should reset the H3 slot.
	if got[3].HeadingPath == "A > B > C > D" {
		t.Errorf("H2 should reset deeper levels: got %q", got[3].HeadingPath)
	}
	// Monotonic Idx.
	for i, c := range got {
		if c.Idx != i {
			t.Errorf("Idx[%d] = %d", i, c.Idx)
		}
	}
}

func TestChunk_H4StaysInline(t *testing.T) {
	doc := "# A\n## B\n#### deep\ndeeper body\n## C\nelse"
	got := Split([]byte(doc))
	if len(got) != 3 {
		t.Fatalf("want 3 chunks, got %d", len(got))
	}
	if got[1].HeadingPath != "A > B" {
		t.Errorf("H4 should not create a section; got path %q", got[1].HeadingPath)
	}
	if !strings.Contains(got[1].Content, "#### deep") {
		t.Errorf("H4 body should stay in H2 section")
	}
}

func TestChunk_HeadingSpecialChars(t *testing.T) {
	doc := "# foo|bar\nbody a\n## a/b\nbody b"
	got := Split([]byte(doc))
	if len(got) != 2 {
		t.Fatalf("want 2 chunks, got %d", len(got))
	}
	if got[0].Heading != "foo|bar" {
		t.Errorf("heading = %q, want foo|bar", got[0].Heading)
	}
	if got[1].HeadingPath != "foo|bar > a/b" {
		t.Errorf("path = %q, want 'foo|bar > a/b'", got[1].HeadingPath)
	}
	if strings.Contains(got[1].HeadingPath, "|") && !strings.Contains(got[1].HeadingPath, " > ") {
		t.Errorf("separator must be ' > ', not pipe")
	}
}

func TestChunk_CodeFenceHashesIgnored(t *testing.T) {
	doc := "# Real\nhello\n```\n## fake heading inside code\n```\nafter\n## Real2\nbody"
	got := Split([]byte(doc))
	if len(got) != 2 {
		t.Fatalf("want 2 chunks, got %d: %+v", len(got), got)
	}
	if got[0].Heading != "Real" || got[1].Heading != "Real2" {
		t.Errorf("unexpected headings: %+v", got)
	}
	if !strings.Contains(got[0].Content, "## fake heading inside code") {
		t.Errorf("fenced '## fake' should remain in first chunk")
	}
}

func TestChunk_FallbackSplitWithOverlap(t *testing.T) {
	// Build a section with 250 words so we can exercise fallback at
	// maxTokens=100, overlap=20.
	words := make([]string, 250)
	for i := range words {
		words[i] = "w" + itoa(i)
	}
	body := strings.Join(words, " ")
	doc := "# H\n" + body

	got := SplitWith([]byte(doc), 100, 20)
	// step = 80. Subchunks: [0:100] [80:180] [160:250] => 3 chunks.
	if len(got) != 3 {
		t.Fatalf("want 3 sub-chunks, got %d", len(got))
	}
	for i, c := range got {
		if c.Heading != "H" {
			t.Errorf("sub-chunk[%d] lost heading", i)
		}
		if !strings.HasPrefix(c.Content, "# H\n") {
			t.Errorf("sub-chunk[%d] should re-prepend heading; got %q...", i, c.Content[:min(20, len(c.Content))])
		}
		if c.Idx != i {
			t.Errorf("Idx[%d] = %d", i, c.Idx)
		}
	}
	// Verify overlap exists between consecutive sub-chunks.
	a := strings.Fields(got[0].Content)
	b := strings.Fields(got[1].Content)
	overlap := 0
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[len(a)-1-i] == b[len(b)-1-i-(len(b)-len(a))] {
			// simple check: look for any shared word in the tail/head boundary
		}
	}
	// More direct: last 20 body words of chunk 0 should equal first 20 body words of chunk 1.
	body0 := strings.TrimPrefix(got[0].Content, "# H\n")
	body1 := strings.TrimPrefix(got[1].Content, "# H\n")
	w0 := strings.Fields(body0)
	w1 := strings.Fields(body1)
	for i := 0; i < 20; i++ {
		if w0[len(w0)-20+i] != w1[i] {
			t.Fatalf("overlap word %d mismatch: %q vs %q", i, w0[len(w0)-20+i], w1[i])
		}
	}
	_ = overlap
}

func TestChunk_Deterministic(t *testing.T) {
	doc := "# A\nhello\n## B\nworld\n### C\nthere\n"
	a := Split([]byte(doc))
	b := Split([]byte(doc))
	if len(a) != len(b) {
		t.Fatalf("non-deterministic length")
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("chunk[%d] not deterministic", i)
		}
	}
}

func TestChunk_HashMatchesContent(t *testing.T) {
	doc := "# X\nfoo bar"
	got := Split([]byte(doc))
	if len(got) != 1 {
		t.Fatalf("want 1 chunk")
	}
	if got[0].ContentHash != sha(got[0].Content) {
		t.Errorf("hash != sha256(content)")
	}
}

func itoa(i int) string { return strconv.Itoa(i) }
