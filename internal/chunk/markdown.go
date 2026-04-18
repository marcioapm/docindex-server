// Package chunk implements a deterministic, heading-aware markdown chunker.
//
// The chunker is PURE: same input bytes → same output slice (byte-identical
// content and identical content_hash). It does not call the embedder or the
// store.
//
// Chunking rules:
//
//  1. Sections are split at H1, H2, and H3 boundaries. Deeper headings
//     (H4+) stay inline within the enclosing H1/H2/H3 section.
//  2. The heading line IS included at the top of the chunk it introduces.
//  3. HeadingPath is the " > "-joined trail of ancestor headings, e.g.
//     "Parent > Child > This". We never use "|" or "/" because both routinely
//     appear inside markdown headings.
//  4. If a section exceeds maxTokens (default 500, where a "token" is a
//     whitespace-separated word), we split it into sub-chunks of maxTokens
//     with overlapTokens (default 50) overlap between adjacent sub-chunks.
//  5. Whitespace-only or empty sections are dropped.
//  6. ContentHash is sha256 of the chunk Content.
package chunk

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// Chunk is one indexable fragment of a markdown document.
type Chunk struct {
	Idx         int
	Heading     string // plain heading text; "" if no heading
	HeadingPath string // "Parent > Child > This"; "" if no heading
	Content     string // full chunk text, heading line included
	ContentHash string // hex sha256 of Content
	Tokens      int    // whitespace-word count of Content
}

// Defaults. Word count (whitespace tokenization) is used as a cheap proxy
// for model tokens. It overestimates for code and underestimates for long
// English words; the goal is a stable cap well within the embedder's 2048
// token window, not perfection.
const (
	defaultMaxTokens     = 500
	defaultOverlapTokens = 50
)

// Split splits markdown into an ordered slice of chunks. See package doc.
//
// (Go does not permit a function and a type to share the same name in a
// package, so the exported function is Split rather than Chunk.)
func Split(markdown []byte) []Chunk {
	return SplitWith(markdown, defaultMaxTokens, defaultOverlapTokens)
}

// SplitWith is Split with configurable fallback thresholds. Exposed for tests.
func SplitWith(markdown []byte, maxTokens, overlap int) []Chunk {
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	if overlap < 0 || overlap >= maxTokens {
		overlap = defaultOverlapTokens
	}

	sections := splitSections(markdown)
	out := make([]Chunk, 0, len(sections))
	idx := 0
	for _, sec := range sections {
		content := strings.TrimRight(sec.content, " \t\r\n")
		if strings.TrimSpace(content) == "" {
			continue
		}
		words := strings.Fields(content)
		if len(words) <= maxTokens {
			out = append(out, makeChunk(idx, sec.heading, sec.headingPath, content))
			idx++
			continue
		}
		// Oversize: split with overlap. The heading line is part of the
		// section content already; we prepend it to each sub-chunk as well
		// so every fragment is self-identifying. We split on word count
		// rather than lines so the fallback is predictable.
		var headingLine string
		rest := content
		if sec.heading != "" {
			if nl := strings.Index(content, "\n"); nl >= 0 {
				headingLine = content[:nl+1]
				rest = content[nl+1:]
			}
		}
		restWords := strings.Fields(rest)
		step := maxTokens - overlap
		if step <= 0 {
			step = 1
		}
		for start := 0; start < len(restWords); start += step {
			end := start + maxTokens
			if end > len(restWords) {
				end = len(restWords)
			}
			body := strings.Join(restWords[start:end], " ")
			full := headingLine + body
			out = append(out, makeChunk(idx, sec.heading, sec.headingPath, full))
			idx++
			if end == len(restWords) {
				break
			}
		}
	}
	return out
}

// section is an intermediate struct produced by splitSections.
type section struct {
	heading     string
	headingPath string
	content     string // includes the heading line as line 0 if heading != ""
}

// splitSections walks the document once, tracking the current H1/H2/H3
// ancestor stack, and emits one section per H1/H2/H3 boundary.
func splitSections(markdown []byte) []section {
	var out []section
	var stack [3]string // [h1, h2, h3]

	var curHeading, curPath string
	var curBuf bytes.Buffer

	flush := func() {
		if curBuf.Len() == 0 && curHeading == "" {
			return
		}
		out = append(out, section{
			heading:     curHeading,
			headingPath: curPath,
			content:     curBuf.String(),
		})
		curBuf.Reset()
	}

	scanner := bufio.NewScanner(bytes.NewReader(markdown))
	// Allow long lines (markdown can legitimately have them).
	scanner.Buffer(make([]byte, 64*1024), 8*1024*1024)
	inFence := false
	for scanner.Scan() {
		line := scanner.Text()
		trim := strings.TrimLeft(line, " \t")
		// Track fenced code blocks so we don't treat "## " inside code as a heading.
		if strings.HasPrefix(trim, "```") || strings.HasPrefix(trim, "~~~") {
			inFence = !inFence
			curBuf.WriteString(line)
			curBuf.WriteByte('\n')
			continue
		}
		if !inFence {
			if level, text, ok := parseHeading(line); ok && level >= 1 && level <= 3 {
				flush()
				// Update the ancestor stack for the new heading.
				stack[level-1] = text
				for i := level; i < len(stack); i++ {
					stack[i] = ""
				}
				curHeading = text
				curPath = joinPath(stack)
				curBuf.WriteString(line)
				curBuf.WriteByte('\n')
				continue
			}
		}
		curBuf.WriteString(line)
		curBuf.WriteByte('\n')
	}
	flush()
	return out
}

func parseHeading(line string) (level int, text string, ok bool) {
	// ATX headings only ("# Heading"). Setext headings ("=====") are rare
	// in modern markdown and not supported here by design.
	trim := strings.TrimLeft(line, " \t")
	if !strings.HasPrefix(trim, "#") {
		return 0, "", false
	}
	i := 0
	for i < len(trim) && trim[i] == '#' {
		i++
	}
	if i == 0 || i > 6 {
		return 0, "", false
	}
	// A valid ATX heading requires a space after the hashes (or EOL).
	if i < len(trim) && trim[i] != ' ' && trim[i] != '\t' {
		return 0, "", false
	}
	text = strings.TrimSpace(trim[i:])
	// Trim trailing closing hashes ("# foo #").
	text = strings.TrimRight(text, " \t#")
	text = strings.TrimSpace(text)
	return i, text, true
}

func joinPath(stack [3]string) string {
	parts := make([]string, 0, 3)
	for _, s := range stack {
		if s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, " > ")
}

func makeChunk(idx int, heading, headingPath, content string) Chunk {
	h := sha256.Sum256([]byte(content))
	return Chunk{
		Idx:         idx,
		Heading:     heading,
		HeadingPath: headingPath,
		Content:     content,
		ContentHash: hex.EncodeToString(h[:]),
		Tokens:      len(strings.Fields(content)),
	}
}
