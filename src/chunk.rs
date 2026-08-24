//! Deterministic, heading-aware markdown chunker.
//!
//! The chunker is PURE: same input bytes -> same output slice (byte-identical
//! content and identical `content_hash`). It does not call the embedder or
//! the store.
//!
//! Rules:
//! 1. Sections split at H1/H2/H3 boundaries. H4+ stay inline within the
//!    enclosing H1/H2/H3 section.
//! 2. The heading line IS included at the top of the chunk it introduces.
//! 3. `heading_path` is the ` > ` joined trail of ancestor headings
//!    (e.g. `Parent > Child > This`). Never `|` or `/` since both appear
//!    in real markdown headings.
//! 4. If a section exceeds `max_tokens` (default 500 words) it's split into
//!    sub-chunks with `overlap_tokens` (default 50) word overlap. The
//!    heading line is re-prepended so every sub-chunk is self-identifying.
//! 5. Empty / whitespace-only sections are dropped.
//! 6. `content_hash` is hex sha256 of `content`.

use sha2::{Digest, Sha256};

use crate::media::MediaType;

/// One indexable fragment of a markdown document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub idx: usize,
    pub heading: String,
    pub heading_path: String,
    pub content: String,
    pub content_hash: String,
    pub tokens: usize,
    pub media_type: MediaType,
    pub mime_type: Option<String>,
    pub media_start: Option<i64>,
    pub media_end: Option<i64>,
    pub media_unit: Option<String>,
    pub truncated: bool,
}

/// Default fallback cap. Whitespace-separated words are used as a token
/// proxy; it overestimates for code and underestimates for long English
/// words, but gives a stable cap well inside the embedder's window.
pub const DEFAULT_MAX_TOKENS: usize = 500;
pub const DEFAULT_OVERLAP_TOKENS: usize = 50;

/// Split markdown into an ordered slice of chunks. See module docs.
pub fn split(markdown: &[u8]) -> Vec<Chunk> {
    split_with(markdown, DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS)
}

/// `split` with configurable thresholds. Exposed for tests.
pub fn split_with(markdown: &[u8], max_tokens: usize, overlap: usize) -> Vec<Chunk> {
    let max_tokens = if max_tokens == 0 {
        DEFAULT_MAX_TOKENS
    } else {
        max_tokens
    };
    let overlap = if overlap >= max_tokens {
        DEFAULT_OVERLAP_TOKENS
    } else {
        overlap
    };

    let sections = split_sections(markdown);
    let mut out = Vec::with_capacity(sections.len());
    let mut idx = 0;

    for sec in sections {
        let content = trim_trailing_ws(&sec.content);
        if content.trim().is_empty() {
            continue;
        }
        let word_count = content.split_whitespace().count();
        if word_count <= max_tokens {
            out.push(make_chunk(idx, &sec.heading, &sec.heading_path, content));
            idx += 1;
            continue;
        }
        // Oversize: peel the heading line (if any), split the body by word
        // count, and re-prepend the heading to every sub-chunk.
        let (heading_line, rest) = if !sec.heading.is_empty() {
            match content.find('\n') {
                Some(nl) => (&content[..=nl], &content[nl + 1..]),
                None => ("", content.as_ref()),
            }
        } else {
            ("", content.as_ref())
        };
        let rest_words: Vec<&str> = rest.split_whitespace().collect();
        let step = max_tokens.saturating_sub(overlap).max(1);
        let mut start = 0;
        while start < rest_words.len() {
            let end = (start + max_tokens).min(rest_words.len());
            let body = rest_words[start..end].join(" ");
            let mut full = String::with_capacity(heading_line.len() + body.len());
            full.push_str(heading_line);
            full.push_str(&body);
            out.push(make_chunk(idx, &sec.heading, &sec.heading_path, full));
            idx += 1;
            if end == rest_words.len() {
                break;
            }
            start += step;
        }
    }
    out
}

struct Section {
    heading: String,
    heading_path: String,
    /// Raw content including the heading line when `heading` is non-empty.
    content: String,
}

/// Walk the document once, tracking the H1/H2/H3 ancestor stack, emitting
/// one `Section` per H1/H2/H3 boundary.
fn split_sections(markdown: &[u8]) -> Vec<Section> {
    let text = std::str::from_utf8(markdown).unwrap_or("");
    let mut out: Vec<Section> = Vec::new();
    let mut stack: [String; 3] = [String::new(), String::new(), String::new()];
    let mut cur_heading = String::new();
    let mut cur_path = String::new();
    let mut cur_buf = String::new();
    let mut in_fence = false;

    let flush =
        |out: &mut Vec<Section>, heading: &mut String, path: &mut String, buf: &mut String| {
            if buf.is_empty() && heading.is_empty() {
                return;
            }
            out.push(Section {
                heading: std::mem::take(heading),
                heading_path: std::mem::take(path),
                content: std::mem::take(buf),
            });
        };

    for line in text.split_inclusive('\n') {
        // Strip the trailing newline for parsing, re-append below.
        let raw = line;
        let line_no_nl = if let Some(s) = line.strip_suffix('\n') {
            s.strip_suffix('\r').unwrap_or(s)
        } else {
            line
        };
        let trim = line_no_nl.trim_start_matches([' ', '\t']);
        if trim.starts_with("```") || trim.starts_with("~~~") {
            in_fence = !in_fence;
            cur_buf.push_str(raw);
            if !raw.ends_with('\n') {
                cur_buf.push('\n');
            }
            continue;
        }
        if !in_fence
            && let Some((level, text)) = parse_heading(line_no_nl)
            && (1..=3).contains(&level)
        {
            // Flush the current section; we're about to start a new one.
            flush(&mut out, &mut cur_heading, &mut cur_path, &mut cur_buf);
            let lvl = level - 1;
            stack[lvl] = text.clone();
            for slot in &mut stack[(lvl + 1)..] {
                slot.clear();
            }
            cur_heading = text.clone();
            cur_path = join_path(&stack);
            cur_buf.push_str(raw);
            if !raw.ends_with('\n') {
                cur_buf.push('\n');
            }
            continue;
        }
        cur_buf.push_str(raw);
        if !raw.ends_with('\n') {
            cur_buf.push('\n');
        }
    }
    flush(&mut out, &mut cur_heading, &mut cur_path, &mut cur_buf);
    out
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    // ATX headings only ("# Heading"). Setext headings are out of scope.
    let trim = line.trim_start_matches([' ', '\t']);
    if !trim.starts_with('#') {
        return None;
    }
    let mut level = 0usize;
    for c in trim.chars() {
        if c == '#' {
            level += 1;
        } else {
            break;
        }
    }
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trim[level..];
    // A valid ATX heading requires a space/tab after the hashes (or EOL).
    if !(rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t')) {
        return None;
    }
    let text = rest.trim().trim_end_matches(['#', ' ', '\t']).trim();
    Some((level, text.to_string()))
}

fn join_path(stack: &[String; 3]) -> String {
    stack
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" > ")
}

fn make_chunk(idx: usize, heading: &str, heading_path: &str, content: String) -> Chunk {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let tokens = content.split_whitespace().count();
    Chunk {
        idx,
        heading: heading.to_string(),
        heading_path: heading_path.to_string(),
        content_hash: hex::encode(hasher.finalize()),
        content,
        tokens,
        media_type: MediaType::Text,
        mime_type: None,
        media_start: None,
        media_end: None,
        media_unit: None,
        truncated: false,
    }
}

fn trim_trailing_ws(s: &str) -> String {
    s.trim_end_matches([' ', '\t', '\r', '\n']).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(s: &str) -> String {
        let mut h = Sha256::new();
        h.update(s.as_bytes());
        hex::encode(h.finalize())
    }

    #[test]
    fn empty_input() {
        assert!(split(b"").is_empty());
        assert!(split(b"").is_empty());
    }

    #[test]
    fn whitespace_only() {
        assert!(split(b"   \n\n\t\n").is_empty());
    }

    #[test]
    fn no_heading() {
        let doc = "just some text\nacross two lines";
        let got = split(doc.as_bytes());
        assert_eq!(got.len(), 1);
        let c = &got[0];
        assert!(c.heading.is_empty());
        assert!(c.heading_path.is_empty());
        assert_eq!(c.content, doc);
        assert_eq!(c.content_hash, sha(&c.content));
        assert_eq!(c.idx, 0);
        assert_eq!(c.tokens, doc.split_whitespace().count());
    }

    #[test]
    fn single_h1() {
        let doc = "# Title\n\nbody text here";
        let got = split(doc.as_bytes());
        assert_eq!(got.len(), 1, "{got:?}");
        let c = &got[0];
        assert_eq!(c.heading, "Title");
        assert_eq!(c.heading_path, "Title");
        assert!(c.content.starts_with("# Title\n"));
    }

    #[test]
    fn nested_h1_h2_h3() {
        let doc = "# A\nintro\n## B\nmid\n### C\nleaf\n## D\nsecond";
        let got = split(doc.as_bytes());
        assert_eq!(got.len(), 4, "{got:?}");
        let want = ["A", "A > B", "A > B > C", "A > D"];
        for (i, w) in want.iter().enumerate() {
            assert_eq!(got[i].heading_path, *w);
        }
        for (i, c) in got.iter().enumerate() {
            assert_eq!(c.idx, i);
        }
    }

    #[test]
    fn h4_stays_inline() {
        let doc = "# A\n## B\n#### deep\ndeeper body\n## C\nelse";
        let got = split(doc.as_bytes());
        assert_eq!(got.len(), 3);
        assert_eq!(got[1].heading_path, "A > B");
        assert!(got[1].content.contains("#### deep"));
    }

    #[test]
    fn heading_special_chars() {
        let doc = "# foo|bar\nbody a\n## a/b\nbody b";
        let got = split(doc.as_bytes());
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].heading, "foo|bar");
        assert_eq!(got[1].heading_path, "foo|bar > a/b");
    }

    #[test]
    fn code_fence_hashes_ignored() {
        let doc = "# Real\nhello\n```\n## fake heading inside code\n```\nafter\n## Real2\nbody";
        let got = split(doc.as_bytes());
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0].heading, "Real");
        assert_eq!(got[1].heading, "Real2");
        assert!(got[0].content.contains("## fake heading inside code"));
    }

    #[test]
    fn fallback_split_with_overlap() {
        let words: Vec<String> = (0..250).map(|i| format!("w{i}")).collect();
        let body = words.join(" ");
        let doc = format!("# H\n{body}");
        let got = split_with(doc.as_bytes(), 100, 20);
        // step = 80, sub-chunks at starts 0, 80, 160 -> 3.
        assert_eq!(got.len(), 3, "{got:?}");
        for (i, c) in got.iter().enumerate() {
            assert_eq!(c.heading, "H");
            assert!(c.content.starts_with("# H\n"));
            assert_eq!(c.idx, i);
        }
        // Overlap: last 20 body words of chunk 0 == first 20 body words of chunk 1.
        let body0: Vec<&str> = got[0]
            .content
            .trim_start_matches("# H\n")
            .split_whitespace()
            .collect();
        let body1: Vec<&str> = got[1]
            .content
            .trim_start_matches("# H\n")
            .split_whitespace()
            .collect();
        let tail0 = &body0[body0.len() - 20..];
        let head1 = &body1[..20];
        assert_eq!(tail0, head1);
    }

    #[test]
    fn deterministic() {
        let doc = "# A\nhello\n## B\nworld\n### C\nthere\n";
        assert_eq!(split(doc.as_bytes()), split(doc.as_bytes()));
    }

    #[test]
    fn hash_matches_content() {
        let got = split(b"# X\nfoo bar");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].content_hash, sha(&got[0].content));
    }
}
