//! Human-readable output formatting for `docindex-search`.

use crate::search::Hit;

/// Terminal width fallback when it can't be detected (piped output, CI).
const DEFAULT_WIDTH: usize = 200;

/// Render one hit as a two- or three-line block:
/// ```text
/// 1. 0.79  path/to/note.md › Heading > Path
///         - snippet text truncated to terminal width...
/// ```
/// For non-text hits a type tag line is inserted between the path line and the
/// snippet:
/// ```text
/// 2. 0.65  scans/receipt.pdf
///         [pdf p3-5]
///         Invoice total: 42.00...
/// ```
/// Text hits render byte-identically to the pre-media format.
pub fn format_hit(rank: usize, hit: &Hit, width: usize) -> String {
    let heading = if hit.heading_path.is_empty() {
        String::new()
    } else {
        format!(" \u{203a} {}", hit.heading_path)
    };
    let snippet_width = width.saturating_sub(8).max(20);
    let snippet = truncate(&hit.snippet, snippet_width);

    match media_tag(hit) {
        None => format!(
            "{rank}. {:.2}  {}{heading}\n        {snippet}",
            hit.score_normalized, hit.path
        ),
        Some(tag) => format!(
            "{rank}. {:.2}  {}{heading}\n        {tag}\n        {snippet}",
            hit.score_normalized, hit.path
        ),
    }
}

/// Return a bracketed type tag for non-text hits, `None` for text hits.
///
/// Tag grammar:
/// - image:               `[image]`
/// - image truncated:     `[image truncated]`
/// - pdf single page:     `[pdf p{n}]`          where n = start+1 (1-based)
/// - pdf page range:      `[pdf p{n}-{end}]`    where end is the exclusive 0-based bound
/// - pdf no range:        `[pdf]`
/// - any + truncated:     ` truncated` appended inside the brackets
///
/// `media_start`/`media_end` are a 0-based half-open page range; display uses
/// 1-based page numbers so `start=0, end=1` → `p1`.
fn media_tag(hit: &Hit) -> Option<String> {
    let trunc = if hit.truncated { " truncated" } else { "" };

    match hit.media_type.as_str() {
        "text" | "" => None,
        "image" => Some(format!("[image{trunc}]")),
        "pdf" => {
            let range = match (hit.media_start, hit.media_end) {
                (Some(s), Some(e)) if e == s + 1 => format!(" p{}", s + 1),
                (Some(s), Some(e)) if e > s + 1 => format!(" p{}-{}", s + 1, e),
                _ => String::new(),
            };
            Some(format!("[pdf{range}{trunc}]"))
        }
        // Unknown future media types fall through as a generic tag.
        other => Some(format!("[{other}{trunc}]")),
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let cap = max_chars.saturating_sub(3);
    let mut out: String = s.chars().take(cap).collect();
    out.push_str("...");
    out
}

/// Terminal width in columns, or [`DEFAULT_WIDTH`] when not a TTY / not
/// detectable.
pub fn terminal_width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| w.0 as usize)
        .unwrap_or(DEFAULT_WIDTH)
        .min(DEFAULT_WIDTH)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baseline text hit used to verify that text rendering is unchanged.
    fn sample_hit() -> Hit {
        Hit {
            path: "Rax/holdouts-prompt.md".into(),
            title: "holdouts-prompt".into(),
            heading_path: "Holdouts Feature > 8. Key Design Decisions".into(),
            snippet: "Holdout split format: [holdout_fraction, 1.0 - holdout_fraction]...".into(),
            score: 0.79,
            score_rrf: 0.79,
            score_normalized: 0.79,
            chunk_id: 42,
            media_type: "text".into(),
            mime_type: None,
            media_start: None,
            media_end: None,
            media_unit: None,
            truncated: false,
        }
    }

    fn image_hit() -> Hit {
        Hit {
            path: "attachments/diagram.png".into(),
            title: "diagram".into(),
            heading_path: String::new(),
            snippet: "Image".into(),
            score: 0.65,
            score_rrf: 0.65,
            score_normalized: 0.65,
            chunk_id: 7,
            media_type: "image".into(),
            mime_type: Some("image/png".into()),
            media_start: None,
            media_end: None,
            media_unit: None,
            truncated: false,
        }
    }

    fn pdf_hit(start: Option<i64>, end: Option<i64>, truncated: bool) -> Hit {
        Hit {
            path: "docs/report.pdf".into(),
            title: "report".into(),
            heading_path: String::new(),
            snippet: "PDF page 3".into(),
            score: 0.50,
            score_rrf: 0.50,
            score_normalized: 0.50,
            chunk_id: 3,
            media_type: "pdf".into(),
            mime_type: Some("application/pdf".into()),
            media_start: start,
            media_end: end,
            media_unit: Some("page".into()),
            truncated,
        }
    }

    // --- text hit: byte-identical output ---

    /// The exact string produced for a text hit must never change.
    ///
    /// Mutation that falsifies: change `media_type` from `"text"` to `"image"`;
    /// the output would gain an extra `[image]` line and the assertion fails.
    #[test]
    fn text_hit_output_is_unchanged() {
        let h = sample_hit();
        let out = format_hit(1, &h, 200);
        assert_eq!(
            out,
            "1. 0.79  Rax/holdouts-prompt.md \u{203a} Holdouts Feature > 8. Key Design Decisions\n        Holdout split format: [holdout_fraction, 1.0 - holdout_fraction]..."
        );
    }

    // --- media tag helper ---

    #[test]
    fn image_hit_renders_image_tag() {
        // Mutation: set `media_type = "text"` → `media_tag` returns `None` and no
        // `[image]` line appears.
        let h = image_hit();
        let out = format_hit(1, &h, 200);
        assert!(
            out.contains("\n        [image]\n"),
            "expected [image] tag line; got:\n{out}"
        );
    }

    #[test]
    fn single_page_pdf_renders_page_number() {
        // Mutation: set `media_end = Some(3)` (two-page range) → tag becomes
        // `[pdf p2-3]` and the single-page assertion fails.
        let h = pdf_hit(Some(1), Some(2), false);
        let out = format_hit(2, &h, 200);
        assert!(
            out.contains("\n        [pdf p2]\n"),
            "expected [pdf p2] tag; got:\n{out}"
        );
    }

    /// Mutation that falsifies: change `media_end` from `Some(5)` to `Some(4)`
    /// → tag becomes `[pdf p3-4]` and the `p3-5` assertion fails.
    #[test]
    fn multi_page_pdf_renders_page_range() {
        let h = pdf_hit(Some(2), Some(5), false);
        let out = format_hit(3, &h, 200);
        assert!(
            out.contains("\n        [pdf p3-5]\n"),
            "expected [pdf p3-5] tag; got:\n{out}"
        );
    }

    #[test]
    fn truncated_image_appends_truncated_in_tag() {
        // Mutation: set `truncated = false` → tag becomes `[image]` and the
        // `[image truncated]` assertion fails.
        let mut h = image_hit();
        h.truncated = true;
        let out = format_hit(1, &h, 200);
        assert!(
            out.contains("\n        [image truncated]\n"),
            "expected [image truncated] tag; got:\n{out}"
        );
    }

    #[test]
    fn pdf_with_no_range_renders_bare_pdf_tag() {
        // Mutation: set `media_start = Some(0)` and `media_end = Some(1)` → tag
        // becomes `[pdf p1]` and the bare `[pdf]` assertion fails.
        let h = pdf_hit(None, None, false);
        let out = format_hit(1, &h, 200);
        assert!(
            out.contains("\n        [pdf]\n"),
            "expected bare [pdf] tag; got:\n{out}"
        );
    }

    // --- pre-existing tests (kept for regression coverage) ---

    #[test]
    fn format_hit_includes_rank_score_path_heading() {
        let h = sample_hit();
        let out = format_hit(1, &h, 200);
        assert!(out.starts_with("1. 0.79  Rax/holdouts-prompt.md"));
        assert!(out.contains("Holdouts Feature > 8. Key Design Decisions"));
        assert!(out.contains("Holdout split format"));
    }

    #[test]
    fn format_hit_no_heading_omits_separator() {
        let mut h = sample_hit();
        h.heading_path.clear();
        let out = format_hit(1, &h, 200);
        assert!(!out.contains('\u{203a}'));
    }

    #[test]
    fn truncate_respects_char_count() {
        let s = "x".repeat(300);
        let t = truncate(&s, 200);
        assert_eq!(t.chars().count(), 200);
        assert!(t.ends_with("..."));
    }

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("short", 200), "short");
    }
}
