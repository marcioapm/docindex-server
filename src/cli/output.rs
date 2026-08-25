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

/// Upper bound on the media-type text inside a generic tag, so an unexpected
/// server value cannot push the tag line past the terminal column budget.
const MAX_TAG_TYPE_CHARS: usize = 24;

/// Return a bracketed type tag for non-text hits, `None` for text hits.
///
/// Tag grammar:
/// - image:               `[image]`
/// - image truncated:     `[image truncated]`
/// - pdf single page:     `[pdf p{first}]`
/// - pdf page range:      `[pdf p{first}-{last}]`
/// - pdf unusable range:  `[pdf]`
/// - any + truncated:     ` truncated` appended inside the brackets
fn media_tag(hit: &Hit) -> Option<String> {
    let trunc = if hit.truncated { " truncated" } else { "" };

    match hit.media_type.as_str() {
        "text" | "" => None,
        "image" => Some(format!("[image{trunc}]")),
        "pdf" => {
            let range = pdf_page_range(hit.media_start, hit.media_end);
            Some(format!("[pdf{range}{trunc}]"))
        }
        other => Some(format!("[{}{trunc}]", truncate(other, MAX_TAG_TYPE_CHARS))),
    }
}

/// Display suffix for a PDF page range stored as the 0-based half-open
/// interval `[start, end)`, or an empty string when the interval is unusable.
///
/// Usable requires both bounds present, `start >= 0` and `end > start`; a
/// malformed row (inverted, zero-length, negative) degrades to a bare `[pdf]`
/// tag rather than a nonsensical page number. Display pages are 1-based and
/// inclusive, so `[0, 1)` renders ` p1` and `[2, 5)` renders ` p3-5`.
///
/// `end > start` also bounds `start` below `i64::MAX`, so `start + 1` cannot
/// overflow.
fn pdf_page_range(start: Option<i64>, end: Option<i64>) -> String {
    let (Some(start), Some(end)) = (start, end) else {
        return String::new();
    };
    if start < 0 || end <= start {
        return String::new();
    }
    let first = start + 1;
    if end == first {
        format!(" p{first}")
    } else {
        format!(" p{first}-{end}")
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

    #[test]
    fn unknown_media_type_renders_generic_tag() {
        let mut h = image_hit();
        h.media_type = "audio".into();
        let out = format_hit(1, &h, 200);
        assert!(
            out.contains("\n        [audio]\n"),
            "unknown media type must still get a tag; got:\n{out}"
        );
    }

    #[test]
    fn unknown_media_type_carries_truncated_marker() {
        let mut h = image_hit();
        h.media_type = "audio".into();
        h.truncated = true;
        let out = format_hit(1, &h, 200);
        assert!(
            out.contains("\n        [audio truncated]\n"),
            "expected [audio truncated]; got:\n{out}"
        );
    }

    #[test]
    fn overlong_media_type_is_clamped_in_tag() {
        let mut h = image_hit();
        h.media_type = "x".repeat(200);
        let out = format_hit(1, &h, 200);
        let tag_line = out
            .lines()
            .find(|l| l.trim_start().starts_with('['))
            .expect("tag line present");
        assert!(
            tag_line.trim().chars().count() <= MAX_TAG_TYPE_CHARS + 2,
            "tag line must stay bounded, got {} chars: {tag_line}",
            tag_line.trim().chars().count()
        );
    }

    #[test]
    fn zero_length_pdf_range_renders_bare_pdf_tag() {
        let h = pdf_hit(Some(2), Some(2), false);
        let out = format_hit(1, &h, 200);
        assert!(
            out.contains("\n        [pdf]\n"),
            "zero-length range must degrade to bare [pdf]; got:\n{out}"
        );
    }

    #[test]
    fn inverted_pdf_range_renders_bare_pdf_tag() {
        let h = pdf_hit(Some(5), Some(2), false);
        let out = format_hit(1, &h, 200);
        assert!(
            out.contains("\n        [pdf]\n"),
            "inverted range must degrade to bare [pdf]; got:\n{out}"
        );
    }

    #[test]
    fn negative_pdf_start_renders_bare_pdf_tag() {
        let h = pdf_hit(Some(-1), Some(3), false);
        let out = format_hit(1, &h, 200);
        assert!(
            out.contains("\n        [pdf]\n"),
            "negative start must degrade to bare [pdf]; got:\n{out}"
        );
    }

    #[test]
    fn pdf_start_at_i64_max_does_not_overflow() {
        let h = pdf_hit(Some(i64::MAX), Some(i64::MAX), false);
        let out = format_hit(1, &h, 200);
        assert!(
            out.contains("\n        [pdf]\n"),
            "i64::MAX start must degrade to bare [pdf] without panicking; got:\n{out}"
        );
    }

    #[test]
    fn pdf_range_with_only_one_bound_renders_bare_pdf_tag() {
        for (start, end) in [(Some(1), None), (None, Some(3))] {
            let h = pdf_hit(start, end, false);
            let out = format_hit(1, &h, 200);
            assert!(
                out.contains("\n        [pdf]\n"),
                "half-specified range {start:?}..{end:?} must degrade to bare [pdf]; got:\n{out}"
            );
        }
    }
}
