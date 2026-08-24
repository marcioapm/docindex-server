//! Human-readable output formatting for `docindex-search`.

use crate::search::Hit;

/// Terminal width fallback when it can't be detected (piped output, CI).
const DEFAULT_WIDTH: usize = 200;

/// Render one hit as the two-line format:
/// ```text
/// 1. 0.79  path/to/note.md › Heading > Path
///         - snippet text truncated to terminal width...
/// ```
pub fn format_hit(rank: usize, hit: &Hit, width: usize) -> String {
    let heading = if hit.heading_path.is_empty() {
        String::new()
    } else {
        format!(" \u{203a} {}", hit.heading_path)
    };
    let snippet_width = width.saturating_sub(8).max(20);
    let snippet = truncate(&hit.snippet, snippet_width);
    format!(
        "{rank}. {:.2}  {}{heading}\n        {snippet}",
        hit.score_normalized, hit.path
    )
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
