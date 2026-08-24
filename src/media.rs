//! Provider-neutral media admission policy and metadata types.

use std::path::{Component, Path};

use globset::{Glob, GlobSet, GlobSetBuilder};
use sha2::{Digest, Sha256};

pub const MEDIA_PROTOCOL_VERSION: &str = "media-v1";
const DEFAULT_MAX_FILE_MB: u64 = 20;
const DEFAULT_PDF_PAGES_PER_CHUNK: u8 = 1;
const DEFAULT_PDF_DPI: u16 = 150;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    Text,
    Image,
    Pdf,
    Audio,
    Video,
}

impl MediaType {
    pub const EXCLUDE_VALUES: [&str; 4] = ["image", "pdf", "audio", "video"];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Pdf => "pdf",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }

    pub fn from_exclude_value(value: &str) -> Option<Self> {
        match value {
            "image" => Some(Self::Image),
            "pdf" => Some(Self::Pdf),
            "audio" => Some(Self::Audio),
            "video" => Some(Self::Video),
            _ => None,
        }
    }

    pub fn for_extension(ext: &str) -> Option<Self> {
        if ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("txt") {
            return Some(Self::Text);
        }
        if ["png", "jpg", "jpeg", "webp", "gif"]
            .iter()
            .any(|known| ext.eq_ignore_ascii_case(known))
        {
            return Some(Self::Image);
        }
        if ext.eq_ignore_ascii_case("pdf") {
            return Some(Self::Pdf);
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct MediaPolicy {
    enabled: bool,
    include: GlobSet,
    exclude: GlobSet,
    include_empty: bool,
    exclude_empty: bool,
    excluded_types: Vec<MediaType>,
    max_file_bytes: u64,
    pub pdf_pages_per_chunk: u8,
    pub pdf_dpi: u16,
}

impl Default for MediaPolicy {
    fn default() -> Self {
        Self::new(
            false,
            &[],
            &[],
            &[],
            DEFAULT_MAX_FILE_MB,
            DEFAULT_PDF_PAGES_PER_CHUNK,
            DEFAULT_PDF_DPI,
        )
        .unwrap_or_else(|_| unreachable!("default media policy is valid"))
    }
}

impl MediaPolicy {
    pub fn new(
        enabled: bool,
        include: &[String],
        exclude: &[String],
        exclude_types: &[String],
        max_file_mb: u64,
        pdf_pages_per_chunk: u8,
        pdf_dpi: u16,
    ) -> Result<Self, String> {
        if max_file_mb == 0 {
            return Err("media.max_file_mb must be > 0".into());
        }
        if !(1..=6).contains(&pdf_pages_per_chunk) {
            return Err(format!(
                "media.pdf_pages_per_chunk {pdf_pages_per_chunk}: must be in 1..=6"
            ));
        }
        if !(72..=300).contains(&pdf_dpi) {
            return Err(format!("media.pdf_dpi {pdf_dpi}: must be in 72..=300"));
        }
        let include_set = compile_globs("media.include", include)?;
        let exclude_set = compile_globs("media.exclude", exclude)?;
        let mut excluded_types = Vec::new();
        for value in exclude_types {
            let canonical = value.to_ascii_lowercase();
            let Some(media_type) = MediaType::from_exclude_value(&canonical) else {
                return Err(format!(
                    "media.exclude_types {value:?}: unknown value; valid values: {}",
                    MediaType::EXCLUDE_VALUES.join(", ")
                ));
            };
            if !excluded_types.contains(&media_type) {
                excluded_types.push(media_type);
            }
        }
        Ok(Self {
            enabled,
            include: include_set,
            exclude: exclude_set,
            include_empty: include.is_empty(),
            exclude_empty: exclude.is_empty(),
            excluded_types,
            max_file_bytes: max_file_mb.saturating_mul(1024 * 1024),
            pdf_pages_per_chunk,
            pdf_dpi,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn classify_path(&self, relative: &Path) -> Option<MediaType> {
        if !is_safe_relative_path(relative) {
            return None;
        }
        let ext = relative.extension()?.to_str()?;
        let media_type = MediaType::for_extension(ext)?;
        if media_type == MediaType::Text {
            return Some(media_type);
        }
        if !self.enabled || self.excluded_types.contains(&media_type) {
            return None;
        }
        let normalized = normalize_relative_path(relative);
        if (!self.include_empty && !self.include.is_match(&normalized))
            || (!self.exclude_empty && self.exclude.is_match(&normalized))
        {
            return None;
        }
        Some(media_type)
    }

    pub fn allows_existing_file(&self, relative: &Path, size: u64) -> Option<MediaType> {
        let media_type = self.classify_path(relative)?;
        if media_type != MediaType::Text && size > self.max_file_bytes {
            return None;
        }
        Some(media_type)
    }

    pub fn allows_remove(&self, relative: &Path) -> bool {
        self.classify_path(relative).is_some()
    }

    pub fn max_file_bytes(&self) -> u64 {
        self.max_file_bytes
    }

    pub fn effective_file_hash(&self, bytes: &[u8], media_type: MediaType) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        if media_type != MediaType::Text {
            hasher.update([0]);
            hasher.update(self.profile());
        }
        hex::encode(hasher.finalize())
    }

    pub fn profile(&self) -> String {
        format!(
            "{MEDIA_PROTOCOL_VERSION}|max_file_mb={}|pdf_pages_per_chunk={}|pdf_dpi={}",
            self.max_file_bytes / (1024 * 1024),
            self.pdf_pages_per_chunk,
            self.pdf_dpi
        )
    }
}

pub fn media_embedding_cache_key(
    media_type: MediaType,
    mime_type: &str,
    processed_bytes: &[u8],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(MEDIA_PROTOCOL_VERSION.as_bytes());
    hasher.update([0]);
    hasher.update(media_type.as_str().as_bytes());
    hasher.update([0]);
    hasher.update(mime_type.as_bytes());
    hasher.update([0]);
    hasher.update(processed_bytes);
    hex::encode(hasher.finalize())
}

pub fn normalize_relative_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

pub fn is_safe_relative_path(path: &Path) -> bool {
    !path.is_absolute()
        && !path.as_os_str().is_empty()
        && path.components().all(|component| match component {
            Component::Normal(name) => {
                let name = name.to_string_lossy();
                !name.starts_with('.')
                    && !matches!(name.as_ref(), ".git" | ".obsidian" | "node_modules")
            }
            Component::CurDir => true,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => false,
        })
}

fn compile_globs(key: &str, patterns: &[String]) -> Result<GlobSet, String> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = Glob::new(pattern)
            .map_err(|error| format!("{key} {pattern:?}: invalid glob: {error}"))?;
        builder.add(glob);
    }
    builder
        .build()
        .map_err(|error| format!("{key}: invalid glob set: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn policy(include: &[&str], exclude: &[&str]) -> MediaPolicy {
        MediaPolicy::new(
            true,
            &include.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            &exclude.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            &[],
            20,
            6,
            150,
        )
        .unwrap()
    }

    #[test]
    fn include_exclude_truth_table_and_precedence() {
        assert!(policy(&[], &[]).classify_path(Path::new("x.png")).is_some());
        assert!(
            policy(&["Attachments/**"], &[])
                .classify_path(Path::new("Attachments/x.png"))
                .is_some()
        );
        assert!(
            policy(&["Attachments/**"], &[])
                .classify_path(Path::new("Papers/x.png"))
                .is_none()
        );
        assert!(
            policy(&[], &["**/Private/**"])
                .classify_path(Path::new("Public/x.png"))
                .is_some()
        );
        assert!(
            policy(&[], &["**/Private/**"])
                .classify_path(Path::new("Papers/Private/x.png"))
                .is_none()
        );
        assert!(
            policy(&["Attachments/**"], &["Attachments/Private/**"])
                .classify_path(Path::new("Attachments/Private/x.png"))
                .is_none()
        );
    }

    #[test]
    fn windows_separators_normalize_for_globs() {
        assert_eq!(
            normalize_relative_path(&PathBuf::from("Attachments\\x.png")),
            "Attachments/x.png"
        );
    }

    #[test]
    fn media_cache_keys_separate_modalities() {
        let bytes = b"same";
        assert_ne!(
            media_embedding_cache_key(MediaType::Image, "image/png", bytes),
            media_embedding_cache_key(MediaType::Pdf, "application/pdf", bytes)
        );
        assert_ne!(
            media_embedding_cache_key(MediaType::Image, "image/png", bytes),
            media_embedding_cache_key(MediaType::Text, "text/plain", bytes)
        );
    }

    #[test]
    fn defaults_are_text_only_and_exclude_types_validate() {
        let default = MediaPolicy::default();
        assert_eq!(
            default.classify_path(Path::new("note.md")),
            Some(MediaType::Text)
        );
        assert!(default.classify_path(Path::new("image.png")).is_none());
        let excluded = MediaPolicy::new(
            true,
            &[],
            &[],
            &["image".into(), "pdf".into(), "audio".into(), "video".into()],
            20,
            6,
            150,
        )
        .unwrap();
        assert!(excluded.classify_path(Path::new("image.png")).is_none());
        assert!(excluded.classify_path(Path::new("paper.pdf")).is_none());
        assert!(
            MediaPolicy::new(true, &[], &[], &["unknown".into()], 20, 6, 150)
                .unwrap_err()
                .contains("valid values")
        );
    }

    #[test]
    fn invalid_glob_names_key_and_pattern() {
        let error = MediaPolicy::new(true, &["[".into()], &[], &[], 20, 6, 150).unwrap_err();
        assert!(error.contains("media.include \"[\""), "{error}");
    }

    #[test]
    fn eligibility_only_changes_do_not_change_hash() {
        let base =
            MediaPolicy::new(true, &["Attachments/**".into()], &[], &[], 20, 6, 150).unwrap();
        let changed = MediaPolicy::new(
            true,
            &["Papers/**".into()],
            &["**/Private/**".into()],
            &["audio".into()],
            20,
            6,
            150,
        )
        .unwrap();
        assert_eq!(
            base.effective_file_hash(b"same", MediaType::Image),
            changed.effective_file_hash(b"same", MediaType::Image)
        );
    }

    #[test]
    fn oversize_existing_media_is_skipped_but_remove_is_path_based() {
        let policy = MediaPolicy::new(true, &[], &[], &[], 1, 6, 150).unwrap();
        let path = Path::new("Attachments/image.png");
        assert!(policy.allows_existing_file(path, 1_048_577).is_none());
        assert!(policy.allows_remove(path));
    }

    #[test]
    fn profile_changes_only_media_effective_hashes() {
        let base = MediaPolicy::default();
        let changed = MediaPolicy::new(false, &[], &[], &[], 20, 5, 150).unwrap();
        assert_eq!(
            base.effective_file_hash(b"same", MediaType::Text),
            changed.effective_file_hash(b"same", MediaType::Text)
        );
        assert_ne!(
            base.effective_file_hash(b"same", MediaType::Pdf),
            changed.effective_file_hash(b"same", MediaType::Pdf)
        );
    }
}
