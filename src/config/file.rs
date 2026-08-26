//! TOML config file discovery, injectable reading, and secret-permission
//! warnings.
//!
//! Both the server and the CLI read a TOML file at one of several
//! well-known locations (or an explicit path from `--config` /
//! `$DOCINDEX_*_CONFIG`). The actual filesystem access goes through
//! [`FileReader`] so tests never touch `$HOME`.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;
use tracing::warn;

use super::Lookup;

#[derive(Debug, Error)]
pub enum FileConfigError {
    #[error("config file {path}: not found")]
    NotFound { path: PathBuf },
    #[error("config file {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
}

/// Contents of a config file plus its Unix permission bits (`None` on
/// non-Unix platforms or when the injected reader doesn't track it).
#[derive(Debug, Clone)]
pub struct FileContent {
    pub text: String,
    pub mode: Option<u32>,
}

/// Function signature for reading a config file, injectable for tests.
/// Returns `None` when the file does not exist (or is otherwise
/// unreadable) — the caller decides whether that's fatal.
pub type FileReader<'a> = dyn Fn(&Path) -> Option<FileContent> + 'a;

/// Real filesystem reader used by `Config::from_env` / the production
/// binaries.
pub fn os_file_reader(path: &Path) -> Option<FileContent> {
    let text = std::fs::read_to_string(path).ok()?;
    let mode = std::fs::metadata(path).ok().map(file_mode);
    Some(FileContent { text, mode })
}

#[cfg(unix)]
fn file_mode(md: std::fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    md.permissions().mode()
}

#[cfg(not(unix))]
fn file_mode(_md: std::fs::Metadata) -> u32 {
    0
}

/// Server-side TOML schema. All fields optional — missing falls through to
/// env, then built-in defaults.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerFile {
    pub vault_dir: Option<String>,
    pub db_path: Option<String>,
    pub listen: Option<String>,
    pub bearer: Option<String>,
    pub bearer_env: Option<String>,
    pub debounce_ms: Option<u64>,
    pub http_timeout_ms: Option<u64>,
    pub log_format: Option<String>,
    pub allow_loopback: Option<bool>,
    #[serde(default)]
    pub embed: EmbedFile,
    #[serde(default)]
    pub media: MediaFile,
    #[serde(default)]
    pub search: SearchFile,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediaFile {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub exclude_types: Vec<String>,
    #[serde(default = "default_max_file_mb")]
    pub max_file_mb: u64,
    #[serde(default = "default_pdf_pages_per_chunk")]
    pub pdf_pages_per_chunk: u8,
    #[serde(default = "default_pdf_dpi")]
    pub pdf_dpi: u16,
}

impl Default for MediaFile {
    fn default() -> Self {
        Self {
            enabled: false,
            include: Vec::new(),
            exclude: Vec::new(),
            exclude_types: Vec::new(),
            max_file_mb: default_max_file_mb(),
            pdf_pages_per_chunk: default_pdf_pages_per_chunk(),
            pdf_dpi: default_pdf_dpi(),
        }
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchFile {
    pub media_lane_enabled: Option<bool>,
    pub media_lane_fraction: Option<f64>,
    pub media_gate_image: Option<f64>,
    pub media_gate_pdf: Option<f64>,
    pub media_display_image_best: Option<f64>,
    pub media_display_image_worst: Option<f64>,
    pub media_display_pdf_best: Option<f64>,
    pub media_display_pdf_worst: Option<f64>,
}

const fn default_max_file_mb() -> u64 {
    20
}
const fn default_pdf_pages_per_chunk() -> u8 {
    6
}
const fn default_pdf_dpi() -> u16 {
    150
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbedFile {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub dim: Option<usize>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
}

/// CLI-side TOML schema (`docindex-search`).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CliFile {
    pub server: Option<String>,
    pub token: Option<String>,
    pub token_env: Option<String>,
    pub limit: Option<usize>,
    pub format: Option<String>,
}

/// `$XDG_CONFIG_HOME` if set, else `$HOME/.config`. `None` if neither
/// resolves (no HOME in the lookup — well-known-path discovery is then
/// simply skipped, env/flag paths still work).
fn config_home(lookup: &Lookup<'_>) -> Option<PathBuf> {
    if let Some(x) = lookup("XDG_CONFIG_HOME")
        && !x.is_empty()
    {
        return Some(PathBuf::from(x));
    }
    lookup("HOME")
        .filter(|h| !h.is_empty())
        .map(|h| PathBuf::from(h).join(".config"))
}

/// Well-known server config locations, in search order, after the explicit
/// `--config` / `$DOCINDEX_CONFIG` layer has been ruled out.
fn well_known_server_paths(lookup: &Lookup<'_>) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(home) = config_home(lookup) {
        v.push(home.join("docindex").join("server.toml"));
    }
    v.push(PathBuf::from("/etc/docindex/server.toml"));
    v
}

/// Well-known CLI config locations (no system-wide `/etc` fallback — the
/// CLI is a per-user tool).
fn well_known_cli_paths(lookup: &Lookup<'_>) -> Vec<PathBuf> {
    config_home(lookup)
        .map(|home| vec![home.join("docindex").join("cli.toml")])
        .unwrap_or_default()
}

fn parse_toml<T: for<'de> Deserialize<'de>>(
    path: &Path,
    content: &FileContent,
) -> Result<T, FileConfigError> {
    toml::from_str(&content.text).map_err(|e| FileConfigError::Parse {
        path: path.to_path_buf(),
        source: e,
    })
}

fn read_required<T, F>(
    reader: &FileReader<'_>,
    path: &Path,
    has_inline_secret: F,
) -> Result<(PathBuf, T), FileConfigError>
where
    T: for<'de> Deserialize<'de>,
    F: Fn(&T) -> bool,
{
    let content = reader(path).ok_or_else(|| FileConfigError::NotFound {
        path: path.to_path_buf(),
    })?;
    let parsed: T = parse_toml(path, &content)?;
    warn_world_readable_secret(path, &content, has_inline_secret(&parsed));
    Ok((path.to_path_buf(), parsed))
}

/// Locate + parse the server config file per the documented search order:
/// `--config` flag > `$DOCINDEX_CONFIG` > `~/.config/docindex/server.toml`
/// (respecting `$XDG_CONFIG_HOME`) > `/etc/docindex/server.toml`.
///
/// An explicit path (flag or env var) that doesn't exist is a hard error;
/// well-known fallback locations are silently skipped when absent so a
/// pure-env deployment (today's production) is unaffected.
pub fn find_server_config(
    lookup: &Lookup<'_>,
    reader: &FileReader<'_>,
    config_flag: Option<&Path>,
) -> Result<Option<(PathBuf, ServerFile)>, FileConfigError> {
    if let Some(p) = config_flag {
        return read_required(reader, p, server_has_inline_secret).map(Some);
    }
    if let Some(v) = lookup("DOCINDEX_CONFIG")
        && !v.is_empty()
    {
        return read_required(reader, &PathBuf::from(v), server_has_inline_secret).map(Some);
    }
    for p in well_known_server_paths(lookup) {
        if let Some(content) = reader(&p) {
            let parsed: ServerFile = parse_toml(&p, &content)?;
            warn_world_readable_secret(&p, &content, server_has_inline_secret(&parsed));
            return Ok(Some((p, parsed)));
        }
    }
    Ok(None)
}

/// Locate + parse the CLI config file per: `--config` flag >
/// `$DOCINDEX_CLI_CONFIG` > `~/.config/docindex/cli.toml`.
pub fn find_cli_config(
    lookup: &Lookup<'_>,
    reader: &FileReader<'_>,
    config_flag: Option<&Path>,
) -> Result<Option<(PathBuf, CliFile)>, FileConfigError> {
    if let Some(p) = config_flag {
        return read_required(reader, p, cli_has_inline_secret).map(Some);
    }
    if let Some(v) = lookup("DOCINDEX_CLI_CONFIG")
        && !v.is_empty()
    {
        return read_required(reader, &PathBuf::from(v), cli_has_inline_secret).map(Some);
    }
    for p in well_known_cli_paths(lookup) {
        if let Some(content) = reader(&p) {
            let parsed: CliFile = parse_toml(&p, &content)?;
            warn_world_readable_secret(&p, &content, cli_has_inline_secret(&parsed));
            return Ok(Some((p, parsed)));
        }
    }
    Ok(None)
}

fn server_has_inline_secret(f: &ServerFile) -> bool {
    f.bearer.as_deref().is_some_and(|s| !s.is_empty())
        || f.embed.api_key.as_deref().is_some_and(|s| !s.is_empty())
}

fn cli_has_inline_secret(f: &CliFile) -> bool {
    f.token.as_deref().is_some_and(|s| !s.is_empty())
}

/// Warn (not refuse) when a config file readable by group/other carries an
/// inline secret. `mode & 0o077 != 0` means group or other has some
/// permission bit set.
fn warn_world_readable_secret(path: &Path, content: &FileContent, has_secret: bool) {
    if !has_secret {
        return;
    }
    if let Some(mode) = content.mode
        && mode & 0o077 != 0
    {
        warn!(
            file = %path.display(),
            mode = format!("{mode:o}"),
            "config file contains an inline secret and is readable by group/other; tighten permissions (chmod 600)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup(map: &HashMap<String, String>) -> impl Fn(&str) -> Option<String> + '_ {
        move |k| map.get(k).cloned()
    }

    fn reader_for(files: HashMap<PathBuf, FileContent>) -> impl Fn(&Path) -> Option<FileContent> {
        move |p: &Path| files.get(p).cloned()
    }

    #[test]
    fn explicit_flag_missing_file_is_error() {
        let env = HashMap::new();
        let reader = reader_for(HashMap::new());
        let err =
            find_server_config(&lookup(&env), &reader, Some(Path::new("/nope.toml"))).unwrap_err();
        assert!(matches!(err, FileConfigError::NotFound { .. }));
    }

    #[test]
    fn well_known_path_used_when_present() {
        let mut env = HashMap::new();
        env.insert("HOME".into(), "/home/u".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/home/u/.config/docindex/server.toml"),
            FileContent {
                text: "vault_dir = \"/v\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = reader_for(files);
        let (path, parsed) = find_server_config(&lookup(&env), &reader, None)
            .unwrap()
            .expect("found");
        assert_eq!(path, PathBuf::from("/home/u/.config/docindex/server.toml"));
        assert_eq!(parsed.vault_dir.as_deref(), Some("/v"));
    }

    #[test]
    fn xdg_config_home_overrides_home_config() {
        let mut env = HashMap::new();
        env.insert("HOME".into(), "/home/u".into());
        env.insert("XDG_CONFIG_HOME".into(), "/custom/xdg".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/custom/xdg/docindex/server.toml"),
            FileContent {
                text: "listen = \"1.2.3.4:7777\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = reader_for(files);
        let (path, parsed) = find_server_config(&lookup(&env), &reader, None)
            .unwrap()
            .expect("found");
        assert_eq!(path, PathBuf::from("/custom/xdg/docindex/server.toml"));
        assert_eq!(parsed.listen.as_deref(), Some("1.2.3.4:7777"));
    }

    #[test]
    fn falls_back_to_etc_when_home_missing() {
        let env = HashMap::new();
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/etc/docindex/server.toml"),
            FileContent {
                text: "db_path = \"/d\"".into(),
                mode: Some(0o644),
            },
        );
        let reader = reader_for(files);
        let (path, parsed) = find_server_config(&lookup(&env), &reader, None)
            .unwrap()
            .expect("found");
        assert_eq!(path, PathBuf::from("/etc/docindex/server.toml"));
        assert_eq!(parsed.db_path.as_deref(), Some("/d"));
    }

    #[test]
    fn no_file_anywhere_is_none() {
        let env = HashMap::new();
        let reader = reader_for(HashMap::new());
        let result = find_server_config(&lookup(&env), &reader, None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn docindex_config_env_takes_priority_over_well_known() {
        let mut env = HashMap::new();
        env.insert("HOME".into(), "/home/u".into());
        env.insert("DOCINDEX_CONFIG".into(), "/explicit/path.toml".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/explicit/path.toml"),
            FileContent {
                text: "bearer = \"secret\"".into(),
                mode: Some(0o600),
            },
        );
        files.insert(
            PathBuf::from("/home/u/.config/docindex/server.toml"),
            FileContent {
                text: "bearer = \"wrong\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = reader_for(files);
        let (path, parsed) = find_server_config(&lookup(&env), &reader, None)
            .unwrap()
            .expect("found");
        assert_eq!(path, PathBuf::from("/explicit/path.toml"));
        assert_eq!(parsed.bearer.as_deref(), Some("secret"));
    }

    #[test]
    fn cli_flag_beats_env_and_well_known() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_CONFIG".into(), "/from-env.toml".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/from-flag.toml"),
            FileContent {
                text: "server = \"http://x\"".into(),
                mode: Some(0o600),
            },
        );
        files.insert(
            PathBuf::from("/from-env.toml"),
            FileContent {
                text: "server = \"http://y\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = reader_for(files);
        let (path, parsed) =
            find_cli_config(&lookup(&env), &reader, Some(Path::new("/from-flag.toml")))
                .unwrap()
                .expect("found");
        assert_eq!(path, PathBuf::from("/from-flag.toml"));
        assert_eq!(parsed.server.as_deref(), Some("http://x"));
    }

    #[test]
    fn malformed_toml_is_parse_error() {
        let env = HashMap::new();
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/bad.toml"),
            FileContent {
                text: "this is not valid toml [[[".into(),
                mode: Some(0o600),
            },
        );
        let reader = reader_for(files);
        let err =
            find_server_config(&lookup(&env), &reader, Some(Path::new("/bad.toml"))).unwrap_err();
        assert!(matches!(err, FileConfigError::Parse { .. }));
    }

    #[test]
    fn unknown_field_is_parse_error() {
        let env = HashMap::new();
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/typo.toml"),
            FileContent {
                text: "vaultdir = \"/v\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = reader_for(files);
        let err =
            find_server_config(&lookup(&env), &reader, Some(Path::new("/typo.toml"))).unwrap_err();
        assert!(matches!(err, FileConfigError::Parse { .. }));
    }
}
