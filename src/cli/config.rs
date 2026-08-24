//! `docindex-search` CLI configuration.
//!
//! Precedence: CLI flags > environment variables > TOML file > defaults.
//! Mirrors the server's [`crate::config`] layering but with a much smaller
//! schema (server URL, bearer token, default limit, output format).

use std::path::PathBuf;

use thiserror::Error;

use crate::config::{
    Lookup,
    file::{FileReader, find_cli_config},
    indirected,
};

/// Resolved CLI configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct CliConfig {
    pub server: String,
    pub token: String,
    pub limit: usize,
    pub format: OutputFormat,
}

impl std::fmt::Debug for CliConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CliConfig")
            .field("server", &self.server)
            .field("token", &"[redacted]")
            .field("limit", &self.limit)
            .field("format", &self.format)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CliConfigError {
    #[error("docindex-search: {0}")]
    Invalid(String),
}

/// CLI-flag-level overrides. `json` is a bool flag that forces
/// `OutputFormat::Json` regardless of what env/file/default say.
#[derive(Default, Clone)]
pub struct CliFlags {
    pub config_path: Option<PathBuf>,
    pub server: Option<String>,
    pub token: Option<String>,
    pub limit: Option<usize>,
    pub json: bool,
}

impl std::fmt::Debug for CliFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CliFlags")
            .field("config_path", &self.config_path)
            .field("server", &self.server)
            .field("token", &self.token.as_ref().map(|_| "[redacted]"))
            .field("limit", &self.limit)
            .field("json", &self.json)
            .finish()
    }
}

const DEFAULT_LIMIT: usize = 10;

impl CliConfig {
    pub fn load(
        lookup: &Lookup<'_>,
        file_reader: &FileReader<'_>,
        flags: &CliFlags,
    ) -> Result<Self, CliConfigError> {
        let found = find_cli_config(lookup, file_reader, flags.config_path.as_deref())
            .map_err(|e| CliConfigError::Invalid(e.to_string()))?;
        let file = found.map(|(_, f)| f).unwrap_or_default();

        let server = flags
            .server
            .clone()
            .or_else(|| non_empty(lookup("DOCINDEX_CLI_SERVER")))
            .or_else(|| non_empty(file.server.clone()))
            .ok_or_else(|| {
                CliConfigError::Invalid(
                    "server URL is required: pass --server, set $DOCINDEX_CLI_SERVER, or add `server` to cli.toml".into(),
                )
            })?;

        let token_file_effective = indirected(&file.token, &file.token_env, lookup);
        let token = flags
            .token
            .clone()
            .or_else(|| non_empty(lookup("DOCINDEX_CLI_TOKEN")))
            .or(token_file_effective)
            .unwrap_or_default();

        let limit = match flags.limit {
            Some(n) => n,
            None => match lookup("DOCINDEX_CLI_LIMIT").filter(|v| !v.is_empty()) {
                Some(v) => v.parse::<usize>().map_err(|e| {
                    CliConfigError::Invalid(format!("DOCINDEX_CLI_LIMIT {v:?}: {e}"))
                })?,
                None => file.limit.unwrap_or(DEFAULT_LIMIT),
            },
        };
        if limit == 0 {
            return Err(CliConfigError::Invalid("limit must be > 0".into()));
        }

        let format = if flags.json {
            OutputFormat::Json
        } else {
            let raw = non_empty(lookup("DOCINDEX_CLI_FORMAT"))
                .or_else(|| non_empty(file.format.clone()))
                .unwrap_or_else(|| "text".to_string());
            match raw.as_str() {
                "text" => OutputFormat::Text,
                "json" => OutputFormat::Json,
                other => {
                    return Err(CliConfigError::Invalid(format!(
                        "format {other:?}: must be 'text' or 'json'"
                    )));
                }
            }
        };

        Ok(Self {
            server,
            token,
            limit,
            format,
        })
    }
}

fn non_empty(v: Option<String>) -> Option<String> {
    v.filter(|s| !s.is_empty())
}

/// Real-filesystem lookup for the CLI binary's default `$HOME`-relative
/// config search.
pub fn env_lookup(k: &str) -> Option<String> {
    std::env::var(k).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::file::FileContent;
    use std::collections::HashMap;
    use std::path::Path;

    fn lookup(map: &HashMap<String, String>) -> impl Fn(&str) -> Option<String> + '_ {
        move |k| map.get(k).cloned()
    }

    fn empty_lookup() -> impl Fn(&str) -> Option<String> {
        |_: &str| None
    }

    fn no_file() -> impl Fn(&Path) -> Option<FileContent> {
        |_: &Path| None
    }

    fn file_reader_for(
        files: HashMap<PathBuf, FileContent>,
    ) -> impl Fn(&Path) -> Option<FileContent> {
        move |p: &Path| files.get(p).cloned()
    }

    #[test]
    fn flag_server_is_used() {
        let flags = CliFlags {
            server: Some("http://x:1".into()),
            ..Default::default()
        };
        let c = CliConfig::load(&empty_lookup(), &no_file(), &flags).expect("valid");
        assert_eq!(c.server, "http://x:1");
        assert_eq!(c.limit, DEFAULT_LIMIT);
        assert_eq!(c.format, OutputFormat::Text);
    }

    #[test]
    fn missing_server_everywhere_errors() {
        let flags = CliFlags::default();
        let err = CliConfig::load(&empty_lookup(), &no_file(), &flags).unwrap_err();
        assert!(format!("{err}").contains("server"));
    }

    #[test]
    fn env_server_used_when_no_flag() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_SERVER".into(), "http://env:2".into());
        let flags = CliFlags::default();
        let c = CliConfig::load(&lookup(&env), &no_file(), &flags).expect("valid");
        assert_eq!(c.server, "http://env:2");
    }

    #[test]
    fn flag_beats_env_and_file() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_SERVER".into(), "http://env:2".into());
        env.insert("DOCINDEX_CLI_CONFIG".into(), "/cfg.toml".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg.toml"),
            FileContent {
                text: "server = \"http://file:3\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = CliFlags {
            server: Some("http://flag:1".into()),
            ..Default::default()
        };
        let c = CliConfig::load(&lookup(&env), &reader, &flags).expect("valid");
        assert_eq!(c.server, "http://flag:1");
    }

    #[test]
    fn file_used_when_no_flag_or_env() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_CONFIG".into(), "/cfg.toml".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg.toml"),
            FileContent {
                text: "server = \"http://file:3\"\nlimit = 25\nformat = \"json\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = CliFlags::default();
        let c = CliConfig::load(&lookup(&env), &reader, &flags).expect("valid");
        assert_eq!(c.server, "http://file:3");
        assert_eq!(c.limit, 25);
        assert_eq!(c.format, OutputFormat::Json);
    }

    #[test]
    fn token_env_indirection() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_CONFIG".into(), "/cfg.toml".into());
        env.insert("MY_TOKEN".into(), "indirected-tok".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg.toml"),
            FileContent {
                text: "server = \"http://x\"\ntoken_env = \"MY_TOKEN\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = CliFlags::default();
        let c = CliConfig::load(&lookup(&env), &reader, &flags).expect("valid");
        assert_eq!(c.token, "indirected-tok");
    }

    #[test]
    fn json_flag_forces_json_even_if_file_says_text() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_CONFIG".into(), "/cfg.toml".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg.toml"),
            FileContent {
                text: "server = \"http://x\"\nformat = \"text\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = CliFlags {
            json: true,
            ..Default::default()
        };
        let c = CliConfig::load(&lookup(&env), &reader, &flags).expect("valid");
        assert_eq!(c.format, OutputFormat::Json);
    }

    #[test]
    fn invalid_format_errors() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_CONFIG".into(), "/cfg.toml".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg.toml"),
            FileContent {
                text: "server = \"http://x\"\nformat = \"yaml\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = CliFlags::default();
        assert!(CliConfig::load(&lookup(&env), &reader, &flags).is_err());
    }

    #[test]
    fn zero_limit_errors() {
        let flags = CliFlags {
            server: Some("http://x".into()),
            limit: Some(0),
            ..Default::default()
        };
        assert!(CliConfig::load(&empty_lookup(), &no_file(), &flags).is_err());
    }

    /// `flags.limit` must beat both `DOCINDEX_CLI_LIMIT` env var and the file
    /// value. If the precedence order were inverted, env (42) or file (5)
    /// would win and the assertion would fail.
    #[test]
    fn flag_limit_beats_env_and_file() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_SERVER".into(), "http://x".into());
        env.insert("DOCINDEX_CLI_LIMIT".into(), "42".into());
        env.insert("DOCINDEX_CLI_CONFIG".into(), "/cfg.toml".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg.toml"),
            FileContent {
                text: "server = \"http://x\"\nlimit = 5".into(),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = CliFlags {
            limit: Some(7),
            ..Default::default()
        };
        let c = CliConfig::load(&lookup(&env), &reader, &flags).expect("valid");
        // flag says 7, env says 42, file says 5 — flag must win.
        assert_eq!(c.limit, 7);
    }

    /// `CliConfig::Debug` must not expose the bearer token.
    #[test]
    fn cli_config_debug_redacts_token() {
        let c = CliConfig {
            server: "http://x".into(),
            token: "secret-bearer".into(),
            limit: 10,
            format: OutputFormat::Text,
        };
        let dbg = format!("{c:?}");
        assert!(
            !dbg.contains("secret-bearer"),
            "token must not appear in Debug output: {dbg}"
        );
        assert!(dbg.contains("[redacted]"), "expected [redacted] in: {dbg}");
        assert!(dbg.contains("http://x"), "{dbg}");
    }

    /// `CliFlags::Debug` must not expose the token when set.
    #[test]
    fn cli_flags_debug_redacts_token() {
        let f = CliFlags {
            token: Some("secret-flag-token".into()),
            ..Default::default()
        };
        let dbg = format!("{f:?}");
        assert!(
            !dbg.contains("secret-flag-token"),
            "token must not appear in Debug output: {dbg}"
        );
        assert!(dbg.contains("[redacted]"), "expected [redacted] in: {dbg}");
    }

    /// `DOCINDEX_CLI_LIMIT` env var overrides the file value when no flag is
    /// present.
    #[test]
    fn env_limit_beats_file() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_SERVER".into(), "http://x".into());
        env.insert("DOCINDEX_CLI_LIMIT".into(), "42".into());
        env.insert("DOCINDEX_CLI_CONFIG".into(), "/cfg.toml".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg.toml"),
            FileContent {
                text: "server = \"http://x\"\nlimit = 5".into(),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = CliFlags::default();
        let c = CliConfig::load(&lookup(&env), &reader, &flags).expect("valid");
        // env says 42, file says 5 — env must win.
        assert_eq!(c.limit, 42);
    }

    /// `DOCINDEX_CLI_FORMAT` env var overrides the file value when no flag is
    /// present.
    #[test]
    fn env_format_beats_file() {
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CLI_SERVER".into(), "http://x".into());
        env.insert("DOCINDEX_CLI_FORMAT".into(), "json".into());
        env.insert("DOCINDEX_CLI_CONFIG".into(), "/cfg.toml".into());
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg.toml"),
            FileContent {
                text: "server = \"http://x\"\nformat = \"text\"".into(),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = CliFlags::default();
        let c = CliConfig::load(&lookup(&env), &reader, &flags).expect("valid");
        // env says json, file says text — env must win.
        assert_eq!(c.format, OutputFormat::Json);
    }
}
