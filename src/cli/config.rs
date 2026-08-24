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
};

/// Resolved CLI configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliConfig {
    pub server: String,
    pub token: String,
    pub limit: usize,
    pub format: OutputFormat,
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
#[derive(Debug, Default, Clone)]
pub struct CliFlags {
    pub config_path: Option<PathBuf>,
    pub server: Option<String>,
    pub token: Option<String>,
    pub limit: Option<usize>,
    pub json: bool,
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

fn indirected(
    inline: &Option<String>,
    env_key: &Option<String>,
    lookup: &Lookup<'_>,
) -> Option<String> {
    if let Some(v) = inline
        && !v.is_empty()
    {
        return Some(v.clone());
    }
    env_key
        .as_ref()
        .filter(|k| !k.is_empty())
        .and_then(|k| lookup(k))
        .filter(|v| !v.is_empty())
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
}
