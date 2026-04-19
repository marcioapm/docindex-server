//! Env-based configuration (12-factor).
//!
//! All runtime configuration comes from environment variables. See
//! `.env.example` for the full list. `0.0.0.0` binds (and the v6 equivalent)
//! are rejected at startup.

use std::{
    net::IpAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use thiserror::Error;

/// Typed, validated runtime configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub vault_dir: PathBuf,
    pub db_path: PathBuf,
    pub listen: String,
    pub bearer: String,
    pub gemini_key: String,
    pub embed_model: String,
    pub embed_dim: usize,
    pub debounce: Duration,
    pub log_format: String,
    pub http_timeout: Duration,
    /// Dev/test-only: when true, `127.0.0.1` binds are allowed. MUST stay
    /// false in production (Tailscale is the perimeter).
    pub allow_loopback: bool,
    /// Which embedder to build. "gemini" in prod, "fake" in tests / when
    /// `GEMINI_API_KEY` is unset.
    pub embed_backend: String,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config: {0}")]
    Invalid(String),
}

/// Function signature matching `std::env::var`, injectable for tests.
pub type Lookup<'a> = dyn Fn(&str) -> Option<String> + 'a;

impl Config {
    /// Load and validate configuration from the process environment.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(&|k| std::env::var(k).ok())
    }

    /// Load and validate configuration from a custom lookup function.
    pub fn from_lookup(lookup: &Lookup<'_>) -> Result<Self, ConfigError> {
        let mut errs: Vec<String> = Vec::new();

        let vault_dir_raw = get_or_default(lookup, "DOCINDEX_VAULT_DIR", "");
        let db_path_raw = get_or_default(lookup, "DOCINDEX_DB_PATH", "");
        let listen = get_or_default(lookup, "DOCINDEX_LISTEN", "");
        let bearer = get_or_default(lookup, "DOCINDEX_BEARER", "");
        let gemini_key = get_or_default(lookup, "GEMINI_API_KEY", "");
        let embed_model = get_or_default(lookup, "DOCINDEX_EMBED_MODEL", "gemini-embedding-001");
        let log_format = get_or_default(lookup, "DOCINDEX_LOG_FORMAT", "json").to_ascii_lowercase();
        let allow_loopback_raw =
            get_or_default(lookup, "DOCINDEX_ALLOW_LOOPBACK", "false").to_ascii_lowercase();
        let allow_loopback = matches!(allow_loopback_raw.as_str(), "1" | "true" | "yes" | "on");
        let embed_backend_raw = get_or_default(lookup, "DOCINDEX_EMBED", "").to_ascii_lowercase();
        let embed_backend = if !embed_backend_raw.is_empty() {
            embed_backend_raw
        } else if !gemini_key.is_empty() {
            "gemini".to_string()
        } else {
            "fake".to_string()
        };

        let embed_dim = parse_int_default(lookup, "DOCINDEX_EMBED_DIM", 3072, &mut errs);
        let debounce_ms = parse_int_default(lookup, "DOCINDEX_DEBOUNCE_MS", 5000, &mut errs);
        let http_timeout_ms =
            parse_int_default(lookup, "DOCINDEX_HTTP_TIMEOUT_MS", 30000, &mut errs);

        let vault_dir = validate_vault_dir(&vault_dir_raw, &mut errs);
        let db_path = validate_db_path(&db_path_raw, &mut errs);

        if listen.is_empty() {
            errs.push("DOCINDEX_LISTEN is required".into());
        } else if let Err(e) = validate_listen(&listen, allow_loopback) {
            errs.push(e);
        }
        if bearer.is_empty() {
            errs.push("DOCINDEX_BEARER is required".into());
        }
        if embed_backend == "gemini" && gemini_key.is_empty() {
            errs.push("GEMINI_API_KEY is required when DOCINDEX_EMBED=gemini".into());
        }
        if embed_backend != "gemini" && embed_backend != "fake" {
            errs.push(format!(
                "DOCINDEX_EMBED {embed_backend:?}: must be 'gemini' or 'fake'"
            ));
        }
        if embed_dim == 0 {
            errs.push("DOCINDEX_EMBED_DIM must be > 0".into());
        }
        if log_format != "json" && log_format != "text" {
            errs.push(format!(
                "DOCINDEX_LOG_FORMAT {log_format:?}: must be 'json' or 'text'"
            ));
        }

        if !errs.is_empty() {
            return Err(ConfigError::Invalid(errs.join("; ")));
        }

        Ok(Self {
            vault_dir: vault_dir.unwrap_or_default(),
            db_path: db_path.unwrap_or_default(),
            listen,
            bearer,
            gemini_key,
            embed_model,
            embed_dim,
            debounce: Duration::from_millis(debounce_ms as u64),
            log_format,
            http_timeout: Duration::from_millis(http_timeout_ms as u64),
            allow_loopback,
            embed_backend,
        })
    }
}

fn get_or_default(lookup: &Lookup<'_>, key: &str, def: &str) -> String {
    match lookup(key) {
        Some(v) if !v.is_empty() => v,
        _ => def.to_string(),
    }
}

fn parse_int_default(lookup: &Lookup<'_>, key: &str, def: i64, errs: &mut Vec<String>) -> usize {
    match lookup(key) {
        Some(v) if !v.is_empty() => match v.parse::<i64>() {
            Ok(n) if n >= 0 => n as usize,
            Ok(_) => {
                errs.push(format!("{key} {v:?}: must be >= 0"));
                def as usize
            }
            Err(e) => {
                errs.push(format!("{key} {v:?}: must be an integer: {e}"));
                def as usize
            }
        },
        _ => def as usize,
    }
}

fn validate_vault_dir(raw: &str, errs: &mut Vec<String>) -> Option<PathBuf> {
    if raw.is_empty() {
        errs.push("DOCINDEX_VAULT_DIR is required".into());
        return None;
    }
    let abs = match absolutize(Path::new(raw)) {
        Ok(p) => p,
        Err(e) => {
            errs.push(format!("DOCINDEX_VAULT_DIR: {e}"));
            return None;
        }
    };
    match std::fs::metadata(&abs) {
        Ok(md) if md.is_dir() => Some(abs),
        Ok(_) => {
            errs.push(format!(
                "DOCINDEX_VAULT_DIR {abs:?} is not a directory",
                abs = abs.display()
            ));
            None
        }
        Err(e) => {
            errs.push(format!("DOCINDEX_VAULT_DIR {}: {e}", abs.display()));
            None
        }
    }
}

fn validate_db_path(raw: &str, errs: &mut Vec<String>) -> Option<PathBuf> {
    if raw.is_empty() {
        errs.push("DOCINDEX_DB_PATH is required".into());
        return None;
    }
    let abs = match absolutize(Path::new(raw)) {
        Ok(p) => p,
        Err(e) => {
            errs.push(format!("DOCINDEX_DB_PATH: {e}"));
            return None;
        }
    };
    let parent = abs.parent().unwrap_or(Path::new("/"));
    match std::fs::metadata(parent) {
        Ok(md) if md.is_dir() => Some(abs),
        Ok(_) => {
            errs.push(format!(
                "DOCINDEX_DB_PATH parent {} is not a directory",
                parent.display()
            ));
            None
        }
        Err(e) => {
            errs.push(format!("DOCINDEX_DB_PATH parent {}: {e}", parent.display()));
            None
        }
    }
}

/// Accept "host:port" where host is not the unspecified / all-interfaces IP.
/// Operators can still supply any specific IP (Tailscale or otherwise);
/// this is a belt-and-suspenders check against the obvious footgun.
///
/// `allow_loopback` permits `127.0.0.1` / `::1` binds for local dev + tests.
/// Production MUST leave it false.
fn validate_listen(addr: &str, allow_loopback: bool) -> Result<(), String> {
    // Support bracketed v6 "[::1]:7777" and bare "1.2.3.4:7777".
    let (host, port) = split_host_port(addr)
        .ok_or_else(|| format!("DOCINDEX_LISTEN {addr:?}: expected host:port"))?;
    if host.is_empty() {
        return Err(format!(
            "DOCINDEX_LISTEN {addr:?}: empty host (refusing to bind to all interfaces)"
        ));
    }
    if port.is_empty() {
        return Err(format!("DOCINDEX_LISTEN {addr:?}: empty port"));
    }
    if port.parse::<u16>().is_err() {
        return Err(format!("DOCINDEX_LISTEN {addr:?}: port not a valid u16"));
    }
    if host == "0.0.0.0" || host == "::" {
        return Err(format!(
            "DOCINDEX_LISTEN {addr:?}: binding to all interfaces is not allowed; use a Tailscale IP"
        ));
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        if ip.is_unspecified() {
            return Err(format!(
                "DOCINDEX_LISTEN {addr:?}: unspecified IP is not allowed"
            ));
        }
        if ip.is_loopback() && !allow_loopback {
            return Err(format!(
                "DOCINDEX_LISTEN {addr:?}: loopback bind requires DOCINDEX_ALLOW_LOOPBACK=true"
            ));
        }
    }
    Ok(())
}

fn split_host_port(addr: &str) -> Option<(&str, &str)> {
    // Bracketed IPv6: "[::1]:7777".
    if let Some(rest) = addr.strip_prefix('[') {
        let end = rest.find(']')?;
        let host = &rest[..end];
        let after = &rest[end + 1..];
        let port = after.strip_prefix(':')?;
        return Some((host, port));
    }
    // Plain host:port — reject if there are multiple colons (unbracketed v6).
    let colon = addr.rfind(':')?;
    let host = &addr[..colon];
    let port = &addr[colon + 1..];
    if host.contains(':') {
        return None;
    }
    Some((host, port))
}

fn absolutize(p: &Path) -> std::io::Result<PathBuf> {
    if p.is_absolute() {
        Ok(p.to_path_buf())
    } else {
        let cwd = std::env::current_dir()?;
        Ok(cwd.join(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn lookup(map: &HashMap<String, String>) -> impl Fn(&str) -> Option<String> + '_ {
        move |k| map.get(k).cloned()
    }

    fn base_env(dir: &TempDir) -> HashMap<String, String> {
        let d = dir.path().to_str().unwrap().to_string();
        HashMap::from([
            ("DOCINDEX_VAULT_DIR".into(), d.clone()),
            ("DOCINDEX_DB_PATH".into(), format!("{}/index.db", d)),
            ("DOCINDEX_LISTEN".into(), "100.83.46.59:7777".into()),
            ("DOCINDEX_BEARER".into(), "secret".into()),
            ("GEMINI_API_KEY".into(), "key".into()),
        ])
    }

    #[test]
    fn valid_defaults_populate() {
        let dir = TempDir::new().unwrap();
        let env = base_env(&dir);
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert_eq!(c.embed_model, "gemini-embedding-001");
        assert_eq!(c.embed_dim, 3072);
        assert_eq!(c.log_format, "json");
        assert_eq!(c.debounce, Duration::from_millis(5000));
    }

    #[test]
    fn missing_required_fields_reported() {
        for key in [
            "DOCINDEX_VAULT_DIR",
            "DOCINDEX_DB_PATH",
            "DOCINDEX_LISTEN",
            "DOCINDEX_BEARER",
        ] {
            let dir = TempDir::new().unwrap();
            let mut env = base_env(&dir);
            env.remove(key);
            let err = Config::from_lookup(&lookup(&env)).expect_err("err");
            assert!(format!("{err}").contains(key), "want {key} in {err}");
        }
    }

    #[test]
    fn gemini_key_required_when_backend_gemini() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.remove("GEMINI_API_KEY");
        env.insert("DOCINDEX_EMBED".into(), "gemini".into());
        let err = Config::from_lookup(&lookup(&env)).expect_err("err");
        assert!(format!("{err}").contains("GEMINI_API_KEY"), "{err}");
    }

    #[test]
    fn rejects_zero_zero_bind() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_LISTEN".into(), "0.0.0.0:7777".into());
        let err = Config::from_lookup(&lookup(&env)).expect_err("err");
        assert!(format!("{err}").contains("0.0.0.0"));
    }

    #[test]
    fn rejects_v6_unspecified() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_LISTEN".into(), "[::]:7777".into());
        assert!(Config::from_lookup(&lookup(&env)).is_err());
    }

    #[test]
    fn vault_must_exist() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert(
            "DOCINDEX_VAULT_DIR".into(),
            format!("{}/nope", dir.path().display()),
        );
        assert!(Config::from_lookup(&lookup(&env)).is_err());
    }

    #[test]
    fn db_parent_must_exist() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert(
            "DOCINDEX_DB_PATH".into(),
            "/definitely/not/a/real/path/x.db".into(),
        );
        assert!(Config::from_lookup(&lookup(&env)).is_err());
    }

    #[test]
    fn relative_paths_are_absolutized() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_VAULT_DIR".into(), ".".into());
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert!(c.vault_dir.is_absolute());
    }

    #[test]
    fn invalid_int_errors() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_EMBED_DIM".into(), "not-a-number".into());
        assert!(Config::from_lookup(&lookup(&env)).is_err());
    }

    #[test]
    fn invalid_log_format_errors() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_LOG_FORMAT".into(), "yaml".into());
        assert!(Config::from_lookup(&lookup(&env)).is_err());
    }

    #[test]
    fn listen_missing_port_errors() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_LISTEN".into(), "100.83.46.59".into());
        assert!(Config::from_lookup(&lookup(&env)).is_err());
    }

    #[test]
    fn loopback_requires_bypass() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_LISTEN".into(), "127.0.0.1:7777".into());
        let err = Config::from_lookup(&lookup(&env)).expect_err("err");
        assert!(
            format!("{err}").contains("DOCINDEX_ALLOW_LOOPBACK"),
            "{err}"
        );
    }

    #[test]
    fn loopback_allowed_with_bypass() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_LISTEN".into(), "127.0.0.1:0".into());
        env.insert("DOCINDEX_ALLOW_LOOPBACK".into(), "true".into());
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert!(c.allow_loopback);
    }

    #[test]
    fn embed_backend_defaults_to_fake_without_key() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.remove("GEMINI_API_KEY");
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert_eq!(c.embed_backend, "fake");
    }

    #[test]
    fn embed_backend_defaults_to_gemini_with_key() {
        let dir = TempDir::new().unwrap();
        let env = base_env(&dir);
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert_eq!(c.embed_backend, "gemini");
    }

    #[test]
    fn embed_backend_explicit_fake_does_not_require_key() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.remove("GEMINI_API_KEY");
        env.insert("DOCINDEX_EMBED".into(), "fake".into());
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert_eq!(c.embed_backend, "fake");
    }

    #[test]
    fn embed_backend_invalid_errors() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_EMBED".into(), "bogus".into());
        assert!(Config::from_lookup(&lookup(&env)).is_err());
    }
}
