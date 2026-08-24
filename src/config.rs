//! Layered configuration: CLI flags > environment variables > TOML file >
//! built-in defaults.
//!
//! The running production service is driven entirely by env vars (see
//! `.env.example`) and that path keeps working unchanged — `Config::from_env`
//! is a thin wrapper over [`Config::load`] with no file and no flag
//! overrides. `0.0.0.0` binds (and the v6 equivalent) are rejected at
//! startup regardless of which layer set `listen`.

pub mod file;

use std::{
    net::IpAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use thiserror::Error;

use crate::embed::registry::{self, EmbedProvider};

pub use file::{
    CliFile, FileReader, ServerFile, find_cli_config, find_server_config, os_file_reader,
};

/// Typed, validated runtime configuration.
///
/// `Debug` is implemented manually to redact `bearer` and `embed_api_key`.
#[derive(Clone)]
pub struct Config {
    pub vault_dir: PathBuf,
    pub db_path: PathBuf,
    pub listen: String,
    pub bearer: String,
    pub embed_provider: EmbedProvider,
    pub embed_model: String,
    pub embed_dim: usize,
    pub embed_api_key: String,
    /// Explicit override for the provider's API base URL (proxy/mock).
    /// Deliberately excluded from the index fingerprint.
    pub embed_base_url: Option<String>,
    pub debounce: Duration,
    pub log_format: String,
    pub http_timeout: Duration,
    /// Dev/test-only: when true, `127.0.0.1` binds are allowed. MUST stay
    /// false in production (Tailscale is the perimeter).
    pub allow_loopback: bool,
    /// Display-side smoothing constant for `score_normalized`. NOT the RRF
    /// constant (ranking is always k=60). Smaller = faster decay past rank-1.
    pub display_k: f64,
    /// Weight of the semantic branch in `score_normalized`.
    pub weight_vec: f64,
    /// Weight of the BM25 branch in `score_normalized`. `weight_vec + weight_bm25`
    /// is validated to sum to 1.0 (± 0.01).
    pub weight_bm25: f64,
    /// `--reembed`: wipe chunks/vectors/FTS and rebuild when the index
    /// fingerprint (provider/model/dim) no longer matches this config.
    pub reembed: bool,
    /// Path of the TOML file this config was loaded from, if any — for
    /// startup logging only.
    pub config_path: Option<PathBuf>,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("vault_dir", &self.vault_dir)
            .field("db_path", &self.db_path)
            .field("listen", &self.listen)
            .field("bearer", &"[redacted]")
            .field("embed_provider", &self.embed_provider)
            .field("embed_model", &self.embed_model)
            .field("embed_dim", &self.embed_dim)
            .field("embed_api_key", &"[redacted]")
            .field("embed_base_url", &self.embed_base_url)
            .field("debounce", &self.debounce)
            .field("log_format", &self.log_format)
            .field("http_timeout", &self.http_timeout)
            .field("allow_loopback", &self.allow_loopback)
            .field("display_k", &self.display_k)
            .field("weight_vec", &self.weight_vec)
            .field("weight_bm25", &self.weight_bm25)
            .field("reembed", &self.reembed)
            .field("config_path", &self.config_path)
            .finish()
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config: {0}")]
    Invalid(String),
}

/// Function signature matching `std::env::var`, injectable for tests.
pub type Lookup<'a> = dyn Fn(&str) -> Option<String> + 'a;

/// CLI-flag-level overrides for the server binary. Only `--config` and
/// `--reembed` are documented server flags — every other field is env/file/
/// default only.
#[derive(Debug, Default, Clone)]
pub struct ConfigFlags {
    pub config_path: Option<PathBuf>,
    pub reembed: bool,
}

impl Config {
    /// Load and validate configuration from the process environment, with
    /// no CLI overrides. Thin wrapper over [`Config::load`] using the real
    /// filesystem for TOML discovery — production deployments have no
    /// config file at the well-known locations, so this resolves to
    /// exactly the pre-TOML env-only behavior.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::load(
            &|k| std::env::var(k).ok(),
            &os_file_reader,
            &ConfigFlags::default(),
        )
    }

    /// Load and validate configuration from a custom lookup function, with
    /// no file layer and no flags. Preserved for existing unit tests.
    pub fn from_lookup(lookup: &Lookup<'_>) -> Result<Self, ConfigError> {
        Self::load(lookup, &|_: &Path| None, &ConfigFlags::default())
    }

    /// Full layered load: CLI flags > env vars > TOML file > defaults.
    pub fn load(
        lookup: &Lookup<'_>,
        file_reader: &FileReader<'_>,
        flags: &ConfigFlags,
    ) -> Result<Self, ConfigError> {
        let mut errs: Vec<String> = Vec::new();

        let found = find_server_config(lookup, file_reader, flags.config_path.as_deref())
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let (config_path, file) = match found {
            Some((p, f)) => (Some(p), f),
            None => (None, ServerFile::default()),
        };

        let vault_dir_raw =
            resolved_str(lookup, "DOCINDEX_VAULT_DIR", file.vault_dir.as_deref(), "");
        let db_path_raw = resolved_str(lookup, "DOCINDEX_DB_PATH", file.db_path.as_deref(), "");
        let listen = resolved_str(lookup, "DOCINDEX_LISTEN", file.listen.as_deref(), "");
        let log_format = resolved_str(
            lookup,
            "DOCINDEX_LOG_FORMAT",
            file.log_format.as_deref(),
            "json",
        )
        .to_ascii_lowercase();
        let allow_loopback = resolved_bool(
            lookup,
            "DOCINDEX_ALLOW_LOOPBACK",
            file.allow_loopback,
            false,
        );

        let bearer_file_effective = indirected(&file.bearer, &file.bearer_env, lookup);
        let bearer = resolved_str(
            lookup,
            "DOCINDEX_BEARER",
            bearer_file_effective.as_deref(),
            "",
        );

        let debounce_ms = resolved_u64(
            lookup,
            "DOCINDEX_DEBOUNCE_MS",
            file.debounce_ms,
            5000,
            &mut errs,
        );
        let http_timeout_ms = resolved_u64(
            lookup,
            "DOCINDEX_HTTP_TIMEOUT_MS",
            file.http_timeout_ms,
            30000,
            &mut errs,
        );

        let display_k = parse_float_default(
            lookup,
            "DOCINDEX_DISPLAY_K",
            crate::search::DEFAULT_DISPLAY_K,
            &mut errs,
        );
        if display_k <= 0.0 {
            errs.push(format!("DOCINDEX_DISPLAY_K {display_k}: must be > 0"));
        }
        let weight_vec = parse_float_default(
            lookup,
            "DOCINDEX_WEIGHT_VEC",
            crate::search::DEFAULT_WEIGHT_VEC,
            &mut errs,
        );
        // If BM25 weight is unset, derive it as (1 - weight_vec). This keeps
        // the common case (operator only tunes the vec weight) one-knob
        // simple while still allowing explicit overrides for experiments.
        let weight_bm25 = match lookup("DOCINDEX_WEIGHT_BM25") {
            Some(v) if !v.is_empty() => match v.parse::<f64>() {
                Ok(n) => n,
                Err(e) => {
                    errs.push(format!("DOCINDEX_WEIGHT_BM25 {v:?}: must be a float: {e}"));
                    crate::search::DEFAULT_WEIGHT_BM25
                }
            },
            _ => 1.0 - weight_vec,
        };
        let weight_sum = weight_vec + weight_bm25;
        if !(0.99..=1.01).contains(&weight_sum) {
            errs.push(format!(
                "DOCINDEX_WEIGHT_VEC ({weight_vec}) + DOCINDEX_WEIGHT_BM25 ({weight_bm25}) = {weight_sum}; must sum to 1.0 (± 0.01)"
            ));
        }
        if !(0.0..=1.0).contains(&weight_vec) {
            errs.push(format!(
                "DOCINDEX_WEIGHT_VEC {weight_vec}: must be in [0.0, 1.0]"
            ));
        }
        if !(0.0..=1.0).contains(&weight_bm25) {
            errs.push(format!(
                "DOCINDEX_WEIGHT_BM25 {weight_bm25}: must be in [0.0, 1.0]"
            ));
        }

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
        if log_format != "json" && log_format != "text" {
            errs.push(format!(
                "DOCINDEX_LOG_FORMAT {log_format:?}: must be 'json' or 'text'"
            ));
        }

        let (embed_provider, embed_model, embed_dim, embed_api_key) =
            resolve_embed(lookup, &file.embed, &mut errs);

        if !errs.is_empty() {
            return Err(ConfigError::Invalid(errs.join("; ")));
        }

        Ok(Self {
            vault_dir: vault_dir.unwrap_or_default(),
            db_path: db_path.unwrap_or_default(),
            listen,
            bearer,
            embed_provider,
            embed_model,
            embed_dim,
            embed_api_key,
            embed_base_url: file.embed.base_url.clone(),
            debounce: Duration::from_millis(debounce_ms),
            log_format,
            http_timeout: Duration::from_millis(http_timeout_ms),
            allow_loopback,
            display_k,
            weight_vec,
            weight_bm25,
            reembed: flags.reembed,
            config_path,
        })
    }
}

/// Resolve the embed provider/model/dim/api_key quadruple, pushing any
/// validation errors onto `errs`. Returns best-effort placeholder values on
/// failure so the caller can keep constructing a `Config` that is discarded
/// once `errs` is non-empty.
fn resolve_embed(
    lookup: &Lookup<'_>,
    file: &file::EmbedFile,
    errs: &mut Vec<String>,
) -> (EmbedProvider, String, usize, String) {
    // Back-compat: DOCINDEX_EMBED is the historical env var name for
    // provider selection ("gemini" | "fake"); it now also accepts "voyage".
    let provider_raw = resolved_str(lookup, "DOCINDEX_EMBED", file.provider.as_deref(), "");
    let provider_raw = if !provider_raw.is_empty() {
        provider_raw
    } else {
        // No explicit provider anywhere: infer from which API key env var
        // is present, defaulting to "fake". Mirrors the pre-registry
        // behavior of defaulting to gemini only when GEMINI_API_KEY is set.
        let gemini_key_present = lookup("GEMINI_API_KEY").is_some_and(|v| !v.is_empty());
        let voyage_key_present = lookup("VOYAGE_API_KEY").is_some_and(|v| !v.is_empty());
        if gemini_key_present {
            "gemini".to_string()
        } else if voyage_key_present {
            "voyage".to_string()
        } else {
            "fake".to_string()
        }
    };

    let provider = match registry::parse_provider(&provider_raw) {
        Ok(p) => p,
        Err(e) => {
            errs.push(e.to_string());
            EmbedProvider::Fake
        }
    };

    let default_model = provider.default_model().to_string();
    let model = resolved_str(
        lookup,
        "DOCINDEX_EMBED_MODEL",
        file.model.as_deref(),
        &default_model,
    );

    let model_info = match registry::lookup(provider, &model) {
        Ok(info) => Some(info),
        Err(e) => {
            errs.push(e.to_string());
            None
        }
    };
    let native_dim = model_info.as_ref().map(|i| i.native_dim).unwrap_or(3072);

    let dim = resolved_u64(
        lookup,
        "DOCINDEX_EMBED_DIM",
        file.dim.map(|d| d as u64),
        native_dim as u64,
        errs,
    ) as usize;
    if dim == 0 {
        errs.push("DOCINDEX_EMBED_DIM must be > 0".into());
    } else if let Some(info) = &model_info
        && let Err(e) = registry::validate_dim(info, dim)
    {
        errs.push(e.to_string());
    }

    let key_file_effective = indirected(&file.api_key, &file.api_key_env, lookup);
    let api_key = match provider.key_env_var() {
        Some(env_var) => {
            let key = resolved_str(lookup, env_var, key_file_effective.as_deref(), "");
            if key.is_empty() {
                errs.push(registry::RegistryError::MissingKey { provider, env_var }.to_string());
            }
            key
        }
        None => key_file_effective.unwrap_or_default(),
    };

    (provider, model, dim, api_key)
}

/// `bearer`/`api_key`-style indirection: prefer the inline value, else look
/// up the named env var from `*_env`, else `None`.
pub(crate) fn indirected(
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

/// Resolve a string field: env var, else TOML file value, else default.
/// (No CLI-flag layer for the server binary — its only flags are
/// `--config` and `--reembed`, handled separately.)
pub(crate) fn resolved_str(
    lookup: &Lookup<'_>,
    env_key: &str,
    file_val: Option<&str>,
    default: &str,
) -> String {
    if let Some(v) = lookup(env_key)
        && !v.is_empty()
    {
        return v;
    }
    if let Some(v) = file_val
        && !v.is_empty()
    {
        return v.to_string();
    }
    default.to_string()
}

pub(crate) fn resolved_bool(
    lookup: &Lookup<'_>,
    env_key: &str,
    file_val: Option<bool>,
    default: bool,
) -> bool {
    if let Some(v) = lookup(env_key) {
        let lc = v.to_ascii_lowercase();
        if !lc.is_empty() {
            return matches!(lc.as_str(), "1" | "true" | "yes" | "on");
        }
    }
    file_val.unwrap_or(default)
}

pub(crate) fn resolved_u64(
    lookup: &Lookup<'_>,
    env_key: &str,
    file_val: Option<u64>,
    default: u64,
    errs: &mut Vec<String>,
) -> u64 {
    if let Some(v) = lookup(env_key)
        && !v.is_empty()
    {
        return match v.parse::<i64>() {
            Ok(n) if n >= 0 => n as u64,
            Ok(_) => {
                errs.push(format!("{env_key} {v:?}: must be >= 0"));
                default
            }
            Err(e) => {
                errs.push(format!("{env_key} {v:?}: must be an integer: {e}"));
                default
            }
        };
    }
    file_val.unwrap_or(default)
}

fn parse_float_default(lookup: &Lookup<'_>, key: &str, def: f64, errs: &mut Vec<String>) -> f64 {
    match lookup(key) {
        Some(v) if !v.is_empty() => match v.parse::<f64>() {
            Ok(n) => n,
            Err(e) => {
                errs.push(format!("{key} {v:?}: must be a float: {e}"));
                def
            }
        },
        _ => def,
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
    fn key_required_when_provider_gemini() {
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
        assert_eq!(c.embed_provider, EmbedProvider::Fake);
    }

    #[test]
    fn embed_backend_defaults_to_gemini_with_key() {
        let dir = TempDir::new().unwrap();
        let env = base_env(&dir);
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert_eq!(c.embed_provider, EmbedProvider::Gemini);
    }

    #[test]
    fn embed_backend_defaults_to_voyage_with_voyage_key_only() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.remove("GEMINI_API_KEY");
        env.insert("VOYAGE_API_KEY".into(), "vk".into());
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert_eq!(c.embed_provider, EmbedProvider::Voyage);
        assert_eq!(c.embed_model, "voyage-4");
        assert_eq!(c.embed_dim, 1024);
    }

    #[test]
    fn embed_backend_explicit_fake_does_not_require_key() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.remove("GEMINI_API_KEY");
        env.insert("DOCINDEX_EMBED".into(), "fake".into());
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert_eq!(c.embed_provider, EmbedProvider::Fake);
    }

    #[test]
    fn embed_backend_invalid_errors() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_EMBED".into(), "bogus".into());
        let err = Config::from_lookup(&lookup(&env)).expect_err("err");
        assert!(format!("{err}").contains("bogus"));
    }

    #[test]
    fn unknown_model_for_provider_errors() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_EMBED_MODEL".into(), "nope-not-a-model".into());
        let err = Config::from_lookup(&lookup(&env)).expect_err("err");
        assert!(format!("{err}").contains("nope-not-a-model"));
    }

    #[test]
    fn bad_dim_for_model_errors() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_EMBED_DIM".into(), "999".into());
        let err = Config::from_lookup(&lookup(&env)).expect_err("err");
        assert!(format!("{err}").contains("999"));
    }

    #[test]
    fn voyage_bad_dim_lists_allowed() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.remove("GEMINI_API_KEY");
        env.insert("DOCINDEX_EMBED".into(), "voyage".into());
        env.insert("VOYAGE_API_KEY".into(), "vk".into());
        env.insert("DOCINDEX_EMBED_DIM".into(), "3072".into());
        let err = Config::from_lookup(&lookup(&env)).expect_err("err");
        let msg = format!("{err}");
        assert!(msg.contains("256"), "{msg}");
        assert!(msg.contains("2048"), "{msg}");
    }

    #[test]
    fn display_defaults_are_035_weights_sum_to_one() {
        let dir = TempDir::new().unwrap();
        let env = base_env(&dir);
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert_eq!(c.display_k, crate::search::DEFAULT_DISPLAY_K);
        assert_eq!(c.weight_vec, crate::search::DEFAULT_WEIGHT_VEC);
        assert!((c.weight_bm25 - crate::search::DEFAULT_WEIGHT_BM25).abs() < 1e-9);
        assert!((c.weight_vec + c.weight_bm25 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn weight_bm25_derived_from_vec_when_unset() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_WEIGHT_VEC".into(), "0.7".into());
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert!((c.weight_vec - 0.7).abs() < 1e-9);
        assert!((c.weight_bm25 - 0.3).abs() < 1e-9);
    }

    #[test]
    fn explicit_weights_that_dont_sum_to_one_error() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_WEIGHT_VEC".into(), "0.7".into());
        env.insert("DOCINDEX_WEIGHT_BM25".into(), "0.2".into());
        let err = Config::from_lookup(&lookup(&env)).expect_err("err");
        let msg = format!("{err}");
        assert!(msg.contains("must sum to 1.0"), "{msg}");
    }

    #[test]
    fn explicit_weights_sum_to_one_ok() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_WEIGHT_VEC".into(), "0.6".into());
        env.insert("DOCINDEX_WEIGHT_BM25".into(), "0.4".into());
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert!((c.weight_vec - 0.6).abs() < 1e-9);
        assert!((c.weight_bm25 - 0.4).abs() < 1e-9);
    }

    #[test]
    fn weight_vec_outside_unit_interval_errors() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_WEIGHT_VEC".into(), "1.5".into());
        env.insert("DOCINDEX_WEIGHT_BM25".into(), "-0.5".into());
        assert!(Config::from_lookup(&lookup(&env)).is_err());
    }

    #[test]
    fn display_k_must_be_positive() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_DISPLAY_K".into(), "0".into());
        assert!(Config::from_lookup(&lookup(&env)).is_err());
    }

    #[test]
    fn display_k_overridable() {
        let dir = TempDir::new().unwrap();
        let mut env = base_env(&dir);
        env.insert("DOCINDEX_DISPLAY_K".into(), "20".into());
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        assert_eq!(c.display_k, 20.0);
    }

    // --- TOML layering ---------------------------------------------------

    fn empty_lookup() -> impl Fn(&str) -> Option<String> {
        |_: &str| None
    }

    fn file_reader_for(
        files: HashMap<PathBuf, file::FileContent>,
    ) -> impl Fn(&Path) -> Option<file::FileContent> {
        move |p: &Path| files.get(p).cloned()
    }

    fn server_toml_file(dir: &TempDir, extra_embed: &str) -> HashMap<PathBuf, file::FileContent> {
        let d = dir.path().to_str().unwrap().to_string();
        let text = format!(
            r#"
vault_dir = "{d}"
db_path = "{d}/index.db"
listen = "100.83.46.59:7777"
bearer = "file-secret"
log_format = "text"

[embed]
{extra_embed}
"#
        );
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg/server.toml"),
            file::FileContent {
                text,
                mode: Some(0o600),
            },
        );
        files
    }

    #[test]
    fn file_only_no_env_boots() {
        let dir = TempDir::new().unwrap();
        let files = server_toml_file(&dir, "provider = \"fake\"");
        let reader = file_reader_for(files);
        let flags = ConfigFlags {
            config_path: Some(PathBuf::from("/cfg/server.toml")),
            reembed: false,
        };
        let c = Config::load(&empty_lookup(), &reader, &flags).expect("valid");
        assert_eq!(c.bearer, "file-secret");
        assert_eq!(c.listen, "100.83.46.59:7777");
        assert_eq!(c.embed_provider, EmbedProvider::Fake);
    }

    #[test]
    fn env_overrides_file() {
        let dir = TempDir::new().unwrap();
        let files = server_toml_file(&dir, "provider = \"fake\"");
        let reader = file_reader_for(files);
        let mut env = HashMap::new();
        env.insert("DOCINDEX_BEARER".into(), "env-secret".into());
        let flags = ConfigFlags {
            config_path: Some(PathBuf::from("/cfg/server.toml")),
            reembed: false,
        };
        let c = Config::load(&lookup(&env), &reader, &flags).expect("valid");
        assert_eq!(c.bearer, "env-secret");
    }

    #[test]
    fn flag_config_path_overrides_env_config_path() {
        let dir = TempDir::new().unwrap();
        let d = dir.path().to_str().unwrap().to_string();
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/from-flag.toml"),
            file::FileContent {
                text: format!(
                    "vault_dir = \"{d}\"\ndb_path = \"{d}/index.db\"\nlisten = \"1.1.1.1:1\"\nbearer = \"flag-wins\"\n\n[embed]\nprovider = \"fake\"\n"
                ),
                mode: Some(0o600),
            },
        );
        files.insert(
            PathBuf::from("/from-env.toml"),
            file::FileContent {
                text: format!(
                    "vault_dir = \"{d}\"\ndb_path = \"{d}/index.db\"\nlisten = \"2.2.2.2:2\"\nbearer = \"env-wins\"\n\n[embed]\nprovider = \"fake\"\n"
                ),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let mut env = HashMap::new();
        env.insert("DOCINDEX_CONFIG".into(), "/from-env.toml".into());
        let flags = ConfigFlags {
            config_path: Some(PathBuf::from("/from-flag.toml")),
            reembed: false,
        };
        let c = Config::load(&lookup(&env), &reader, &flags).expect("valid");
        assert_eq!(c.bearer, "flag-wins");
        assert_eq!(c.listen, "1.1.1.1:1");
    }

    #[test]
    fn bearer_env_indirection_reads_named_var() {
        let dir = TempDir::new().unwrap();
        let d = dir.path().to_str().unwrap().to_string();
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg/server.toml"),
            file::FileContent {
                text: format!(
                    "vault_dir = \"{d}\"\ndb_path = \"{d}/index.db\"\nlisten = \"1.1.1.1:1\"\nbearer_env = \"MY_BEARER\"\n\n[embed]\nprovider = \"fake\"\n"
                ),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let mut env = HashMap::new();
        env.insert("MY_BEARER".into(), "indirected-secret".into());
        let flags = ConfigFlags {
            config_path: Some(PathBuf::from("/cfg/server.toml")),
            reembed: false,
        };
        let c = Config::load(&lookup(&env), &reader, &flags).expect("valid");
        assert_eq!(c.bearer, "indirected-secret");
    }

    #[test]
    fn api_key_env_indirection_for_voyage() {
        let dir = TempDir::new().unwrap();
        let d = dir.path().to_str().unwrap().to_string();
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg/server.toml"),
            file::FileContent {
                text: format!(
                    "vault_dir = \"{d}\"\ndb_path = \"{d}/index.db\"\nlisten = \"1.1.1.1:1\"\nbearer = \"b\"\n\n[embed]\nprovider = \"voyage\"\napi_key_env = \"MY_VOYAGE_KEY\"\n"
                ),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let mut env = HashMap::new();
        env.insert("MY_VOYAGE_KEY".into(), "vk-from-indirection".into());
        let flags = ConfigFlags {
            config_path: Some(PathBuf::from("/cfg/server.toml")),
            reembed: false,
        };
        let c = Config::load(&lookup(&env), &reader, &flags).expect("valid");
        assert_eq!(c.embed_api_key, "vk-from-indirection");
    }

    #[test]
    fn missing_key_for_voyage_names_env_var() {
        let dir = TempDir::new().unwrap();
        let d = dir.path().to_str().unwrap().to_string();
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg/server.toml"),
            file::FileContent {
                text: format!(
                    "vault_dir = \"{d}\"\ndb_path = \"{d}/index.db\"\nlisten = \"1.1.1.1:1\"\nbearer = \"b\"\n\n[embed]\nprovider = \"voyage\"\n"
                ),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = ConfigFlags {
            config_path: Some(PathBuf::from("/cfg/server.toml")),
            reembed: false,
        };
        let err = Config::load(&empty_lookup(), &reader, &flags).expect_err("err");
        assert!(format!("{err}").contains("VOYAGE_API_KEY"));
    }

    #[test]
    fn base_url_read_from_file() {
        let dir = TempDir::new().unwrap();
        let d = dir.path().to_str().unwrap().to_string();
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("/cfg/server.toml"),
            file::FileContent {
                text: format!(
                    "vault_dir = \"{d}\"\ndb_path = \"{d}/index.db\"\nlisten = \"1.1.1.1:1\"\nbearer = \"b\"\n\n[embed]\nprovider = \"fake\"\nbase_url = \"http://proxy.local\"\n"
                ),
                mode: Some(0o600),
            },
        );
        let reader = file_reader_for(files);
        let flags = ConfigFlags {
            config_path: Some(PathBuf::from("/cfg/server.toml")),
            reembed: false,
        };
        let c = Config::load(&empty_lookup(), &reader, &flags).expect("valid");
        assert_eq!(c.embed_base_url.as_deref(), Some("http://proxy.local"));
    }

    #[test]
    fn debug_redacts_bearer_and_api_key() {
        let dir = TempDir::new().unwrap();
        let env = base_env(&dir);
        let c = Config::from_lookup(&lookup(&env)).expect("valid");
        let dbg = format!("{c:?}");
        // "secret" is the bearer value; "key" is the GEMINI_API_KEY value.
        // The field names (bearer, embed_api_key) are allowed to appear; only
        // the values must be redacted.
        assert!(
            !dbg.contains("\"secret\""),
            "bearer value must not appear in Debug output: {dbg}"
        );
        assert!(
            !dbg.contains("\"key\""),
            "api_key value must not appear in Debug output: {dbg}"
        );
        assert!(dbg.contains("[redacted]"), "expected [redacted] in: {dbg}");
    }
}
