//! `docindex-search` — CLI client for a running docindex server.
//!
//! ```text
//! docindex-search "some query"              # default subcommand = search
//! docindex-search search "q" -n 5 --json
//! docindex-search similar path/to/note.md
//! docindex-search health
//! ```
//!
//! Exit codes: 0 ok, 1 usage/config error, 2 network/server error,
//! 3 authentication failure (401/403 or health reports `authenticated=false`),
//! 4 no results. Errors go to stderr, stdout is reserved for results.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use docindex::cli::{CliConfig, CliFlags, Client, ClientError, OutputFormat, config, output};
use docindex::media::MediaType;
use docindex::search::Hit;

#[derive(Parser)]
#[command(name = "docindex-search", version, about = "Query a docindex server")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Bare query text — shorthand for `search <query>` when no subcommand
    /// is given.
    #[arg(hide = true)]
    bare_query: Vec<String>,

    #[command(flatten)]
    global: GlobalArgs,
}

impl std::fmt::Debug for Cli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cli")
            .field("command", &self.command)
            .field("bare_query", &self.bare_query)
            .field("global", &self.global)
            .finish()
    }
}

#[derive(clap::Args, Default)]
struct GlobalArgs {
    /// Result count. Defaults to the configured limit, else 10.
    #[arg(short = 'n', long = "limit", global = true)]
    limit: Option<usize>,
    /// Emit the server response verbatim as JSON instead of formatted text.
    #[arg(long, global = true)]
    json: bool,
    /// Server base URL, e.g. http://100.64.0.1:7777.
    #[arg(long, global = true)]
    server: Option<String>,
    /// Bearer token for authentication.
    ///
    /// Intended for local development and testing only. argv is visible to
    /// other processes on the host via /proc and ps(1). For production use,
    /// set $DOCINDEX_CLI_TOKEN or add `token`/`token_env` to cli.toml.
    #[arg(long, global = true)]
    token: Option<String>,
    /// Path to a CLI TOML config file.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Return only media results, optionally restricted to comma-separated types.
    #[arg(long, global = true, num_args = 0..=1, default_missing_value = "")]
    media: Option<String>,
    /// Client-side filter: only show hits whose path starts with this
    /// prefix.
    #[arg(long, global = true)]
    path_filter: Option<String>,
}

impl std::fmt::Debug for GlobalArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalArgs")
            .field("limit", &self.limit)
            .field("json", &self.json)
            .field("server", &self.server)
            .field("token", &"[redacted]")
            .field("config", &self.config)
            .field("media", &self.media)
            .field("path_filter", &self.path_filter)
            .finish()
    }
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a hybrid search query.
    Search {
        query: Vec<String>,
        #[command(flatten)]
        args: GlobalArgs,
    },
    /// Find chunks similar to an already-indexed path.
    Similar {
        path: String,
        #[command(flatten)]
        args: GlobalArgs,
    },
    /// Verify server reachability and bearer authentication.
    Health {
        #[command(flatten)]
        args: GlobalArgs,
    },
}

fn merge(base: GlobalArgs, over: GlobalArgs) -> GlobalArgs {
    GlobalArgs {
        limit: over.limit.or(base.limit),
        json: over.json || base.json,
        server: over.server.or(base.server),
        token: over.token.or(base.token),
        config: over.config.or(base.config),
        media: over.media.or(base.media),
        path_filter: over.path_filter.or(base.path_filter),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("docindex-search: build runtime: {e}");
            return ExitCode::from(1);
        }
    };
    rt.block_on(run(cli))
}

async fn run(cli: Cli) -> ExitCode {
    let (verb, arg, global) = match cli.command {
        Some(Command::Search { query, args }) => {
            ("search", query.join(" "), merge(cli.global, args))
        }
        Some(Command::Similar { path, args }) => ("similar", path, merge(cli.global, args)),
        Some(Command::Health { args }) => ("health", String::new(), merge(cli.global, args)),
        None => {
            if cli.bare_query.is_empty() {
                eprintln!("docindex-search: expected a query, or one of: search, similar, health");
                return ExitCode::from(1);
            }
            ("search", cli.bare_query.join(" "), cli.global)
        }
    };

    if global.media.is_some() && verb != "search" {
        eprintln!("docindex-search: --media is valid only with a search query");
        return ExitCode::from(1);
    }

    let media_types = match global.media.as_deref() {
        Some(value) => match parse_media_types(value) {
            Ok(types) => types,
            Err(message) => {
                eprintln!("docindex-search: {message}");
                return ExitCode::from(1);
            }
        },
        None => Vec::new(),
    };

    let flags = CliFlags {
        config_path: global.config.clone(),
        server: global.server.clone(),
        token: global.token.clone(),
        limit: global.limit,
        json: global.json,
    };
    let cfg = match CliConfig::load(
        &config::env_lookup,
        &docindex::config::os_file_reader,
        &flags,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };

    let client = match Client::new(&cfg.server, &cfg.token) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("docindex-search: {e}");
            return ExitCode::from(1);
        }
    };

    match verb {
        "health" => run_health(&client, cfg.format).await,
        "similar" => {
            run_search_like(cfg.format, global.path_filter.as_deref(), || {
                client.similar(&arg, cfg.limit)
            })
            .await
        }
        _ => {
            if arg.trim().is_empty() {
                eprintln!("docindex-search: empty query");
                return ExitCode::from(1);
            }
            run_search_like(cfg.format, global.path_filter.as_deref(), || {
                client.search(&arg, cfg.limit, global.media.is_some(), &media_types)
            })
            .await
        }
    }
}

fn parse_media_types(value: &str) -> Result<Vec<String>, String> {
    if value.is_empty() {
        return Ok(Vec::new());
    }

    let mut media_types = Vec::new();
    for media_type in value.split(',') {
        if MediaType::from_exclude_value(media_type).is_none() {
            // A value containing whitespace is almost always the query text
            // captured by `--media`, not a mistyped media type, so the hint
            // names the orderings that keep the query positional.
            let hint = if media_type.contains(char::is_whitespace) {
                "; to search all media types put the query first \
                 (`search \"<query>\" --media`) or separate it \
                 (`search --media -- \"<query>\"`)"
            } else {
                ""
            };
            return Err(format!(
                "--media: unknown value {media_type:?}; valid: {}{hint}",
                MediaType::EXCLUDE_VALUES.join(", ")
            ));
        }
        if !media_types.iter().any(|known| known == media_type) {
            media_types.push(media_type.to_string());
        }
    }
    Ok(media_types)
}

async fn run_health(client: &Client, format: OutputFormat) -> ExitCode {
    match client.health().await {
        Ok(h) if !h.authenticated => {
            eprintln!(
                "docindex-search: server is reachable but the bearer token is missing or invalid"
            );
            ExitCode::from(3)
        }
        Ok(h) => {
            if format == OutputFormat::Json {
                match serde_json::to_string(&h) {
                    Ok(s) => println!("{s}"),
                    Err(e) => {
                        eprintln!("docindex-search: encode health response: {e}");
                        return ExitCode::from(2);
                    }
                }
            } else {
                println!(
                    "ok={} authenticated={} indexed_chunks={} last_reindex_ms={} embedding_model={} dim={}",
                    h.ok,
                    h.authenticated,
                    h.indexed_chunks.unwrap_or_default(),
                    h.last_reindex_ms.unwrap_or_default(),
                    h.embedding_model.as_deref().unwrap_or_default(),
                    h.dim.unwrap_or_default(),
                );
            }
            ExitCode::from(0)
        }
        Err(e) => exit_for_client_error(&e),
    }
}

async fn run_search_like<F, Fut>(
    format: OutputFormat,
    path_filter: Option<&str>,
    call: F,
) -> ExitCode
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<docindex::cli::client::SearchResponse, ClientError>>,
{
    let resp = match call().await {
        Ok(r) => r,
        Err(e) => return exit_for_client_error(&e),
    };
    let hits: Vec<Hit> = match path_filter {
        Some(prefix) => resp
            .hits
            .into_iter()
            .filter(|h| h.path.starts_with(prefix))
            .collect(),
        None => resp.hits,
    };
    let is_empty = hits.is_empty();

    if format == OutputFormat::Json {
        match serde_json::to_string(&docindex::cli::client::SearchResponse { hits }) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("docindex-search: encode response: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        let width = output::terminal_width();
        for (i, h) in hits.iter().enumerate() {
            println!("{}", output::format_hit(i + 1, h, width));
        }
    }

    if is_empty {
        ExitCode::from(4)
    } else {
        ExitCode::from(0)
    }
}

fn exit_for_client_error(e: &ClientError) -> ExitCode {
    eprintln!("docindex-search: {e}");
    match e {
        ClientError::Auth(_) => ExitCode::from(3),
        ClientError::Network(_) | ClientError::Server { .. } => ExitCode::from(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_args_debug_redacts_token() {
        let args = GlobalArgs {
            token: Some("super-secret-token".into()),
            server: Some("http://example.com".into()),
            ..Default::default()
        };
        let dbg = format!("{args:?}");
        assert!(
            !dbg.contains("super-secret-token"),
            "token value must not appear in Debug output: {dbg}"
        );
        assert!(dbg.contains("[redacted]"), "expected [redacted] in: {dbg}");
        // Non-secret fields are still visible.
        assert!(dbg.contains("http://example.com"), "{dbg}");
    }

    #[test]
    fn cli_debug_redacts_nested_token() {
        let cli = Cli {
            command: None,
            bare_query: vec!["q".into()],
            global: GlobalArgs {
                token: Some("hidden".into()),
                ..Default::default()
            },
        };
        let dbg = format!("{cli:?}");
        assert!(!dbg.contains("hidden"), "token must not appear: {dbg}");
        assert!(dbg.contains("[redacted]"), "expected [redacted] in: {dbg}");
    }

    #[test]
    fn media_flag_parses_for_search() {
        let cli = Cli::try_parse_from(["docindex-search", "search", "sunset", "--media"]).unwrap();

        match cli.command {
            Some(Command::Search { args, .. }) => assert_eq!(args.media.as_deref(), Some("")),
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn media_flag_parses_for_bare_query() {
        let cli = Cli::try_parse_from(["docindex-search", "sunset", "--media"]).unwrap();

        assert_eq!(cli.global.media.as_deref(), Some(""));
        assert_eq!(cli.bare_query, ["sunset"]);
    }

    #[test]
    fn media_flag_parses_types() {
        let cli = Cli::try_parse_from([
            "docindex-search",
            "search",
            "sunset",
            "--media",
            "pdf,image",
        ])
        .unwrap();

        match cli.command {
            Some(Command::Search { args, .. }) => {
                assert_eq!(args.media.as_deref(), Some("pdf,image"));
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn parse_media_types_rejects_unknown_type() {
        let error = parse_media_types("jpeg").unwrap_err();

        assert_eq!(
            error,
            "--media: unknown value \"jpeg\"; valid: image, pdf, audio, video"
        );
    }

    #[tokio::test]
    async fn media_flag_is_rejected_for_similar_before_config_loading() {
        let cli =
            Cli::try_parse_from(["docindex-search", "similar", "note.md", "--media"]).unwrap();

        assert_eq!(run(cli).await, ExitCode::from(1));
    }
}
