//! docindex binary entry point.
//!
//! Thin shell around [`docindex::server::run`]: parse config, init tracing,
//! hand off to the server.

use anyhow::{Context, Result};
use clap::Parser;
use docindex::{
    Config,
    config::{self, ConfigFlags},
    server,
};
use tracing_subscriber::{EnvFilter, fmt};

/// docindex — semantic + BM25 search server for a markdown vault.
#[derive(Parser, Debug)]
#[command(name = "docindex", version)]
struct Cli {
    /// Path to a server TOML config file. Overrides $DOCINDEX_CONFIG and
    /// the well-known search locations.
    #[arg(long)]
    config: Option<std::path::PathBuf>,
    /// Wipe chunks/vectors/FTS rows and rebuild the index when the stored
    /// embedding fingerprint (provider/model/dim) no longer matches the
    /// effective config.
    #[arg(long)]
    reembed: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // Multi-thread runtime: one thread for the HTTP executor, another for
    // the indexer + watcher tasks. spawn_blocking offloads SQL regardless.
    let rt = tokio::runtime::Runtime::new().context("build tokio runtime")?;
    rt.block_on(run(cli))
}

async fn run(cli: Cli) -> Result<()> {
    let flags = ConfigFlags {
        config_path: cli.config,
        reembed: cli.reembed,
    };
    let cfg = Config::load(&|k| std::env::var(k).ok(), &config::os_file_reader, &flags)
        .context("load config")?;
    init_tracing(&cfg.log_format);
    server::run(cfg).await
}

fn init_tracing(format: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = fmt().with_env_filter(filter).with_writer(std::io::stderr);
    if format == "text" {
        let _ = builder.try_init();
    } else {
        let _ = builder.json().try_init();
    }
}
