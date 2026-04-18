//! docindex binary entry point.
//!
//! Thin shell around [`docindex::server::run`]: parse config, init tracing,
//! hand off to the server.

use anyhow::{Context, Result};
use docindex::{Config, server};
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> Result<()> {
    // Multi-thread runtime: one thread for the HTTP executor, another for
    // the indexer + watcher tasks. spawn_blocking offloads SQL regardless.
    let rt = tokio::runtime::Runtime::new().context("build tokio runtime")?;
    rt.block_on(run())
}

async fn run() -> Result<()> {
    let cfg = Config::from_env().context("load config")?;
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
