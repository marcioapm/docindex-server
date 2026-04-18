//! docindex binary entry point.
//!
//! Phase 1 behavior: parse configuration, initialize tracing, open the
//! store (loading `sqlite-vec` as an auto-extension), log status, exit 0.
//! HTTP and the file watcher land in Phase 2.

use anyhow::{Context, Result};
use docindex::{Config, store::Store};
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> Result<()> {
    // The embedder (Phase 1 surface) is async; build a single-thread tokio
    // runtime so tests and future Phase 2 HTTP handlers share the same
    // async contract without requiring a heavy executor in Phase 1.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;
    rt.block_on(run())
}

async fn run() -> Result<()> {
    let cfg = Config::from_env().context("load config")?;
    init_tracing(&cfg.log_format);

    let store = Store::open(&cfg.db_path).context("open store")?;
    let schema_version = store
        .get_meta("schema_version")
        .context("read schema_version")?
        .unwrap_or_default();

    tracing::info!(
        vault_dir = %cfg.vault_dir.display(),
        db_path = %cfg.db_path.display(),
        listen = %cfg.listen,
        embed_model = %cfg.embed_model,
        embed_dim = cfg.embed_dim,
        schema_version = %schema_version,
        "docindex-server ready (phase 1: no http/watcher yet)"
    );
    Ok(())
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
