//! docindex — Rust library crate.
//!
//! Phase 2 scope (on top of Phase 1): HTTP API (`axum`), filesystem watcher
//! (`notify`), hybrid BM25 + semantic search ranker with Reciprocal Rank
//! Fusion, and a single indexer task consuming dirty paths from both the
//! startup walker and the watcher.

pub mod api;
pub mod chunk;
pub mod config;
pub mod embed;
pub mod indexer;
pub mod search;
pub mod server;
pub mod store;
pub mod walk;
pub mod watch;

pub use config::Config;
