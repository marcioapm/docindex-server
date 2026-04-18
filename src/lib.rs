//! docindex — Rust library crate.
//!
//! Phase 1 scope: config parsing, directory walking, heading-aware markdown
//! chunking, embedding (Gemini + deterministic fake), and a SQLite store
//! with `sqlite-vec` loaded as a runtime extension.
//!
//! HTTP, the watcher, and the hybrid search ranker are Phase 2.

pub mod chunk;
pub mod config;
pub mod embed;
pub mod store;
pub mod walk;

pub use config::Config;
