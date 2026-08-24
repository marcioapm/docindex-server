//! Shared library surface for the `docindex-search` CLI binary.

pub mod client;
pub mod config;
pub mod output;

pub use client::{Client, ClientError};
pub use config::{CliConfig, CliFlags, OutputFormat};
