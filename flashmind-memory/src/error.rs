use thiserror::Error;

/// Errors produced by the flashmind_memory crate.
#[derive(Debug, Error)]
pub enum FlashmemError {
    /// Vector memory operation failed.
    #[error("Memory error: {0}")]
    Memory(String),

    /// Invalid or missing configuration.
    #[error("Configuration error: {0}")]
    Config(String),

    /// HTTP request failed.
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Filesystem I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// SQLite database error.
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Async SQLite database error.
    #[error("Async database error: {0}")]
    DatabaseAsync(#[from] tokio_rusqlite::Error),
}

/// Convenience alias for `Result<T, FlashmemError>`.
pub type Result<T> = std::result::Result<T, FlashmemError>;
