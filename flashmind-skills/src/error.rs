//! Skill-specific error types.

/// Errors that can occur during skill operations.
#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("skill not found: {0}")]
    NotFound(String),
    #[error("invalid SKILL.md: {0}")]
    InvalidFormat(String),
    #[error("skill execution failed: {0}")]
    ExecutionFailed(String),
    #[error("skill execution timed out after {0}s")]
    Timeout(u64),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
