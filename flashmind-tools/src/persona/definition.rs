//! The persona type, its errors and name rules.

use std::io;
use std::path::PathBuf;

use flashmind_types::ReasoningLevel;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A named persona: a system prompt with optional model and reasoning defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Persona {
    /// Unique name, compared case-insensitively.
    pub name: String,
    /// Short summary shown in persona catalogs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Compatibility notes from importing the persona, such as frontmatter
    /// settings that are not applied. Hosts show them to the user.
    #[serde(skip)]
    pub notices: Vec<String>,
    /// The system prompt the persona runs with.
    pub system_prompt: String,
    /// Model string (`provider:name`). `None` keeps the current model.
    ///
    /// Kept as written so a persona with a model the host cannot parse still
    /// loads and reports the problem when it is applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Reasoning level. `None` keeps the current level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningLevel>,
}

/// Why a persona name was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NameError {
    /// The name is empty or only whitespace.
    #[error("name cannot be empty")]
    Empty,
    /// The name contains a space, dot or slash.
    #[error("name cannot contain spaces, dots, or slashes")]
    InvalidCharacter,
    /// The name matches one of the host's reserved names.
    #[error("'{0}' is a reserved command name")]
    Reserved(String),
}

/// Why a persona could not be loaded.
#[derive(Debug, Error)]
pub enum PersonaError {
    /// The file does not start with a `---` frontmatter block.
    #[error("expected YAML frontmatter")]
    MissingFrontmatter,
    /// The frontmatter block is never closed.
    #[error("missing closing frontmatter delimiter")]
    UnclosedFrontmatter,
    /// The frontmatter is not valid persona metadata.
    #[error("invalid agent metadata: {0}")]
    Metadata(#[from] serde_yaml::Error),
    /// The name breaks the naming rules.
    #[error(transparent)]
    Name(#[from] NameError),
    /// The name has surrounding whitespace or the description is empty.
    #[error(
        "agent name and description must be nonempty and name must not have surrounding whitespace"
    )]
    Header,
    /// Nothing follows the frontmatter.
    #[error("agent prompt is empty")]
    EmptyPrompt,
    /// The model string is not `provider:name`.
    #[error("invalid model '{model}' for persona: {reason}")]
    Model {
        /// The model string as written.
        model: String,
        /// Why it did not parse.
        reason: String,
    },
    /// The file could not be read.
    #[error("cannot read {}: {source}", path.display())]
    Read {
        /// The file that failed.
        path: PathBuf,
        /// The underlying error.
        source: io::Error,
    },
}

/// Check a candidate persona name.
///
/// Names may not be empty or contain spaces, dots or slashes, and may not
/// match any of `reserved` (compared case-insensitively). Hosts that invoke
/// personas as commands pass their command names as `reserved`.
pub fn validate_name<S: AsRef<str>>(name: &str, reserved: &[S]) -> Result<(), NameError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(NameError::Empty);
    }
    if name.contains(['/', '\\', '.', ' ']) {
        return Err(NameError::InvalidCharacter);
    }
    let lower = name.to_lowercase();
    if reserved
        .iter()
        .any(|word| word.as_ref().to_lowercase() == lower)
    {
        return Err(NameError::Reserved(name.to_owned()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_name_rules() {
        let reserved = ["model", "agents"];
        assert!(validate_name("reviewer", &reserved).is_ok());
        assert_eq!(validate_name("", &reserved), Err(NameError::Empty));
        assert_eq!(
            validate_name("with space", &reserved),
            Err(NameError::InvalidCharacter)
        );
        assert!(validate_name("a/b", &reserved).is_err());
        assert!(validate_name("dot.name", &reserved).is_err());
        assert!(validate_name("model", &reserved).is_err());
        assert_eq!(
            validate_name("Agents", &reserved),
            Err(NameError::Reserved("Agents".into()))
        );
        assert!(validate_name::<&str>("model", &[]).is_ok());
    }
}
