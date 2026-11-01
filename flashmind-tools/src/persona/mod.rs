//! Named personas for sessions and delegated workers.
//!
//! A [`Persona`] is a name, a description, a system prompt and optional model
//! and reasoning defaults. Personas are read from Claude-style Markdown files
//! (`.claude/agents/*.md`, `.agents/*.md`) found in directories the caller
//! passes to [`MarkdownPersonas`]; this module never looks at the home or
//! working directory on its own. [`project_dirs`] and [`scope_dirs`] build the
//! usual search list for a project folder.
//!
//! Later directories override earlier ones, and [`merge`] applies the same
//! rule when a host layers its own personas (for example saved ones) on top.
//! [`PersonaDelegate`] wraps the `delegate` tool so a model can start a worker
//! from a persona, and [`Persona::spawn_builder`] does the same for host code
//! that spawns through [`AgentManager`](flashmind_core::AgentManager) directly.

mod catalog;
mod definition;
mod delegate;
mod markdown;

pub use catalog::{PersonaSource, catalog, delegate_instructions, merge};
pub use definition::{NameError, Persona, PersonaError, validate_name};
pub use delegate::PersonaDelegate;
pub use markdown::{MarkdownPersonas, project_dirs, scope_dirs};
