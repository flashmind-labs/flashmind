//! Persona sources, precedence and the catalog shown to models.

use std::collections::BTreeMap;

use super::definition::Persona;

/// Somewhere personas come from.
///
/// [`MarkdownPersonas`](super::MarkdownPersonas) implements it; a host that
/// keeps its own personas (saved files, a database) implements it too,
/// usually by combining layers with [`merge`].
pub trait PersonaSource: Send + Sync {
    /// Every persona, sorted by name, with precedence already applied.
    fn personas(&self) -> Vec<Persona>;

    /// Look up a persona by name, ignoring case.
    fn get(&self, name: &str) -> Option<Persona> {
        let want = name.to_lowercase();
        self.personas()
            .into_iter()
            .find(|persona| persona.name.to_lowercase() == want)
    }
}

/// Combine personas so a later one replaces an earlier one with the same name
/// (case-insensitive). The result is sorted by name.
pub fn merge(personas: impl IntoIterator<Item = Persona>) -> Vec<Persona> {
    let mut by_name = BTreeMap::new();
    for persona in personas {
        by_name.insert(persona.name.to_lowercase(), persona);
    }
    by_name.into_values().collect()
}

/// One `- name: description` line per persona. Prompts are left out so the
/// parent context stays small.
pub fn catalog(personas: &[Persona]) -> String {
    personas
        .iter()
        .map(|persona| {
            let description = persona.description.as_deref().unwrap_or("Custom agent");
            format!(
                "- {}: {}",
                persona.name,
                description.split_whitespace().collect::<Vec<_>>().join(" ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// System prompt section telling the model how to delegate to `personas`.
/// Empty when there are none.
pub fn delegate_instructions(personas: &[Persona]) -> String {
    let catalog = catalog(personas);
    if catalog.is_empty() {
        return String::new();
    }
    format!(
        "\n\n## Agent personas\n\nUse a listed persona when its description matches an independent task. Call `delegate` with `persona` set to its name and `task` containing the full assignment. The worker loads that persona's instructions automatically. `name` only labels the worker; do not combine `persona` with `system_prompt`.\n{catalog}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn persona(name: &str, prompt: &str) -> Persona {
        Persona {
            name: name.into(),
            description: None,
            notices: Vec::new(),
            system_prompt: prompt.into(),
            model: None,
            reasoning: None,
        }
    }

    #[test]
    fn later_personas_override_and_results_sort_by_name() {
        let merged = merge([
            persona("Reviewer", "first"),
            persona("planner", "plan"),
            persona("reviewer", "second"),
        ]);
        let names: Vec<&str> = merged.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["planner", "reviewer"]);
        assert_eq!(merged[1].system_prompt, "second");
        assert_eq!(
            catalog(&merged),
            "- planner: Custom agent\n- reviewer: Custom agent"
        );
        assert!(delegate_instructions(&[]).is_empty());
    }
}
