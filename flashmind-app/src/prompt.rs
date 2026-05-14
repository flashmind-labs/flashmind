//! System prompt builder.

use flashmind_skills::SkillProvider;

use crate::config::AppConfig;

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

impl AppConfig {
    /// Assemble the full system prompt from SOUL.md, project instructions,
    /// git context, skills, and memory instructions.
    pub fn system_prompt(&self, skills: &dyn SkillProvider) -> String {
        let mut prompt = if let Some(ref p) = self.agent.system_prompt {
            p.clone()
        } else {
            let soul = Self::soul_path();
            if soul.exists()
                && let Ok(text) = std::fs::read_to_string(&soul)
                && !text.trim().is_empty()
            {
                text
            } else {
                crate::config::DEFAULT_SOUL.to_string()
            }
        };

        // Project instructions from CLAUDE.md / AGENTS.md
        let cwd = std::env::current_dir().unwrap_or_default();
        let project = build_project_instructions(&cwd);
        if !project.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&project);
        }

        // Git context
        let git = build_git_context(&cwd);
        if !git.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&git);
        }

        // Skills
        let skills_section = build_skills_section(skills);
        if !skills_section.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&skills_section);
        }

        // Memory instructions (when embedder is available)
        if self.build_embedder().is_some() {
            prompt.push_str("\n\n");
            prompt.push_str(flashmind_prompts::MEMORY_INSTRUCTIONS);
        }

        prompt
    }
}

// ---------------------------------------------------------------------------
// Project instructions
// ---------------------------------------------------------------------------

/// Scan `workspace` for `CLAUDE.md` / `AGENTS.md` and build instructions.
pub fn build_project_instructions(workspace: &std::path::Path) -> String {
    let names = ["CLAUDE.md", "AGENTS.md"];
    let found: Vec<&str> = names
        .iter()
        .filter(|n| workspace.join(n).is_file())
        .copied()
        .collect();

    if found.is_empty() {
        return String::new();
    }

    let mut out = String::from("## Project Instructions\n\nThe following project files exist:\n\n");
    for f in &found {
        out.push_str(&format!("- `{f}` — read this on your first turn\n"));
    }
    out.push_str(
        "\nWhen the user asks you to change, add, fix, or configure anything — \
         including tools, limits, features, or behavior — they mean modify the \
         source code. Do not confuse yourself with the software being built.\n",
    );
    out
}

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------

/// Build a skills section listing available skills by name and description.
pub fn build_skills_section(registry: &dyn SkillProvider) -> String {
    let skills = registry.list();
    if skills.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "## Skills\n\n\
         Use `skill_load` to read a skill's full instructions before using it.\n\n",
    );
    for skill in &skills {
        let name = &skill.meta.name;
        match skill.meta.description.as_deref() {
            Some(desc) => out.push_str(&format!("- **{name}** — {desc}\n")),
            None => out.push_str(&format!("- **{name}**\n")),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Git context
// ---------------------------------------------------------------------------

/// Collect basic git repo info (repo name, branch, uncommitted changes).
pub fn build_git_context(cwd: &std::path::Path) -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(cwd)
        .output();
    if !matches!(output, Ok(ref o) if o.status.success()) {
        return String::new();
    }

    let mut parts = Vec::new();

    if let Ok(o) = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(cwd)
        .output()
    {
        let root = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if let Some(name) = std::path::Path::new(&root).file_name() {
            parts.push(format!("Repository: {}", name.to_string_lossy()));
        }
    }

    if let Ok(o) = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(cwd)
        .output()
    {
        let branch = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !branch.is_empty() {
            parts.push(format!("Branch: {branch}"));
        }
    }

    if let Ok(o) = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(cwd)
        .output()
    {
        let status = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !status.is_empty() {
            let lines: Vec<&str> = status.lines().collect();
            parts.push(format!("{} uncommitted change(s)", lines.len()));
        }
    }

    if parts.is_empty() {
        return String::new();
    }

    format!("## Git Context\n\n{}", parts.join("\n"))
}
