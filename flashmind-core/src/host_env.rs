//! Host environment information for system prompts.
//!
//! Provides a common function to build host environment details that can be
//! included in system prompts for both main agents and subagents.

use std::{env, path::Path};

/// Build the host environment info section for the system prompt.
///
/// Includes OS, architecture, hostname, user, home directory, workspace,
/// and path to TOOLS.md.
pub fn build_host_env_info(workspace: Option<&Path>, base_dir: &Path) -> String {
    let os = env::consts::OS;
    let arch = env::consts::ARCH;
    let hostname = env::var("HOSTNAME")
        .or_else(|_| env::var("HOST"))
        .unwrap_or_else(|_| "unknown".into());
    let user = env::var("USER")
        .or_else(|_| env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".into());
    let tools_md = base_dir.join("TOOLS.md").display().to_string();

    let mut info = format!(
        "## Environment\n\n\
         - OS: {os} ({arch})\n\
         - Hostname: {hostname}\n\
         - User: {user}"
    );

    if let Some(ws) = workspace {
        info.push_str(&format!(
            "\n- **Workspace**: {ws}/\n\n\
                This is YOUR workspace, isolated from other chats. All files you create, download, modify, or repos you clone MUST stay inside this directory. \
                ALWAYS use relative paths. Your shell and tools already run in this directory. \
                NEVER write files outside your workspace *{ws}*. You don't need to use cd, commands are already cwd in {ws}. \
                **Important**: Each chat (Telegram thread, Slack channel, REPL session) has its own isolated workspace — work done here will NOT collide with other conversations.",
            ws = ws.display()
        ));
    }

    info.push_str(&format!(
        "\n- Notes: {tools_md} (writable scratchpad for brief notes about tools, integrations, and learnings)"
    ));

    info
}

/// Build project-specific instructions from CLAUDE.md or AGENTS.md files.
///
/// Tells the agent which project instruction files exist so it can read them.
/// Does not inline file contents — the agent reads them on its first turn.
pub fn build_project_instructions(workspace: &Path) -> String {
    let files: Vec<String> = ["CLAUDE.md", "AGENTS.md"]
        .iter()
        .filter(|name| workspace.join(name).is_file())
        .map(|name| name.to_string())
        .collect();

    if files.is_empty() {
        return String::new();
    }

    let mut instructions =
        String::from("## Project Instructions\n\nThe following project files exist:\n\n");
    for f in &files {
        instructions.push_str(&format!("- `{f}` — read this on your first turn\n"));
    }
    instructions
}
