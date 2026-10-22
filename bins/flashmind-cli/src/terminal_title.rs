use std::io::{self, IsTerminal, Write};

/// Saves the terminal title for the lifetime of an interactive session.
pub(crate) struct TerminalTitle {
    enabled: bool,
}

impl TerminalTitle {
    pub(crate) fn new() -> Self {
        let mut stdout = io::stdout().lock();
        let enabled = stdout.is_terminal()
            && write!(stdout, "\x1b[22;0t")
                .and_then(|_| stdout.flush())
                .is_ok();
        Self { enabled }
    }

    pub(crate) fn set(&self, task: &str) {
        if self.enabled {
            let title = task_title(task);
            let mut stdout = io::stdout().lock();
            let _ = write!(stdout, "\x1b]0;{title}\x07").and_then(|_| stdout.flush());
        }
    }
}

impl Drop for TerminalTitle {
    fn drop(&mut self) {
        if self.enabled {
            let mut stdout = io::stdout().lock();
            let _ = write!(stdout, "\x1b[23;0t").and_then(|_| stdout.flush());
        }
    }
}

fn task_title(task: &str) -> String {
    let mut title = String::from("flsh");
    // Keep OSC terminators and other control characters out of the title.
    let task: String = task
        .chars()
        .filter(|c| !c.is_control() || c.is_whitespace())
        .collect();
    let task = task.split_whitespace().collect::<Vec<_>>().join(" ");
    if !task.is_empty() {
        title.push_str(": ");
        title.extend(task.chars().take(80));
    }
    title
}

#[cfg(test)]
mod tests {
    use super::task_title;

    #[test]
    fn task_text_cannot_terminate_the_title_sequence() {
        let title = task_title("Fix\n the\tparser\x07\x1b\\\u{009c} bug");
        assert_eq!(title, "flsh: Fix the parser\\ bug");
        assert!(!title.chars().any(char::is_control));
        assert_eq!(task_title(" \n\t\x07"), "flsh");
    }

    #[test]
    fn long_unicode_tasks_are_bounded_without_splitting_characters() {
        let title = task_title(&"界".repeat(100));
        assert_eq!(title, format!("flsh: {}", "界".repeat(80)));
    }
}
