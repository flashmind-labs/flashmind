//! Protected path validation and destructive command detection.
//!
//! Guards critical files (config, SOUL.md, service units, SSH private keys) and
//! system directories from writes, recursive deletes, and permission changes.
//! Also detects self-termination attempts and disk-wipe commands.

use regex::Regex;
use std::path::{Path, PathBuf};

/// Directories that should never be recursively deleted, chmod'd, or chown'd.
const FORBIDDEN_DIRS: &[&str] = &[
    "/",
    "/home",
    "/Users",
    "/root",
    "/usr",
    "/etc",
    "/var",
    "/sys",
    "/proc",
    "/boot",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/opt",
    "/dev",
    "/Applications",
    "/System",
    "/Library",
];

pub struct ProtectedPaths {
    pub paths: Vec<PathBuf>,
    pub own_pid: u32,
    pub executable: PathBuf,
    /// Directory protected except for files matching allowed extensions.
    ssh_dir: Option<PathBuf>,
    /// Glob patterns for write-protected paths (e.g. `**/canvases/**/.canvas.toml`).
    write_protected_globs: Vec<glob::Pattern>,
}

impl ProtectedPaths {
    pub fn new(base_dir: &Path) -> Self {
        let executable = std::env::current_exe().unwrap_or_default();
        let home = dirs::home_dir();

        let mut paths = vec![
            executable.clone(),
            base_dir.join("config.toml"),
            base_dir.join("SOUL.md"),
            // Linux service file
            PathBuf::from("/etc/systemd/system/flash.service"),
        ];

        // macOS user-level LaunchAgent
        if let Some(ref home) = home {
            paths.push(home.join("Library/LaunchAgents/com.flash.plist"));
        }

        let base_canonical = base_dir
            .canonicalize()
            .unwrap_or_else(|_| base_dir.to_path_buf());
        let canvases_glob = format!(
            "{}/**/.canvas.toml",
            base_canonical.join("canvases").display()
        );

        Self {
            paths,
            own_pid: std::process::id(),
            executable,
            ssh_dir: home.map(|h| h.join(".ssh")),
            write_protected_globs: vec![canvases_glob]
                .into_iter()
                .filter_map(|g| glob::Pattern::new(&g).ok())
                .collect(),
        }
    }

    /// Check if a path is protected from reads.
    /// Blocks reads on sensitive files: config.toml (API keys), SSH private keys,
    /// and files excluded by `.gitignore`.
    pub fn is_read_protected(&self, path: &Path) -> bool {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

        if is_gitignored(&canonical) {
            return true;
        }

        // SSH private keys are read-protected (not .pub)
        if let Some(ref ssh_dir) = self.ssh_dir {
            let ssh_canonical = ssh_dir.canonicalize().unwrap_or_else(|_| ssh_dir.clone());
            if canonical.starts_with(&ssh_canonical) {
                let is_pub = canonical.extension().is_some_and(|ext| ext == "pub");
                return !is_pub;
            }
        }

        // config.toml contains API keys — read-protected
        if canonical.file_name().is_some_and(|f| f == "config.toml") {
            return self.paths.iter().any(|p| {
                let protected = p.canonicalize().unwrap_or_else(|_| p.clone());
                canonical == protected
            });
        }

        false
    }

    /// Check if a path is protected from writes/modifications.
    /// More restrictive than read protection: includes config, SOUL.md, service files,
    /// SSH keys, own executable, canvas metadata, and files excluded by `.gitignore`.
    pub fn is_write_protected(&self, path: &Path) -> bool {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

        if is_gitignored(&canonical) {
            return true;
        }

        // Glob-based write protection (e.g. canvases/**/.canvas.toml)
        let glob_opts = glob::MatchOptions {
            require_literal_separator: false,
            require_literal_leading_dot: false,
            ..Default::default()
        };
        if self
            .write_protected_globs
            .iter()
            .any(|g| g.matches_path_with(&canonical, glob_opts))
        {
            return true;
        }

        // SSH dir is write-protected (except .pub)
        if let Some(ref ssh_dir) = self.ssh_dir {
            let ssh_canonical = ssh_dir.canonicalize().unwrap_or_else(|_| ssh_dir.clone());
            if canonical.starts_with(&ssh_canonical) {
                let is_pub = canonical.extension().is_some_and(|ext| ext == "pub");
                return !is_pub;
            }
        }

        // All protected paths (config.toml, SOUL.md, executable, service files)
        self.paths.iter().any(|p| {
            let protected = p.canonicalize().unwrap_or_else(|_| p.clone());
            canonical == protected || canonical.starts_with(&protected)
        })
    }

    pub fn can_kill_pid(&self, pid: u32) -> bool {
        pid != self.own_pid
    }

    /// Check if a bash command would write to any protected path.
    /// Catches common patterns: redirects (`>`/`>>`), `tee`, `cp`, `mv`, `sed -i`.
    pub fn command_writes_protected(&self, command: &str) -> Option<String> {
        // Collect protected path strings for substring matching.
        // Include both original and canonicalized forms to handle symlinks (e.g. /var → /private/var).
        let mut protected_strs: Vec<String> = Vec::new();
        for p in &self.paths {
            let original = p.to_string_lossy().into_owned();
            protected_strs.push(original);
            if let Ok(canonical) = p.canonicalize() {
                let canonical_str = canonical.to_string_lossy().into_owned();
                if !protected_strs.contains(&canonical_str) {
                    protected_strs.push(canonical_str);
                }
            }
        }

        for protected in &protected_strs {
            if !command.contains(protected.as_str()) {
                continue;
            }

            // Protected path appears in the command — check if it's a write target.
            // Be conservative: block if the command contains any write indicator.
            let has_write_indicator = command.contains('>')
                || command.contains("tee ")
                || command.contains("cp ")
                || command.contains("mv ")
                || command.contains("sed -i")
                || command.contains("sed -e")
                || command.contains("install ")
                || command.contains("cat <<")
                || command.contains("dd ")
                || command.contains("truncate ")
                || command.contains("python")
                || command.contains("ruby")
                || command.contains("perl")
                || command.contains("node ");

            if has_write_indicator {
                return Some(format!(
                    "Blocked: command would write to protected file: {protected}"
                ));
            }
        }

        None
    }

    pub fn is_self_termination_attempt(&self, command: &str) -> bool {
        let patterns = [
            format!(r"kill\s+{}", self.own_pid),
            format!(r"kill\s+-\d+\s+{}", self.own_pid),
            r"launchctl\s+unload.*flash".to_string(),
            r"launchctl\s+stop.*flash".to_string(),
            r"systemctl\s+stop\s+flash".to_string(),
            r"systemctl\s+restart\s+flash".to_string(),
            r"pkill.*flash".to_string(),
        ];

        patterns
            .iter()
            .any(|p| Regex::new(p).map(|r| r.is_match(command)).unwrap_or(false))
    }

    /// Check if a command would be destructive (recursive delete/overwrite of system dirs).
    ///
    /// Returns `Some(reason)` if blocked, `None` if safe.
    /// `working_dir` is used to resolve relative paths in the command.
    /// Tracks `cd` across compound commands (e.g. `cd /home/user && rm -rf ../../`).
    pub fn is_destructive_command(
        command: &str,
        working_dir: Option<&str>,
    ) -> Option<&'static str> {
        let parts = split_shell_commands(command);
        let mut effective_wd = working_dir.map(|s| s.to_string());

        for part in &parts {
            let tokens: Vec<&str> = part.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            // Track `cd` commands to update effective working directory
            if let Some(cd_path) = extract_cd_path(&tokens) {
                let resolved = resolve_path(&cd_path, effective_wd.as_deref());
                effective_wd = Some(resolved.to_string_lossy().to_string());
                continue;
            }

            let wd_ref = effective_wd.as_deref();

            // Check for disk-wiping commands
            if check_disk_wipe(&tokens) {
                return Some("Blocked: disk-level destructive command (mkfs/dd to device)");
            }

            // Check `rm` with recursive flags
            if check_destructive_rm(&tokens, wd_ref) {
                return Some("Blocked: recursive rm targeting a system or root-level directory");
            }

            // Check `chmod`/`chown` with recursive flags on system dirs
            if check_destructive_perm_change(&tokens, wd_ref) {
                return Some(
                    "Blocked: recursive permission change targeting a system or root-level directory",
                );
            }
        }

        None
    }
}

/// Split a shell command string on `&&`, `||`, `;`, and `|` to get individual commands.
/// Tracks `cd` to update an effective working directory for subsequent commands.
fn split_shell_commands(command: &str) -> Vec<String> {
    // Split on &&, ||, ;, | (but not inside quotes — simplified: split on delimiters)
    let re = Regex::new(r"\s*(?:&&|\|\||[;|])\s*").unwrap();
    re.split(command).map(|s| s.trim().to_string()).collect()
}

/// Resolve a path string to an absolute path, expanding `~`, `$HOME`, and resolving `..`.
fn resolve_path(path_str: &str, working_dir: Option<&str>) -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));

    // Expand ~ and $HOME
    let expanded = path_str
        .replace("$HOME", home.to_str().unwrap_or("/"))
        .replace('~', home.to_str().unwrap_or("/"));

    let path = Path::new(&expanded);

    // Make relative paths absolute using working_dir
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else if let Some(wd) = working_dir {
        let wd = Path::new(wd);
        wd.join(path)
    } else if let Ok(cwd) = std::env::current_dir() {
        cwd.join(path)
    } else {
        path.to_path_buf()
    };

    // Normalize the path (resolve `.` and `..` without touching the filesystem)
    normalize_path(&absolute)
}

/// Normalize a path by resolving `.` and `..` components lexically.
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                // Pop the last component if possible
                if !components.is_empty() {
                    components.pop();
                }
            }
            std::path::Component::CurDir => {}
            other => components.push(other),
        }
    }
    components.iter().collect()
}

/// Check if tokens represent a disk-wiping command (mkfs, dd to device).
fn check_disk_wipe(tokens: &[&str]) -> bool {
    let cmd = tokens[0];

    // mkfs, mkfs.ext4, etc.
    if cmd.starts_with("mkfs") {
        return true;
    }

    // dd with of=/dev/...
    if cmd == "dd" {
        return tokens.iter().any(|t| {
            t.starts_with("of=/dev/") && !t.starts_with("of=/dev/null")
                || t.starts_with("of=/dev/sd")
                || t.starts_with("of=/dev/nvme")
                || t.starts_with("of=/dev/vd")
                || t.starts_with("of=/dev/hd")
        });
    }

    false
}

/// Extract the effective working directory from a `cd` prefix in a command part.
/// E.g., "cd /home/user" returns Some("/home/user").
fn extract_cd_path(tokens: &[&str]) -> Option<String> {
    if tokens.len() >= 2 && tokens[0] == "cd" {
        Some(tokens[1].to_string())
    } else {
        None
    }
}

/// Check if tokens represent a destructive `rm` command targeting a forbidden directory.
fn check_destructive_rm(tokens: &[&str], working_dir: Option<&str>) -> bool {
    let cmd = tokens[0];
    if cmd != "rm" {
        return false;
    }

    // Check for recursive flags (-r, -R, -rf, -fr, etc.)
    let has_recursive = tokens.iter().any(|t| {
        t.starts_with('-') && !t.starts_with("--") && (t.contains('r') || t.contains('R'))
    }) || tokens.contains(&"--recursive");

    if !has_recursive {
        return false;
    }

    // Check each non-flag argument as a path
    for token in tokens.iter().skip(1) {
        if token.starts_with('-') {
            continue;
        }

        // Check glob patterns like /* or ~/*
        let path_str = token.trim_end_matches('*').trim_end_matches('/');
        if path_str.is_empty() {
            // `rm -rf /*` — the path_str after stripping is empty, meaning root
            return true;
        }

        let resolved = resolve_path(path_str, working_dir);
        if is_forbidden_path(&resolved) {
            return true;
        }
    }

    false
}

/// Check if tokens represent a destructive chmod/chown on a forbidden directory.
fn check_destructive_perm_change(tokens: &[&str], working_dir: Option<&str>) -> bool {
    let cmd = tokens[0];
    if cmd != "chmod" && cmd != "chown" {
        return false;
    }

    // Check for recursive flag
    let has_recursive =
        tokens.contains(&"-R") || tokens.contains(&"--recursive") || tokens.contains(&"-r");

    if !has_recursive {
        return false;
    }

    // Check paths (last arguments, skip flags and the mode/owner arg)
    for token in tokens.iter().skip(1) {
        if token.starts_with('-') {
            continue;
        }
        // Skip the mode (e.g., "777") or owner (e.g., "root:root") argument
        // by checking if it looks like a path
        if !token.contains('/') && !token.starts_with('~') && !token.starts_with('.') {
            continue;
        }

        let path_str = token.trim_end_matches('/');
        if path_str.is_empty() {
            // Bare "/" after trimming → root
            return true;
        }
        let resolved = resolve_path(path_str, working_dir);
        if is_forbidden_path(&resolved) {
            return true;
        }
    }

    false
}

/// Check if a path is excluded by a `.gitignore` file in its directory ancestry.
/// Walks up from the file's parent to the repo root (`.git` directory).
fn is_gitignored(path: &Path) -> bool {
    let mut dir = path.parent();
    while let Some(current_dir) = dir {
        let gitignore_path = current_dir.join(".gitignore");
        if gitignore_path.is_file()
            && let Ok(contents) = std::fs::read_to_string(&gitignore_path)
            && let Ok(rel_path) = path.strip_prefix(current_dir)
            && matches_gitignore_patterns(&contents, rel_path)
        {
            return true;
        }

        if current_dir.join(".git").exists() {
            break;
        }
        dir = current_dir.parent();
    }

    false
}

/// Check if a relative path matches any pattern in gitignore contents.
fn matches_gitignore_patterns(contents: &str, rel_path: &Path) -> bool {
    let filename = rel_path.file_name().unwrap_or_default().to_string_lossy();
    let rel_str = rel_path.to_string_lossy();

    let glob_opts = glob::MatchOptions {
        require_literal_separator: false,
        require_literal_leading_dot: false,
        ..Default::default()
    };

    let mut ignored = false;

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let (negated, pattern) = if let Some(rest) = line.strip_prefix('!') {
            (true, rest)
        } else {
            (false, line)
        };

        let pattern = pattern.trim_end_matches('/');
        if pattern.is_empty() {
            continue;
        }

        let matches = if pattern.contains('/') {
            let pattern = pattern.strip_prefix('/').unwrap_or(pattern);
            glob::Pattern::new(pattern)
                .map(|p| p.matches_with(&rel_str, glob_opts))
                .unwrap_or(false)
        } else {
            glob::Pattern::new(pattern)
                .map(|p| p.matches_with(&filename, glob_opts))
                .unwrap_or(false)
        };

        if matches {
            ignored = !negated;
        }
    }

    ignored
}

/// Check if a resolved path matches or is a parent of a forbidden directory.
fn is_forbidden_path(path: &Path) -> bool {
    let path_str = path.to_string_lossy();
    let normalized = path_str.trim_end_matches('/');

    // Check exact match with forbidden dirs
    for forbidden in FORBIDDEN_DIRS {
        let forbidden_trimmed = forbidden.trim_end_matches('/');
        if normalized == forbidden_trimmed {
            return true;
        }
    }

    // Also block the home directory itself (but not subdirectories)
    if let Some(home) = dirs::home_dir() {
        let home_str = home.to_string_lossy();
        let home_trimmed = home_str.trim_end_matches('/');
        if normalized == home_trimmed {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_protected_paths() {
        let dir = tempdir().unwrap();
        let protected = ProtectedPaths::new(dir.path());

        // config.toml is both read-protected (API keys) and write-protected
        assert!(protected.is_read_protected(&dir.path().join("config.toml")));
        assert!(protected.is_write_protected(&dir.path().join("config.toml")));

        // SOUL.md is write-protected but readable
        assert!(!protected.is_read_protected(&dir.path().join("SOUL.md")));
        assert!(protected.is_write_protected(&dir.path().join("SOUL.md")));

        // TOOLS.md and MEMORY.md are neither
        assert!(!protected.is_read_protected(&dir.path().join("TOOLS.md")));
        assert!(!protected.is_write_protected(&dir.path().join("TOOLS.md")));
    }

    #[test]
    fn test_command_writes_protected() {
        let dir = tempdir().unwrap();
        // Create the config.toml so canonicalize works
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "").unwrap();
        let soul_path = dir.path().join("SOUL.md");
        std::fs::write(&soul_path, "").unwrap();

        let protected = ProtectedPaths::new(dir.path());
        let config_str = config_path.to_string_lossy().to_string();
        let soul_str = soul_path.to_string_lossy().to_string();

        // Redirects
        assert!(
            protected
                .command_writes_protected(&format!("echo 'x' > {config_str}"))
                .is_some()
        );
        assert!(
            protected
                .command_writes_protected(&format!("echo 'x' >> {config_str}"))
                .is_some()
        );

        // tee
        assert!(
            protected
                .command_writes_protected(&format!("echo 'x' | tee {config_str}"))
                .is_some()
        );

        // cp / mv
        assert!(
            protected
                .command_writes_protected(&format!("cp /tmp/evil {config_str}"))
                .is_some()
        );
        assert!(
            protected
                .command_writes_protected(&format!("mv /tmp/evil {soul_str}"))
                .is_some()
        );

        // sed -i
        assert!(
            protected
                .command_writes_protected(&format!("sed -i 's/a/b/' {config_str}"))
                .is_some()
        );

        // Reading is fine (no write indicator)
        assert!(
            protected
                .command_writes_protected(&format!("cat {config_str}"))
                .is_none()
        );
        assert!(
            protected
                .command_writes_protected(&format!("grep foo {config_str}"))
                .is_none()
        );

        // Unrelated file is fine
        assert!(
            protected
                .command_writes_protected("echo 'x' > /tmp/whatever.txt")
                .is_none()
        );
    }

    #[test]
    fn test_ssh_dir_protected() {
        let dir = tempdir().unwrap();
        let protected = ProtectedPaths::new(dir.path());

        if let Some(ref ssh_dir) = protected.ssh_dir {
            // Private keys are read-protected and write-protected
            assert!(protected.is_read_protected(&ssh_dir.join("id_rsa")));
            assert!(protected.is_read_protected(&ssh_dir.join("id_ed25519")));
            assert!(protected.is_read_protected(&ssh_dir.join("config")));
            assert!(protected.is_read_protected(&ssh_dir.join("known_hosts")));
            assert!(protected.is_write_protected(&ssh_dir.join("id_rsa")));
            assert!(protected.is_write_protected(&ssh_dir.join("id_ed25519")));

            // .pub files are allowed for both read and write
            assert!(!protected.is_read_protected(&ssh_dir.join("id_rsa.pub")));
            assert!(!protected.is_read_protected(&ssh_dir.join("id_ed25519.pub")));
            assert!(!protected.is_write_protected(&ssh_dir.join("id_rsa.pub")));
            assert!(!protected.is_write_protected(&ssh_dir.join("id_ed25519.pub")));
        }
    }

    #[test]
    fn test_can_kill_pid() {
        let dir = tempdir().unwrap();
        let protected = ProtectedPaths::new(dir.path());

        assert!(!protected.can_kill_pid(protected.own_pid));
        assert!(protected.can_kill_pid(99999));
    }

    #[test]
    fn test_self_termination_detection() {
        let dir = tempdir().unwrap();
        let protected = ProtectedPaths::new(dir.path());
        let pid = protected.own_pid;

        assert!(protected.is_self_termination_attempt(&format!("kill {}", pid)));
        assert!(protected.is_self_termination_attempt(&format!("kill -9 {}", pid)));
        assert!(protected.is_self_termination_attempt("pkill flash"));
        assert!(protected.is_self_termination_attempt("systemctl stop flash"));
        assert!(!protected.is_self_termination_attempt("echo hello"));
        assert!(!protected.is_self_termination_attempt("kill 99999"));
    }

    // --- Destructive command tests ---

    fn blocked(cmd: &str) -> bool {
        ProtectedPaths::is_destructive_command(cmd, None).is_some()
    }

    fn blocked_wd(cmd: &str, wd: &str) -> bool {
        ProtectedPaths::is_destructive_command(cmd, Some(wd)).is_some()
    }

    #[test]
    fn test_rm_rf_root() {
        assert!(blocked("rm -rf /"));
        assert!(blocked("rm -rf /*"));
        assert!(blocked("rm -Rf /"));
        assert!(blocked("rm -fr /"));
    }

    #[test]
    fn test_rm_rf_system_dirs() {
        assert!(blocked("rm -rf /home"));
        assert!(blocked("rm -rf /Users"));
        assert!(blocked("rm -rf /etc"));
        assert!(blocked("rm -rf /usr"));
        assert!(blocked("rm -rf /var"));
        assert!(blocked("rm -rf /boot"));
        assert!(blocked("rm -rf /opt"));
    }

    #[test]
    fn test_rm_rf_home_dir() {
        assert!(blocked("rm -rf ~"));
        assert!(blocked("rm -rf $HOME"));
    }

    #[test]
    fn test_rm_rf_safe_paths() {
        assert!(!blocked("rm -rf ~/Projects/idk"));
        assert!(!blocked("rm -rf /tmp/test"));
        assert!(!blocked("rm -rf ./node_modules"));
        assert!(!blocked("rm file.txt")); // no recursive flag
    }

    #[test]
    fn test_rm_non_recursive_is_safe() {
        assert!(!blocked("rm /etc/something"));
        assert!(!blocked("rm -f /usr/something"));
    }

    #[test]
    fn test_cd_then_rm_relative_resolves() {
        // cd /home/user && rm -rf ../../ resolves to /
        assert!(blocked("cd /home/user && rm -rf ../../"));
        // cd /usr/local && rm -rf ../.. resolves to /
        assert!(blocked("cd /usr/local && rm -rf ../.."));
        // cd /home/user && rm -rf .. resolves to /home
        assert!(blocked("cd /home/user && rm -rf .."));
    }

    #[test]
    fn test_cd_then_rm_safe_relative() {
        // cd /home/user/projects && rm -rf ./build is fine
        assert!(!blocked("cd /home/user/projects && rm -rf ./build"));
    }

    #[test]
    fn test_working_dir_relative_paths() {
        // rm -rf ../../ with working_dir=/home/user resolves to /
        assert!(blocked_wd("rm -rf ../../", "/home/user"));
        // rm -rf .. with working_dir=/home/user resolves to /home
        assert!(blocked_wd("rm -rf ..", "/home/user"));
        // rm -rf ./build with working_dir=/home/user is fine
        assert!(!blocked_wd("rm -rf ./build", "/home/user"));
    }

    #[test]
    fn test_compound_commands() {
        assert!(blocked("echo hello && rm -rf /"));
        assert!(blocked("ls; rm -rf /home"));
        assert!(!blocked("echo hello && echo world"));
    }

    #[test]
    fn test_disk_wipe_commands() {
        assert!(blocked("mkfs.ext4 /dev/sda1"));
        assert!(blocked("mkfs /dev/vda"));
        assert!(blocked("dd if=/dev/zero of=/dev/sda"));
        assert!(!blocked("dd if=/dev/zero of=/dev/null"));
        assert!(!blocked("dd if=file.iso of=/tmp/out.img"));
    }

    #[test]
    fn test_chmod_chown_system_dirs() {
        assert!(blocked("chmod -R 777 /"));
        assert!(blocked("chown -R root:root /usr"));
        assert!(blocked("chmod -R 755 /etc"));
        assert!(!blocked("chmod -R 755 /tmp/mydir"));
        assert!(!blocked("chmod 644 /etc/config")); // no recursive flag
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(
            normalize_path(Path::new("/home/user/../..")),
            PathBuf::from("/")
        );
        assert_eq!(
            normalize_path(Path::new("/usr/local/..")),
            PathBuf::from("/usr")
        );
        assert_eq!(
            normalize_path(Path::new("/a/b/c/../../d")),
            PathBuf::from("/a/d")
        );
    }

    #[test]
    fn test_split_shell_commands() {
        let parts = split_shell_commands("cd /tmp && rm -rf foo; echo done");
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], "cd /tmp");
        assert_eq!(parts[1], "rm -rf foo");
        assert_eq!(parts[2], "echo done");
    }

    #[test]
    fn test_rm_recursive_flag_variants() {
        assert!(blocked("rm --recursive /home"));
        assert!(blocked("rm -rfi /home"));
        assert!(blocked("rm -Rf /etc"));
    }

    #[test]
    fn test_canvas_toml_write_protection() {
        let dir = tempdir().unwrap();
        let canvas_dir = dir.path().join("canvases").join("my-canvas");
        std::fs::create_dir_all(&canvas_dir).unwrap();
        let canvas_config = canvas_dir.join(".canvas.toml");
        std::fs::write(&canvas_config, "").unwrap();

        let protected = ProtectedPaths::new(dir.path());

        // .canvas.toml is NOT protected from reads
        assert!(
            !protected.is_read_protected(&canvas_config),
            ".canvas.toml should not be read-protected"
        );

        // .canvas.toml IS protected from writes (matched by glob)
        assert!(
            protected.is_write_protected(&canvas_config),
            ".canvas.toml should be write-protected"
        );

        // .canvas.toml outside canvases/ is NOT write-protected
        let other_dir = dir.path().join("other");
        std::fs::create_dir_all(&other_dir).unwrap();
        let other_config = other_dir.join(".canvas.toml");
        std::fs::write(&other_config, "").unwrap();
        assert!(
            !protected.is_write_protected(&other_config),
            ".canvas.toml outside canvases/ should not be write-protected"
        );
    }

    #[test]
    fn test_gitignore_basic_patterns() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join(".gitignore"),
            "*.env\nsecrets/\n!public.env\n",
        )
        .unwrap();
        // Fake a git repo so the walk stops here
        std::fs::create_dir(dir.path().join(".git")).unwrap();

        let protected = ProtectedPaths::new(dir.path());

        // .env files are gitignored → protected
        let env_file = dir.path().join(".env");
        std::fs::write(&env_file, "SECRET=x").unwrap();
        assert!(protected.is_read_protected(&env_file));
        assert!(protected.is_write_protected(&env_file));

        let prod_env = dir.path().join("production.env");
        std::fs::write(&prod_env, "").unwrap();
        assert!(protected.is_read_protected(&prod_env));

        // Negated pattern: public.env is NOT ignored
        let public_env = dir.path().join("public.env");
        std::fs::write(&public_env, "").unwrap();
        assert!(!protected.is_read_protected(&public_env));
        assert!(!protected.is_write_protected(&public_env));

        // Non-matching file is fine
        let readme = dir.path().join("README.md");
        std::fs::write(&readme, "").unwrap();
        assert!(!protected.is_read_protected(&readme));
    }

    #[test]
    fn test_gitignore_subdirectory() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "*.secret\n").unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();

        let sub = dir.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();
        let secret = sub.join("creds.secret");
        std::fs::write(&secret, "").unwrap();

        let protected = ProtectedPaths::new(dir.path());
        assert!(protected.is_read_protected(&secret));
    }

    #[test]
    fn test_gitignore_path_pattern() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "build/output\n").unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();

        let build_dir = dir.path().join("build");
        std::fs::create_dir(&build_dir).unwrap();
        let output = build_dir.join("output");
        std::fs::write(&output, "").unwrap();

        let protected = ProtectedPaths::new(dir.path());
        assert!(protected.is_read_protected(&output));

        // A file named "output" not under build/ should not match
        let other = dir.path().join("output");
        std::fs::write(&other, "").unwrap();
        assert!(!protected.is_read_protected(&other));
    }

    #[test]
    fn test_no_gitignore_no_protection() {
        let dir = tempdir().unwrap();
        // No .gitignore file
        let env_file = dir.path().join(".env");
        std::fs::write(&env_file, "").unwrap();

        let protected = ProtectedPaths::new(dir.path());
        assert!(!protected.is_read_protected(&env_file));
    }

    #[test]
    fn test_write_protected_includes_regular_protected() {
        let dir = tempdir().unwrap();
        let protected = ProtectedPaths::new(dir.path());

        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "").unwrap();

        // config.toml is both read-protected and write-protected
        assert!(protected.is_read_protected(&config_path));
        assert!(protected.is_write_protected(&config_path));
    }
}
