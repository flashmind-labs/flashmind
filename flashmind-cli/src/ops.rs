//! Operational subcommands: `logs` and `clean`.
//!
//! Logs are emitted by `tracing_appender::rolling::daily` into
//! `~/.flashmind/logs/flsh.log.YYYY-MM-DD`.  Sessions live in per-cwd SQLite
//! databases under `~/.flashmind/sessions/<hash>/sessions.db`; display logs are
//! `~/.flashmind/display-<key>.jsonl`.

use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};

use flashmind_memory::session::SessionStore;

use crate::config::config_dir;

// ---------------------------------------------------------------------------
// logs
// ---------------------------------------------------------------------------

/// Resolve a user-supplied date token ("today" | "yesterday" | YYYY-MM-DD) to
/// a calendar date string in the same format used by `tracing_appender`.
fn resolve_date(token: &str) -> Result<String> {
    let today = chrono::Utc::now();
    match token {
        "today" => Ok(today.format("%Y-%m-%d").to_string()),
        "yesterday" => Ok((today - chrono::Duration::days(1))
            .format("%Y-%m-%d")
            .to_string()),
        other => {
            // Validate it parses as a date.
            chrono::NaiveDate::parse_from_str(other, "%Y-%m-%d")
                .with_context(|| format!("invalid date: {other} (expected YYYY-MM-DD)"))?;
            Ok(other.to_string())
        }
    }
}

fn logs_dir() -> Result<PathBuf> {
    Ok(config_dir()?.join("logs"))
}

/// Print the last `n` lines of `path` to stdout.
fn tail_file(path: &PathBuf, n: usize) -> Result<()> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = io::BufReader::new(file);
    let mut all = String::new();
    reader.read_to_string(&mut all)?;
    let lines: Vec<&str> = all.lines().collect();
    let start = lines.len().saturating_sub(n);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in &lines[start..] {
        writeln!(out, "{line}")?;
    }
    Ok(())
}

/// Follow `path` like `tail -f`: emit new bytes as they arrive.
fn follow_file(path: &PathBuf) -> Result<()> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // Start from the end so we only print *new* lines.
    file.seek(SeekFrom::End(0))?;
    let stdout = io::stdout();
    loop {
        let mut buf = [0u8; 4096];
        match file.read(&mut buf) {
            Ok(0) => {
                std::thread::sleep(Duration::from_millis(250));
            }
            Ok(n) => {
                let mut out = stdout.lock();
                out.write_all(&buf[..n])?;
                out.flush()?;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
}

/// Implementation of `flashmind logs [date] [--follow] [--lines N]`.
pub fn run_logs(date: String, follow: bool, lines: Option<usize>) -> Result<()> {
    let dir = logs_dir()?;
    let target = resolve_date(&date)?;
    let file = dir.join(format!("flsh.log.{target}"));
    if !file.exists() {
        anyhow::bail!("no log file for {target} in {}", dir.display());
    }
    let n = lines.unwrap_or(50);
    if follow {
        // Show the tail first, then follow.
        tail_file(&file, n)?;
        follow_file(&file)
    } else {
        tail_file(&file, n)
    }
}

// ---------------------------------------------------------------------------
// clean
// ---------------------------------------------------------------------------

/// List log files (`flsh.log.*`) in the logs directory.
fn list_log_files(dir: &PathBuf) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("flsh.log."))
            {
                out.push(p);
            }
        }
    }
    out
}

/// List display logs (`display-*.jsonl`) directly under the config dir.
fn list_display_logs() -> Vec<PathBuf> {
    let dir = match config_dir() {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("display-") && n.ends_with(".jsonl"))
            {
                out.push(p);
            }
        }
    }
    out
}

/// Open a `SessionStore` from an explicit database path.
async fn open_store_at(db_path: PathBuf) -> Result<SessionStore> {
    let conn = tokio_rusqlite::Connection::open(&db_path).await?;
    conn.call(
        |c| -> std::result::Result<(), flashmind_memory::rusqlite::Error> {
            flashmind_memory::session::schema::init_session_schema(c)?;
            Ok(())
        },
    )
    .await?;
    Ok(SessionStore::new(conn))
}

/// Implementation of `flashmind clean --days N [--dry-run]`.
pub async fn run_clean(days: u64, dry_run: bool) -> Result<()> {
    let cutoff = SystemTime::now() - Duration::from_secs(days * 86400);
    let mut removed = 0usize;
    let mut inspected = 0usize;

    // Log files older than `days` (by modification time).
    let logs = logs_dir()?;
    for f in list_log_files(&logs) {
        inspected += 1;
        let old = fs::metadata(&f)
            .and_then(|m| m.modified())
            .is_ok_and(|m| m < cutoff);
        if old {
            println!("  log    {}", f.display());
            if !dry_run {
                let _ = fs::remove_file(&f);
            }
            removed += 1;
        }
    }

    // Display logs older than `days`.
    for f in list_display_logs() {
        inspected += 1;
        let old = fs::metadata(&f)
            .and_then(|m| m.modified())
            .is_ok_and(|m| m < cutoff);
        if old {
            println!("  display {}", f.display());
            if !dry_run {
                let _ = fs::remove_file(&f);
            }
            removed += 1;
        }
    }

    // Sessions: prune every per-cwd session database.
    let sessions_root = config_dir()?.join("sessions");
    if sessions_root.exists() {
        for entry in fs::read_dir(&sessions_root)? {
            let entry = entry?;
            let db = entry.path().join("sessions.db");
            if !db.exists() {
                continue;
            }
            inspected += 1;
            match open_store_at(db.clone()).await {
                Ok(store) => match store.prune(days as u32).await {
                    Ok(n) if n > 0 => {
                        println!("  sessions {n} pruned from {}", entry.path().display());
                        if !dry_run {
                            removed += n as usize;
                        } else {
                            println!("    (dry-run; not deleted)");
                        }
                    }
                    Ok(_) => {}
                    Err(e) => println!("  ! failed to prune {}: {e}", entry.path().display()),
                },
                Err(e) => println!("  ! failed to open {}: {e}", db.display()),
            }
        }
    }

    let verb = if dry_run { "would remove" } else { "removed" };
    println!("\n  inspected {inspected}, {verb} {removed} (retention: {days}d)");
    Ok(())
}
