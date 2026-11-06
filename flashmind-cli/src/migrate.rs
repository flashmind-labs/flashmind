//! Import legacy sessions across all saved project directories.

use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use flashmind_memory::session::SessionStore;

#[derive(Default)]
struct MigrationReport {
    conversations: usize,
    migrated: usize,
    skipped: usize,
    failures: Vec<(PathBuf, anyhow::Error)>,
}

pub async fn run() -> Result<()> {
    let report = migrate_all(&crate::paths::data_dir()?.join("sessions")).await?;
    for (path, error) in &report.failures {
        eprintln!("{}: {error:#}", path.display());
    }
    println!(
        "Saved conversations: {}. Project stores: {} migrated, {} already migrated, {} failed.",
        report.conversations,
        report.migrated,
        report.skipped,
        report.failures.len()
    );
    ensure!(
        report.failures.is_empty(),
        "some session stores could not be migrated; rerun `flsh migrate` to retry"
    );
    Ok(())
}

async fn migrate_all(root: &Path) -> Result<MigrationReport> {
    let mut report = MigrationReport::default();
    if !root.exists() {
        return Ok(report);
    }
    let mut directories = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() && entry.file_name() != "display" {
            directories.push(entry.path());
        }
    }
    directories.sort();
    for directory in directories {
        let source = directory.join("sessions.db");
        if !source.is_file() {
            continue;
        }
        let already_imported = directory.join(".sqlite-imported").exists();
        let result = async {
            let store = SessionStore::open(&directory)?;
            if !already_imported {
                store.import_sqlite(&source).await?;
            }
            store.list_sessions().await
        }
        .await;
        match result {
            Ok(sessions) => {
                report.conversations += sessions.len();
                if already_imported {
                    report.skipped += 1;
                } else {
                    report.migrated += 1;
                }
            }
            Err(error) => report.failures.push((source, error)),
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use flashmind_memory::rusqlite::Connection;

    fn legacy_store(directory: &Path) -> Vec<u8> {
        std::fs::create_dir_all(directory).unwrap();
        let path = directory.join("sessions.db");
        let connection = Connection::open(&path).unwrap();
        flashmind_memory::session::schema::init_session_schema(&connection).unwrap();
        connection
            .execute(
                "INSERT INTO sessions (chat_key,entry_kind,content,turn_index,created_at) VALUES
             ('example','{\"type\":\"user\"}','question',0,100),
             ('example','{\"type\":\"assistant\"}','answer',1,101),
             ('another','{\"type\":\"user\"}','another question',0,102)",
                [],
            )
            .unwrap();
        drop(connection);
        std::fs::read(path).unwrap()
    }

    #[tokio::test]
    async fn migrates_all_projects_and_can_be_repeated_after_partial_failure() {
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("first");
        let second = root.path().join("second");
        let broken = root.path().join("broken");
        let original = legacy_store(&first);
        legacy_store(&second);
        // Display logs are not project stores, even if a database is present.
        legacy_store(&root.path().join("display"));
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("sessions.db"), b"invalid database").unwrap();
        let report = migrate_all(root.path()).await.unwrap();
        assert_eq!(report.migrated, 2);
        assert_eq!(report.conversations, 4);
        assert_eq!(report.failures.len(), 1);
        assert!(!root.path().join("display/.sqlite-imported").exists());
        assert_eq!(std::fs::read(first.join("sessions.db")).unwrap(), original);
        for path in [&first, &second] {
            assert_eq!(
                SessionStore::open(path)
                    .unwrap()
                    .load("example")
                    .await
                    .unwrap()[0]
                    .content,
                "question"
            );
        }
        let first_store = SessionStore::open(&first).unwrap();
        first_store.delete_session("example").await.unwrap();
        std::fs::remove_file(broken.join("sessions.db")).unwrap();
        legacy_store(&broken);
        let repeated = migrate_all(root.path()).await.unwrap();
        assert_eq!(repeated.migrated, 1);
        assert_eq!(repeated.skipped, 2);
        assert_eq!(repeated.conversations, 5);
        assert!(repeated.failures.is_empty());
        assert!(first_store.load("example").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn missing_sessions_directory_has_nothing_to_migrate() {
        let root = tempfile::tempdir().unwrap();
        let report = migrate_all(&root.path().join("missing")).await.unwrap();
        assert_eq!(report.migrated, 0);
        assert_eq!(report.conversations, 0);
        assert!(report.failures.is_empty());
    }

    #[test]
    fn migration_command_needs_no_project_or_model() {
        let cli = crate::Cli::try_parse_from(["flsh", "migrate"]).unwrap();
        assert!(matches!(cli.command, Some(crate::Command::Migrate)));
    }
}
