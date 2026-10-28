//! Persistent turn history and project file snapshots. Git is only used to list files.
//! Home and filesystem roots skip file snapshots without interrupting conversation history.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, Metadata, Permissions};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use flashmind_core::Conversation;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use uuid::Uuid;

use crate::interactive::DisplayEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TurnState {
    pub conversation: Conversation,
    pub display: Vec<DisplayEvent>,
    pub working_dir: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct Record {
    before: TurnState,
    after: TurnState,
    files: Option<Changes>,
}

#[derive(Default, Serialize, Deserialize)]
pub(super) struct History {
    past: Vec<Record>,
    future: Vec<Record>,
}

impl History {
    pub fn open(dir: &Path) -> Result<Self> {
        match fs::read(dir.join("history.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("reading undo history"),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error.into()),
        }
    }

    fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;
        let mut file = tempfile::NamedTempFile::new_in(dir)?;
        serde_json::to_writer(&mut file, self)?;
        file.as_file().sync_all()?;
        file.persist(dir.join("history.json"))?;
        Ok(())
    }

    pub fn record(
        &mut self,
        dir: &Path,
        before: TurnState,
        after: TurnState,
        files: Option<Changes>,
    ) -> Result<()> {
        self.past.push(Record {
            before,
            after,
            files,
        });
        let future = std::mem::take(&mut self.future);
        if let Err(error) = self.save(dir) {
            self.past.pop();
            self.future = future;
            return Err(error);
        }
        Ok(())
    }

    pub fn undo(&mut self, dir: &Path) -> Result<Option<TurnState>> {
        self.restore(dir, false)
    }

    pub fn redo(&mut self, dir: &Path) -> Result<Option<TurnState>> {
        self.restore(dir, true)
    }

    fn restore(&mut self, dir: &Path, redo: bool) -> Result<Option<TurnState>> {
        let source = if redo {
            &mut self.future
        } else {
            &mut self.past
        };
        let Some(record) = source.last() else {
            return Ok(None);
        };
        if let Some(files) = &record.files {
            files.restore(redo)?;
        }
        let state = if redo { &record.after } else { &record.before }.clone();
        let record = source.pop().context("undo record disappeared")?;
        if redo {
            self.past.push(record);
        } else {
            self.future.push(record);
        }
        if let Err(error) = self.save(dir) {
            let source = if redo {
                &mut self.past
            } else {
                &mut self.future
            };
            let record = source.pop().context("undo record disappeared")?;
            let rollback = record
                .files
                .as_ref()
                .map_or(Ok(()), |files| files.restore(!redo));
            if redo {
                self.future.push(record);
            } else {
                self.past.push(record);
            }
            rollback.context("could not roll back files after failing to save undo history")?;
            return Err(error);
        }
        Ok(Some(state))
    }

    pub fn clear(dir: &Path) -> Result<()> {
        Self::default().save(dir)
    }

    pub fn fork(source: &Path, destination: &Path) -> Result<()> {
        Self::open(source)?.save(destination)
    }
}

pub(super) async fn restore(session: &str, redo: bool) -> Result<Option<TurnState>> {
    let dir = history_dir(session)?;
    tokio::task::spawn_blocking(move || {
        let mut history = History::open(&dir)?;
        if redo {
            history.redo(&dir)
        } else {
            history.undo(&dir)
        }
    })
    .await?
}

pub(super) fn history_dir(session: &str) -> Result<PathBuf> {
    // Keep UUID paths compatible with histories saved before prefixed session keys.
    let key = match session.parse::<Uuid>() {
        Ok(id) => id.to_string(),
        Err(_) => {
            ensure!(
                !session.is_empty()
                    && session
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
                "invalid undo session ID"
            );
            session.to_owned()
        }
    };
    Ok(crate::paths::data_dir()?.join("sessions/undo").join(key))
}

#[derive(Clone, Serialize, Deserialize)]
struct Version {
    blob: Option<PathBuf>,
    symlink: Option<PathBuf>,
    mode: u32,
}

type Files = BTreeMap<PathBuf, Version>;

const MAX_SNAPSHOT_FILES: usize = 10_000;
const MAX_SNAPSHOT_BYTES: u64 = 100 * 1024 * 1024;
const MAX_SNAPSHOT_DURATION: Duration = Duration::from_secs(2);

struct SnapshotBudget {
    started: Instant,
    files: usize,
    bytes: u64,
}

impl SnapshotBudget {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            files: 0,
            bytes: 0,
        }
    }

    fn check_time(&self) -> Result<()> {
        ensure!(
            self.started.elapsed() < MAX_SNAPSHOT_DURATION,
            "file snapshot exceeded its time limit"
        );
        Ok(())
    }

    fn add_file(&mut self, size: u64) -> Result<()> {
        self.check_time()?;
        ensure!(
            self.files < MAX_SNAPSHOT_FILES,
            "too many files to snapshot"
        );
        ensure!(
            size <= MAX_SNAPSHOT_BYTES - self.bytes,
            "file snapshot exceeds 100 MiB"
        );
        self.files += 1;
        self.bytes += size;
        Ok(())
    }
}

pub(super) struct Capture {
    root: PathBuf,
    directory: TempDir,
    files: Files,
}

struct SnapshotFile {
    relative: PathBuf,
    metadata: Metadata,
    symlink: Option<PathBuf>,
}

impl Capture {
    pub fn start(cwd: &Path, dir: &Path) -> Result<Option<Self>> {
        fs::create_dir_all(dir)?;
        let output = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .current_dir(cwd)
            .output()?;
        let root = if output.status.success() {
            PathBuf::from(String::from_utf8(output.stdout)?.trim_end())
        } else {
            cwd.canonicalize()?
        };
        let root = root.canonicalize()?;
        if root.parent().is_none() || dirs::home_dir().as_deref() == Some(root.as_path()) {
            return Ok(None);
        }
        Self::at_root(root, dir, &BTreeSet::new()).map(Some)
    }

    fn at_root(root: PathBuf, dir: &Path, extra: &BTreeSet<PathBuf>) -> Result<Self> {
        let mut budget = SnapshotBudget::new();
        let directory = tempfile::Builder::new()
            .prefix("snapshot-")
            .tempdir_in(dir)?;
        let mut files = BTreeMap::new();
        let paths = project_files(&root, dir, &budget)?
            .union(extra)
            .cloned()
            .collect::<BTreeSet<_>>();
        // Check the complete file set before copying any contents.
        let mut versions = Vec::new();
        for relative in paths {
            // State may live under the project when FLASHMIND_HOME is customized.
            if root.join(&relative).starts_with(dir) {
                continue;
            }
            check_path(&root, &relative)?;
            let path = root.join(&relative);
            let metadata = match fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            let symlink = metadata
                .is_symlink()
                .then(|| fs::read_link(&path))
                .transpose()?;
            if !metadata.is_file() && symlink.is_none() {
                continue;
            }
            budget.add_file(if metadata.is_file() {
                metadata.len()
            } else {
                0
            })?;
            versions.push(SnapshotFile {
                relative,
                metadata,
                symlink,
            });
        }
        for SnapshotFile {
            relative,
            metadata,
            symlink,
        } in versions
        {
            budget.check_time()?;
            let path = root.join(&relative);
            let blob = if symlink.is_none() {
                let blob = directory.path().join(files.len().to_string());
                fs::copy(&path, &blob)
                    .with_context(|| format!("snapshotting {}", relative.display()))?;
                Some(blob)
            } else {
                None
            };
            files.insert(
                relative,
                Version {
                    blob,
                    symlink,
                    mode: mode(&metadata.permissions()),
                },
            );
        }
        Ok(Self {
            root,
            directory,
            files,
        })
    }

    pub(super) fn finish(mut self, dir: &Path) -> Result<Changes> {
        let mut after = Self::at_root(
            self.root.clone(),
            dir,
            &self.files.keys().cloned().collect(),
        )?;
        let paths: BTreeSet<_> = self
            .files
            .keys()
            .chain(after.files.keys())
            .cloned()
            .collect();
        let mut changes = BTreeMap::new();
        for path in paths {
            if versions_equal(self.files.get(&path), after.files.get(&path))? {
                for files in [&mut self.files, &mut after.files] {
                    if let Some(version) = files.remove(&path)
                        && let Some(blob) = version.blob
                    {
                        fs::remove_file(blob)?;
                    }
                }
            } else {
                changes.insert(
                    path.clone(),
                    Change {
                        before: self.files.remove(&path),
                        after: after.files.remove(&path),
                    },
                );
            }
        }
        // Changed file versions remain available across session resumes.
        let _before_directory = self.directory.keep();
        let _after_directory = after.directory.keep();
        Ok(Changes {
            root: self.root,
            files: changes,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Change {
    before: Option<Version>,
    after: Option<Version>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Changes {
    root: PathBuf,
    files: BTreeMap<PathBuf, Change>,
}

impl Changes {
    fn restore(&self, redo: bool) -> Result<()> {
        for (path, change) in &self.files {
            check_path(&self.root, path)?;
            let expected = if redo { &change.before } else { &change.after };
            let target = if redo { &change.after } else { &change.before };
            if !matches_disk(&self.root.join(path), expected.as_ref())? {
                bail!(
                    "{} changed since this turn; leaving files untouched",
                    path.display()
                );
            }
            if let Some(blob) = target.as_ref().and_then(|version| version.blob.as_ref()) {
                File::open(blob).context("undo snapshot is missing")?;
            }
        }
        let mut applied = Vec::new();
        for (path, change) in &self.files {
            let target = if redo { &change.after } else { &change.before };
            if let Err(error) = restore_file(&self.root.join(path), target.as_ref()) {
                for restored in applied.into_iter().rev() {
                    let original = &self.files[restored];
                    let version = if redo {
                        &original.before
                    } else {
                        &original.after
                    };
                    restore_file(&self.root.join(restored), version.as_ref())
                        .context("could not roll back a partial file restore")?;
                }
                return Err(error);
            }
            applied.push(path);
        }
        Ok(())
    }
}

fn project_files(root: &Path, dir: &Path, budget: &SnapshotBudget) -> Result<BTreeSet<PathBuf>> {
    let data_dir = crate::paths::data_dir()?;
    let snapshot_dir = dir.to_path_buf();
    let included =
        move |path: &Path| !path.starts_with(&snapshot_dir) && !path.starts_with(&data_dir);
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output()?;
    if output.status.success() {
        let mut paths = BTreeSet::new();
        for bytes in output
            .stdout
            .split(|byte| *byte == 0)
            .filter(|bytes| !bytes.is_empty())
        {
            budget.check_time()?;
            let path = PathBuf::from(std::str::from_utf8(bytes)?);
            if included(&root.join(&path)) {
                paths.insert(path);
                ensure!(
                    paths.len() <= MAX_SNAPSHOT_FILES,
                    "too many files to snapshot"
                );
            }
        }
        return Ok(paths);
    }
    let mut paths = BTreeSet::new();
    for entry in ignore::WalkBuilder::new(root)
        .hidden(false)
        .filter_entry(move |entry| included(entry.path()))
        .build()
    {
        budget.check_time()?;
        let entry = entry?;
        if entry
            .file_type()
            .is_some_and(|kind| kind.is_file() || kind.is_symlink())
        {
            paths.insert(entry.path().strip_prefix(root)?.to_path_buf());
            ensure!(
                paths.len() <= MAX_SNAPSHOT_FILES,
                "too many files to snapshot"
            );
        }
    }
    Ok(paths)
}

fn check_path(root: &Path, relative: &Path) -> Result<()> {
    if root.canonicalize()? != root {
        bail!("project directory moved or became a symlink");
    }
    if relative.as_os_str().is_empty() {
        bail!("empty snapshot path");
    }
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("invalid snapshot path");
    }
    let mut parent = root.to_path_buf();
    let components: Vec<_> = relative.components().collect();
    for component in &components[..components.len().saturating_sub(1)] {
        parent.push(component);
        match fs::symlink_metadata(&parent) {
            Ok(metadata) if metadata.is_symlink() => bail!("{} is a symlink", parent.display()),
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn versions_equal(left: Option<&Version>, right: Option<&Version>) -> Result<bool> {
    match (left, right) {
        (None, None) => Ok(true),
        (Some(left), Some(right)) if left.mode == right.mode && left.symlink == right.symlink => {
            match (&left.blob, &right.blob) {
                (Some(left), Some(right)) => files_equal(left, right),
                (None, None) => Ok(true),
                _ => Ok(false),
            }
        }
        _ => Ok(false),
    }
}

fn files_equal(left: &Path, right: &Path) -> Result<bool> {
    let mut left = File::open(left)?;
    let mut right = File::open(right)?;
    if left.metadata()?.len() != right.metadata()?.len() {
        return Ok(false);
    }
    let mut a = [0; 65536];
    let mut b = [0; 65536];
    loop {
        let n = left.read(&mut a)?;
        right.read_exact(&mut b[..n])?;
        if a[..n] != b[..n] {
            return Ok(false);
        }
        if n == 0 {
            return Ok(true);
        }
    }
}

fn matches_disk(path: &Path, version: Option<&Version>) -> Result<bool> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(version.is_none()),
        Err(error) => return Err(error.into()),
    };
    let Some(version) = version else {
        return Ok(false);
    };
    if metadata.is_symlink() {
        return Ok(version.symlink.as_ref() == Some(&fs::read_link(path)?));
    }
    if !metadata.is_file()
        || version.symlink.is_some()
        || mode(&metadata.permissions()) != version.mode
    {
        return Ok(false);
    }
    match &version.blob {
        Some(blob) => files_equal(path, blob),
        None => Ok(false),
    }
}

fn restore_file(path: &Path, version: Option<&Version>) -> Result<()> {
    let Some(version) = version else {
        fs::remove_file(path)?;
        return Ok(());
    };
    let parent = path.parent().context("file has no parent")?;
    fs::create_dir_all(parent)?;
    if let Some(target) = &version.symlink {
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let temporary = tempfile::NamedTempFile::new_in(parent)?.into_temp_path();
            fs::remove_file(&temporary)?;
            symlink(target, &temporary)?;
            fs::rename(&temporary, path)?;
        }
        #[cfg(not(unix))]
        bail!(
            "restoring symlinks is not supported on this platform: {}",
            target.display()
        );
    } else {
        let blob = version.blob.as_ref().context("snapshot has no contents")?;
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        io::copy(&mut File::open(blob)?, &mut file)?;
        set_mode(file.path(), version.mode)?;
        file.persist(path)?;
    }
    Ok(())
}

#[cfg(unix)]
fn mode(permissions: &Permissions) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    permissions.mode()
}
#[cfg(not(unix))]
fn mode(permissions: &Permissions) -> u32 {
    u32::from(permissions.readonly())
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, Permissions::from_mode(mode))?;
    Ok(())
}
#[cfg(not(unix))]
fn set_mode(path: &Path, mode: u32) -> Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(mode != 0);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_skip_home_and_filesystem_root_without_an_error() {
        let history = tempfile::tempdir().unwrap();
        let home = dirs::home_dir().unwrap();
        for root in [home.as_path(), home.ancestors().last().unwrap()] {
            assert!(Capture::start(root, history.path()).unwrap().is_none());
            assert_eq!(fs::read_dir(history.path()).unwrap().count(), 0);
        }
    }

    #[test]
    fn oversized_projects_are_rejected_before_copying_contents() {
        let project = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        fs::write(project.path().join("a-small"), "contents").unwrap();
        File::create(project.path().join("z-large"))
            .unwrap()
            .set_len(MAX_SNAPSHOT_BYTES + 1)
            .unwrap();

        let error = Capture::start(project.path(), history.path())
            .err()
            .unwrap();

        assert!(error.to_string().contains("100 MiB"));
        assert_eq!(fs::read_dir(history.path()).unwrap().count(), 0);
    }

    #[test]
    fn large_file_sets_are_rejected_before_copying_contents() {
        let project = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        for index in 0..=MAX_SNAPSHOT_FILES {
            File::create(project.path().join(index.to_string())).unwrap();
        }

        let error = Capture::start(project.path(), history.path())
            .err()
            .unwrap();

        assert!(error.to_string().contains("too many files"));
        assert_eq!(fs::read_dir(history.path()).unwrap().count(), 0);
    }

    #[test]
    fn snapshots_inside_the_project_are_excluded_from_later_captures() {
        let project = tempfile::tempdir().unwrap();
        let history = project.path().join(".history");
        fs::create_dir(&history).unwrap();
        File::create(history.join("old-snapshot"))
            .unwrap()
            .set_len(MAX_SNAPSHOT_BYTES + 1)
            .unwrap();
        fs::write(project.path().join("file"), "before").unwrap();
        let capture = Capture::start(project.path(), &history).unwrap().unwrap();
        fs::write(project.path().join("file"), "after").unwrap();
        let mut journal = History::default();
        journal
            .record(
                &history,
                state(project.path(), "before"),
                state(project.path(), "after"),
                Some(capture.finish(&history).unwrap()),
            )
            .unwrap();

        journal.undo(&history).unwrap();

        assert_eq!(
            fs::read_to_string(project.path().join("file")).unwrap(),
            "before"
        );
        assert!(history.join("old-snapshot").exists());
    }

    #[test]
    fn conversation_undo_and_redo_work_without_file_snapshots() {
        let project = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        let mut journal = History::default();
        journal
            .record(
                history.path(),
                state(project.path(), "before"),
                state(project.path(), "after"),
                None,
            )
            .unwrap();
        let mut journal = History::open(history.path()).unwrap();
        assert_eq!(
            journal
                .undo(history.path())
                .unwrap()
                .unwrap()
                .conversation
                .entries()[0]
                .content(),
            "before"
        );
        assert_eq!(
            journal
                .redo(history.path())
                .unwrap()
                .unwrap()
                .conversation
                .entries()[0]
                .content(),
            "after"
        );
    }

    #[test]
    fn undo_history_accepts_new_and_imported_session_keys() {
        for session in [
            crate::session::new_session_key(),
            "pi_example-session".into(),
        ] {
            let dir = history_dir(&session).unwrap();
            assert_eq!(
                dir,
                crate::paths::data_dir()
                    .unwrap()
                    .join("sessions/undo")
                    .join(&session)
            );
        }
    }

    #[test]
    fn undo_history_preserves_existing_uuid_paths() {
        let id = Uuid::new_v4();
        let expected = crate::paths::data_dir()
            .unwrap()
            .join("sessions/undo")
            .join(id.to_string());
        assert_eq!(history_dir(&id.to_string()).unwrap(), expected);
        assert_eq!(history_dir(&id.as_simple().to_string()).unwrap(), expected);
    }

    #[test]
    fn undo_history_rejects_paths_as_session_keys() {
        for session in ["", ".", "..", "../example", "/example", "a/b", "a\\b"] {
            assert!(history_dir(session).is_err(), "accepted {session:?}");
        }
    }

    fn state(root: &Path, text: &str) -> TurnState {
        TurnState {
            conversation: Conversation::with_system(text),
            display: vec![],
            working_dir: root.to_path_buf(),
        }
    }

    #[test]
    fn undo_redo_preserves_dirty_files_and_survives_resume() {
        let project = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        let root = project.path();
        fs::write(root.join("existing"), b"user edits\x00").unwrap();
        fs::write(root.join("deleted"), "keep me").unwrap();
        let capture = Capture::start(root, history.path()).unwrap().unwrap();
        fs::write(root.join("existing"), "agent edits").unwrap();
        fs::write(root.join("created"), "new").unwrap();
        fs::remove_file(root.join("deleted")).unwrap();
        let mut journal = History::default();
        journal
            .record(
                history.path(),
                state(root, "before"),
                state(root, "after"),
                Some(capture.finish(history.path()).unwrap()),
            )
            .unwrap();
        let mut journal = History::open(history.path()).unwrap();
        let restored = journal.undo(history.path()).unwrap().unwrap();
        assert_eq!(restored.conversation.entries()[0].content(), "before");
        assert_eq!(fs::read(root.join("existing")).unwrap(), b"user edits\x00");
        assert_eq!(fs::read_to_string(root.join("deleted")).unwrap(), "keep me");
        assert!(!root.join("created").exists());
        let mut journal = History::open(history.path()).unwrap();
        let restored = journal.redo(history.path()).unwrap().unwrap();
        assert_eq!(restored.conversation.entries()[0].content(), "after");
        assert_eq!(
            fs::read_to_string(root.join("existing")).unwrap(),
            "agent edits"
        );
        assert!(root.join("created").exists());
        assert!(!root.join("deleted").exists());
    }

    #[test]
    fn multiple_turns_and_forks_restore_their_own_conversations() {
        let project = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        let fork = tempfile::tempdir().unwrap();
        let root = project.path();
        let mut journal = History::default();
        for (before, after) in [("zero", "one"), ("one", "two")] {
            fs::write(root.join("file"), before).unwrap();
            let capture = Capture::start(root, history.path()).unwrap().unwrap();
            fs::write(root.join("file"), after).unwrap();
            journal
                .record(
                    history.path(),
                    state(root, before),
                    state(root, after),
                    Some(capture.finish(history.path()).unwrap()),
                )
                .unwrap();
        }
        History::fork(history.path(), fork.path()).unwrap();
        assert_eq!(
            journal
                .undo(history.path())
                .unwrap()
                .unwrap()
                .conversation
                .entries()[0]
                .content(),
            "one"
        );
        assert_eq!(
            journal
                .undo(history.path())
                .unwrap()
                .unwrap()
                .conversation
                .entries()[0]
                .content(),
            "zero"
        );
        assert_eq!(fs::read_to_string(root.join("file")).unwrap(), "zero");
        journal.redo(history.path()).unwrap();
        journal.redo(history.path()).unwrap();
        let mut forked = History::open(fork.path()).unwrap();
        assert!(forked.undo(fork.path()).unwrap().is_some());
        assert_eq!(fs::read_to_string(root.join("file")).unwrap(), "one");
    }

    #[test]
    fn tracked_files_and_index_survive_changes_to_ignore_rules() {
        let project = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        let root = project.path();
        assert!(
            Command::new("git")
                .arg("init")
                .arg(root)
                .output()
                .unwrap()
                .status
                .success()
        );
        fs::write(root.join("file"), "staged").unwrap();
        assert!(
            Command::new("git")
                .args(["add", "file"])
                .current_dir(root)
                .output()
                .unwrap()
                .status
                .success()
        );
        let index = fs::read(root.join(".git/index")).unwrap();
        fs::write(root.join("file"), "dirty").unwrap();
        fs::write(root.join("untracked"), "before").unwrap();
        let capture = Capture::start(root, history.path()).unwrap().unwrap();
        fs::write(root.join("file"), "agent").unwrap();
        fs::write(root.join("untracked"), "after").unwrap();
        fs::write(root.join(".gitignore"), "untracked\n").unwrap();
        let mut journal = History::default();
        journal
            .record(
                history.path(),
                state(root, "before"),
                state(root, "after"),
                Some(capture.finish(history.path()).unwrap()),
            )
            .unwrap();
        journal.undo(history.path()).unwrap();
        assert_eq!(fs::read(root.join(".git/index")).unwrap(), index);
        assert_eq!(fs::read_to_string(root.join("file")).unwrap(), "dirty");
        assert_eq!(
            fs::read_to_string(root.join("untracked")).unwrap(),
            "before"
        );
    }

    #[cfg(unix)]
    #[test]
    fn executable_bits_and_symlinks_round_trip_without_following_links() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let project = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        let root = project.path();
        fs::write(root.join("file"), "before").unwrap();
        fs::set_permissions(root.join("file"), Permissions::from_mode(0o755)).unwrap();
        symlink("file", root.join("link")).unwrap();
        let capture = Capture::start(root, history.path()).unwrap().unwrap();
        fs::set_permissions(root.join("file"), Permissions::from_mode(0o644)).unwrap();
        fs::remove_file(root.join("link")).unwrap();
        fs::write(root.join("link"), "regular file").unwrap();
        let mut journal = History::default();
        journal
            .record(
                history.path(),
                state(root, "before"),
                state(root, "after"),
                Some(capture.finish(history.path()).unwrap()),
            )
            .unwrap();
        journal.undo(history.path()).unwrap();
        assert_eq!(
            fs::metadata(root.join("file"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
        assert_eq!(fs::read_link(root.join("link")).unwrap(), Path::new("file"));
        journal.redo(history.path()).unwrap();
        assert_eq!(
            fs::read_to_string(root.join("link")).unwrap(),
            "regular file"
        );
    }

    #[test]
    fn conflicts_leave_all_files_and_history_untouched() {
        let project = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        let root = project.path();
        fs::write(root.join("a"), "before").unwrap();
        fs::write(root.join("z"), "before").unwrap();
        let capture = Capture::start(root, history.path()).unwrap().unwrap();
        fs::write(root.join("a"), "after").unwrap();
        fs::write(root.join("z"), "after").unwrap();
        let mut journal = History::default();
        journal
            .record(
                history.path(),
                state(root, "before"),
                state(root, "after"),
                Some(capture.finish(history.path()).unwrap()),
            )
            .unwrap();
        fs::write(root.join("z"), "user edit").unwrap();
        assert!(
            journal
                .undo(history.path())
                .unwrap_err()
                .to_string()
                .contains("changed since")
        );
        assert_eq!(fs::read_to_string(root.join("a")).unwrap(), "after");
        assert_eq!(journal.past.len(), 1);
        assert!(journal.future.is_empty());
    }

    #[test]
    fn a_new_turn_discards_redo_and_ignored_files_stay_untouched() {
        let project = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        let root = project.path();
        assert!(
            Command::new("git")
                .arg("init")
                .arg(root)
                .output()
                .unwrap()
                .status
                .success()
        );
        fs::write(root.join(".gitignore"), "ignored\n").unwrap();
        fs::write(root.join("ignored"), "before").unwrap();
        let capture = Capture::start(root, history.path()).unwrap().unwrap();
        fs::write(root.join("new"), "first").unwrap();
        fs::write(root.join("ignored"), "after").unwrap();
        let mut journal = History::default();
        journal
            .record(
                history.path(),
                state(root, "before"),
                state(root, "after"),
                Some(capture.finish(history.path()).unwrap()),
            )
            .unwrap();
        journal.undo(history.path()).unwrap();
        assert_eq!(fs::read_to_string(root.join("ignored")).unwrap(), "after");
        let capture = Capture::start(root, history.path()).unwrap().unwrap();
        fs::write(root.join("new"), "second").unwrap();
        journal
            .record(
                history.path(),
                state(root, "before"),
                state(root, "second"),
                Some(capture.finish(history.path()).unwrap()),
            )
            .unwrap();
        assert!(journal.redo(history.path()).unwrap().is_none());
    }
}
