use std::process::Command;

#[test]
fn version_is_available_without_loading_config() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("config.toml"), "invalid = [").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_flsh"))
        .arg("--version")
        .env("FLASHMIND_HOME", directory.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("flsh {}\n", env!("CARGO_PKG_VERSION"))
    );
}
