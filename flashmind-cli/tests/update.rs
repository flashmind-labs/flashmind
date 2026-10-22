#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

#[test]
fn update_streams_cargo_output_and_reports_failure_without_loading_config() {
    let directory = tempfile::tempdir().unwrap();
    let cargo = directory.path().join("cargo");
    let arguments = directory.path().join("arguments");
    fs::write(directory.path().join("config.toml"), "invalid = [").unwrap();
    fs::write(
        &cargo,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$UPDATE_ARGS\"\nprintf 'Example build output\\n'\nexit \"$UPDATE_STATUS\"\n",
    )
    .unwrap();
    fs::set_permissions(&cargo, fs::Permissions::from_mode(0o755)).unwrap();

    for status in [0, 7] {
        let output = Command::new(env!("CARGO_BIN_EXE_flsh"))
            .arg("update")
            .env("PATH", directory.path())
            .env("FLASHMIND_HOME", directory.path())
            .env("UPDATE_ARGS", &arguments)
            .env("UPDATE_STATUS", status.to_string())
            .output()
            .unwrap();
        assert_eq!(output.status.success(), status == 0);
        assert_eq!(output.stdout, b"Example build output\n");
        assert_eq!(
            fs::read_to_string(&arguments)
                .unwrap()
                .lines()
                .collect::<Vec<_>>(),
            [
                "install",
                "--git",
                env!("CARGO_PKG_REPOSITORY"),
                "--locked",
                "--force",
                "flashmind-cli"
            ]
        );
        if status != 0 {
            let error = String::from_utf8(output.stderr).unwrap();
            assert!(error.contains("cargo install failed"), "{error}");
            assert!(error.contains('7'), "{error}");
        }
    }
}
