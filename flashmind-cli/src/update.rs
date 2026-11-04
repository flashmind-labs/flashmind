use std::io::Write;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

const BASE: &str = "https://releases.flashmind.com/flashmind/cli";

fn target() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Ok("aarch64-apple-darwin"),
        ("macos", "x86_64") => Ok("x86_64-apple-darwin"),
        ("linux", "aarch64") => Ok("aarch64-unknown-linux-gnu"),
        ("linux", "x86_64") => Ok("x86_64-unknown-linux-gnu"),
        ("windows", "x86_64") => Ok("x86_64-pc-windows-msvc"),
        _ => bail!("no flsh release for this operating system and architecture"),
    }
}

fn valid_version(version: &str) -> bool {
    let parts: Vec<_> = version.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

fn verify(binary: &[u8], expected: &str) -> Result<()> {
    let actual: String = Sha256::digest(binary)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if actual != expected.trim() {
        bail!("release checksum does not match; update cancelled");
    }
    Ok(())
}

pub async fn run() -> Result<()> {
    let target = target()?;
    let client = reqwest::Client::new();
    let version = client
        .get(format!("{BASE}/latest.txt"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let version = version.trim();
    if !valid_version(version) {
        bail!("invalid release version: {version:?}");
    }
    let name = if cfg!(windows) {
        format!("flsh-{target}.exe")
    } else {
        format!("flsh-{target}")
    };
    let url = format!("{BASE}/{version}/{name}");
    let checksum = client
        .get(format!("{url}.sha256"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let binary = client
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    verify(&binary, &checksum)?;
    let version = version.to_owned();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let mut file = tempfile::NamedTempFile::new().context("creating temporary update file")?;
        file.write_all(&binary).context("writing update")?;
        file.flush()?;
        self_replace::self_replace(file.path()).context("replacing flsh executable")?;
        Ok(())
    })
    .await??;
    println!("Installed flsh {version}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_release_data() {
        assert!(valid_version("0.2.0"));
        assert!(!valid_version("../0.2.0"));
        assert!(!valid_version("0.2"));
        let digest = "9a3a45d01531a20e89ac6ae10b0b0beb0492acd7216a368aa062d1a5fecaf9cd";
        assert!(verify(b"binary", &digest).is_ok());
        assert!(verify(b"changed", &digest).is_err());
    }
}
