#!/bin/sh
set -eu

base=https://releases.flashmind.com/flashmind/cli
case "$(uname -s):$(uname -m)" in
  Darwin:arm64) target=aarch64-apple-darwin ;;
  Darwin:x86_64) target=x86_64-apple-darwin ;;
  Linux:aarch64|Linux:arm64) target=aarch64-unknown-linux-gnu ;;
  Linux:x86_64) target=x86_64-unknown-linux-gnu ;;
  *) echo 'No flsh binary is available for this platform.' >&2; exit 1 ;;
esac
version=$(curl -fsSL "$base/latest.txt")
case "$version" in *[!0-9.]*|'') echo 'Invalid release version.' >&2; exit 1 ;; esac
name="flsh-$target"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
curl -fsSL "$base/$version/$name" -o "$tmp/flsh"
curl -fsSL "$base/$version/$name.sha256" -o "$tmp/checksum"
expected=$(tr -d ' \t\r\n' < "$tmp/checksum")
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$tmp/flsh" | cut -d ' ' -f 1)
else
  actual=$(shasum -a 256 "$tmp/flsh" | cut -d ' ' -f 1)
fi
[ "$expected" = "$actual" ] || { echo 'Checksum mismatch.' >&2; exit 1; }
install_dir=${FLSH_INSTALL_DIR:-"$HOME/.local/bin"}
mkdir -p "$install_dir"
install -m 755 "$tmp/flsh" "$install_dir/flsh"
echo "Installed flsh $version to $install_dir/flsh"
case ":$PATH:" in *":$install_dir:"*) ;; *) echo "Add $install_dir to PATH to run flsh." ;; esac
