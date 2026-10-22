#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around cargo-release for publishing Flashmind crates.
#
# Usage:
#   ./scripts/release-crates.sh              # dry run of current versions
#   ./scripts/release-crates.sh --execute    # publish current versions
#   ./scripts/release-crates.sh minor        # dry run of a minor bump
#   ./scripts/release-crates.sh patch --execute  # publish patch bump
#
# Install cargo-release first:
#   cargo install cargo-release

if ! command -v cargo-release &>/dev/null && ! cargo release --version &>/dev/null 2>&1; then
  echo "error: cargo-release not found. Install with: cargo install cargo-release" >&2
  exit 1
fi

echo "==> Running workspace checks"
cargo fmt --check
cargo check --workspace --all-features
cargo test --workspace --all-features -q
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps -q

echo
echo "==> cargo release --workspace $*"
cargo release --workspace "$@"
