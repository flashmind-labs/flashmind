#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around cargo-release for publishing Flashmind crates.
#
# Usage:
#   ./scripts/release-crates.sh              # dry run (patch bump)
#   ./scripts/release-crates.sh --execute    # actually publish
#   ./scripts/release-crates.sh minor        # dry run minor bump
#   ./scripts/release-crates.sh patch --execute  # publish patch bump
#
# Install cargo-release first:
#   cargo install cargo-release

if ! command -v cargo-release &>/dev/null && ! cargo release --version &>/dev/null 2>&1; then
  echo "error: cargo-release not found. Install with: cargo install cargo-release" >&2
  exit 1
fi

LEVEL="${1:-patch}"
shift 2>/dev/null || true

echo "==> Running workspace checks"
cargo fmt --check
cargo check --workspace --all-features
cargo test --workspace --no-run -q
cargo doc --workspace --all-features --no-deps -q

echo
echo "==> cargo release $LEVEL --workspace $*"
cargo release "$LEVEL" --workspace "$@"
