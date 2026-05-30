#!/usr/bin/env bash
set -euo pipefail

CRATES=(
  flashmind-types
  flashmind-prompts
  flashmind-core
  flashmind-llm
  flashmind-memory
  flashmind-cron
  flashmind-skills
  flashmind-tailscale
  flashmind-tools
  flashmind-tui
  flashmind
)

PUBLISH=0
SKIP_CHECKS=0
NO_VERIFY=0
START_CRATE=""
PUBLISH_DELAY_SECONDS="${PUBLISH_DELAY_SECONDS:-30}"

usage() {
  cat <<'USAGE'
Usage: scripts/release-crates.sh [OPTIONS]

Prepare or publish Flashmind workspace crates in dependency order.

Default mode is a dry run: runs checks, then prints publish commands.
No crate is published unless --publish is passed.

Options:
  --publish       Publish crates after checks/package verification.
  --skip-checks   Skip workspace-wide checks.
  --no-verify     Pass --no-verify to cargo publish. Only applies with --publish.
  --crate NAME    Start at NAME in the release order, useful for resuming.
  -h, --help      Show this help.

Environment:
  PUBLISH_DELAY_SECONDS  Delay after each successful publish, default: 30.
USAGE
}

die() {
  echo "error: $*" >&2
  echo >&2
  usage >&2
  exit 2
}

contains_crate() {
  local needle="$1"
  local crate
  for crate in "${CRATES[@]}"; do
    [[ "$crate" == "$needle" ]] && return 0
  done
  return 1
}

selected_crates() {
  local started=0
  local crate

  if [[ -z "$START_CRATE" ]]; then
    printf '%s\n' "${CRATES[@]}"
    return 0
  fi

  for crate in "${CRATES[@]}"; do
    if [[ "$crate" == "$START_CRATE" ]]; then
      started=1
    fi

    if [[ "$started" -eq 1 ]]; then
      printf '%s\n' "$crate"
    fi
  done
}

run_workspace_checks() {
  echo "==> Running workspace checks"
  cargo fmt --check
  cargo check --workspace
  cargo check --workspace --all-features
  cargo test --workspace --no-run
  cargo doc --workspace --all-features --no-deps
}

print_publish_commands() {
  local crate
  local verify_arg=()

  if [[ "$NO_VERIFY" -eq 1 ]]; then
    verify_arg=(--no-verify)
  fi

  echo
  echo "Dry run complete. To publish manually, run these commands in order:"
  while IFS= read -r crate; do
    printf 'cargo publish -p %q' "$crate"
    if [[ "${#verify_arg[@]}" -gt 0 ]]; then
      printf ' %s' "${verify_arg[@]}"
    fi
    printf '\n'
  done < <(selected_crates)
}

publish_crates() {
  local crate
  local reply
  local publish_args=()

  if [[ "$NO_VERIFY" -eq 1 ]]; then
    publish_args+=(--no-verify)
  fi

  echo
  echo "Publish mode enabled. Each crate requires confirmation."
  while IFS= read -r crate; do
    echo "==> Verifying package for $crate"
    cargo package -p "$crate" --allow-dirty

    printf 'Publish %s to crates.io? Type "publish %s" to continue: ' "$crate" "$crate"
    read -r reply
    if [[ "$reply" != "publish $crate" ]]; then
      echo "Skipping $crate"
      continue
    fi

    echo "==> cargo publish -p $crate ${publish_args[*]:-}"
    cargo publish -p "$crate" "${publish_args[@]}"

    if [[ "$PUBLISH_DELAY_SECONDS" != "0" ]]; then
      echo "Waiting ${PUBLISH_DELAY_SECONDS}s for crates.io indexing"
      sleep "$PUBLISH_DELAY_SECONDS"
    fi
  done < <(selected_crates)
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --publish)
      PUBLISH=1
      shift
      ;;
    --skip-checks)
      SKIP_CHECKS=1
      shift
      ;;
    --no-verify)
      NO_VERIFY=1
      shift
      ;;
    --crate)
      [[ "$#" -ge 2 ]] || die "--crate requires a crate name"
      START_CRATE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

if [[ -n "$START_CRATE" ]] && ! contains_crate "$START_CRATE"; then
  die "unknown crate: $START_CRATE"
fi

if [[ "$NO_VERIFY" -eq 1 && "$PUBLISH" -eq 0 ]]; then
  echo "warning: --no-verify only affects --publish mode" >&2
fi

echo "Release order:"
while IFS= read -r crate; do
  echo "  - $crate"
done < <(selected_crates)
echo

if [[ "$SKIP_CHECKS" -eq 0 ]]; then
  run_workspace_checks
else
  echo "==> Skipping workspace checks"
fi

if [[ "$PUBLISH" -eq 1 ]]; then
  publish_crates
else
  print_publish_commands
fi
