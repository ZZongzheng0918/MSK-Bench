#!/usr/bin/env bash

set -euo pipefail

REMOTE="public"
BRANCH="main"
MESSAGE="Sync public release"
NO_PUSH=0

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_public_repo.sh [--remote NAME] [--branch NAME] [--message TEXT] [--no-push]

Create a new snapshot commit on top of an existing public remote branch without
exposing private development history. The script publishes the current HEAD tree
using `git archive`, so only committed files are included.

Options:
  --remote NAME    Remote that points to the public repository (default: public)
  --branch NAME    Remote branch to update (default: main)
  --message TEXT   Commit message for the public sync commit
  --no-push        Build the sync commit locally but do not push it
  -h, --help       Show this help text

Requirements:
  - Run from inside the private/dev repository.
  - The working tree must be clean.
  - The target remote branch must already exist.
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      [[ $# -ge 2 ]] || die "--remote requires a value"
      REMOTE="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || die "--branch requires a value"
      BRANCH="$2"
      shift 2
      ;;
    --message)
      [[ $# -ge 2 ]] || die "--message requires a value"
      MESSAGE="$2"
      shift 2
      ;;
    --no-push)
      NO_PUSH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
cd "$REPO_ROOT"

git remote get-url "$REMOTE" >/dev/null 2>&1 || die "remote '$REMOTE' does not exist"

if [[ -n "$(git status --porcelain)" ]]; then
  die "working tree is not clean; commit or stash changes before syncing"
fi

printf 'Fetching %s/%s...\n' "$REMOTE" "$BRANCH"
git fetch "$REMOTE" "$BRANCH"

REMOTE_REF="refs/remotes/${REMOTE}/${BRANCH}"
git rev-parse --verify --quiet "$REMOTE_REF" >/dev/null || die "remote branch '$REMOTE/$BRANCH' does not exist"

WORKTREE=""
cleanup() {
  if [[ -n "${WORKTREE}" && -d "${WORKTREE}" ]]; then
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || rm -rf "$WORKTREE"
  fi
}
trap cleanup EXIT

WORKTREE="$(mktemp -d "${TMPDIR:-/tmp}/sync-public.XXXXXXXX")"
printf 'Creating temporary worktree at %s...\n' "$WORKTREE"
git worktree add --detach "$WORKTREE" "$REMOTE_REF" >/dev/null

cd "$WORKTREE"

if [[ -n "$(git ls-files)" ]]; then
  git rm -r -q .
fi
git clean -fdx -q

printf 'Exporting HEAD snapshot from %s...\n' "$REPO_ROOT"
git -C "$REPO_ROOT" archive --format=tar HEAD | tar -xf - -C "$WORKTREE"

git add -A

if git diff --cached --quiet; then
  printf 'No public sync needed; %s/%s already matches HEAD.\n' "$REMOTE" "$BRANCH"
  exit 0
fi

printf 'Creating sync commit...\n'
git commit -m "$MESSAGE"

SYNC_COMMIT="$(git rev-parse --short HEAD)"
printf 'Created commit %s on top of %s/%s.\n' "$SYNC_COMMIT" "$REMOTE" "$BRANCH"

if [[ "$NO_PUSH" -eq 1 ]]; then
  printf 'Skipping push because --no-push was specified.\n'
  exit 0
fi

printf 'Pushing %s to %s/%s...\n' "$SYNC_COMMIT" "$REMOTE" "$BRANCH"
git push "$REMOTE" "HEAD:refs/heads/${BRANCH}"

printf 'Public repo updated successfully.\n'
