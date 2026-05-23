#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-origin/codex/fix_base_model_protocol}"
REPO_URL="${REPO_URL:-https://github.com/EnemoCE/long_rlp.git}"
REPO_DIR="${REPO_DIR:-verl}"

if [[ "$BRANCH" != origin/* ]]; then
  echo "Expected branch argument like origin/codex/fix_base_model_protocol, got: $BRANCH" >&2
  exit 2
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  set +x
  git config --global credential.helper 'store --file ~/.git-credentials'
  git config --global credential.https://github.com.useHttpPath true
  umask 077

  if [[ ! -f ~/.git-credentials ]]; then
    read -rsp "GitHub PAT for ${REPO_URL}: " GITHUB_PAT
    echo
    printf "protocol=https\nhost=github.com\npath=EnemoCE/long_rlp.git\nusername=x-access-token\npassword=%s\n\n" \
      "$GITHUB_PAT" | git credential approve
    unset GITHUB_PAT
    chmod 600 ~/.git-credentials
  fi

  git clone "$REPO_URL" "$REPO_DIR"
else
  echo "Repository already exists at $REPO_DIR; reusing it."
fi

cd "$REPO_DIR"
git fetch origin

LOCAL_BRANCH="${BRANCH#origin/}"
if git show-ref --verify --quiet "refs/heads/${LOCAL_BRANCH}"; then
  git switch "$LOCAL_BRANCH"
else
  git switch --track "$BRANCH"
fi
git pull --ff-only

echo
echo "Repository ready: $(pwd)"
echo "Branch: $(git branch --show-current)"
echo
echo "Next step:"
echo "  bash exps/scripts/setup_prime_env.sh"
