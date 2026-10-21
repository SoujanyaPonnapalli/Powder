#!/usr/bin/env bash
# One unattended Claude Code work session on the VM (started by powder-runner.service).
# Works on branch $POWDER_WORK_BRANCH, does the next unchecked item of the
# progress checklist in docs/plans/exec-01-canonical-failure-data.md (§8), commits, pushes.
set -euo pipefail
source "$HOME/.powder-runner.env"
cd "$HOME/Powder"
export PATH="$HOME/.cargo/bin:$PATH"
LOG=$HOME/powder-logs; mkdir -p "$LOG"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BRANCH=${POWDER_WORK_BRANCH:-exec01-fault-library-vm}
BASE=${POWDER_BASE_BRANCH:-exec01-fault-library}

if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${CLAUDE_CODE_USE_BEDROCK:-}" ]]; then
  echo "no Claude auth configured in ~/.powder-runner.env" | tee -a "$LOG/runner-$STAMP.log"; exit 0
fi

git fetch -q origin
if git show-ref -q "refs/remotes/origin/$BRANCH"; then
  git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" "origin/$BRANCH"
  git pull -q --rebase origin "$BRANCH"
else
  git checkout -q -B "$BRANCH" "origin/$BASE"
fi

if ! grep -q '^- \[ \]' docs/plans/exec-01-canonical-failure-data.md; then
  echo "checklist complete; nothing to do" >> "$LOG/runner-$STAMP.log"; exit 0
fi

timeout "${POWDER_SESSION_TIMEOUT:-3h}" claude -p "$(cat infra/aws/runner_prompt.md)" \
  --permission-mode acceptEdits \
  --allowedTools "Bash" "Read" "Edit" "Write" "Glob" "Grep" \
  --max-turns "${POWDER_MAX_TURNS:-200}" \
  --output-format text > "$LOG/runner-$STAMP.log" 2>&1 || echo "claude exited $?" >> "$LOG/runner-$STAMP.log"

git push -q origin "$BRANCH" || echo "push failed" >> "$LOG/runner-$STAMP.log"
