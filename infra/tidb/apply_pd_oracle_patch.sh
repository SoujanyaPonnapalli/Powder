#!/usr/bin/env bash
# Apply the opt-in Powder repair hook to the pinned, clean PD submodule.
# The patch lives in the parent repository so the upstream submodule pin stays reproducible.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PD=$(cd "$HERE/../../third_party/pd" && pwd)
PATCH="$HERE/patches/pd-powder-repair.patch"
BASE=4ecadf2f59e4fa04fe0e611c6d7808e8b1806ce7

case "${1:---apply}" in
  --apply|--check) ;;
  *) printf 'usage: %s [--check|--apply]\n' "$0" >&2; exit 2 ;;
esac
if [[ "$(git -C "$PD" rev-parse HEAD)" != "$BASE" ]]; then
  printf 'PD must be checked out at %s\n' "$BASE" >&2
  exit 1
fi
if [[ -n "$(git -C "$PD" status --porcelain)" ]]; then
  printf 'PD checkout has local changes; refusing to overwrite them\n' >&2
  exit 1
fi
git -C "$PD" apply --check "$PATCH"
if [[ "${1:---apply}" == --check ]]; then
  printf 'Powder PD patch applies cleanly\n'
else
  git -C "$PD" apply "$PATCH"
  printf 'Powder PD repair hook applied (not committed in the submodule)\n'
fi
