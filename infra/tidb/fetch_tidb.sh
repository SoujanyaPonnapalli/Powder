#!/usr/bin/env bash
# Download the pinned TiDB-ecosystem tarballs and verify them against third_party/tidb/VERSIONS.
#
#   infra/tidb/fetch_tidb.sh                    # fetch every pinned artifact into the cache
#   infra/tidb/fetch_tidb.sh pd tikv            # only these components
#   infra/tidb/fetch_tidb.sh --check            # verify what is already cached; download nothing
#   infra/tidb/fetch_tidb.sh --extract pd       # also unpack to <dest>/<component>-<version>/
#   infra/tidb/fetch_tidb.sh --print-version    # the pinned TiDB version, e.g. v8.5.8
#
# Cache: ${POWDER_CACHE:-$HOME/.cache/powder}/tidb/<tidb-version>/ (never inside the repo; override with --dest).
# Mirror: ${POWDER_TIUP_MIRROR:-https://tiup-mirrors.pingcap.com}.
# A tarball whose sha256 differs from the pin is deleted and the script exits non-zero: the mirror must not
# change an artifact under a published version, and if it does we want to know before running experiments.
# Re-pin deliberately with infra/tidb/pin_versions.py. Platform: linux/amd64 only (the AWS experiment hosts).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
VERSIONS=${POWDER_TIDB_VERSIONS:-$ROOT/third_party/tidb/VERSIONS}
MIRROR=${POWDER_TIUP_MIRROR:-https://tiup-mirrors.pingcap.com}

CHECK=0 EXTRACT=0 DEST="" WANT=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK=1 ;;
    --extract) EXTRACT=1 ;;
    --dest) DEST=$2; shift ;;
    --print-version) awk '$1=="tidb"{print $2}' "$VERSIONS"; exit 0 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) WANT+=("$1") ;;
  esac
  shift
done

[[ -f "$VERSIONS" ]] || { echo "missing $VERSIONS (run infra/tidb/pin_versions.py)" >&2; exit 1; }
TIDB_VERSION=$(awk '$1=="tidb"{print $2}' "$VERSIONS")
[[ -n "$TIDB_VERSION" ]] || { echo "no tidb row in $VERSIONS" >&2; exit 1; }
DEST=${DEST:-${POWDER_CACHE:-$HOME/.cache/powder}/tidb/$TIDB_VERSION}
mkdir -p "$DEST"

log() { printf '\033[1m[fetch_tidb]\033[0m %s\n' "$*"; }

wanted() {
  [[ ${#WANT[@]} -eq 0 ]] && return 0
  local w; for w in "${WANT[@]}"; do [[ "$w" == "$1" ]] && return 0; done
  return 1
}

sha_of() { sha256sum "$1" | awk '{print $1}'; }

status=0 seen=0
while read -r comp ver plat sha size url; do
  [[ -z "${comp:-}" || "$comp" == \#* ]] && continue
  wanted "$comp" || continue
  seen=$((seen + 1))
  file=$DEST/$(basename "$url")

  if [[ -f "$file" && "$(sha_of "$file")" == "$sha" ]]; then
    log "ok      $comp $ver (cached, hash matches)"
  elif [[ $CHECK -eq 1 ]]; then
    log "MISSING or WRONG HASH: $file"; status=1; continue
  else
    log "fetch   $comp $ver ($((size / 1000000)) MB)"
    rm -f "$file" "$file.part"
    curl -fsSL --retry 3 --retry-delay 2 -o "$file.part" "$MIRROR$url"
    got=$(sha_of "$file.part")
    if [[ "$got" != "$sha" ]]; then
      rm -f "$file.part"
      echo "HASH MISMATCH for $comp $ver: pinned $sha, mirror served $got." >&2
      echo "The mirror changed an artifact under a published version. Investigate; do not just re-pin." >&2
      exit 1
    fi
    mv "$file.part" "$file"
  fi

  if [[ $EXTRACT -eq 1 ]]; then
    out=$DEST/$comp-$ver
    if [[ ! -d "$out" ]]; then
      mkdir -p "$out.part" && tar -xzf "$file" -C "$out.part" && mv "$out.part" "$out"
      log "extract $comp -> $out"
    fi
  fi
done < "$VERSIONS"

if [[ $seen -eq 0 ]]; then echo "no pinned component matches: ${WANT[*]:-<none>}" >&2; exit 1; fi
if [[ $status -ne 0 ]]; then exit $status; fi
log "done. cache: $DEST"
