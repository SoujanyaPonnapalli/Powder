#!/bin/bash
# Local check (no AWS): with quorum lost, does KILL QUERY end a TiDB write at the 5 s SLO, and is the connection
# reusable afterwards? Starts a tiup playground (1 PD, 3 TiKV, 1 TiDB) from the pinned tiup in ~/.cache/powder,
# waits until every Region has 3 replicas, pauses 2 TiKV for 75 s and runs two writes: one left alone, one killed
# after 5 s. Measured 2026-11-04 (v8.5.8): no kill 53.3 s ("Region is unavailable"), killed 5.03 s, both connections
# reusable; performance.max-txn-ttl = 5000 changed nothing (53.7 s).
#
#   infra/tidb/local_kill_check/run.sh [label] [tidb-config.toml]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LABEL=${1:-default}; CONF=${2:-}
WORK=$(mktemp -d)
export TIUP_HOME=$HOME/.cache/powder/tiup-home
TIUP=$TIUP_HOME/bin/tiup
GO=$(command -v go || echo "$HOME/.cache/powder/go-root/go/bin/go")
(cd "$HERE" && GOFLAGS=-mod=mod CGO_ENABLED=0 "$GO" build -o "$WORK/local_kill_check" .) || exit 1
BIN=$WORK/local_kill_check
EXTRA=()
[ -n "$CONF" ] && EXTRA=(--db.config "$CONF")
"$TIUP" playground v8.5.8 --db 1 --pd 1 --kv 3 --tiflash 0 --without-monitor --host 127.0.0.1 --tag "kc-$LABEL" \
  "${EXTRA[@]}" > "$WORK/playground.log" 2>&1 &
PG=$!
trap 'pkill -CONT -f "data/kc-$LABEL/" 2>/dev/null; kill $PG 2>/dev/null; sleep 8; pkill -9 -f "data/kc-$LABEL/" 2>/dev/null; rm -rf "$TIUP_HOME/data/kc-$LABEL"' EXIT
for i in $(seq 1 150); do "$BIN" -mode setup >/dev/null 2>&1 && break; sleep 2; done
"$BIN" -mode setup || { echo "no TiDB on :4000"; exit 1; }
echo "[$LABEL] TiDB config: $(curl -s 127.0.0.1:10080/config | python3 -c "import json,sys; c=json.load(sys.stdin); print('max-txn-ttl', c['performance']['max-txn-ttl'], 'commit-timeout', c['tikv-client']['commit-timeout'])")"
PD=$(pgrep -fa "data/kc-$LABEL/pd-0" | grep -o -- '--client-urls=[^ ]*' | head -1 | cut -d= -f2)
# a fresh playground places Regions on one store first: pausing two TiKV before replication completes can leave the
# test row's Region with its only replica alive (writes then succeed in milliseconds)
for i in $(seq 1 90); do
  miss=$(curl -s "$PD/pd/api/v1/regions/check/miss-peer" | python3 -c "import json,sys; print(json.load(sys.stdin).get('count', 1))")
  [ "$miss" = "0" ] && break; sleep 2
done
echo "[$LABEL] Regions missing a replica: $miss"
coproc KC { "$BIN" -kill-after 5s -wait 90s; }
read -r line <&"${KC[0]}"; echo "[$LABEL] $line"
pkill -STOP -f "data/kc-$LABEL/tikv-[01]/"
echo "[$LABEL] paused: $(pgrep -fa "data/kc-$LABEL/tikv-[01]/" | grep -o -- '--addr=[^ ]*' | tr '\n' ' ')"
echo paused >&"${KC[1]}"
sleep 75
pkill -CONT -f "data/kc-$LABEL/tikv-[01]/"
echo resumed >&"${KC[1]}"
while read -r line <&"${KC[0]}"; do echo "[$LABEL] $line"; done
