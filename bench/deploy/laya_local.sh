#!/usr/bin/env bash
# Laya (ConvAI Innovations) as a local `/v1/systemone` server, for the benchmark's `laya` arms.
#
#     bench/deploy/laya_local.sh                  # PyTorch on MPS (Apple silicon), http://127.0.0.1:8010
#     bench/deploy/laya_local.sh --port 8011
#     Ctrl-C (or SIGTERM to the script) to stop it: the trap stops the server and everything under it
#
# Pinned for the benchmark's provenance:
# - code: laya[serve]==0.3.27 from PyPI, run through `uvx` (the `laya-serve` entry point);
# - checkpoint: convaiinnovations/laya @ 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851, the revision `laya` itself lists
#   as reviewed; `LAYA_REVISION` pins every checkpoint load to it, so nothing moves with the Hub's default branch.
#
# Every file lands in bench/models/hf (git-ignored), through HF_HUB_CACHE, so the benchmark's pre-send check reads
# the same tokenizer.json the server uses. The server binds to loopback and has no authentication.
# The script owns the LAYA_* variables: it unsets every one it inherits (LAYA_API_KEY, LAYA_JEV_STRICT, the budget
# caps, the AMP switches...) and sets only the ones below, so a caller's environment cannot change the pins.
# `/health` therefore answers in full, and the reply keeps `routing` and the extended `usage` (input tokens,
# truncation, collapsed options) that the benchmark's strict mode reads.
# Laya answers one inference at a time behind a lock, so the benchmark asks it serially.
# On exit the script asks the server to stop, waits up to 30 s, then kills whatever is still alive.
# A shell that starts the script with `&` makes bash ignore SIGINT, and an ignored signal cannot be trapped: stop it
# with SIGTERM there.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PORT=8010

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

for name in $(compgen -v LAYA_); do
    unset "$name"
done
export HF_HUB_CACHE="$REPO/bench/models/hf"
export LAYA_HOST=127.0.0.1
export LAYA_PORT="$PORT"
export LAYA_MODELS=english
export LAYA_DEFAULT_MODEL=english
export LAYA_AUTO_TASK=0
export LAYA_REVISION=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
export LAYA_DEVICE=mps

descendants() {
    local child
    for child in $(pgrep -P "$1" 2>/dev/null); do
        echo "$child"
        descendants "$child"
    done
}

stop() {
    trap - EXIT INT TERM
    set +e
    local family pid i
    family="$(descendants "$SERVER")" # read before the stop: orphans are re-parented and lose the link
    kill -TERM "$SERVER" 2>/dev/null  # uvx forwards it to laya-serve and waits for it
    for i in $(seq 1 60); do
        kill -0 "$SERVER" 2>/dev/null || break
        sleep 0.5
    done
    for pid in "$SERVER" $family; do
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
    done
    wait "$SERVER" 2>/dev/null
}

# In the background so that the trap can run while the server does; $SERVER is uvx itself, not a subshell.
uvx --python 3.12 --from 'laya[serve]==0.3.27' laya-serve &
SERVER=$!
trap stop EXIT INT TERM
wait "$SERVER"
