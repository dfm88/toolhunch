#!/usr/bin/env bash
# Strands Decider 2B (AWS Strands Labs) as a local `/v1/systemone` server, for the benchmark's `strands` arms.
#
#     bench/deploy/strands_local.sh                  # MLX on Apple silicon, http://127.0.0.1:8000
#     bench/deploy/strands_local.sh --device mps --port 8001
#     Ctrl-C (or kill the process) to stop it
#
# Pinned for the benchmark's provenance:
# - code: strands-labs/strands-decider @ 75c9fd3 (the mlx extra is not on PyPI yet, so it installs from git);
# - checkpoint: StrandsAgents/strands-decider-2B-hobson-v19 @ bb282d7, downloaded to bench/models/ and served from
#   there, because `serve` takes a path or a Hub id without a revision;
# - base: Qwen/Qwen3.5-2B-Base @ b1485b2, the revision the checkpoint's provenance.json names, which the loader uses.
#
# Every file lands in bench/models/ (git-ignored). The server binds to loopback and has no authentication.
# `--strict-window` makes a prompt over the 4,096-token window fail with HTTP 422 instead of losing its state.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CODE="git+https://github.com/strands-labs/strands-decider@75c9fd32e664954cdc18481434018aa507eee8fb"
CHECKPOINT="StrandsAgents/strands-decider-2B-hobson-v19"
REVISION="bb282d786bc251fd4e3068de3ada9ddbb38127cd"
MODEL_NAME="strands-decider-2B-hobson-v19"
DEVICE="mlx"
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

export HF_HUB_CACHE="$REPO/bench/models/hf"
LOCAL="$REPO/bench/models/$MODEL_NAME"
if [[ ! -f "$LOCAL/.revision" || "$(cat "$LOCAL/.revision")" != "$REVISION" ]]; then
    uvx --from "huggingface_hub[cli]>=1.5" hf download "$CHECKPOINT" --revision "$REVISION" --local-dir "$LOCAL"
    echo "$REVISION" > "$LOCAL/.revision"
fi

EXTRA="mlx"
[[ "$DEVICE" == "mlx" ]] || EXTRA="$DEVICE"
exec uvx --python 3.12 --from "strands-decider[$EXTRA] @ $CODE" \
    strands-decider serve "$LOCAL" --device "$DEVICE" --strict-window --port "$PORT" --model-name "$MODEL_NAME"
