#!/usr/bin/env bash
# Laya (ConvAI Innovations) as a local `/v1/systemone` server, for the benchmark's `laya` arms.
#
#     bench/deploy/laya_local.sh                  # PyTorch on MPS (Apple silicon), http://127.0.0.1:8010
#     bench/deploy/laya_local.sh --port 8011
#     Ctrl-C (or kill the process) to stop it
#
# Pinned for the benchmark's provenance:
# - code: laya[serve]==0.3.27 from PyPI, run through `uvx` (the `laya-serve` entry point);
# - checkpoint: convaiinnovations/laya @ 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851, the revision `laya` itself lists
#   as reviewed; `LAYA_REVISION` pins every checkpoint load to it, so nothing moves with the Hub's default branch.
#
# Every file lands in bench/models/hf (git-ignored), through HF_HUB_CACHE, so the benchmark's pre-send check reads
# the same tokenizer.json the server uses. The server binds to loopback and has no authentication.
# Neither LAYA_API_KEY nor LAYA_JEV_STRICT is set: `/health` answers in full, and the reply keeps `routing` and the
# extended `usage` (input tokens, truncation, collapsed options) that the benchmark's strict mode reads.
# Laya answers one inference at a time behind a lock, so the benchmark asks it serially.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PORT=8010

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

export HF_HUB_CACHE="$REPO/bench/models/hf"
export LAYA_HOST=127.0.0.1
export LAYA_PORT="$PORT"
export LAYA_MODELS=english
export LAYA_DEFAULT_MODEL=english
export LAYA_AUTO_TASK=0
export LAYA_REVISION=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
export LAYA_DEVICE=mps

exec uvx --python 3.12 --from 'laya[serve]==0.3.27' laya-serve
