#!/usr/bin/env bash
# rizzo-flow (Rizzo AI Academy) as a local `/v1/systemone` server, for the benchmark's `rizzo` arms.
#
#     bench/deploy/rizzo_local.sh                  # llama.cpp on Metal (Apple silicon), http://127.0.0.1:8017
#     bench/deploy/rizzo_local.sh --port 8018
#     Ctrl-C (or kill the script) to stop it; the trap stops the server it started
#
# Pinned for the benchmark's provenance:
# - code: Rizzo-AI-Academy/rizzo-flow @ b9ba007ee4d2928bbab5b1d8bfe9009c3696b6de (not on PyPI, so it installs from git);
# - weights: rizzoaiacademy/rizzo-flow @ 55633c8, spark-x2.5-4b-rizzo-flow-lora-q8_0.gguf (~4.4 GB), which
#   `rizzo download` checks against the SHA-256 it pins;
# - runtime: llama.cpp b11081 (161755f), downloaded and checked by `rizzo download`.
#
# Every file lands in bench/models/rizzo (git-ignored): the GGUF there, and the llama.cpp runtime in its `runtimes/`
# folder, because `rizzo` resolves `runtimes/` against the working directory and has no flag for it, so the script
# runs from that folder. `--destination` is the weights FILE, not a folder. The server binds to loopback.
# `--ctx 8192` makes a question over 8,192 tokens (state, question and the prompt template) fail with HTTP 422
# instead of losing its state. The wire accepts at most 26 options per choice question.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CODE="git+https://github.com/Rizzo-AI-Academy/rizzo-flow@b9ba007ee4d2928bbab5b1d8bfe9009c3696b6de"
HOME_DIR="$REPO/bench/models/rizzo"
WEIGHTS="$HOME_DIR/spark-x2.5-4b-rizzo-flow-lora-q8_0.gguf"
PORT=8017

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# `rizzo download` fetches with urllib, and a python.org build of Python has no CA bundle until its
# "Install Certificates" step has run (CERTIFICATE_VERIFY_FAILED); macOS ships one.
if [[ -z "${SSL_CERT_FILE:-}" && -f /etc/ssl/cert.pem ]]; then
    export SSL_CERT_FILE=/etc/ssl/cert.pem
fi

mkdir -p "$HOME_DIR"
cd "$HOME_DIR"
rizzo() { uvx --python 3.12 --from "$CODE" rizzo "$@"; }

# Idempotent: both files are fetched only when missing or failing their SHA-256.
rizzo download --size 4b --quant q8_0 --runtime metal --destination "$WEIGHTS"

rizzo serve --model "$WEIGHTS" --device metal --ctx 8192 --host 127.0.0.1 --port "$PORT" &
SERVER=$!
stop() {
    trap - EXIT INT TERM
    # uvx runs the server as its child: stop the child first, then uvx, then wait for Metal to release the GPU.
    pkill -TERM -P "$SERVER" 2>/dev/null || true
    kill -TERM "$SERVER" 2>/dev/null || true
    wait "$SERVER" 2>/dev/null || true
}
trap stop EXIT INT TERM
wait "$SERVER"
