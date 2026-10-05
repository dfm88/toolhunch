#!/usr/bin/env bash
# rizzo-flow (Rizzo AI Academy) as a local `/v1/systemone` server, for the benchmark's `rizzo` arms.
#
#     bench/deploy/rizzo_local.sh                  # llama.cpp on Metal (Apple silicon), http://127.0.0.1:8017
#     bench/deploy/rizzo_local.sh --port 8018
#     Ctrl-C (or SIGTERM to the script) to stop it: the trap stops the server and everything under it
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
# The script owns the RIZZO_* variables: it unsets every one it inherits (RIZZO_LLAMA_DIR would swap the pinned
# llama.cpp for another build, RIZZO_API_KEY would add a bearer check), so a caller's environment cannot change the
# pins. On exit it asks the server to stop, waits up to 30 s, then kills whatever is still alive.
# A shell that starts the script with `&` makes bash ignore SIGINT, and an ignored signal cannot be trapped: stop it
# with SIGTERM there.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CODE="git+https://github.com/Rizzo-AI-Academy/rizzo-flow@b9ba007ee4d2928bbab5b1d8bfe9009c3696b6de"
MODELS_DIR="$REPO/bench/models/rizzo"
WEIGHTS="$MODELS_DIR/spark-x2.5-4b-rizzo-flow-lora-q8_0.gguf"
PORT=8017

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

for name in $(compgen -v RIZZO_); do
    unset "$name"
done

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"
RIZZO=(uvx --python 3.12 --from "$CODE" rizzo)

# `rizzo download` fetches with urllib, and a python.org build of Python has no CA bundle until its
# "Install Certificates" step has run (CERTIFICATE_VERIFY_FAILED); macOS ships one. Only this step gets it.
# Idempotent: both files are fetched only when missing or failing their SHA-256.
SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/cert.pem}" \
    "${RIZZO[@]}" download --size 4b --quant q8_0 --runtime metal --destination "$WEIGHTS"

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
    kill -TERM "$SERVER" 2>/dev/null  # uvx forwards it to rizzo and waits for Metal to release the GPU
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
"${RIZZO[@]}" serve --model "$WEIGHTS" --device metal --ctx 8192 --host 127.0.0.1 --port "$PORT" &
SERVER=$!
trap stop EXIT INT TERM
wait "$SERVER"
