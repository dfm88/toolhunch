#!/usr/bin/env bash
# CLM (Contrastive-LM/CLM-v0.1-8B) on a Mac, for the benchmark's `clm-local` arms: the authors' `clm-serve`
# unchanged, with encoder.py in place of the vLLM pooling server that needs Linux and NVIDIA.
#
#     bench/deploy/clm_local/serve.sh            # CLM on http://127.0.0.1:8700, no auth
#     Ctrl-C (or kill the process) to stop both servers
#
# Pinned: contrastive-lm 0.1.0 (installed without its vllm dependency), the reference head CLM_v0.1-8B.pt,
# Qwen/Qwen3-8B @ b968826 (in encoder.py). Everything lands in bench/models/ (git-ignored): the Hub cache, the
# head, and the virtual environment clm-serve runs in. Both servers bind to loopback only.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
MODELS="$REPO/bench/models"
VENV="$MODELS/.venv-clm"
export HF_HUB_CACHE="$MODELS/hf"
export CLM_CKPT_DIR="$MODELS/clm"
export CLM_DEVICE="mps"
unset CLM_API_KEY  # local and loopback only: no auth

if [[ ! -x "$VENV/bin/clm-serve" ]]; then
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" --no-deps "contrastive-lm==0.1.0"
    uv pip install --python "$VENV/bin/python" torch fastapi uvicorn numpy requests huggingface_hub
fi
"$VENV/bin/clm-download" --dest "$CLM_CKPT_DIR"

uv run --script "$REPO/bench/deploy/clm_local/encoder.py" --port 8090 &
ENCODER_PID=$!
CLM_PID=""
# A signal interrupts `wait` (not a foreground process), so the trap stops both servers at once.
trap 'kill $ENCODER_PID $CLM_PID 2>/dev/null; wait 2>/dev/null' EXIT INT TERM

ready="import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=5)"
until "$VENV/bin/python" -c "$ready" 2>/dev/null; do
    kill -0 "$ENCODER_PID" 2>/dev/null || { echo "the encoder exited" >&2; exit 1; }
    sleep 2
done
"$VENV/bin/clm-serve" --host 127.0.0.1 --port 8700 --emb-url http://127.0.0.1:8090/v1/embeddings --no-download &
CLM_PID=$!
wait "$CLM_PID"
