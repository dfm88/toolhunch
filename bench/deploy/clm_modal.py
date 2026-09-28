"""CLM (Contrastive-LM/CLM-v0.1-8B) as a private HTTP API on Modal, for the benchmark's CLM arms.

The authors' setup, unchanged: a vLLM Qwen3-8B pooling server feeds `clm-serve`, which answers
TypeSafe-style `POST /v1/systemone` requests. vLLM listens on localhost only; `clm-serve` is the one
public port and requires `Authorization: Bearer $CLM_API_KEY` on `/v1/*` (the playground at `/` has a
field for the key). One container at most, stopped after ten idle minutes; Qwen3-8B's weights stay in
a Volume so only the first start downloads them.

    uvx modal secret create clm CLM_API_KEY=<a long random key>   # once
    uvx modal deploy bench/deploy/clm_modal.py                     # prints the URL
    uvx modal app stop clm                                         # when done

Versions are pinned for the benchmark's provenance: vllm 0.30.0 was the latest release when
contrastive-lm 0.1.0 shipped (2026-09-24).
"""

from __future__ import annotations

import subprocess

import modal

ENCODER = "Qwen/Qwen3-8B"
EMBEDDER_PORT = 8090
CLM_PORT = 8700
MINUTES = 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0", "contrastive-lm==0.1.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_commands("clm-download")  # bakes the 75 MB reference head into ~/.cache/clm
)
hf_cache = modal.Volume.from_name("clm-hf-cache", create_if_missing=True)
HF_CACHE = "/hf-cache"  # not ~/.cache/huggingface: the image build writes there, and a Volume needs an empty path
app = modal.App("clm")

# Start the encoder, wait until it answers, then start CLM: requests reach CLM only when it can serve.
LAUNCH = f"""
HF_HOME={HF_CACHE} vllm serve {ENCODER} --served-model-name qwen3-8b --runner pooling --max-model-len 2048 \\
    --enable-prefix-caching --enforce-eager --gpu-memory-utilization 0.85 --max-num-seqs 32 \\
    --host 127.0.0.1 --port {EMBEDDER_PORT} &
ready="import urllib.request; urllib.request.urlopen('http://127.0.0.1:{EMBEDDER_PORT}/v1/models', timeout=5)"
until python -c "$ready" 2>/dev/null; do sleep 2; done
exec clm-serve --host 0.0.0.0 --port {CLM_PORT} --emb-url http://127.0.0.1:{EMBEDDER_PORT}/v1/embeddings --no-download
"""


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE: hf_cache},
    secrets=[modal.Secret.from_name("clm")],
    scaledown_window=10 * MINUTES,
    timeout=10 * MINUTES,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=CLM_PORT, startup_timeout=15 * MINUTES)
def serve() -> None:
    """Run the Qwen3-8B encoder and `clm-serve` in one container; Modal routes traffic to CLM's port."""
    subprocess.Popen(["bash", "-c", LAUNCH])
