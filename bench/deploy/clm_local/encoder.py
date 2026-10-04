# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.7",
#     "transformers>=4.56,<6",
#     "fastapi>=0.115",
#     "uvicorn>=0.30",
#     "numpy>=1.26",
# ]
# ///
"""CLM's encoder on a Mac: Qwen3-8B last-token embeddings behind the OpenAI `/v1/embeddings` shape `clm-serve` calls.

The authors serve the encoder with vLLM (`vllm serve Qwen/Qwen3-8B --runner pooling`), which needs Linux and an
NVIDIA GPU. This replaces only that process and keeps vLLM's choices, because CLM's head was trained on them:

- bf16 weights (`--dtype float32` for a precision check), the final hidden state after the model's last norm,
  at the last token of each text;
- the tokenizer's own defaults (Qwen3 adds no special token);
- `truncate_prompt_tokens` keeps the **first** tokens: vLLM 0.30.0 truncates from the right for the pooling
  runner (`vllm/tokenizers/registry.py`, lines 140-144; from the left only for generation);
- each text is encoded on its own positions: batches are right-padded, so no position shifts.

`clm-serve` L2-normalises what it gets, so whether vLLM normalises does not matter. Run it through
`bench/deploy/clm_local/serve.sh`; `GET /health` reports the revision, the device and the library versions.

    uv run --script bench/deploy/clm_local/encoder.py --port 8090
"""

from __future__ import annotations

import argparse
import base64
import threading
from typing import Any

import numpy as np
import torch
import transformers
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

ENCODER = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BATCH = 8


class EmbeddingRequest(BaseModel):
    """The fields of an OpenAI embeddings request that `clm-serve` sends."""

    model: str = "qwen3-8b"
    input: str | list[str]
    encoding_format: str = "float"
    truncate_prompt_tokens: int | None = None


def build(device: str, *, dtype: str = "bfloat16") -> FastAPI:
    """Load the encoder on `device` in `dtype` and return the app that serves it."""
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION)
    model = AutoModel.from_pretrained(ENCODER, revision=REVISION, dtype=getattr(torch, dtype)).to(device).eval()
    lock = threading.Lock()  # one forward at a time on the GPU
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "encoder": ENCODER,
            "revision": REVISION,
            "device": device,
            "dtype": dtype,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"data": [{"id": "qwen3-8b", "object": "model"}]}

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        texts = [request.input] if isinstance(request.input, str) else request.input
        ids: list[list[int]] = [tokenizer(text)["input_ids"] for text in texts]
        if request.truncate_prompt_tokens:
            ids = [row[: request.truncate_prompt_tokens] for row in ids]
        vectors: list[np.ndarray] = []
        with lock, torch.inference_mode():
            for start in range(0, len(ids), BATCH):
                rows = ids[start : start + BATCH]
                width = max(len(row) for row in rows)
                pad = tokenizer.pad_token_id or 0
                input_ids = torch.tensor([row + [pad] * (width - len(row)) for row in rows], device=device)
                mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows], device=device)
                hidden = model(input_ids=input_ids, attention_mask=mask).last_hidden_state
                last = torch.tensor([len(row) - 1 for row in rows], device=device)
                picked = hidden[torch.arange(len(rows), device=device), last]
                vectors.extend(picked.float().cpu().numpy())
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": base64.b64encode(vector.astype(np.float32).tobytes()).decode()
                if request.encoding_format == "base64"
                else vector.astype(np.float32).tolist(),
            }
            for index, vector in enumerate(vectors)
        ]
        tokens = sum(len(row) for row in ids)
        usage = {"prompt_tokens": tokens, "total_tokens": tokens}
        return {"object": "list", "data": data, "model": request.model, "usage": usage}

    return app


def main() -> None:
    """Serve the encoder on loopback."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()
    uvicorn.run(build(args.device, dtype=args.dtype), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
