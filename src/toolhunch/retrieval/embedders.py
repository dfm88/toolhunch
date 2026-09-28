"""Text embedders for dense retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx2
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Embedder", "EmbeddingBatch", "EmbeddingError", "EmbeddingKind", "OpenAIEmbedder"]

type EmbeddingKind = Literal["query", "document"]
"""Whether a text is a search query or an indexed document; some models embed them differently."""


class EmbeddingError(Exception):
    """An embeddings request could not be made or failed."""


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Vectors in input order and the tokens the provider billed for them."""

    vectors: tuple[tuple[float, ...], ...]
    input_tokens: int


class Embedder(Protocol):
    """Turns texts into vectors."""

    @property
    def model_id(self) -> str:
        """Identifies the vector space; vectors from different ids must never be mixed."""
        ...

    async def embed(self, texts: Sequence[str], /, *, kind: EmbeddingKind) -> EmbeddingBatch:
        """Embed `texts`, returning one vector per text in the same order."""
        ...


class _Item(BaseModel):
    index: int
    embedding: list[float]


class _Usage(BaseModel):
    prompt_tokens: int


class _Response(BaseModel):
    data: list[_Item]
    usage: _Usage


class OpenAIEmbedder:
    """Embeddings from an OpenAI-compatible `POST {base_url}/embeddings` endpoint.

    Speaks the wire protocol directly over httpx2, so any compatible server works through
    `base_url`. The key is read from `api_key` or, at call time, from the `api_key_env` variable;
    it is never logged or shown in `repr`. Pass `api_key_env=None` for servers without auth. `kind`
    is ignored: OpenAI models embed queries and documents alike.

    Inputs longer than `max_input_bytes` UTF-8 bytes are cut before sending, since the API rejects an
    over-long input instead of truncating it, and one such card would fail every index build.
    OpenAI's `text-embedding-3-*` models accept 8,191 tokens per input (OpenAI cookbook,
    "Embedding texts that are longer than the model's maximum context length", checked 2026-09-28);
    a byte-level BPE token covers at least one byte, so the default can never exceed that limit.
    English runs about four bytes per token, so text past roughly 2,000 tokens loses its tail: raise
    the limit for servers that accept more, or pass `None` when texts are already cut.

    No retries yet: a failed request raises [`EmbeddingError`][toolhunch.retrieval.EmbeddingError].
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        dimensions: int | None = None,
        batch_size: int = 512,
        max_input_bytes: int | None = 8191,
        http_client: httpx2.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Configure the endpoint; `http_client` is used as is and never closed by the embedder."""
        self._model = model
        self._url = base_url.rstrip("/") + "/embeddings"
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_input_bytes = max_input_bytes
        self._client = http_client
        self._owns_client = http_client is None
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        """`model`, or `model@dimensions` when the output is shortened."""
        return self._model if self._dimensions is None else f"{self._model}@{self._dimensions}"

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self._model!r}, url={self._url!r}, dimensions={self._dimensions!r})"

    async def embed(self, texts: Sequence[str], /, *, kind: EmbeddingKind) -> EmbeddingBatch:
        """Embed `texts` in batches of `batch_size`; see [`Embedder.embed`][toolhunch.retrieval.Embedder.embed]."""
        if not texts:
            return EmbeddingBatch(vectors=(), input_tokens=0)
        key = self._key()
        headers = {} if key is None else {"Authorization": f"Bearer {key}"}
        client = self._http_client()
        vectors: list[tuple[float, ...]] = []
        tokens = 0
        for start in range(0, len(texts), self._batch_size):
            batch = [self._fit(text) for text in texts[start : start + self._batch_size]]
            payload: dict[str, Any] = {"model": self._model, "input": batch, "encoding_format": "float"}
            if self._dimensions is not None:
                payload["dimensions"] = self._dimensions
            try:
                response = await client.post(self._url, json=payload, headers=headers)
            except httpx2.HTTPError as error:  # transport failures: connection, timeout, protocol
                raise EmbeddingError(f"embeddings request failed: {type(error).__name__}: {error}") from error
            # Bodies are redacted: a proxy may echo the request, key included.
            if response.status_code >= 400:
                body = _redact(response.text, key)[:500]
                raise EmbeddingError(f"embeddings request failed: HTTP {response.status_code}: {body}")
            try:
                parsed = _Response.model_validate_json(response.content)
            except ValidationError as error:
                raise EmbeddingError(f"unexpected embeddings response: {_redact(str(error), key)}") from None
            if len(parsed.data) != len(batch):
                raise EmbeddingError(f"asked for {len(batch)} embeddings, got {len(parsed.data)}")
            vectors.extend(tuple(item.embedding) for item in sorted(parsed.data, key=lambda item: item.index))
            tokens += parsed.usage.prompt_tokens
        return EmbeddingBatch(vectors=tuple(vectors), input_tokens=tokens)

    async def aclose(self) -> None:
        """Close the HTTP client if the embedder created it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _fit(self, text: str) -> str:
        if self._max_input_bytes is None or len(encoded := text.encode()) <= self._max_input_bytes:
            return text
        return encoded[: self._max_input_bytes].decode(errors="ignore")  # drops a character cut in half

    def _key(self) -> str | None:
        if self._api_key is not None or self._api_key_env is None:
            return self._api_key
        if not (key := os.environ.get(self._api_key_env)):
            raise EmbeddingError(f"no API key: set {self._api_key_env} or pass api_key")
        return key

    def _http_client(self) -> httpx2.AsyncClient:
        if self._client is None:
            self._client = httpx2.AsyncClient(timeout=self._timeout)
        return self._client


def _redact(text: str, key: str | None) -> str:
    return text if key is None else text.replace(key, "***")
