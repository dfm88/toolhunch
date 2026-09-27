"""Embedding helpers for paid runs: a persistent cache (each text is paid once), truncation, cost estimates."""

from __future__ import annotations

import hashlib
import sqlite3
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tiktoken
from genai_prices import Usage, calc_price

from toolhunch.retrieval import EmbeddingBatch
from toolhunch_bench import BENCH_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from toolhunch.retrieval import Embedder, EmbeddingKind

__all__ = ["EMBEDDING_CACHE_PATH", "CachedEmbedder", "CostEstimate", "TruncatingEmbedder", "estimate_embedding_cost"]

EMBEDDING_CACHE_PATH = BENCH_DIR / "runs" / "cache" / "embeddings.sqlite"
_LOOKUP_CHUNK = 500  # keys per `IN (...)`, well under SQLite's variable limit


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CachedEmbedder:
    """An [`Embedder`][toolhunch.retrieval.Embedder] that keeps every vector in SQLite.

    Vectors are keyed by `(model_id, kind, sha256(text))` and stored as float32, the precision
    `DenseRetriever` keeps anyway; fresh vectors are rounded the same way, so a cold and a warm run
    rank identically. Only misses reach the wrapped embedder, `chunk_size` texts at a time, and each
    chunk is stored before the next is requested: a failed run keeps what it paid for, and running
    it again resumes. `billed_tokens` adds up what the wrapped embedder reported.
    """

    def __init__(self, inner: Embedder, *, path: Path = EMBEDDING_CACHE_PATH, chunk_size: int = 2048) -> None:
        """Open (or create) the cache at `path` in front of `inner`."""
        self._inner = inner
        self._chunk_size = chunk_size
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (model_id TEXT NOT NULL, kind TEXT NOT NULL, "
            "text_sha256 TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY (model_id, kind, text_sha256))"
        )
        self.billed_tokens = 0

    @property
    def model_id(self) -> str:
        """The wrapped embedder's model id."""
        return self._inner.model_id

    def __repr__(self) -> str:
        return f"CachedEmbedder({self._inner!r})"

    async def embed(self, texts: Sequence[str], /, *, kind: EmbeddingKind) -> EmbeddingBatch:
        """Embed `texts`, asking the wrapped embedder only for texts not in the cache."""
        keys = [_key(text) for text in texts]
        vectors = self._load(keys, kind=kind)
        pending = list({key: text for key, text in zip(keys, texts, strict=True) if key not in vectors}.items())
        billed = 0
        for start in range(0, len(pending), self._chunk_size):
            chunk = pending[start : start + self._chunk_size]
            batch = await self._inner.embed([text for _, text in chunk], kind=kind)
            billed += batch.input_tokens
            self.billed_tokens += batch.input_tokens
            stored = [(key, array("f", vector)) for (key, _), vector in zip(chunk, batch.vectors, strict=True)]
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                [(self.model_id, kind, key, vector.tobytes()) for key, vector in stored],
            )
            self._db.commit()
            vectors.update((key, tuple(vector)) for key, vector in stored)
        return EmbeddingBatch(vectors=tuple(vectors[key] for key in keys), input_tokens=billed)

    def missing(self, texts: Iterable[str], *, kind: EmbeddingKind) -> list[str]:
        """The distinct `texts` with no cached vector, in first-seen order."""
        unique = {_key(text): text for text in texts}
        cached = {key for key, _ in self._rows(list(unique), kind=kind, with_vectors=False)}
        return [text for key, text in unique.items() if key not in cached]

    def close(self) -> None:
        """Close the database; the wrapped embedder is left open."""
        self._db.close()

    def _load(self, keys: Sequence[str], *, kind: EmbeddingKind) -> dict[str, tuple[float, ...]]:
        vectors: dict[str, tuple[float, ...]] = {}
        for key, blob in self._rows(list(dict.fromkeys(keys)), kind=kind, with_vectors=True):
            vector = array("f")
            vector.frombytes(blob)
            vectors[key] = tuple(vector)
        return vectors

    def _rows(self, keys: list[str], *, kind: EmbeddingKind, with_vectors: bool) -> Iterable[tuple[str, bytes]]:
        columns = "text_sha256, vector" if with_vectors else "text_sha256, NULL"
        for start in range(0, len(keys), _LOOKUP_CHUNK):
            chunk = keys[start : start + _LOOKUP_CHUNK]
            yield from self._db.execute(
                f"SELECT {columns} FROM embeddings WHERE model_id = ? AND kind = ? "
                f"AND text_sha256 IN ({', '.join('?' * len(chunk))})",
                (self.model_id, kind, *chunk),
            )


class TruncatingEmbedder:
    """An [`Embedder`][toolhunch.retrieval.Embedder] that cuts each text to its first `max_tokens` tokens.

    OpenAI's embedding models reject an input over 8,192 tokens (HTTP 400) instead of truncating it;
    counting with the model's own tiktoken encoding makes the cut exact. The model id is unchanged:
    texts within the limit embed exactly as before.
    """

    def __init__(self, inner: Embedder, *, max_tokens: int, encoding: tiktoken.Encoding) -> None:
        """Wrap `inner`, counting tokens with `encoding` (`cl100k_base` for `text-embedding-3-*`)."""
        self._inner = inner
        self._max_tokens = max_tokens
        self._encoding = encoding

    @property
    def model_id(self) -> str:
        """The wrapped embedder's model id."""
        return self._inner.model_id

    def __repr__(self) -> str:
        return f"TruncatingEmbedder({self._inner!r}, max_tokens={self._max_tokens}, encoding={self._encoding.name!r})"

    async def embed(self, texts: Sequence[str], /, *, kind: EmbeddingKind) -> EmbeddingBatch:
        """Embed `texts`, each cut to at most `max_tokens` tokens."""
        return await self._inner.embed([self._cut(text) for text in texts], kind=kind)

    def _cut(self, text: str) -> str:
        tokens = self._encoding.encode(text, disallowed_special=())
        return text if len(tokens) <= self._max_tokens else self._encoding.decode(tokens[: self._max_tokens])


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """What embedding the texts missing from the cache would cost.

    `max_tokens_per_text` is the longest text before truncation; `truncated` counts the texts over
    the cap, whose tokens are counted as the cap.
    """

    texts: int
    tokens: int
    usd: float
    max_tokens_per_text: int
    truncated: int


def estimate_embedding_cost(
    texts: Iterable[str],
    *,
    model: str,
    cache: CachedEmbedder | None,
    kind: EmbeddingKind = "document",
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
) -> CostEstimate:
    """Estimate the cost of embedding the distinct `texts` that `cache` does not hold yet.

    Tokens are counted with `count_tokens`, by default tiktoken's `cl100k_base` (the tokenizer of
    OpenAI's `text-embedding-3-*` models), capped at `max_tokens` per text when given, and priced
    with `genai-prices` for OpenAI.
    """
    counter = count_tokens if count_tokens is not None else _cl100k_tokens
    pending = cache.missing(texts, kind=kind) if cache is not None else list(dict.fromkeys(texts))
    counts = [counter(text) for text in pending]
    cap = max_tokens if max_tokens is not None else max(counts, default=0)
    billed = sum(min(count, cap) for count in counts)
    price = calc_price(Usage(input_tokens=billed), model_ref=model, provider_id="openai")
    return CostEstimate(
        texts=len(pending),
        tokens=billed,
        usd=float(price.total_price),
        max_tokens_per_text=max(counts, default=0),
        truncated=sum(count > cap for count in counts),
    )


def _cl100k_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
