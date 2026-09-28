from collections.abc import Sequence
from pathlib import Path

import pytest
import tiktoken

from toolhunch.retrieval import EmbeddingBatch, EmbeddingKind
from toolhunch_bench.embedding_cache import CachedEmbedder, TruncatingEmbedder, estimate_embedding_cost

pytestmark = pytest.mark.anyio


class CountingEmbedder:
    """Vector = (characters, words, 1/3); one billed token per word. Records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[EmbeddingKind, list[str]]] = []

    @property
    def model_id(self) -> str:
        return "fake-model"

    async def embed(self, texts: Sequence[str], /, *, kind: EmbeddingKind) -> EmbeddingBatch:
        self.calls.append((kind, list(texts)))
        return EmbeddingBatch(
            vectors=tuple((float(len(text)), float(len(text.split())), 1 / 3) for text in texts),
            input_tokens=sum(len(text.split()) for text in texts),
        )


async def test_only_misses_are_embedded_and_the_cache_survives_a_reopen(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    inner = CountingEmbedder()
    cache = CachedEmbedder(inner, path=path, chunk_size=2)

    first = await cache.embed(["get weather", "send email", "get weather", "read diary"], kind="document")
    second = await cache.embed(["send email", "list users now"], kind="document")
    await cache.embed(["send email"], kind="query")  # queries are kept apart from documents
    cache.close()

    assert inner.calls == [
        ("document", ["get weather", "send email"]),  # chunks of 2, each stored before the next is asked
        ("document", ["read diary"]),
        ("document", ["list users now"]),
        ("query", ["send email"]),
    ]
    assert (first.input_tokens, second.input_tokens, cache.billed_tokens) == (6, 3, 11)
    assert first.vectors[0] == first.vectors[2]

    reopened_inner = CountingEmbedder()
    reopened = CachedEmbedder(reopened_inner, path=path)
    again = await reopened.embed(["read diary", "get weather"], kind="document")
    reopened.close()

    assert reopened_inner.calls == []
    assert again == EmbeddingBatch(vectors=(first.vectors[3], first.vectors[0]), input_tokens=0)  # float32 either way


async def test_estimate_prices_only_what_is_not_cached(tmp_path: Path) -> None:
    cache = CachedEmbedder(CountingEmbedder(), path=tmp_path / "embeddings.sqlite")
    await cache.embed(["get weather"], kind="document")

    estimate = estimate_embedding_cost(
        ["get weather", "send an email", "send an email", "read the diary today"],
        model="text-embedding-3-small",
        cache=cache,
        kind="document",
        count_tokens=lambda text: len(text.split()),
        max_tokens=3,
    )
    cache.close()

    assert (estimate.texts, estimate.tokens, estimate.max_tokens_per_text, estimate.truncated) == (2, 6, 4, 1)
    assert estimate.usd == pytest.approx(6 * 0.02 / 1_000_000)  # $0.02 per million input tokens


async def test_long_texts_are_cut_to_the_input_limit() -> None:
    byte_level = tiktoken.Encoding(
        name="bytes", pat_str=r"\S+|\s+", mergeable_ranks={bytes([b]): b for b in range(256)}, special_tokens={}
    )  # one token per byte, built offline
    inner = CountingEmbedder()
    embedder = TruncatingEmbedder(inner, max_tokens=5, encoding=byte_level)

    await embedder.embed(["short", "much longer text", "\u00e9\u00e9\u00e9"], kind="document")

    # Five bytes of "\u00e9\u00e9\u00e9" end inside a character and decode to "\u00e9\u00e9\ufffd" (7 tokens): back off.
    assert inner.calls == [("document", ["short", "much ", "\u00e9\u00e9"])]
    assert embedder.model_id == inner.model_id
