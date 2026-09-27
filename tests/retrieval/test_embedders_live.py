import pytest

from toolhunch.retrieval import OpenAIEmbedder

pytestmark = [pytest.mark.anyio, pytest.mark.live]


async def test_openai_embeddings(openai_api_key: str) -> None:
    embedder = OpenAIEmbedder("text-embedding-3-small", api_key=openai_api_key)
    try:
        batch = await embedder.embed(["send an email"], kind="query")
    finally:
        await embedder.aclose()
    assert len(batch.vectors) == 1
    assert len(batch.vectors[0]) == 1536
    assert batch.input_tokens > 0
