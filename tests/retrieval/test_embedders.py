import json

import httpx2
import pytest

from toolhunch.retrieval import EmbeddingError, OpenAIEmbedder

pytestmark = pytest.mark.anyio

SECRET = "sk-test-not-a-real-key"


def fake_openai(requests: list[httpx2.Request], *, status: int = 200) -> httpx2.AsyncClient:
    """An OpenAI-shaped /embeddings endpoint that returns the vectors in reverse order."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if status != 200:
            return httpx2.Response(status, json={"error": {"message": "invalid api key"}})
        texts: list[str] = json.loads(request.content)["input"]
        data = [{"index": i, "embedding": [float(len(text)), float(i)]} for i, text in enumerate(texts)]
        return httpx2.Response(200, json={"data": data[::-1], "usage": {"prompt_tokens": 10 * len(texts)}})

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def test_wire_format_batching_and_order() -> None:
    requests: list[httpx2.Request] = []
    embedder = OpenAIEmbedder(
        "text-embedding-3-small",
        api_key=SECRET,
        dimensions=256,
        batch_size=2,
        base_url="https://example.test/v1",
        http_client=fake_openai(requests),
    )

    batch = await embedder.embed(["a", "bb", "ccc"], kind="document")

    assert batch.vectors == ((1.0, 0.0), (2.0, 1.0), (3.0, 0.0))  # input order despite reversed data
    assert batch.input_tokens == 30
    assert [str(request.url) for request in requests] == ["https://example.test/v1/embeddings"] * 2
    assert requests[0].headers["authorization"] == f"Bearer {SECRET}"
    assert json.loads(requests[0].content) == {
        "model": "text-embedding-3-small",
        "input": ["a", "bb"],
        "encoding_format": "float",
        "dimensions": 256,
    }
    assert embedder.model_id == "text-embedding-3-small@256"


async def test_no_texts_no_request() -> None:
    requests: list[httpx2.Request] = []
    batch = await OpenAIEmbedder(api_key=SECRET, http_client=fake_openai(requests)).embed([], kind="query")
    assert (batch.vectors, batch.input_tokens, requests) == ((), 0, [])


async def test_errors_never_leak_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOLHUNCH_TEST_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="TOOLHUNCH_TEST_KEY") as missing:
        await OpenAIEmbedder(api_key_env="TOOLHUNCH_TEST_KEY", http_client=fake_openai([])).embed(["a"], kind="query")

    embedder = OpenAIEmbedder(api_key=SECRET, http_client=fake_openai([], status=401))
    with pytest.raises(EmbeddingError, match="401") as rejected:
        await embedder.embed(["a"], kind="query")

    for text in (str(missing.value), str(rejected.value), repr(embedder)):
        assert SECRET not in text


async def test_inputs_are_cut_to_the_declared_byte_limit() -> None:
    requests: list[httpx2.Request] = []
    long_ascii, long_accented = "x" * 9000, "é" * 5000  # 9,000 and 10,000 UTF-8 bytes

    await OpenAIEmbedder(api_key=SECRET, http_client=fake_openai(requests)).embed(
        [long_ascii, long_accented, "ok"], kind="document"
    )
    await OpenAIEmbedder(api_key=SECRET, max_input_bytes=None, http_client=fake_openai(requests)).embed(
        [long_ascii], kind="document"
    )

    cut, uncut = (json.loads(request.content)["input"] for request in requests)
    assert cut == ["x" * 8191, "é" * 4095, "ok"]  # at most 8,191 bytes, never inside a character
    assert uncut == [long_ascii]


async def test_transport_failures_and_echoed_keys_surface_as_embedding_errors() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    def echo(request: httpx2.Request) -> httpx2.Response:  # some proxies echo the request back
        return httpx2.Response(400, text=f"bad request, headers: {dict(request.headers)}")

    for handler in (refuse, echo):
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        with pytest.raises(EmbeddingError) as failure:
            await OpenAIEmbedder(api_key=SECRET, http_client=client).embed(["a"], kind="query")
        assert SECRET not in str(failure.value)
