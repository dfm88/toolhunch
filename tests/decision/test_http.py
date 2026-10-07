import json
from collections.abc import Sequence
from typing import Any

import anyio
import httpx2
import pytest

from toolhunch.decision import DecisionError
from toolhunch.decision._http import JsonPoster

pytestmark = pytest.mark.anyio

SECRET = "sk-test-not-a-real-key"
URL = "https://h.test/x"


def make(
    responses: Sequence[httpx2.Response | Exception],
    *,
    max_retries: int = 3,
    api_key: str | None = SECRET,
    pause: float = 0.0,
) -> tuple[JsonPoster, list[httpx2.Request], list[float]]:
    """A poster over a mock transport that serves `responses` in order (an exception is raised instead).

    Returns the poster, every request it made and every wait it asked for; `pause` is how long each wait
    really lasts.
    """
    queue = list(responses)
    requests: list[httpx2.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if pause:
            await anyio.sleep(pause)

    poster = JsonPoster(
        URL,
        model_id="m@h",
        api_key=api_key,
        api_key_env=None,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        timeout=1.0,
        max_retries=max_retries,
        sleep=sleep,
    )
    return poster, requests, sleeps


def bare(
    *, api_key: str | None = None, api_key_env: str | None = None, http_client: httpx2.AsyncClient | None = None
) -> JsonPoster:
    """A poster that makes one attempt and never waits, for tests that build their own client."""
    return JsonPoster(
        URL,
        model_id="m@h",
        api_key=api_key,
        api_key_env=api_key_env,
        http_client=http_client,
        timeout=7.0,
        max_retries=0,
    )


async def test_retry_after_is_honoured_then_the_reply_returns() -> None:
    poster, requests, sleeps = make(
        [httpx2.Response(429, headers={"Retry-After": "2"}), httpx2.Response(200, json={"ok": True})]
    )
    reply = await poster.post({"x": 1})
    assert reply.body == {"ok": True}
    assert (sleeps, len(requests)) == ([2.0], 2)
    assert requests[0].headers["authorization"] == f"Bearer {SECRET}"


async def test_backoff_doubles_then_gives_up() -> None:
    poster, requests, sleeps = make([httpx2.Response(503) for _ in range(4)])
    with pytest.raises(DecisionError, match=r"m@h.*HTTP 503"):
        await poster.post({})
    assert (sleeps, len(requests)) == ([0.5, 1.0, 2.0], 4)


async def test_the_final_error_carries_the_wait_its_reply_asked_for() -> None:
    poster, _, _ = make([httpx2.Response(429, headers={"Retry-After": "7"})], max_retries=0)
    with pytest.raises(DecisionError) as caught:
        await poster.post({})
    assert (caught.value.status, caught.value.retry_after) == (429, 7.0)


@pytest.mark.parametrize("status", [400, 401, 403, 422])
async def test_client_errors_are_final(status: int) -> None:
    poster, requests, sleeps = make([httpx2.Response(status, json={"detail": "no"})])
    with pytest.raises(DecisionError, match=str(status)):
        await poster.post({})
    assert (len(requests), sleeps) == (1, [])


async def test_transport_errors_are_retried() -> None:
    poster, requests, sleeps = make([httpx2.ReadTimeout("slow"), httpx2.Response(200, json={})])
    assert (await poster.post({})).body == {}
    assert (sleeps, len(requests)) == ([0.5], 2)


async def test_retry_after_is_capped_at_30_seconds() -> None:
    poster, _, sleeps = make([httpx2.Response(429, headers={"Retry-After": "120"}), httpx2.Response(200, json={})])
    await poster.post({})
    assert sleeps == [30.0]


async def test_the_key_never_leaks(monkeypatch: pytest.MonkeyPatch) -> None:
    poster, _, _ = make([httpx2.Response(401, text=f"bad key {SECRET}")])
    with pytest.raises(DecisionError) as error:
        await poster.post({})
    assert SECRET not in str(error.value)
    assert "***" in str(error.value)
    assert SECRET not in repr(poster)
    monkeypatch.delenv("TOOLHUNCH_TEST_KEY", raising=False)
    keyless = JsonPoster(
        "https://h.test/x",
        model_id="m@h",
        api_key=None,
        api_key_env="TOOLHUNCH_TEST_KEY",
        http_client=None,
        timeout=1.0,
        max_retries=0,
    )
    with pytest.raises(DecisionError, match="TOOLHUNCH_TEST_KEY"):
        await keyless.post({})


async def test_a_non_json_body_is_an_error() -> None:
    poster, _, _ = make([httpx2.Response(200, text="<html>")])
    with pytest.raises(DecisionError, match="JSON"):
        await poster.post({})


@pytest.mark.parametrize("text", ["[1]", "null", '"text"', ""])
async def test_a_reply_that_is_not_a_json_object_is_an_error(text: str) -> None:
    poster, _, _ = make([httpx2.Response(200, text=text)])
    with pytest.raises(DecisionError, match=r"m@h: HTTP 200: not a JSON object"):
        await poster.post({})


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
async def test_transient_statuses_are_retried(status: int) -> None:
    poster, requests, sleeps = make([httpx2.Response(status), httpx2.Response(200, json={"ok": True})])
    assert (await poster.post({})).body == {"ok": True}
    assert (sleeps, len(requests)) == ([0.5], 2)


@pytest.mark.parametrize("status", [301, 501])
async def test_other_statuses_outside_the_retry_list_are_final(status: int) -> None:
    poster, requests, sleeps = make([httpx2.Response(status, text="elsewhere")])
    with pytest.raises(DecisionError, match=rf"m@h: HTTP {status}: elsewhere"):
        await poster.post({})
    assert (len(requests), sleeps) == (1, [])


async def test_exhausted_transport_errors_name_the_model_and_the_attempts() -> None:
    poster, requests, sleeps = make([httpx2.ConnectError("refused") for _ in range(4)])
    with pytest.raises(DecisionError) as error:
        await poster.post({})
    assert str(error.value) == "m@h: ConnectError after 4 attempts"
    assert (sleeps, len(requests)) == ([0.5, 1.0, 2.0], 4)


async def test_zero_retries_means_one_attempt() -> None:
    poster, requests, sleeps = make([httpx2.Response(503)], max_retries=0)
    with pytest.raises(DecisionError, match="HTTP 503"):
        await poster.post({})
    assert (len(requests), sleeps) == (1, [])


async def test_an_http_error_that_is_not_a_transport_error_is_wrapped_and_not_retried() -> None:
    poster, requests, sleeps = make([httpx2.DecodingError("corrupt gzip")])
    with pytest.raises(DecisionError, match=r"m@h: DecodingError: corrupt gzip"):
        await poster.post({})
    assert (len(requests), sleeps) == (1, [])


@pytest.mark.parametrize("value", ["soon", "Wed, 21 Oct 2026 07:28:00 GMT", "-5", "nan"])
async def test_an_unusable_retry_after_falls_back_to_the_backoff(value: str) -> None:
    poster, _, sleeps = make([httpx2.Response(429, headers={"Retry-After": value}), httpx2.Response(200, json={})])
    await poster.post({})
    assert sleeps == [0.5]


async def test_the_reply_carries_the_headers_and_the_time_of_every_attempt() -> None:
    poster, _, sleeps = make(
        [
            httpx2.Response(503, headers={"Retry-After": "0"}),
            httpx2.Response(200, json={}, headers={"X-CLM-Latency-Ms": "12"}),
        ],
        pause=0.05,
    )
    reply = await poster.post({})
    assert sleeps == [0.0]  # a Retry-After of zero is honoured, not mistaken for a missing header
    assert reply.headers["x-clm-latency-ms"] == reply.headers["X-CLM-Latency-Ms"] == "12"
    assert reply.seconds >= 0.04  # the wait between the attempts counts


async def test_the_payload_is_posted_as_json() -> None:
    poster, requests, _ = make([httpx2.Response(200, json={})])
    payload: dict[str, Any] = {"state": "café", "questions": {"q": {"type": "noul"}}}
    await poster.post(payload)
    (request,) = requests
    assert (request.method, str(request.url)) == ("POST", URL)
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == payload


@pytest.mark.parametrize("api_key", [None, ""])
async def test_without_a_key_no_authorization_is_sent(api_key: str | None) -> None:
    poster, requests, _ = make([httpx2.Response(400, text="bad request")], api_key=api_key)
    with pytest.raises(DecisionError) as error:
        await poster.post({})
    assert "authorization" not in requests[0].headers
    assert str(error.value) == "m@h: HTTP 400: bad request"  # an empty key must not garble the redaction


@pytest.mark.parametrize("key", [f"{SECRET}\n", f" {SECRET}", f"{SECRET}\r\nX-Injected: 1", f"{SECRET}é"])
async def test_a_key_that_a_header_cannot_carry_is_refused_before_anything_is_sent(key: str) -> None:
    # httpx2 rejects such a header with an error that quotes it, and the chained error would print the key.
    poster, requests, sleeps = make([httpx2.Response(200, json={})], api_key=key)
    with pytest.raises(DecisionError, match=r"m@h: api_key is not a usable API key") as error:
        await poster.post({})
    assert (requests, sleeps) == ([], [])
    assert SECRET not in str(error.value)


async def test_a_malformed_key_in_the_environment_is_named_but_not_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOLHUNCH_TEST_KEY", f"{SECRET}\n")
    with pytest.raises(DecisionError, match="TOOLHUNCH_TEST_KEY") as error:
        await bare(api_key_env="TOOLHUNCH_TEST_KEY").post({})
    assert SECRET not in str(error.value)


async def test_error_bodies_are_cut_to_500_characters_after_the_key_is_redacted() -> None:
    # The key straddles the cut: cutting first would leave a readable prefix of it.
    poster, _, _ = make([httpx2.Response(400, text="x" * 495 + SECRET + "tail")])
    with pytest.raises(DecisionError) as error:
        await poster.post({})
    assert str(error.value) == "m@h: HTTP 400: " + "x" * 495 + "***ta"


async def test_an_owned_client_is_created_lazily_and_closed_but_an_injected_one_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[httpx2.AsyncClient] = []
    real_client = httpx2.AsyncClient

    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        client = real_client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json={})), **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(httpx2, "AsyncClient", factory)
    monkeypatch.delenv("TOOLHUNCH_TEST_KEY", raising=False)

    with pytest.raises(DecisionError, match="TOOLHUNCH_TEST_KEY"):
        await bare(api_key_env="TOOLHUNCH_TEST_KEY").post({})
    assert created == []  # the key is checked before a client exists

    owned = bare(api_key=SECRET)
    assert created == []  # and a client is created by the first call, not by the constructor
    await owned.post({})
    (client,) = created
    assert client.timeout == httpx2.Timeout(7.0)
    await owned.aclose()
    await owned.aclose()  # closing twice is harmless
    assert client.is_closed

    injected = real_client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json={})))
    borrowed = bare(api_key=SECRET, http_client=injected)
    await borrowed.post({})
    await borrowed.aclose()
    assert not injected.is_closed
    await injected.aclose()
