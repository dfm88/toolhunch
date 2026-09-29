"""Private helpers for decision models that speak JSON over HTTP: the endpoint, auth, retries and redaction."""

from __future__ import annotations

import contextlib
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import anyio
import httpx2

from toolhunch.decision.base import DecisionError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

__all__ = ["RETRY_STATUSES", "Endpoint", "JsonPoster", "JsonReply"]

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
"""Rate limiting and transient server faults; any other status outside 2xx is final."""

_BACKOFF_SECONDS = 0.5
_MAX_RETRY_AFTER_SECONDS = 30.0
_BODY_CHARACTERS = 500
_VISIBLE_ASCII = re.compile(r"[\x21-\x7e]+")


@dataclass(frozen=True, slots=True)
class Endpoint:
    """The API root a decision model talks to, and the identity derived from it.

    Attributes:
        base_url: The API root, without a trailing slash.
        model_id: `"<model>@<host>"`; the host keeps its port, as in `localhost:8000`.
    """

    base_url: str
    model_id: str

    @classmethod
    def parse(cls, base_url: str, *, model: str) -> Endpoint:
        """Check `base_url` and derive the identity of `model` behind it.

        Raises:
            ValueError: `base_url` is not an absolute http(s) URL, or it carries credentials
                (`user:password@host`). A secret there would reach `repr`, the model id, threshold keys
                and manifests, so the URL is refused, and never quoted in the message.
        """
        parts = urlsplit(base_url)
        if "@" in parts.netloc:
            raise ValueError("base_url must not carry credentials (user:password@host): pass the key as api_key")
        if parts.scheme not in ("http", "https") or not parts.netloc:
            # Without a scheme, "user:secret@host" has no netloc for the check above to find: do not quote it.
            shown = "" if "@" in base_url else f", got {base_url!r}"
            raise ValueError(f"base_url must be an absolute http(s) URL{shown}")
        return cls(base_url=base_url.rstrip("/"), model_id=f"{model}@{parts.netloc}")


@dataclass(frozen=True, slots=True)
class JsonReply:
    """A successful reply.

    Attributes:
        body: The decoded JSON object.
        headers: The response headers; lookups ignore case.
        seconds: Client wall time from the first attempt to the reply, retry waits included.
    """

    body: dict[str, Any]
    headers: Mapping[str, str]
    seconds: float


class JsonPoster:
    """POSTs JSON to one URL on behalf of a decision model: auth, retries and redaction.

    The key is `api_key` or, read at call time, the `api_key_env` variable; with neither, no
    `Authorization` header is sent, for a server without auth. An empty `api_key` counts as not given.
    The key never appears in `repr` or in an error message.

    A transport error or one of `RETRY_STATUSES` is retried up to `max_retries` times, so a call makes at
    most `max_retries + 1` attempts. Each wait goes through `sleep` and lasts the reply's `Retry-After`
    seconds, capped at 30, or else a backoff that starts at 0.5 s and doubles with every retry. Every
    other failure raises `DecisionError`, and its message starts with `model_id`.

    An injected `http_client` is used as is (`timeout` applies only to a client the poster creates) and is
    never closed. A client the poster creates on its first request is closed by `aclose()`. Calls share no
    state, so concurrent `post` calls on one poster are safe.
    """

    def __init__(
        self,
        url: str,
        *,
        model_id: str,
        api_key: str | None,
        api_key_env: str | None,
        http_client: httpx2.AsyncClient | None,
        timeout: float,
        max_retries: int,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    ) -> None:
        self._url = url
        self._model_id = model_id
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._client = http_client
        self._owns_client = http_client is None
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"JsonPoster(model_id={self._model_id!r}, url={self._url!r})"

    async def post(self, payload: Mapping[str, Any]) -> JsonReply:
        """POST `payload` as JSON and return the decoded reply, retrying transient failures.

        Raises:
            DecisionError: No key is available, the call failed after its retries, the status is
                final, or the body is not a JSON object.
        """
        key = self._key()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        client = self._http_client()
        started = time.perf_counter()
        attempt = 0
        while True:
            backoff = _BACKOFF_SECONDS * 2**attempt
            try:
                response = await client.post(self._url, json=dict(payload), headers=headers)
            except httpx2.TransportError as error:
                if attempt >= self._max_retries:
                    raise DecisionError(
                        f"{self._model_id}: {type(error).__name__} after {attempt + 1} attempts"
                    ) from error
                delay = backoff
            except httpx2.HTTPError as error:  # not a transport fault: a retry would fail the same way
                raise DecisionError(
                    f"{self._model_id}: {type(error).__name__}: {self._shown(str(error), key)}"
                ) from error
            else:
                if response.is_success:
                    return self._reply(response, key=key, seconds=time.perf_counter() - started)
                status = response.status_code
                if status not in RETRY_STATUSES or attempt >= self._max_retries:
                    raise DecisionError(f"{self._model_id}: HTTP {status}: {self._shown(response.text, key)}")
                retry_after = self._retry_after(response)
                delay = backoff if retry_after is None else retry_after
            await self._sleep(delay)
            attempt += 1

    async def aclose(self) -> None:
        """Close the HTTP client if the poster created it; an injected client stays open."""
        if self._owns_client and self._client is not None:
            client, self._client = self._client, None
            await client.aclose()

    def _key(self) -> str | None:
        if self._api_key:
            key, source = self._api_key, "api_key"
        elif self._api_key_env is None:
            return None
        else:
            source = self._api_key_env
            if not (key := os.environ.get(source)):
                raise DecisionError(f"{self._model_id}: no API key: set {source} or pass api_key")
        if not _VISIBLE_ASCII.fullmatch(key):
            # An HTTP library rejecting the header value quotes it, key included, in its own error.
            raise DecisionError(
                f"{self._model_id}: {source} is not a usable API key: it may only hold visible ASCII "
                "characters, so no space, line break or accent"
            )
        return key

    def _http_client(self) -> httpx2.AsyncClient:
        if self._client is None:
            self._client = httpx2.AsyncClient(timeout=self._timeout)
        return self._client

    def _reply(self, response: httpx2.Response, *, key: str | None, seconds: float) -> JsonReply:
        try:
            body: Any = response.json()
        except ValueError:  # a JSONDecodeError, or bytes that are not valid text
            body = None
        if not isinstance(body, dict):
            raise DecisionError(
                f"{self._model_id}: HTTP {response.status_code}: not a JSON object: {self._shown(response.text, key)}"
            )
        return JsonReply(body=cast("dict[str, Any]", body), headers=response.headers, seconds=seconds)

    @staticmethod
    def _retry_after(response: httpx2.Response) -> float | None:
        """The seconds a reply asks to wait, capped at 30; `None` when the header is absent or unusable."""
        with contextlib.suppress(ValueError):  # absent (""), an HTTP-date or any other text is not a number
            # NaN and negative values fail the comparison; infinity is capped.
            if (seconds := float(response.headers.get("retry-after", ""))) >= 0:
                return min(seconds, _MAX_RETRY_AFTER_SECONDS)
        return None

    @staticmethod
    def _shown(text: str, key: str | None) -> str:
        # Redact before cutting: a key that straddles the cut would leave a readable prefix.
        return (text.replace(key, "***") if key else text)[:_BODY_CHARACTERS]
