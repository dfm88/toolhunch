"""Provider accounting and a shared spend guard for direct-choice runs."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import httpx2
from genai_prices import Usage, calc_price
from openai import APIConnectionError

from toolhunch.decision import DecisionError, DecisionUsage
from toolhunch_bench.ledger import BUDGET

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from toolhunch.decision import DecisionModel, DecisionRequest, DecisionResponse, ModelLimits, QuestionKind
    from toolhunch.retrieval import Embedder, EmbeddingBatch, EmbeddingKind

AGENT_MODEL = "gpt-4.1-mini-2025-04-14"
EMBEDDING_MODEL = "text-embedding-3-small"
P1_CAP_USD = 7.0


def openai_usd(*, model: str, input_tokens: int, output_tokens: int = 0, cache_read_tokens: int = 0) -> float:
    """Price inclusive input usage, with cache reads at their cached rate."""
    if not 0 <= cache_read_tokens <= input_tokens:
        raise ValueError("cache reads must be between zero and the inclusive input total")
    return float(
        calc_price(
            Usage(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read_tokens),
            model_ref=model,
            provider_id="openai",
        ).total_price
    )


class RunStopped(Exception):
    """A run cannot usefully go on: it stops, keeps what it did and ledgers what it was charged."""


class SpendLimit(RunStopped):
    """The next provider attempt cannot fit inside the remaining budget."""


class ProviderStopped(RunStopped):
    """A provider refused the account, or stopped answering: every further attempt would only be charged."""


_REFUSED_STATUSES = frozenset({401, 402, 403})
"""A key, a payment or a permission: no retry can fix them."""
OUTAGE_LIMIT = 3
"""Failed attempts in a row, without an answer, after which a provider is taken as out (a quota, a limit, an outage)."""


def attempt_failure(error: BaseException) -> tuple[int | None, bool]:
    """The HTTP status a failed attempt ended with, if any, and whether a reply arrived at all."""
    status = getattr(error, "status", None) if isinstance(error, DecisionError) else getattr(error, "status_code", None)
    cause = error.__cause__ if isinstance(error, DecisionError) else error
    reached = not isinstance(cause, httpx2.TransportError | APIConnectionError | TimeoutError | ConnectionError)
    return (status if isinstance(status, int) else None), reached


class ProviderFailure(DecisionError):
    """A provider request failed; the public message contains only an exception class or a fixed reason.

    It is a `DecisionError`, so a decision run records it as a failed search and goes on.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderCall:
    """One real provider attempt, distinct from an answer replayed locally.

    Failed attempts have unknown usage and price. Their reservation remains charged to the guard, so a timeout
    cannot free budget that may have been spent upstream. The ledger reports these reserves separately in its note.
    A failed attempt also records the HTTP status it ended with, if any, and whether a reply arrived at all.
    """

    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    seconds: float
    usd: float | None
    list_usd: float | None
    budget_charge_usd: float
    error: str | None = None
    status: int | None = None
    reached: bool = True


class SpendGuard:
    """Reserve a conservative upper bound before every attempt, then account for the returned usage.

    Every provider shares this guard. An unsuccessful attempt keeps its reservation, including before a retry.
    `sink` writes each completed attempt immediately; crashes therefore retain an audit trail. The guard stops the
    run (`ProviderStopped`) when a provider refuses with HTTP 401, 402 or 403, or after `OUTAGE_LIMIT` failed attempts
    in a row at one provider that ended with HTTP 429, a server fault or no reply. Any other reply, an unusable one
    included, shows the provider answering and starts the count again.
    """

    def __init__(
        self,
        *,
        prior_usd: float = 0.0,
        cap_usd: float = BUDGET.cap_usd,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.prior_usd = prior_usd
        self.cap_usd = cap_usd
        self.calls: list[dict[str, Any]] = []
        self.context: Mapping[str, Any] = {}
        self._sink = sink
        self._unanswered: dict[str, int] = {}

    @property
    def run_usd(self) -> float:
        """Verified usage and conservative failure reserves charged during this run."""
        return sum(call["budget_charge_usd"] for call in self.calls)

    def before(self, upper_usd: float) -> None:
        """Refuse an attempt whose upper bound would cross the cap."""
        if upper_usd < 0 or self.prior_usd + self.run_usd + upper_usd > self.cap_usd:
            raise SpendLimit(f"next provider attempt would exceed the ${self.cap_usd:.2f} cap")

    def record(self, call: ProviderCall) -> None:
        """Retain one attempt and publish it with its request context."""
        record = dict(self.context) | asdict(call)
        self.calls.append(record)
        if self._sink is not None:
            self._sink(record)
        if self.prior_usd + self.run_usd > self.cap_usd:
            raise SpendLimit("provider usage exceeded its conservative reservation; run stopped")
        if call.status in _REFUSED_STATUSES:
            raise ProviderStopped(f"{call.provider} refused with HTTP {call.status} ({call.model})")
        status = call.status or 0
        if call.error is None or (call.reached and status != 429 and status < 500):
            self._unanswered[call.provider] = 0
            return
        unanswered = self._unanswered[call.provider] = self._unanswered.get(call.provider, 0) + 1
        if unanswered >= OUTAGE_LIMIT:
            raise ProviderStopped(f"{unanswered} failed attempts in a row at {call.provider} ({call.model})")


_UNPRICED = "priced reply without input tokens"


class GuardedDecisionModel:
    """A decision model with each real attempt guarded, priced and recorded.

    The price comes from the wrapped model's declared limits, and each attempt reserves what its largest declared
    request would cost. The wrapped adapter must have internal retries disabled; the optional retry here obtains its
    own reservation. A priced reply that reports no input tokens is a failure: it could not be priced, so the guard
    cannot account for it.
    """

    def __init__(
        self, inner: DecisionModel, *, guard: SpendGuard, provider: str = "typesafe", retries: int = 1
    ) -> None:
        """Wrap `inner`, billed by `provider`.

        Raises:
            ValueError: `inner` declares no input price, or a price without a request token cap to reserve from.
        """
        limits = inner.limits
        self._local = provider == "local"  # a model on this machine: recorded, timed, never priced
        if limits.price_input_per_mtok is None and not self._local:
            raise ValueError(f"{inner.model_id} declares no input price to guard")
        if limits.max_request_tokens is None and (limits.price_input_per_mtok or 0) > 0 and not self._local:
            raise ValueError(f"{inner.model_id} declares no request token cap to reserve from")
        self._inner, self._guard, self._provider, self._retries = inner, guard, provider, retries

    @property
    def model_id(self) -> str:
        """The wrapped model identity."""
        return self._inner.model_id

    @property
    def limits(self) -> ModelLimits:
        """The wrapped model's declared limits."""
        return self._inner.limits

    @property
    def question_kinds(self) -> frozenset[QuestionKind]:
        """The question kinds accepted by the wrapped model."""
        return self._inner.question_kinds

    @property
    def prompt_version(self) -> str | None:
        """The wrapped model's prompt version."""
        return self._inner.prompt_version

    def __repr__(self) -> str:
        return repr(self._inner)

    def _usd(self, usage: DecisionUsage) -> float:
        return 0.0 if self._local else self.limits.estimate_usd(usage) or 0.0

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Ask with a separately guarded reservation for each attempt."""
        priced = not self._local and (self.limits.price_input_per_mtok or 0.0) > 0
        upper = self._usd(DecisionUsage(1, self.limits.max_request_tokens or 0, 0))
        for attempt in range(self._retries + 1):
            self._guard.before(upper)
            started = time.perf_counter()
            status, reached = None, True
            try:
                response = await self._inner.ask(request, **options)
            except DecisionError as error:
                failure = type(error).__name__
                status, reached = attempt_failure(error)
            else:
                if not (priced and response.usage.input_tokens <= 0):
                    usd = self._usd(response.usage)
                    self._guard.record(
                        ProviderCall(
                            provider=self._provider,
                            model=self.model_id,
                            input_tokens=response.usage.input_tokens,
                            output_tokens=response.usage.output_tokens,
                            cache_read_tokens=0,
                            seconds=time.perf_counter() - started,
                            usd=usd,
                            list_usd=usd,
                            budget_charge_usd=usd,
                        )
                    )
                    return response
                failure = _UNPRICED
            self._guard.record(
                ProviderCall(
                    provider=self._provider,
                    model=self.model_id,
                    input_tokens=None,
                    output_tokens=None,
                    cache_read_tokens=None,
                    seconds=time.perf_counter() - started,
                    usd=None,
                    list_usd=None,
                    budget_charge_usd=upper,
                    error=failure,
                    status=status,
                    reached=reached,
                )
            )
            # A reply the provider billed but did not account for is not retried: a retry would be billed again.
            if attempt == self._retries or failure == _UNPRICED:
                raise ProviderFailure(failure)
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        """Close the wrapped model's client, if it has one."""
        if (close := getattr(self._inner, "aclose", None)) is not None:
            await close()


class GuardedEmbedder:
    """Guard each real embedding batch; place this inside the persistent embedding cache.

    Configure the wrapped embedder and cache for one physical batch per `embed` call. UTF-8 byte counts bound BPE
    tokens without downloading a tokenizer; the estimate uses the model's actual tokenizer separately.
    """

    def __init__(self, inner: Embedder, *, guard: SpendGuard) -> None:
        self._inner, self._guard = inner, guard

    @property
    def model_id(self) -> str:
        """The embedding model identity."""
        return self._inner.model_id

    async def embed(self, texts: Sequence[str], /, *, kind: EmbeddingKind) -> EmbeddingBatch:
        """Guard and record one real embedding batch, without hidden retries."""
        upper = openai_usd(model=self.model_id, input_tokens=sum(len(text.encode()) for text in texts))
        self._guard.before(upper)
        started = time.perf_counter()
        try:
            response = await self._inner.embed(texts, kind=kind)
        except Exception as error:
            status, reached = attempt_failure(error)
            self._guard.record(
                ProviderCall(
                    provider="openai",
                    model=self.model_id,
                    input_tokens=None,
                    output_tokens=None,
                    cache_read_tokens=None,
                    seconds=time.perf_counter() - started,
                    usd=None,
                    list_usd=None,
                    budget_charge_usd=upper,
                    error=type(error).__name__,
                    status=status,
                    reached=reached,
                )
            )
            raise ProviderFailure(type(error).__name__) from None
        usd = openai_usd(model=self.model_id, input_tokens=response.input_tokens)
        self._guard.record(
            ProviderCall(
                provider="openai",
                model=self.model_id,
                input_tokens=response.input_tokens,
                output_tokens=0,
                cache_read_tokens=0,
                seconds=time.perf_counter() - started,
                usd=usd,
                list_usd=usd,
                budget_charge_usd=usd,
            )
        )
        return response
