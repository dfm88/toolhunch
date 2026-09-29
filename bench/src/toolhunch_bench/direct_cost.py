"""Provider accounting and a shared spend guard for direct-choice runs."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from genai_prices import Usage, calc_price

from toolhunch.decision import DecisionError
from toolhunch_bench.decision import jev_usd

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


class SpendLimit(Exception):
    """The next provider attempt cannot fit inside the remaining budget."""


class ProviderFailure(Exception):
    """A provider request failed; the public message contains only an exception class."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderCall:
    """One real provider attempt, distinct from an answer replayed locally.

    Failed attempts have unknown usage and price. Their reservation remains charged to the guard, so a timeout
    cannot free budget that may have been spent upstream. The ledger reports these reserves separately in its note.
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


class SpendGuard:
    """Reserve a conservative upper bound before every attempt, then account for the returned usage.

    Every provider shares this guard. An unsuccessful attempt keeps its reservation, including before a retry.
    `sink` writes each completed attempt immediately; crashes therefore retain an audit trail.
    """

    def __init__(
        self,
        *,
        prior_usd: float = 0.0,
        cap_usd: float = P1_CAP_USD,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.prior_usd = prior_usd
        self.cap_usd = cap_usd
        self.calls: list[dict[str, Any]] = []
        self.context: Mapping[str, Any] = {}
        self._sink = sink

    @property
    def run_usd(self) -> float:
        """Verified usage and conservative failure reserves charged during this run."""
        return sum(call["budget_charge_usd"] for call in self.calls)

    def before(self, upper_usd: float) -> None:
        """Refuse an attempt whose upper bound would cross the cap."""
        if upper_usd < 0 or self.prior_usd + self.run_usd + upper_usd > self.cap_usd:
            raise SpendLimit("next provider attempt would exceed the P1 budget")

    def record(self, call: ProviderCall) -> None:
        """Retain one attempt and publish it with its request context."""
        record = dict(self.context) | asdict(call)
        self.calls.append(record)
        if self._sink is not None:
            self._sink(record)
        if self.prior_usd + self.run_usd > self.cap_usd:
            raise SpendLimit("provider usage exceeded its conservative reservation; run stopped")


class GuardedDecisionModel:
    """A decision model with each real attempt guarded, priced and recorded.

    The wrapped adapter must have internal retries disabled. The optional retry here obtains its own reservation.
    Jev's declared request cap bounds the tokens it can bill for one attempt, including its own prompt.
    """

    def __init__(self, inner: DecisionModel, *, guard: SpendGuard, retries: int = 1) -> None:
        self._inner, self._guard, self._retries = inner, guard, retries

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

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Ask with a separately guarded reservation for each attempt."""
        if self.limits.max_request_tokens is None:
            raise ValueError("direct runs require a declared Jev request token cap")
        upper = jev_usd(self.limits.max_request_tokens)
        for attempt in range(self._retries + 1):
            self._guard.before(upper)
            started = time.perf_counter()
            try:
                response = await self._inner.ask(request, **options)
            except DecisionError as error:
                self._guard.record(
                    ProviderCall(
                        provider="typesafe",
                        model=self.model_id,
                        input_tokens=None,
                        output_tokens=None,
                        cache_read_tokens=None,
                        seconds=time.perf_counter() - started,
                        usd=None,
                        list_usd=None,
                        budget_charge_usd=upper,
                        error=type(error).__name__,
                    )
                )
                if attempt == self._retries:
                    raise ProviderFailure(type(error).__name__) from None
            else:
                usd = jev_usd(response.usage.input_tokens)
                self._guard.record(
                    ProviderCall(
                        provider="typesafe",
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
        raise AssertionError("unreachable")


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
