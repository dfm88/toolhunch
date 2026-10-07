"""Decision runs: every decider arm answers every task, on positives and negatives, over one shared retrieval.

An arm is the hybrid retriever at one K, alone (`hybrid@K`) or followed by a decider (`hybrid+<name>@K`). Each task is
searched with each query source: the ToolRet request alone (`plain`), or what an agent searched for (`model`, the
queries `toolhunch-bench queries write` froze). Each search runs as a positive, and as a negative with the task's gold
tools taken out of the candidates, where the right answer is that none of them fits. The retrieval is shared: every
arm, variant and repeat of one query list reuses one hybrid retrieval, so the arms differ only in their decider.

Nothing is paid before `estimate_decisions` has priced the same searches with stand-in models, which ask nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import anyio
import httpx2
from genai_prices import Usage, calc_price

from toolhunch import DetailLevel, HeuristicTokenizer, ToolSearchPipeline, default_search_text
from toolhunch.decision import (
    JEV_LIMITS,
    Abstention,
    CandidatesDoNotFit,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionRefused,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    check_request,
)
from toolhunch.decision.planner import PROMPT_VERSION
from toolhunch.retrieval import Retrieval
from toolhunch.retrieval.base import FUSION_DEPTH, check_k, clean_queries
from toolhunch_bench.deciders import (
    CLM_DEPLOYMENT,
    CLM_MODEL,
    JEV_MODEL,
    LOGPROB_MODEL,
    LUNA_MODEL,
    DeciderName,
    choice_decider,
    planner_tokenizer,
)
from toolhunch_bench.deciders import DECIDERS as REGISTRY
from toolhunch_bench.decision_cache import CachedDecisionModel
from toolhunch_bench.direct_cost import RunStopped
from toolhunch_bench.embedding_cache import estimate_embedding_cost
from toolhunch_bench.ledger import F2A_CAP_EUR
from toolhunch_bench.retrieval import repo_path, run_provenance

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
    from pathlib import Path

    from toolhunch import ToolCatalog
    from toolhunch.decision import Decider, Decision, DecisionModel, ModelLimits, QuestionKind
    from toolhunch.retrieval import Retriever
    from toolhunch_bench.datasets.model_queries import WrittenQueries
    from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask
    from toolhunch_bench.embedding_cache import CachedEmbedder

__all__ = [
    "CLM_DEPLOYMENT",
    "CLM_MODEL",
    "DECIDERS",
    "F2A_CAP_EUR",
    "JEV_MODEL",
    "JEV_REQUEST_OVERHEAD_TOKENS",
    "LOGPROB_MODEL",
    "LUNA_MODEL",
    "QUERY_SOURCES",
    "CacheOnlyRetrieval",
    "CountedModel",
    "DecisionArm",
    "EstimateLine",
    "ExcludingRetriever",
    "QuerySource",
    "SharedRetrieval",
    "Split",
    "build_decision_arms",
    "clm_usd",
    "estimate_decisions",
    "jev_usd",
    "run_decisions",
    "warm_up_clm",
]

type QuerySource = Literal["plain", "model"]
"""Where a search's queries come from: the ToolRet request alone, or what an agent searched for."""
type Split = Literal["dev", "heldout"]
"""The task split of a run: settings and thresholds are chosen on dev, and results are reported on held-out."""

DECIDERS: tuple[str, ...] = tuple(DeciderName)
"""The deciders a run can compare, each asking its own model: every `DeciderName`."""
QUERY_SOURCES: tuple[QuerySource, ...] = ("plain", "model")
"""Every query source."""
JEV_REQUEST_OVERHEAD_TOKENS = 300
"""What Jev bills beyond the text of a request, for its own prompt: a round figure, for estimates only."""

_CLM_GPU_SECONDS_PER_CALL = 0.3  # estimates only: CLM is billed by GPU time, never by tokens
_HEALTH_TIMEOUT_SECONDS = 30.0
_WARM_UP_REQUEST = DecisionRequest(
    state="Warm-up request.",
    questions={
        "warm-up": ChoiceQuestion(
            instructions="Pick either option.", options={"a": "The first option.", "b": "The second option."}
        )
    },
)


def jev_usd(input_tokens: int) -> float:
    """What Jev bills for `input_tokens`, at the input price `JEV_LIMITS` declares."""
    return input_tokens * _declared_input_price(JEV_LIMITS) / 1_000_000


def clm_usd(seconds: float) -> float:
    """What `seconds` of the CLM deployment's GPU cost at its list price, in Modal credits."""
    return seconds * float(CLM_DEPLOYMENT["usd_per_gpu_hour"]) / 3600


def _declared_input_price(limits: ModelLimits) -> float:
    if limits.price_input_per_mtok is None:
        raise ValueError(f"no input price is declared in {limits.source}")
    return limits.price_input_per_mtok


class SharedRetrieval:
    """A `Retriever` that retrieves each list of queries once, and serves every later request from what it kept.

    The first request for `queries` in a catalog calls `inner` at depth `max(depth, k)`; a later request with `k` up to
    that depth gets the first `k` of what was kept, and no usage. The hybrid retriever fuses at depth `max(k, 100)`, so
    with the default depth its first `k` of a depth-100 retrieval are what it returns for `k` itself, for any `k` up to
    100. A request deeper than what was kept calls `inner` again, at the new depth.

    Attributes:
        config: What the run manifest records about `inner`.
        depth: The least depth `inner` is asked for.
    """

    def __init__(self, inner: Retriever, *, depth: int = FUSION_DEPTH, config: Mapping[str, Any] | None = None) -> None:
        """Share `inner`'s retrievals, each at least `depth` deep."""
        check_k(depth, name="depth")
        self._inner = inner
        self.depth = depth
        self.config: Mapping[str, Any] = {} if config is None else config
        self._kept: dict[tuple[tuple[str, ...], str], tuple[int, Retrieval]] = {}
        self._first_seconds: dict[tuple[str, ...], float] = {}
        self._stand_in_queries: set[tuple[str, ...]] = set()

    @property
    def stand_in_queries(self) -> frozenset[tuple[str, ...]]:
        """Query lists this wrapper observed using the inner free estimator's fallback."""
        return frozenset(self._stand_in_queries)

    def uses_stand_in(self, queries: Sequence[str], catalog: ToolCatalog) -> bool:
        """Whether this exact query/catalog search used the free fallback."""
        diagnostic = getattr(self._inner, "uses_stand_in", None)
        return diagnostic(queries, catalog) if diagnostic is not None else False

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """The first `k` matches of the kept retrieval of `queries`, retrieving it when it is missing or too shallow."""
        check_k(k)
        key = (tuple(queries), catalog.fingerprint)
        kept = self._kept.get(key)
        if kept is not None and k <= kept[0]:
            return Retrieval(matches=kept[1].matches[:k])
        depth = max(self.depth, k)
        started = time.perf_counter()
        retrieval = await self._inner.retrieve(queries, catalog, k=depth)
        if self.uses_stand_in(queries, catalog):
            self._stand_in_queries.add(tuple(clean_queries(queries)))
        self._first_seconds.setdefault(key[0], time.perf_counter() - started)
        self._kept[key] = (depth, retrieval)
        return Retrieval(matches=retrieval.matches[:k], usage=retrieval.usage)

    def first_seconds(self, queries: Sequence[str]) -> float:
        """How long the first retrieval of `queries` took, measured around the call to `inner`.

        Raises:
            KeyError: `queries` were never retrieved.
        """
        return self._first_seconds[tuple(queries)]


class ExcludingRetriever:
    """A `Retriever` that leaves some cards out, standing in for a catalog without them.

    It asks `inner` for `k` plus one match per excluded id, drops the excluded ids and keeps the first `k`, in `inner`'s
    order. The other cards can rank a little differently than in a catalog that really lacks the excluded ones: their
    ranks there would shift where those cards were fused in, and BM25's document frequencies would move slightly.
    """

    def __init__(self, inner: Retriever, *, exclude: Collection[str]) -> None:
        """Leave the cards whose ids are in `exclude` out of `inner`'s matches."""
        self._inner = inner
        self._exclude = frozenset(exclude)

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """At most `k` matches of `inner` for `queries`, none of them excluded."""
        check_k(k)
        retrieval = await self._inner.retrieve(queries, catalog, k=k + len(self._exclude))
        kept = [match for match in retrieval.matches if match.card.id not in self._exclude]
        return Retrieval(matches=tuple(kept[:k]), usage=retrieval.usage)


class CacheOnlyRetrieval:
    """A `Retriever` for estimates, which never pays for an embedding.

    A search runs on `hybrid`, whose dense half embeds through `embeddings`, only when that cache already holds a
    vector for every card of the catalog (their `default_search_text`) and for every query of the search: then no text
    reaches the paid API. Any other search gets `stand_in`'s matches, such as BM25's alone.

    Attributes:
        searched: The query lists retrieved, cleaned as the retrievers clean them.
        stand_in_queries: Those among them that `stand_in` retrieved.
    """

    def __init__(self, hybrid: Retriever, *, stand_in: Retriever, embeddings: CachedEmbedder) -> None:
        """Search with `hybrid` where `embeddings` holds every vector it needs, and with `stand_in` elsewhere."""
        self._hybrid = hybrid
        self._stand_in = stand_in
        self._embeddings = embeddings
        self._cards_cached: dict[str, bool] = {}
        self.searched: set[tuple[str, ...]] = set()
        self.stand_in_queries: set[tuple[str, ...]] = set()
        self._stand_in_searches: set[tuple[tuple[str, ...], str]] = set()

    def uses_stand_in(self, queries: Sequence[str], catalog: ToolCatalog) -> bool:
        """Whether the exact cleaned query list and catalog required a stand-in."""
        return (tuple(clean_queries(queries)), catalog.fingerprint) in self._stand_in_searches

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """`hybrid`'s matches when every vector they need is cached, `stand_in`'s otherwise."""
        cleaned = clean_queries(queries)
        self.searched.add(tuple(cleaned))
        if self._cards_cached.get(catalog.fingerprint) is None:
            texts = (default_search_text(card) for card in catalog)
            self._cards_cached[catalog.fingerprint] = not self._embeddings.missing(texts, kind="document")
        if self._cards_cached[catalog.fingerprint] and not self._embeddings.missing(cleaned, kind="query"):
            return await self._hybrid.retrieve(queries, catalog, k=k)
        self.stand_in_queries.add(tuple(cleaned))
        self._stand_in_searches.add((tuple(cleaned), catalog.fingerprint))
        return await self._stand_in.retrieve(queries, catalog, k=k)


class OrderedRetrieval:
    """Present the same retrieval in a reproducible task-seeded order; seed zero keeps its order."""

    def __init__(self, inner: Retriever, *, task_id: str, seed: int) -> None:
        self._inner, self._task_id, self._seed = inner, task_id, seed

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """Keep scores and usage while changing only presentation order."""
        retrieval = await self._inner.retrieve(queries, catalog, k=k)
        matches = list(retrieval.matches)
        if self._seed:
            digest = hashlib.sha256(f"tool-order-v1:{self._task_id}:{self._seed}".encode()).digest()
            random.Random(int.from_bytes(digest, "big")).shuffle(matches)
        return Retrieval(matches=tuple(matches), usage=retrieval.usage)


@dataclass(frozen=True, slots=True)
class DecisionArm:
    """One arm of a decision run: the shared hybrid retrieval at `k`, then a decider or nothing.

    Attributes:
        name: `hybrid@<k>`, or `hybrid+<decider>@<k>`.
        decider_name: One of `DECIDERS`; `None` for retrieval alone.
        k: The candidates retrieved, all of which the decider ranks.
        decider: What decides; `None` for retrieval alone.
        config: What the run manifest records about the arm.
        model: The model the decider asks, which an estimate replaces with a stand-in.
        asked: How many asks the decider has sent to `model` so far; `None` when they are not counted. A search tells
            from it whether a `CandidatesDoNotFit` came before or after its first ask.
    """

    name: str
    decider_name: str | None
    k: int
    decider: Decider | None
    config: Mapping[str, Any]
    model: DecisionModel | None = None
    asked: Callable[[], int] | None = None


class CountedModel:
    """`inner`, with a count of the asks sent through it.

    A search tells from the count whether a `CandidatesDoNotFit` came before or after its decider's first ask.
    """

    def __init__(self, inner: DecisionModel) -> None:
        self._inner = inner
        self._asks = 0

    @property
    def model_id(self) -> str:
        """The wrapped model's identity."""
        return self._inner.model_id

    @property
    def limits(self) -> ModelLimits:
        """The wrapped model's limits."""
        return self._inner.limits

    @property
    def question_kinds(self) -> frozenset[QuestionKind]:
        """The wrapped model's question kinds."""
        return self._inner.question_kinds

    @property
    def prompt_version(self) -> str | None:
        """The wrapped model's prompt version."""
        return self._inner.prompt_version

    def __repr__(self) -> str:
        return repr(self._inner)

    def count(self) -> int:
        """How many asks were sent so far, failed ones and answers from a cache included."""
        return self._asks

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Count the ask, then ask the wrapped model."""
        self._asks += 1
        return await self._inner.ask(request, **options)


def build_decision_arms(
    deciders: Sequence[str],
    *,
    ks: Sequence[int],
    models: Mapping[str, DecisionModel],
    max_detail: Mapping[str, DetailLevel],
    min_detail: Mapping[str, DetailLevel] | None = None,
    reserved_option: bool = True,
) -> list[DecisionArm]:
    """Build, for each k in `ks`, the arm `hybrid@<k>` and then `hybrid+<name>@<k>` for each name in `deciders`.

    Each decider is the registry's `choice_decider` over `models[name]`, planning with the tokenizer its entry names,
    that records the full final distribution: abstention at threshold 0, with the reserved option unless
    `reserved_option` is off, and cards at `max_detail[name]` at most (`FULL` for a name it leaves out) and
    `min_detail[name]` at least (`NAME` for a name it leaves out). Each arm's config records both levels and the
    `repr` of the planner's tokenizer.

    Raises:
        ValueError: A name is not one of `DECIDERS` or has no model, a k is below 1, or a floor is above its
            decider's most detail.
    """
    if unknown := sorted(set(deciders) - set(DECIDERS)):
        raise ValueError(f"unknown deciders {unknown}; choose from {', '.join(DECIDERS)}")
    if missing := [name for name in deciders if name not in models]:
        raise ValueError(f"no model for the deciders {missing}")
    arms: list[DecisionArm] = []
    for k in ks:
        check_k(k)
        baseline: dict[str, Any] = {
            "k": k,
            "decider": None,
            "model_id": None,
            "model": None,
            "limits": None,
            "prompt_version": None,
            "max_detail": None,
            "min_detail": None,
            "tokenizer": None,
            "reserved_option": None,
        }
        arms.append(DecisionArm(f"hybrid@{k}", None, k, None, baseline))
        for name in deciders:
            model = models[name]
            detail = max_detail.get(name, DetailLevel.FULL)
            floor = (min_detail or {}).get(name, DetailLevel.NAME)
            own_version = model.prompt_version
            config = baseline | {
                "decider": name,
                "model_id": model.model_id,
                "model": repr(model),
                "limits": model.limits.model_dump(mode="json"),
                # As `Decision.key` writes it: the planner's version, then the model's own after a "+".
                "prompt_version": PROMPT_VERSION if own_version is None else f"{PROMPT_VERSION}+{own_version}",
                "max_detail": detail.name,
                "min_detail": floor.name,
                "tokenizer": repr(planner_tokenizer(name)),
                "reserved_option": reserved_option,
            }
            abstention = Abstention(threshold=0.0, reserved_option=reserved_option)
            counted = CountedModel(model)
            decider = choice_decider(name, counted, abstention=abstention, max_detail=detail, min_detail=floor)
            arms.append(DecisionArm(f"hybrid+{name}@{k}", name, k, decider, config, model, counted.count))
    return arms


async def warm_up_clm(
    model: DecisionModel,
    *,
    base_url: str,
    timeout: float = 900.0,
    poll: float = 10.0,
    client: httpx2.AsyncClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
) -> float:
    """Wake a CLM server that scales to zero; return the seconds it took.

    It asks `GET {base_url}/health` every `poll` seconds, waiting through `sleep`, until the answer is 200; a
    transport error or any other status means not yet. Then it asks `model` one throwaway two-option question, so that
    the run's first timed request does not meet a server that has only just started. Pass the model itself, not a
    cache in front of it: an answer replayed from a cache wakes nothing.

    `client` is used as is and never closed; without one, a client is created for the polls and closed after them.

    Raises:
        TimeoutError: The server did not answer 200 within `timeout` seconds.
        DecisionError: The throwaway question failed.
    """
    started = time.perf_counter()
    http = client if client is not None else httpx2.AsyncClient(timeout=_HEALTH_TIMEOUT_SECONDS)
    url = f"{base_url.rstrip('/')}/health"
    try:
        while True:
            with contextlib.suppress(httpx2.TransportError):  # the container is still starting
                if (await http.get(url)).status_code == 200:
                    break
            if time.perf_counter() - started + poll > timeout:
                raise TimeoutError(f"the CLM server did not answer 200 on /health within {timeout:.0f} s")
            await sleep(poll)
    finally:
        if client is None:
            await http.aclose()
    await model.ask(_WARM_UP_REQUEST)
    return time.perf_counter() - started


@dataclass(frozen=True, slots=True)
class EstimateLine:
    """What one provider is expected to bill for a run.

    Attributes:
        provider: Who bills: `typesafe`, `openai` or `modal`.
        model: What is billed.
        calls: The asks a decision model gets, or the texts an embedder gets.
        input_tokens: The input tokens priced.
        usd: The expected cost; `None` for CLM, which Modal credits pay and the cap leaves out.
        note: How the figure was reached; for CLM, the GPU seconds.
        not_applicable: The searches whose candidates two rounds cannot hold, which the run records as not
            applicable and asks nothing for.
        would_fail: The searches with a card that no question can show, which the run records as failed.
    """

    provider: str
    model: str
    calls: int
    input_tokens: int
    usd: float | None
    note: str
    not_applicable: int = 0
    would_fail: int = 0


class _StandIn:
    """What `model` would be asked, answered without asking it: an estimate's decision model.

    It takes `model`'s id, limits, question kinds and prompt version, checks each request against them as the adapters
    do, and answers every choice question with a uniform distribution. `calls` and `input_tokens` add up what it was
    asked: the state, the instructions and every option's key and text, counted by `HeuristicTokenizer`.
    """

    def __init__(self, model: DecisionModel) -> None:
        self._model = model
        self._tokenizer = HeuristicTokenizer()
        self.calls = 0
        self.input_tokens = 0
        self.options = 0

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @property
    def limits(self) -> ModelLimits:
        return self._model.limits

    @property
    def question_kinds(self) -> frozenset[QuestionKind]:
        return self._model.question_kinds

    @property
    def prompt_version(self) -> str | None:
        return self._model.prompt_version

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        check_request(request, limits=self.limits, kinds=self.question_kinds)
        count = self._tokenizer.count
        tokens = count(request.state)
        answers: dict[str, ChoiceAnswer] = {}
        for key, question in request.questions.items():
            if not isinstance(question, ChoiceQuestion):
                raise ValueError(f"the stand-in answers choice questions, not {question.kind} questions")
            tokens += count(question.instructions)
            tokens += sum(count(option) + count(text) for option, text in question.options.items())
            self.options += len(question.options)
            answers[key] = ChoiceAnswer(probabilities=dict.fromkeys(question.options, 1 / len(question.options)))
        self.calls += 1
        self.input_tokens += tokens
        return DecisionResponse(answers=answers, usage=DecisionUsage(1, tokens, 0), seconds=0.0, raw={})


async def estimate_decisions(
    arms: Sequence[DecisionArm],
    data: ToolRetData,
    tasks: Sequence[ToolRetTask],
    *,
    retriever: Retriever,
    sources: Sequence[QuerySource],
    model_queries: Mapping[str, WrittenQueries] | None,
    negatives: bool,
    repeat: int,
    embeddings: CachedEmbedder | None = None,
    order_seeds: Sequence[int] | None = None,
) -> list[EstimateLine]:
    """Estimate what `run_decisions` would bill for these arms and searches, asking no model.

    Every decider arm runs its searches once, as `run_decisions` would, with a stand-in for its model: the stand-in
    has the model's limits, so the questions are planned as they will be, but it answers uniformly and asks nothing.
    The asks and their heuristic token counts, times `repeat`, are priced per decider, in one line each:

    - a token-billed decider: the tokens plus its entry's `request_overhead_tokens` per ask, at the input price its
      limits declare (Jev, Clef), or priced by genai-prices with its entry's output tokens per ask (logprob, luna);
    - CLM on Modal: no dollars in the cap, 0.3 GPU seconds per ask in the note;
    - a local decider: no charge.

    Every ask is counted, including those the decision cache would answer, so the estimate leans high. A search that
    raises `CandidatesDoNotFit` is counted in its line as the run records it: not applicable when the plan raised it
    before any ask, and as one that would fail when it names a card or came after an ask. With `embeddings`, one more
    line prices the query and card texts that cache lacks.

    Args:
        arms: The arms, from `build_decision_arms`; each decider arm's model is replaced by a stand-in.
        data: The loaded dataset; its catalog is searched.
        tasks: The tasks to search.
        retriever: What the searches retrieve with. It must not pay for anything: a `CacheOnlyRetrieval`, shared.
        sources: The query sources.
        model_queries: The written queries of every task, by task id; needed for the `model` source.
        negatives: Whether each search also runs with its gold tools taken out.
        repeat: How many times the run makes each search.
        embeddings: The cache the run's embedder goes through, to price what it lacks.
        order_seeds: Optional candidate presentation seeds; zero is identity. Every order is estimated separately.

    Raises:
        ValueError: A decider arm has no model, or the `model` source has no queries.
    """
    counted: dict[str, list[_StandIn]] = {}
    unfit: dict[str, list[bool]] = {}  # per decider, whether each unfit search is not applicable
    for arm in arms:
        if arm.decider_name is None:
            continue
        if arm.model is None:
            raise ValueError(f"arm {arm.name} has no model to stand in for")
        stand_in = _StandIn(arm.model)
        decider = choice_decider(
            arm.decider_name,
            stand_in,
            abstention=Abstention(threshold=0.0, reserved_option=arm.config["reserved_option"]),
            max_detail=DetailLevel[arm.config["max_detail"]],
            # An arm built by hand, as the direct runs build theirs, may leave its floor out.
            min_detail=DetailLevel[arm.config.get("min_detail", DetailLevel.NAME.name)],
        )
        for search in _searches(
            tasks, sources=sources, model_queries=model_queries, negatives=negatives, retriever=retriever
        ):
            for seed in [None] if order_seeds is None else order_seeds:
                presented = (
                    search.retriever
                    if seed is None
                    else OrderedRetrieval(search.retriever, task_id=search.task.id, seed=seed)
                )
                pipeline = ToolSearchPipeline(presented, decider=decider, k=arm.k)
                calls = stand_in.calls
                try:
                    await pipeline.search(list(search.queries), data.catalog, context=search.task.query)
                except CandidatesDoNotFit as error:
                    inapplicable = error.card_id is None and stand_in.calls == calls
                    unfit.setdefault(arm.decider_name, []).append(inapplicable)
        counted.setdefault(arm.decider_name, []).append(stand_in)
    lines = [
        replace(
            _decider_line(
                name,
                calls=repeat * sum(stand_in.calls for stand_in in stand_ins),
                tokens=repeat * sum(stand_in.input_tokens for stand_in in stand_ins),
                options=repeat * sum(stand_in.options for stand_in in stand_ins),
                limits=stand_ins[0].limits,
            ),
            not_applicable=repeat * sum(unfit.get(name, [])),
            would_fail=repeat * sum(not inapplicable for inapplicable in unfit.get(name, [])),
        )
        for name, stand_ins in counted.items()
    ]
    if embeddings is not None:
        queries = [
            query
            for source in sources
            for task in tasks
            for query in clean_queries(_source_queries(task, source, model_queries)[0])
        ]
        lines.append(_embedding_line(embeddings, queries=queries, catalog=data.catalog))
    return lines


def _decider_line(name: str, *, calls: int, tokens: int, options: int, limits: ModelLimits) -> EstimateLine:
    spec = REGISTRY[DeciderName(name)]
    overhead = spec.request_overhead_tokens
    priced = tokens + overhead * calls + spec.option_overhead_tokens * options
    if spec.billing == "local":
        note = f"{calls:,} asks, {tokens:,} input tokens by the heuristic count, on this machine: no charge"
        return EstimateLine("local", spec.model, calls, tokens, 0.0, note)
    if spec.billing == "gpu-time":
        seconds = calls * _CLM_GPU_SECONDS_PER_CALL
        note = (
            f"{calls:,} asks, {tokens:,} input tokens by the heuristic count; {seconds:,.0f} GPU seconds at "
            f"{_CLM_GPU_SECONDS_PER_CALL} s per ask, about ${clm_usd(seconds):.4f} of Modal credits"
        )
        return EstimateLine("modal", spec.model, calls, tokens, None, note)
    if spec.priced_by == "genai-prices":
        output = spec.estimate_output_tokens
        price = calc_price(
            Usage(input_tokens=priced, output_tokens=output * calls), model_ref=spec.model, provider_id="openai"
        )
        note = (
            f"{calls:,} asks, {priced:,} input tokens (the heuristic count plus {overhead} per ask) and {output} "
            "output tokens per ask, priced by genai-prices"
        )
        return EstimateLine("openai", spec.model, calls, priced, float(price.total_price), note)
    usd = priced * _declared_input_price(limits) / 1_000_000
    per_option = f" and {spec.option_overhead_tokens} per option" if spec.option_overhead_tokens else ""
    note = (
        f"{calls:,} asks, {priced:,} input tokens: the heuristic count plus {overhead} per ask{per_option}, at the "
        f"declared ${_declared_input_price(limits)}/M"
    )
    return EstimateLine(spec.provider, spec.model, calls, priced, usd, note)


def _embedding_line(embeddings: CachedEmbedder, *, queries: Sequence[str], catalog: ToolCatalog) -> EstimateLine:
    model = embeddings.model_id
    query_cost = estimate_embedding_cost(queries, model=model, cache=embeddings, kind="query")
    cards = (default_search_text(card) for card in catalog)
    card_cost = estimate_embedding_cost(cards, model=model, cache=embeddings, kind="document")
    note = (
        f"{query_cost.texts:,} query texts and {card_cost.texts:,} card texts not in the cache yet, "
        f"{query_cost.tokens + card_cost.tokens:,} tokens by tiktoken cl100k_base, priced by genai-prices"
    )
    return EstimateLine(
        "openai",
        model,
        query_cost.texts + card_cost.texts,
        query_cost.tokens + card_cost.tokens,
        query_cost.usd + card_cost.usd,
        note,
    )


@dataclass(frozen=True, slots=True)
class _Search:
    """One search of one task in one arm: its queries, its variant and the retriever that finds its candidates."""

    source: QuerySource
    task: ToolRetTask
    queries: tuple[str, ...]
    fallback: bool | None
    variant: Literal["positive", "negative"]
    retriever: Retriever


def _searches(
    tasks: Sequence[ToolRetTask],
    *,
    sources: Sequence[QuerySource],
    model_queries: Mapping[str, WrittenQueries] | None,
    negatives: bool,
    retriever: Retriever,
) -> Iterator[_Search]:
    """Each source, then each task, then the positive and, with `negatives`, the negative: one arm's searches."""
    for source in sources:
        for task in tasks:
            queries, fallback = _source_queries(task, source, model_queries)
            yield _Search(source, task, queries, fallback, "positive", retriever)
            if negatives:
                without_gold = ExcludingRetriever(retriever, exclude=task.relevant)
                yield _Search(source, task, queries, fallback, "negative", without_gold)


def _source_queries(
    task: ToolRetTask, source: QuerySource, model_queries: Mapping[str, WrittenQueries] | None
) -> tuple[tuple[str, ...], bool | None]:
    """The queries `task` is searched with from `source`, and whether they fell back to the request (model only)."""
    if source == "plain":
        return (task.query,), None
    if model_queries is None:
        raise ValueError("the model source needs the written queries of every task")
    written = model_queries[task.id]
    return written.queries, written.fallback


async def run_decisions(
    arms: Sequence[DecisionArm],
    data: ToolRetData,
    tasks: Sequence[ToolRetTask],
    *,
    retriever: SharedRetrieval,
    split: Split,
    sources: Sequence[QuerySource],
    model_queries: Mapping[str, WrittenQueries] | None,
    negatives: bool,
    repeat: int,
    out_dir: Path,
    task_file: Path,
    model_queries_file: Path | None,
    run_id: str | None = None,
    before_arm: Callable[[DecisionArm], Awaitable[None]] | None = None,
    order_seeds: Sequence[int] | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
    before_search: Callable[[Mapping[str, Any]], None] | None = None,
    stop_on_error: bool = True,
) -> Path:
    """Run every arm on every search; returns the run directory `out_dir/<run_id>`.

    The run is arm-major: an arm makes all of its searches before the next arm starts, one search at a time. Within an
    arm it goes through each source, task, variant (the positive, then the negative with the gold tools taken out when
    `negatives` is on) and repeat, and searches through a `ToolSearchPipeline` with the arm's decider over its K
    candidates, the task's request as the decider's context. A `DecisionError` is recorded in the search's `error` and
    the run goes on; the candidates of that search come from the shared retrieval. So is a `CandidatesDoNotFit` that
    names a card no question can show, whose id goes in `failed_card`. One without a card, raised by the planner before
    the first ask because two rounds cannot hold the candidates, is recorded in `not_applicable` instead, with no error
    and no ranking. Raised after an ask, by the decider's guard against a final that does not fit after all, it is a
    failed search: round one was asked, and its calls are not in the record.

    The directory gets `manifest.json` (provenance, written first) and `run.jsonl`: one `search` record per search,
    and after each arm's searches one `arm` record with its wall seconds, searches and errors, and, when the arm's
    model is a `CachedDecisionModel`, the asks it answered from the cache (`cache_hits`) and the asks it sent
    (`cache_misses`) during the arm; both are null otherwise. A `search` record holds
    the identity of the search (`arm`, `decider`, `k`, `source`, `split`, `variant`, `repeat` from 0, `task`), its
    input (`queries`, `fallback` for the model source, `context`, `candidates` in retrieval order, `relevant`,
    `gold_in_candidates`), the decision (`ranked`, `probabilities`, `none_probability`, `abstained`, `key`, `shape`,
    `state_cut`, `exchanges`, each with the `key_only_options` it sent with an empty text), its cost and timing
    (`usage`, `decision_seconds` on the critical path, `decision_sequential_seconds` with every call added,
    `server_seconds`, `retrieval_seconds`, the first retrieval's measured time), `not_applicable`, `failed_card`,
    `error` and `refused` (the model declined to answer, so the error is its answer). The retrieval-only arms rank by
    retrieval and have no probabilities and no decision; a search that failed or was not applicable has no ranking.

    Args:
        arms: What to run, from `build_decision_arms`.
        data: The loaded dataset; its catalog is searched.
        tasks: The tasks, usually read from `task_file`.
        retriever: The hybrid retrieval every arm shares.
        split: The split of `task_file`.
        sources: The query sources.
        model_queries: The written queries of every task, by task id; needed for the `model` source.
        negatives: Whether each search also runs with its gold tools taken out.
        repeat: How many times each search is made.
        out_dir: Parent of the run directory, usually `bench/runs`.
        task_file: The task file, hashed into the manifest.
        model_queries_file: The file `model_queries` were read from, hashed into the manifest.
        run_id: Directory name; a UTC timestamp by default.
        before_arm: Awaited with each arm before its first search, outside its wall time: to wake a server that may
            have scaled to zero during the arms before it, for example.
        order_seeds: Opt-in identity and four task-seeded candidate shuffles, with fresh decisions for every order.
        manifest_extra: Additional experiment provenance written before the run starts.
        before_search: Receives each search's identity and presented candidates before its decision.
        stop_on_error: In order mode, stop at the first failed search; off, record it and go on. A search the model
            refused (`DecisionRefused`) never stops a run: the refusal is its answer, recorded as `refused`.

    Raises:
        ValueError: The `model` source has no queries or no queries file, or `repeat` is below 1.
    """
    if "model" in sources and (model_queries is None or model_queries_file is None):
        raise ValueError("the model source needs the written queries and the file they come from")
    check_k(repeat, name="repeat")
    if order_seeds is not None:
        if list(order_seeds) != [0, 1, 2, 3, 4] or repeat != 1:
            raise ValueError("order mode needs identity plus seeds 1-4, with no additional repeats")
        if any(isinstance(arm.model, CachedDecisionModel) and not arm.model.bypasses_cache for arm in arms):
            raise ValueError("all order-mode decision models must bypass local cache")
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True)
    manifest = run_provenance(data, tasks, task_file=task_file, run_id=run_id) | {
        "split": split,
        "sources": list(sources),
        "negatives": negatives,
        "repeat": repeat,
        "model_queries": None if model_queries_file is None else _queries_file(model_queries_file),
        "embedding_model": retriever.config.get("embedding_model"),
        "retriever": {**retriever.config, "shared_depth": retriever.depth},
        "clm_deployment": dict(CLM_DEPLOYMENT) if any(arm.decider_name == "clm" for arm in arms) else None,
        "arms": {arm.name: dict(arm.config) for arm in arms},
    }
    if order_seeds is not None:
        manifest.pop("clm_deployment", None)
        manifest["orders"] = {
            "version": "tool-order-v1",
            "seeds": list(order_seeds),
            "identity_seed": 0,
            "seed_derivation": "SHA256(tool-order-v1:<task_id>:<seed>), Python Random.shuffle",
            "cache_bypass": True,
            "reserved_position": "last",
        }
        manifest["task_ids"] = [task.id for task in tasks]
    manifest.update(manifest_extra or {})
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with (run_dir / "run.jsonl").open("w", encoding="utf-8") as out:
            for arm in arms:
                if before_arm is not None:
                    await before_arm(arm)
                cached = arm.model if isinstance(arm.model, CachedDecisionModel) else None
                hits, misses = (0, 0) if cached is None else (cached.hits, cached.misses)
                started = time.perf_counter()
                searches = errors = 0
                for search in _searches(
                    tasks, sources=sources, model_queries=model_queries, negatives=negatives, retriever=retriever
                ):
                    for index, seed in enumerate([None] * repeat if order_seeds is None else order_seeds):
                        presented = (
                            search.retriever
                            if seed is None
                            else OrderedRetrieval(search.retriever, task_id=search.task.id, seed=seed)
                        )
                        context: dict[str, Any] = {
                            "arm": arm.name,
                            "task": search.task.id,
                            "variant": search.variant,
                            "order_seed": seed,
                            "source": search.source,
                        }
                        if before_search is not None:
                            before_search(context)
                        if seed is not None:
                            candidates = await presented.retrieve(list(search.queries), data.catalog, k=arm.k)
                            if len(candidates.matches) != arm.k:
                                raise ValueError("order mode needs exactly K candidates before any decision is paid")
                            context["presented_candidates"] = [match.card.id for match in candidates.matches]
                        pipeline = ToolSearchPipeline(presented, decider=arm.decider, k=arm.k)
                        if before_search is not None:
                            before_search(context)
                        record = await _search_record(
                            pipeline,
                            arm,
                            search,
                            data=data,
                            split=split,
                            repeat=index,
                            shared=retriever,
                            presented=presented if seed is not None else None,
                        )
                        if seed is not None:
                            record["order_seed"] = seed
                            record["replayed"] = False
                        out.write(json.dumps(record) + "\n")
                        if seed is not None:
                            out.flush()
                        searches += 1
                        errors += record["error"] is not None
                        if seed is not None and stop_on_error and record["error"] is not None and not record["refused"]:
                            raise DecisionError("order experiment stopped after an errored search")
                wall_seconds = time.perf_counter() - started
                summary = {
                    "record": "arm",
                    "arm": arm.name,
                    "wall_seconds": wall_seconds,
                    "searches": searches,
                    "errors": errors,
                    "cache_hits": None if cached is None else cached.hits - hits,
                    "cache_misses": None if cached is None else cached.misses - misses,
                }
                out.write(json.dumps(summary) + "\n")
                out.flush()
    except RunStopped as stop:  # the searches so far stay in run.jsonl; the manifest says why the rest is missing
        manifest["stop_reason"] = str(stop)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    return run_dir


def _queries_file(path: Path) -> dict[str, Any]:
    return {
        "file": repo_path(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "writer": json.loads(path.read_text(encoding="utf-8"))["writer"],
    }


async def _search_record(
    pipeline: ToolSearchPipeline,
    arm: DecisionArm,
    search: _Search,
    *,
    data: ToolRetData,
    split: Split,
    repeat: int,
    shared: SharedRetrieval,
    presented: Retriever | None = None,
) -> dict[str, Any]:
    queries = list(search.queries)
    decision: Decision | None = None
    error: str | None = None
    refused = False
    not_applicable: str | None = None
    failed_card: str | None = None
    asks = None if arm.asked is None else arm.asked()
    try:
        decision = (await pipeline.search(queries, data.catalog, context=search.task.query)).decision
    except DecisionError as failure:
        error, refused = str(failure), isinstance(failure, DecisionRefused)
    except CandidatesDoNotFit as unfit:
        if unfit.card_id is None and (arm.asked is None or arm.asked() == asks):
            not_applicable = str(unfit)
        else:
            error, failed_card = str(unfit), unfit.card_id
    # The search's own retrieval again: the shared retrieval kept it, so it costs nothing and outlives an error.
    retrieval = await (search.retriever if presented is None else presented).retrieve(queries, data.catalog, k=arm.k)
    candidates = [match.card.id for match in retrieval.matches]
    return {
        "record": "search",
        "arm": arm.name,
        "decider": arm.decider_name,
        "k": arm.k,
        "source": search.source,
        "split": split,
        "variant": search.variant,
        "repeat": repeat,
        "task": search.task.id,
        "queries": queries,
        "fallback": search.fallback,
        "context": search.task.query,
        "candidates": candidates,
        "relevant": sorted(search.task.relevant),
        "gold_in_candidates": not search.task.relevant.isdisjoint(candidates),
        **_decision_fields(
            decision, candidates=candidates, decided=arm.decider is not None, record_requests=presented is not None
        ),
        "retrieval_seconds": shared.first_seconds(queries),
        "not_applicable": not_applicable,
        "failed_card": failed_card,
        "error": error,
        "refused": refused,
    }


def _decision_fields(
    decision: Decision | None, *, candidates: list[str], decided: bool, record_requests: bool = False
) -> dict[str, Any]:
    if decision is None:
        # Retrieval alone ranks by retrieval; a decider that failed, or had nothing it could ask, ranked nothing.
        return {
            "ranked": None if decided else candidates,
            "probabilities": {},
            "none_probability": None,
            "abstained": None,
            "key": None,
            "shape": None,
            "state_cut": None,
            "exchanges": None,
            "usage": None,
            "decision_seconds": None,
            "decision_sequential_seconds": None,
            "server_seconds": None,
        }
    return {
        "ranked": [match.card.id for match in decision.ranked],
        "probabilities": dict(decision.probabilities),
        "none_probability": decision.none_probability,
        "abstained": decision.abstained,
        "key": str(decision.key),
        "shape": dict(decision.shape),
        "state_cut": decision.state_cut,
        "exchanges": [
            {
                "round": exchange.round,
                "detail": exchange.detail.name,
                "estimated_input_tokens": exchange.estimated_input_tokens,
                "input_tokens": exchange.response.usage.input_tokens,
                "output_tokens": exchange.response.usage.output_tokens,
                "seconds": exchange.response.seconds,
                "server_seconds": exchange.response.server_seconds,
                "key_only_options": [
                    key
                    for question in exchange.request.questions.values()
                    if isinstance(question, ChoiceQuestion)
                    for key, text in question.options.items()
                    if text == ""
                ],
                **(
                    {
                        "request": {
                            "state": exchange.request.state,
                            "questions": {
                                key: {"instructions": question.instructions, "options": list(question.options.items())}
                                for key, question in exchange.request.questions.items()
                                if isinstance(question, ChoiceQuestion)
                            },
                        },
                        "option_card_ids": dict(exchange.option_card_ids),
                        "candidate_order": list(exchange.option_card_ids.values()),
                        "response": {
                            "answers": {
                                key: dict(answer.probabilities)
                                for key, answer in exchange.response.answers.items()
                                if isinstance(answer, ChoiceAnswer)
                            },
                            "raw": dict(exchange.response.raw),
                        },
                    }
                    if record_requests
                    else {}
                ),
            }
            for exchange in decision.exchanges
        ],
        "usage": asdict(decision.usage),
        "decision_seconds": decision.seconds,
        "decision_sequential_seconds": decision.sequential_seconds,
        "server_seconds": decision.server_seconds,
    }
