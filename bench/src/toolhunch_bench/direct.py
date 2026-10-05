"""Direct selection from per-source ToolRet catalogs, with five fixed strategies."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Literal

from toolhunch import DetailLevel, ToolCatalog, ToolSearchPipeline, default_search_text
from toolhunch.decision import Abstention, CandidatesDoNotFit, ChoiceQuestion
from toolhunch.decision.planner import PROMPT_VERSION, build_state, fit_state, plan_rounds
from toolhunch.retrieval import Retrieval, ScoredCard
from toolhunch.retrieval.base import clean_queries
from toolhunch.tokens import HeuristicTokenizer
from toolhunch_bench.datasets.toolret import TOOLRET_CORPUS_SHA256, TOOLRET_DATASET, TOOLRET_REVISION, ToolRetData
from toolhunch_bench.deciders import DECIDERS, DeciderName, SerialDecisionModel, choice_decider, planner_tokenizer
from toolhunch_bench.decision import LUNA_MODEL, CountedModel, DecisionArm, SharedRetrieval, estimate_decisions
from toolhunch_bench.decision_cache import CachedDecisionModel
from toolhunch_bench.direct_agent import (
    AGENT_MAX_OUTPUT_TOKENS,
    AGENT_PROMPT_VERSION,
    CachedAgent,
    function_cards,
    wire_request,
)
from toolhunch_bench.direct_cost import (
    AGENT_MODEL,
    EMBEDDING_MODEL,
    GuardedDecisionModel,
    ProviderFailure,
    RunStopped,
    SpendGuard,
    openai_usd,
)
from toolhunch_bench.embedding_cache import estimate_embedding_cost

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from toolhunch import Retriever, ToolCard
    from toolhunch.decision import DecisionModel
    from toolhunch.retrieval import EmbeddingKind
    from toolhunch.tokens import Tokenizer
    from toolhunch_bench.datasets.toolret import ToolRetTask
    from toolhunch_bench.embedding_cache import CachedEmbedder

CATALOGS: Mapping[str, str] = {
    "webtools_spotify": "restgpt-spotify",
    "webtools_tmdb": "restgpt-tmdb",
    "tooleyes": "tooleyes",
    "apibank": "apibank",
    "metatool_which": "metatool",
}
DIRECT_ARMS = ("hybrid@20", "hybrid@20+jev", "jev-all", "agent@20", "agent-all")
LUNA_ARMS = ("agent-luna@20", "agent-luna-all")
"""The agent arms again with GPT-6 Luna, reasoning off: a later run, reported beside `DIRECT_ARMS`."""
AGENT_MODELS: Mapping[str, str] = {
    "agent@20": AGENT_MODEL,
    "agent-all": AGENT_MODEL,
    "agent-luna@20": LUNA_MODEL,
    "agent-luna-all": LUNA_MODEL,
}
SEARCHING_ARMS = frozenset({"hybrid@20", "hybrid@20+jev", "agent@20", "agent-luna@20"})
_DECIDER_SEARCH = "hybrid@20+"
_DECIDER_ALL = "-all"
_ABSTENTION = Abstention()  # every decider of a direct run: the reserved option, threshold 0
_FINALISTS_PER_CHUNK = 2  # `ChoiceDecider`'s default, which the deciders of a direct run keep


def decider_arms(names: Sequence[str]) -> tuple[str, ...]:
    """Each decider's two arms: search then the decider (`hybrid@20+<name>`), and the whole catalog (`<name>-all`)."""
    return tuple(arm for name in names for arm in (f"{_DECIDER_SEARCH}{name}", f"{name}{_DECIDER_ALL}"))


def arm_decider(arm: str) -> str | None:
    """The decider an arm asks, such as `jev` for `hybrid@20+jev` and `jev-all`; `None` for search or an agent."""
    if arm.startswith(_DECIDER_SEARCH):
        return arm.removeprefix(_DECIDER_SEARCH)
    if arm.endswith(_DECIDER_ALL) and not arm.startswith("agent"):
        return arm.removesuffix(_DECIDER_ALL)
    return None


def arm_searches(arm: str) -> bool:
    """Whether an arm searches first: hybrid alone, a decider after search, or an agent over 20 searched tools."""
    return arm in SEARCHING_ARMS or arm.startswith(_DECIDER_SEARCH)


NOT_APPLICABLE: Mapping[tuple[str, str], Mapping[str, str]] = {
    ("agent-all", "metatool_which"): {
        "reason": "OpenAI Chat Completions rejected 200 function tools: HTTP 400 "
        "array_above_max_length, param tools (4/4 attempts); exact maximum not established",
        "source": "bench/runs/20260929T221520Z-direct-pilot/run.jsonl",
        "date": "2026-09-30",
    },
    ("agent-luna-all", "metatool_which"): {
        "reason": "OpenAI Chat Completions rejected 200 function tools for gpt-6-luna: HTTP 400 "
        "array_above_max_length, param tools (4/4 attempts)",
        "source": "bench/runs/20260930T165646Z-direct-luna-pilot/run.jsonl",
        "date": "2026-09-30",
    },
}


@dataclass(frozen=True, slots=True)
class DirectCatalog:
    """One fixed positive catalog, its scored requests, and the rejected-positive count."""

    source: str
    subtask: str
    catalog: ToolCatalog
    positives: tuple[ToolRetTask, ...]
    negatives: tuple[ToolRetTask, ...]
    dropped: int

    def requests(self) -> Sequence[tuple[ToolRetTask, str, ToolCatalog, str]]:
        """Positives in fixed order followed by gold-removed negatives; only positives define cold/warm phases."""
        return [
            (task, "positive", self.catalog, "cold" if position == 0 else "warm")
            for position, task in enumerate(self.positives)
        ] + [
            (task, "negative", ToolCatalog(card for card in self.catalog if card.id not in task.relevant), "negative")
            for task in self.negatives
        ]


def direct_catalogs(
    data: ToolRetData,
    *,
    pilot: bool = False,
    sources: Mapping[str, str] = CATALOGS,
) -> list[DirectCatalog]:
    """Select all contained positive tasks and deterministic negatives, allocating odd remainders in source order.

    The full run has floor(total positives / 2) negatives; an odd catalog receives its extra request before later
    odd catalogs. Pilot requests are stable subsets of the full request lists, so they can be replayed in the full run.
    """
    catalogs = {source: ToolCatalog(card for card in data.catalog if card.source == source) for source in sources}
    positives: dict[str, tuple[ToolRetTask, ...]] = {}
    dropped: dict[str, int] = {}
    for source, subtask in sources.items():
        ids = {card.id for card in catalogs[source]}
        tasks = [task for task in data.tasks if task.subtask == subtask]
        positives[source] = tuple(task for task in tasks if task.relevant and task.relevant <= ids)
        dropped[source] = len(tasks) - len(positives[source])
        if not catalogs[source] or not positives[source]:
            raise ValueError(f"no contained tasks or tools for source {source}")
    counts = {source: len(tasks) // 2 for source, tasks in positives.items()}
    remaining = sum(map(len, positives.values())) // 2 - sum(counts.values())
    for source in sources:
        if remaining and len(positives[source]) % 2:
            counts[source] += 1
            remaining -= 1
    selected: list[DirectCatalog] = []
    for source, subtask in sources.items():
        negatives = tuple(random.Random(0).sample(positives[source], k=counts[source]))
        selected.append(
            DirectCatalog(
                source=source,
                subtask=subtask,
                catalog=catalogs[source],
                positives=positives[source][:10] if pilot else positives[source],
                negatives=negatives[:5] if pilot else negatives,
                dropped=dropped[source],
            )
        )
    return selected


def planned_not_applicable(
    catalogs: Sequence[DirectCatalog],
    *,
    arms: Sequence[str],
    models: Mapping[str, DecisionModel],
    max_detail: Mapping[str, DetailLevel] | None = None,
    min_detail: Mapping[str, DetailLevel] | None = None,
) -> dict[tuple[str, str], dict[str, str]]:
    """The `<name>-all` arms and catalogs with a request that does not fit the two-round policy, found asking nothing.

    Each request of each catalog is planned as the run would ask it: the state `ToolSearchPipeline` builds from the
    request, cut by `fit_state`, then `plan_rounds` over the whole catalog with the reserved option, the decider's
    declared limits and `planner_tokenizer`, and cards at `max_detail[name]` at most (`FULL` when left out) and
    `min_detail[name]` at least (`NAME`). Every request of an `-all` arm shares its catalog, so the first
    `CandidatesDoNotFit` makes the pair not applicable, with the planner's message as its reason, as `NOT_APPLICABLE`
    records a pair. Searching arms are left out: their candidates are known only once searched.

    Raises:
        ValueError: A request's state does not fit its model's state budget even once cut (see `fit_state`).
    """
    today = datetime.now(UTC).date().isoformat()
    found: dict[tuple[str, str], dict[str, str]] = {}
    for arm in arms:
        if (name := arm_decider(arm)) is None or arm_searches(arm):
            continue
        limits, tokenizer = models[name].limits, _CountOnce(planner_tokenizer(name))
        top = (max_detail or {}).get(name, DetailLevel.FULL)
        floor = (min_detail or {}).get(name, DetailLevel.NAME)
        # `plan_rounds` reads the state only through its token count, so a request with the same cards and as long a
        # state has the same plan: the positives of a catalog differ in their state alone.
        planned: set[tuple[int, tuple[str, ...]]] = set()
        for selected in catalogs:
            for task, _variant, catalog, _phase in selected.requests():
                cards = list(catalog)
                if len(cards) + _ABSTENTION.reserved_option < 2:
                    continue  # the decider asks nothing
                state, _ = fit_state(
                    build_state(task.query, clean_queries([task.query])), limits=limits, tokenizer=tokenizer
                )
                key = (tokenizer.count(state), tuple(card.id for card in cards))
                if key in planned:
                    continue
                planned.add(key)
                try:
                    plan_rounds(
                        state,
                        cards,
                        reserved=_ABSTENTION.reserved_option,
                        limits=limits,
                        tokenizer=tokenizer,
                        max_detail=top,
                        min_detail=floor,
                        finalists_per_chunk=_FINALISTS_PER_CHUNK,
                    )
                except CandidatesDoNotFit as unfit:
                    found[(arm, selected.source)] = {"reason": str(unfit), "source": "planner", "date": today}
                    break
    return found


def direct_decision_model(
    name: str, adapter: DecisionModel, *, guard: SpendGuard | None, path: Path
) -> CachedDecisionModel:
    """What a direct run asks for the decider `name`: `adapter`, through `guard` when given, under the decision cache.

    The guard reserves, records and times every attempt. An entry asked one request at a time takes its lock outside
    the guard, so each call is timed alone, without its wait for the lock: a first round's questions are asked at the
    same time. The cache keys the entry's `cache_namespace`, which keeps apart entries that ask one model id.
    """
    spec = DECIDERS[DeciderName(name)]
    model = adapter if guard is None else GuardedDecisionModel(adapter, guard=guard, provider=spec.provider)
    return CachedDecisionModel(
        SerialDecisionModel(model) if spec.serial else model, path=path, namespace=spec.cache_namespace
    )


class _CountOnce:
    """`inner`, counting each distinct text once: planning a catalog counts the same card texts again and again."""

    def __init__(self, inner: Tokenizer) -> None:
        self._inner = inner
        self._counts: dict[str, int] = {}

    def count(self, text: str, /) -> int:
        if (tokens := self._counts.get(text)) is None:
            tokens = self._counts[text] = self._inner.count(text)
        return tokens


class CatalogOrderRetriever:
    """Return the entire catalog in its declared order, with no search or paid usage."""

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """Keep catalog order for the direct Jev question."""
        return Retrieval(matches=tuple(ScoredCard(card=card, score=0) for card in catalog)[:k])


@dataclass(frozen=True, slots=True)
class DirectEstimate:
    """Prepaid estimate; legacy request totals describe the workload before per-arm applicability exclusions."""

    usd: float
    requests_per_arm: int
    positives_per_arm: int
    negatives_per_arm: int
    lines: tuple[Mapping[str, Any], ...]
    stand_in_searches: int
    planned_requests_by_arm: Mapping[str, int]
    candidate_bound_searches: Mapping[str, int]
    candidate_bound_method: str


class _DirectEstimateRetrieval:
    """Bound short free stand-ins for direct estimates without changing actual retrieval."""

    def __init__(self, inner: Retriever, *, card_size: Callable[[ToolCard], int]) -> None:
        self._inner, self._card_size = inner, card_size
        self.bounded: set[tuple[tuple[str, ...], str]] = set()
        self._longest: dict[str, tuple[ScoredCard, ...]] = {}

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        retrieval = await self._inner.retrieve(queries, catalog, k=k)
        diagnostic = getattr(self._inner, "uses_stand_in", None)
        if len(retrieval.matches) >= min(k, len(catalog)) or diagnostic is None or not diagnostic(queries, catalog):
            return retrieval
        self.bounded.add((tuple(clean_queries(queries)), catalog.fingerprint))
        if catalog.fingerprint not in self._longest:
            self._longest[catalog.fingerprint] = tuple(
                ScoredCard(card=card, score=0) for card in sorted(catalog, key=self._card_size, reverse=True)
            )
        return Retrieval(matches=self._longest[catalog.fingerprint][:k], usage=retrieval.usage)


async def estimate_direct(
    catalogs: Sequence[DirectCatalog],
    *,
    retriever: Retriever,
    models: Mapping[str, DecisionModel],
    embeddings: CachedEmbedder | None = None,
    not_applicable: Mapping[tuple[str, str], Mapping[str, str]] = NOT_APPLICABLE,
    arms: Sequence[str] = DIRECT_ARMS,
    max_detail: Mapping[str, DetailLevel] | None = None,
    min_detail: Mapping[str, DetailLevel] | None = None,
) -> DirectEstimate:
    """Estimate identical planned requests without contacting a decision model or a missing embedding.

    Each decider's planning uses the existing decision estimator, with its own declared limits, its tokenizer and
    the run's detail ceiling and floor (`FULL` and `NAME` for a decider `max_detail` or `min_detail` leaves out).
    Agent input is tokenized from prompt-bearing wire fields; output is bounded by its configured limit and cache
    savings are not assumed. Replay savings are not assumed. Missing dense vectors with fewer than K lexical matches
    use the longest applicable FULL cards, separately measured for Jev's heuristic text and the agent's function
    schema. The actual Jev planner still applies limits and detail reduction. Full-length lexical stand-ins remain
    approximate, not a guaranteed whole-run upper bound.
    """
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    jev_retrieval = _DirectEstimateRetrieval(
        retriever, card_size=lambda card: HeuristicTokenizer().count(card.render(DetailLevel.FULL))
    )
    agent_retrieval = _DirectEstimateRetrieval(
        retriever,
        card_size=lambda card: len(
            encoding.encode(
                json.dumps(wire_request("", function_cards([card]))["tools"][0], ensure_ascii=False),
                disallowed_special=(),
            )
        ),
    )
    counts: dict[str, dict[str, Any]] = {}
    query_texts: list[str] = []
    document_texts: list[str] = []
    positives = negatives = 0
    planned: dict[str, int] = dict.fromkeys(arms, 0)
    for selected in catalogs:
        positives += len(selected.positives)
        negatives += len(selected.negatives)
        searches = any((arm, selected.source) not in not_applicable for arm in arms if arm_searches(arm))
        if searches:
            document_texts += [default_search_text(card) for card in selected.catalog]
        for arm in arms:
            if (arm, selected.source) not in not_applicable:
                planned[arm] += len(selected.positives) + len(selected.negatives)
        for task, _variant, catalog, _ in selected.requests():
            if searches:
                query_texts.append(task.query)
            data = ToolRetData(catalog=catalog, tasks=(task,), raw_text={}, mapping_stats={})
            decider_arms_here = [
                DecisionArm(
                    name=name,
                    decider_name=decider,
                    k=20 if name.startswith(_DECIDER_SEARCH) else len(catalog),
                    decider=None,
                    config={
                        "max_detail": (max_detail or {}).get(decider, DetailLevel.FULL).name,
                        "min_detail": (min_detail or {}).get(decider, DetailLevel.NAME).name,
                        "reserved_option": _ABSTENTION.reserved_option,
                    },
                    model=models[decider],
                )
                for name in arms
                if (decider := arm_decider(name)) is not None
            ]
            for arm in decider_arms_here:
                if (arm.name, selected.source) in not_applicable:
                    continue
                lines = await estimate_decisions(
                    [arm],
                    data,
                    [task],
                    retriever=jev_retrieval if arm.name.startswith(_DECIDER_SEARCH) else CatalogOrderRetriever(),
                    sources=("plain",),
                    model_queries=None,
                    negatives=False,
                    repeat=1,
                )
                for line in lines:
                    key = arm.name
                    row = counts.setdefault(
                        key,
                        {
                            "provider": line.provider,
                            "model": line.model,
                            "arm": arm.name,
                            "requests": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "usd": 0.0,
                        },
                    )
                    row["requests"] += line.calls
                    row["input_tokens"] += line.input_tokens
                    row["usd"] += line.usd or 0.0
            agent_arms = [arm for arm in arms if arm in AGENT_MODELS and (arm, selected.source) not in not_applicable]
            candidates = []
            if any(arm.endswith("@20") for arm in agent_arms):
                retrieved = await agent_retrieval.retrieve([task.query], catalog, k=20)
                candidates = [match.card for match in retrieved.matches]
            for agent_arm in agent_arms:
                cards = candidates if agent_arm.endswith("@20") else list(catalog)
                model = AGENT_MODELS[agent_arm]
                payload = wire_request(task.query, function_cards(cards))
                tokens = len(encoding.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=())) + 120
                row = counts.setdefault(
                    agent_arm,
                    {
                        "provider": "openai",
                        "model": model,
                        "arm": agent_arm,
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "usd": 0.0,
                    },
                )
                row["requests"] += 1
                row["input_tokens"] += tokens
                row["output_tokens"] += AGENT_MAX_OUTPUT_TOKENS
                row["usd"] += openai_usd(model=model, input_tokens=tokens, output_tokens=AGENT_MAX_OUTPUT_TOKENS)
    batches: tuple[tuple[EmbeddingKind, list[str]], ...] = (("query", query_texts), ("document", document_texts))
    for kind, texts in batches:
        price = estimate_embedding_cost(
            texts,
            model=EMBEDDING_MODEL,
            cache=embeddings,
            kind=kind,
            max_tokens=8191,
        )
        row = counts.setdefault(
            "embeddings",
            {
                "provider": "openai",
                "model": EMBEDDING_MODEL,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "usd": 0.0,
            },
        )
        row["requests"] += price.texts
        row["input_tokens"] += price.tokens
        row["usd"] += price.usd
    return DirectEstimate(
        usd=sum(row["usd"] for row in counts.values()),
        requests_per_arm=positives + negatives,
        positives_per_arm=positives,
        negatives_per_arm=negatives,
        lines=tuple(counts.values()),
        stand_in_searches=len(getattr(retriever, "stand_in_queries", ())),
        planned_requests_by_arm=planned,
        candidate_bound_searches={
            **{arm: len(jev_retrieval.bounded) for arm in arms if arm.startswith(_DECIDER_SEARCH)},
            "agent@20": len(agent_retrieval.bounded),
        },
        candidate_bound_method="Short/empty missing-dense BM25 stand-ins: min(K, applicable catalog size) longest "
        "FULL cards by Jev heuristic text or agent wire tokens; actual Jev planner limits/detail retained. "
        "Complete cached hybrid matches unchanged; other lexical stand-ins approximate, not a guaranteed bound.",
    )


class DirectRunner:
    """Run every catalog and arm in fixed order, sharing search but recording each strategy's search overhead.

    Each decider is the registry's `choice_decider`, planning with its entry's tokenizer, with cards at
    `max_detail[name]` at most (`FULL` for a name it leaves out) and `min_detail[name]` at least (`NAME`).
    The manifest records `query_embeddings`, whether the retriever embedded each query afresh or read it from the
    embedding cache, and `agent_cache`, the name of the agents' replay file.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        models: Mapping[str, DecisionModel],
        agent: CachedAgent | None,
        guard: SpendGuard,
        not_applicable: Mapping[tuple[str, str], Mapping[str, str]] = NOT_APPLICABLE,
        arms: Sequence[str] = DIRECT_ARMS,
        luna_agent: CachedAgent | None = None,
        provenance: Mapping[str, Mapping[str, Any]] | None = None,
        max_detail: Mapping[str, DetailLevel] | None = None,
        min_detail: Mapping[str, DetailLevel] | None = None,
        query_embeddings: Literal["fresh", "cached"] = "cached",
        agent_cache: str | None = None,
    ) -> None:
        self.retriever = SharedRetrieval(retriever)
        self.models, self.agent, self.guard = models, agent, guard
        self.arms, self.luna_agent = tuple(arms), luna_agent
        self.not_applicable = not_applicable
        self.provenance = dict(provenance or {})
        self.max_detail, self.min_detail = dict(max_detail or {}), dict(min_detail or {})
        self.query_embeddings: Literal["fresh", "cached"] = query_embeddings
        self.agent_cache = agent_cache
        self._searches: dict[tuple[str, str], tuple[Retrieval, float, float]] = {}

    async def run(
        self,
        catalogs: Sequence[DirectCatalog],
        *,
        out_dir: Path,
        run_id: str,
        pilot: bool,
        estimate: DirectEstimate,
    ) -> Path:
        """Write requests incrementally and stop on budget exhaustion, a crash, or more than 5% errors in an arm.

        Only provider errors count towards the 5%: a search the planner could not ask about is recorded and reported,
        and the run goes on. As in decision runs, a search whose candidates do not fit the two-round policy has its
        reason in `not_applicable` and no error; one with a card no question can show names it in `failed_card`.
        """
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = _manifest(
            catalogs,
            run_id=run_id,
            pilot=pilot,
            estimate=estimate,
            arms=self.arms,
            query_embeddings=self.query_embeddings,
            agent_cache=self.agent_cache,
        )
        manifest["not_applicable"] = [
            {"arm": arm, "catalog": source, **evidence}
            for (arm, source), evidence in self.not_applicable.items()
            if any(c.source == source for c in catalogs)
        ]
        planned = {
            arm: sum(
                len(c.positives) + len(c.negatives) for c in catalogs if (arm, c.source) not in self.not_applicable
            )
            for arm in self.arms
        }
        manifest["planned_requests_by_arm"] = planned
        # Rounds are not assumed: each record lists its exchanges, one per question asked.
        manifest["deciders"] = {
            name: {
                "model": model.model_id,
                "limits": model.limits.model_dump(mode="json"),
                "max_detail": self.max_detail.get(name, DetailLevel.FULL).name,
                "min_detail": self.min_detail.get(name, DetailLevel.NAME).name,
                "tokenizer": repr(planner_tokenizer(name)),
                "abstention": {"reserved_option": True, "threshold": 0},
                "provenance": dict(self.provenance.get(name, {})),
            }
            for name, model in self.models.items()
            if any(arm_decider(arm) == name for arm in self.arms)
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        completed = False
        errors = dict.fromkeys(self.arms, 0)
        provider_errors = dict.fromkeys(self.arms, 0)
        counts = dict.fromkeys(self.arms, 0)
        stop_reason: str | None = None
        try:
            with (run_dir / "run.jsonl").open("w") as output:
                for selected in catalogs:
                    for arm in self.arms:
                        if (arm, selected.source) in self.not_applicable:
                            continue
                        for task, variant, catalog, phase in selected.requests():
                            self.guard.context = {
                                "catalog": selected.source,
                                "arm": arm,
                                "task": task.id,
                                "variant": variant,
                                "phase": phase,
                            }
                            record, provider_failed = await self._request(
                                selected, arm=arm, task=task, variant=variant, catalog=catalog, phase=phase
                            )
                            output.write(json.dumps(record) + "\n")
                            output.flush()
                            counts[arm] += 1
                            errors[arm] += record["error"] is not None
                            provider_errors[arm] += provider_failed
                            if provider_errors[arm] / planned[arm] > 0.05:
                                raise ProviderFailure(f"more than 5% errored requests in {arm}")
                completed = True
        except (ProviderFailure, RunStopped) as error:
            stop_reason = str(error)
        except Exception as error:
            stop_reason = f"crash: {type(error).__name__}"
        finally:
            manifest |= {
                "completed": completed,
                "stop_reason": stop_reason,
                "counts": counts,
                "errors": errors,
                "provider_errors": provider_errors,
                "budget_charge_usd": self.guard.run_usd,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            with (run_dir / "calls.jsonl").open("w") as calls:
                for call in self.guard.calls:
                    calls.write(json.dumps(call) + "\n")
        return run_dir

    async def _retrieval(self, task: ToolRetTask, catalog: ToolCatalog) -> tuple[Retrieval, float, float]:
        key = (task.query, catalog.fingerprint)
        if key not in self._searches:
            before = len(self.guard.calls)
            started = time.perf_counter()
            retrieval = await self.retriever.retrieve([task.query], catalog, k=20)
            usd = sum(call["usd"] or 0 for call in self.guard.calls[before:])
            self._searches[key] = retrieval, time.perf_counter() - started, usd
        return self._searches[key]

    async def _request(
        self,
        selected: DirectCatalog,
        *,
        arm: str,
        task: ToolRetTask,
        variant: str,
        catalog: ToolCatalog,
        phase: str,
    ) -> tuple[dict[str, Any], bool]:
        """One request's record, and whether a provider failed it."""
        record: dict[str, Any] = {
            "catalog": selected.source,
            "arm": arm,
            "task": task.id,
            "variant": variant,
            "phase": phase,
            "catalog_fingerprint": catalog.fingerprint,
            "relevant": sorted(task.relevant),
            "pick": None,
            "abstained": None,
            "ranked": [],
            "candidates": [],
            "error": None,
            "replayed": False,
            "search_seconds": 0.0,
            "search_usd": 0.0,
            "decision_seconds": None,
            "extra_calls": 0,
            "text_is_none": None,
            "detail": [],
            "key_only_options": [],
            "card_options": [],
            "not_applicable": None,
            "failed_card": None,
            "decision_key": None,
            "historical_usage": None,
            "provider_calls": [],
        }
        before: int | None = None
        provider_failed = False
        try:
            if arm_searches(arm):
                retrieval, seconds, usd = await self._retrieval(task, catalog)
                cards = [match.card for match in retrieval.matches]
                record |= {"search_seconds": seconds, "search_usd": usd}
            else:
                cards = list(catalog)
            record["candidates"] = [card.id for card in cards]
            before = len(self.guard.calls)
            if arm == "hybrid@20":
                record |= {"ranked": record["candidates"], "pick": cards[0].id if cards else None, "abstained": False}
            elif (decider := arm_decider(arm)) is not None:
                counted = CountedModel(self.models[decider])
                pipeline = ToolSearchPipeline(
                    CatalogOrderRetriever(),
                    k=max(1, len(cards)),
                    decider=choice_decider(
                        decider,
                        counted,
                        abstention=_ABSTENTION,
                        max_detail=self.max_detail.get(decider, DetailLevel.FULL),
                        min_detail=self.min_detail.get(decider, DetailLevel.NAME),
                    ),
                )
                try:
                    result = await pipeline.search([task.query], ToolCatalog(cards), context=task.query)
                except CandidatesDoNotFit as unfit:
                    # As in decision runs: raised before any ask without a card, the list does not fit the two-round
                    # policy. With a card, or after round one was asked, the search failed.
                    if unfit.card_id is None and counted.count() == 0:
                        record["not_applicable"] = str(unfit)
                    else:
                        record |= {"error": str(unfit), "failed_card": unfit.card_id}
                else:
                    assert result.decision is not None
                    decision = result.decision
                    record |= {
                        "ranked": result.ids,
                        "pick": None if result.abstained or not result.ids else result.ids[0],
                        "abstained": result.abstained,
                        "decision_key": str(decision.key),
                        "detail": [exchange.detail.name for exchange in decision.exchanges],
                        # Per exchange, like `detail`: the options sent as their key alone, with an empty text, and
                        # the card options sent, the reserved option left out.
                        "key_only_options": [
                            [
                                key
                                for question in exchange.request.questions.values()
                                if isinstance(question, ChoiceQuestion)
                                for key, text in question.options.items()
                                if text == ""
                            ]
                            for exchange in decision.exchanges
                        ],
                        "card_options": [len(exchange.option_card_ids) for exchange in decision.exchanges],
                        "state_cut": decision.state_cut,
                    }
                    record["replayed"] = len(self.guard.calls) == before and bool(decision.exchanges)
                    record["historical_usage"] = asdict(decision.usage)
            else:
                agent = self.luna_agent if arm in LUNA_ARMS else self.agent
                assert agent is not None, f"no agent for {arm}"
                answer, replayed = await agent.ask(task.query, function_cards(cards), catalog_name=selected.source)
                record |= {
                    "pick": answer.pick,
                    "abstained": answer.pick is None,
                    "extra_calls": answer.extra_calls,
                    "text_is_none": answer.text_is_none,
                    "replayed": replayed,
                    "detail": ["FULL"],
                    "historical_usage": {
                        "input_tokens": answer.input_tokens,
                        "cache_read_tokens": answer.cache_read_tokens,
                        "output_tokens": answer.output_tokens,
                    },
                }
        except ProviderFailure as error:
            record["error"] = str(error)
            provider_failed = True
        finally:
            if before is not None:
                own = self.guard.calls[before:]
                record["provider_calls"] = own
                record["decision_seconds"] = sum(call["seconds"] for call in own) if own else None
        return record, provider_failed


def _manifest(
    catalogs: Sequence[DirectCatalog],
    *,
    run_id: str,
    pilot: bool,
    estimate: DirectEstimate,
    arms: Sequence[str],
    query_embeddings: Literal["fresh", "cached"],
    agent_cache: str | None,
) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10).stdout.strip()

    models = sorted({AGENT_MODELS[arm] for arm in arms if arm in AGENT_MODELS})
    tasks = {
        selected.source: {
            "positive": [task.id for task in selected.positives],
            "negative": [task.id for task in selected.negatives],
        }
        for selected in catalogs
    }
    return {
        "run_id": run_id,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "pilot": pilot,
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
        "versions": {package: version(package) for package in ("pydantic-ai-slim", "genai-prices", "toolhunch")},
        "dataset": {"name": TOOLRET_DATASET, "revision": TOOLRET_REVISION, "corpus_sha256": TOOLRET_CORPUS_SHA256},
        "catalogs": [
            {
                "source": c.source,
                "subtask": c.subtask,
                "tools": len(c.catalog),
                "fingerprint": c.catalog.fingerprint,
                "positives": len(c.positives),
                "negatives": len(c.negatives),
                "dropped": c.dropped,
            }
            for c in catalogs
        ],
        "tasks": tasks,
        "tasks_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
        "arms": list(arms),
        "query_embeddings": query_embeddings,
        "agent_cache": agent_cache,
        "order": "catalog, arm, positives, negatives",
        "decision_basis": "sum of calls",
        "concurrency": 1,
        "negative_seed": 0,
        "negative_allocation": "largest remainder, stable source order",
        "prompt_versions": {"jev": PROMPT_VERSION, "agent": AGENT_PROMPT_VERSION},
        "agent": {
            "model": models[0] if len(models) == 1 else models,
            **({"reasoning_effort": "none"} if LUNA_MODEL in models else {}),
            "api": "openai-chat-completions",
            "endpoint": "api.openai.com",
            "description_shape": "JSON object with original name and complete description",
            "parameters": "names as string properties",
            "max_output_tokens": AGENT_MAX_OUTPUT_TOKENS,
            "retry": 1,
            "cache_key": "catalog source",
        },
        "estimate": asdict(estimate),
        "completed": False,
    }
