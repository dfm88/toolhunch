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
from typing import TYPE_CHECKING, Any

from toolhunch import DetailLevel, ToolCatalog, ToolSearchPipeline, default_search_text
from toolhunch.decision import Abstention, ChoiceDecider
from toolhunch.decision.planner import PROMPT_VERSION
from toolhunch.retrieval import Retrieval, ScoredCard
from toolhunch_bench.datasets.toolret import TOOLRET_CORPUS_SHA256, TOOLRET_DATASET, TOOLRET_REVISION, ToolRetData
from toolhunch_bench.decision import DecisionArm, SharedRetrieval, estimate_decisions
from toolhunch_bench.direct_agent import (
    AGENT_MAX_OUTPUT_TOKENS,
    AGENT_PROMPT_VERSION,
    CachedAgent,
    agent_payload,
    function_cards,
)
from toolhunch_bench.direct_cost import (
    AGENT_MODEL,
    EMBEDDING_MODEL,
    ProviderFailure,
    SpendGuard,
    SpendLimit,
    openai_usd,
)
from toolhunch_bench.embedding_cache import estimate_embedding_cost

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from toolhunch import Retriever
    from toolhunch.decision import DecisionModel
    from toolhunch.retrieval import EmbeddingKind
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


class CatalogOrderRetriever:
    """Return the entire catalog in its declared order, with no search or paid usage."""

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """Keep catalog order for the direct Jev question."""
        return Retrieval(matches=tuple(ScoredCard(card=card, score=0) for card in catalog)[:k])


@dataclass(frozen=True, slots=True)
class DirectEstimate:
    """Prepaid estimate, with per-provider lines and the number of scored requests in each arm."""

    usd: float
    requests_per_arm: int
    positives_per_arm: int
    negatives_per_arm: int
    lines: tuple[Mapping[str, Any], ...]
    stand_in_searches: int


async def estimate_direct(
    catalogs: Sequence[DirectCatalog],
    *,
    retriever: Retriever,
    jev_model: DecisionModel,
    embeddings: CachedEmbedder | None = None,
) -> DirectEstimate:
    """Estimate identical planned requests without contacting a decision model or a missing embedding.

    Jev planning uses the existing decision estimator. Agent input is tokenized from its canonical function request;
    output is bounded by its configured limit and cache savings are not assumed. Replay savings are not assumed.
    """
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    counts: dict[str, dict[str, Any]] = {}
    query_texts: list[str] = []
    document_texts: list[str] = []
    positives = negatives = 0
    for selected in catalogs:
        positives += len(selected.positives)
        negatives += len(selected.negatives)
        document_texts += [default_search_text(card) for card in selected.catalog]
        for task, _variant, catalog, _ in selected.requests():
            query_texts.append(task.query)
            data = ToolRetData(catalog=catalog, tasks=(task,), raw_text={}, mapping_stats={})
            models = [
                DecisionArm(
                    name=name,
                    decider_name="jev",
                    k=20 if name.endswith("+jev") else len(catalog),
                    decider=None,
                    config={"max_detail": "FULL", "reserved_option": True},
                    model=jev_model,
                )
                for name in ("hybrid@20+jev", "jev-all")
            ]
            for arm in models:
                lines = await estimate_decisions(
                    [arm],
                    data,
                    [task],
                    retriever=retriever if arm.name == "hybrid@20+jev" else CatalogOrderRetriever(),
                    sources=("plain",),
                    model_queries=None,
                    negatives=False,
                    repeat=1,
                )
                for line in lines:
                    key = "jev"
                    row = counts.setdefault(
                        key,
                        {
                            "provider": line.provider,
                            "model": line.model,
                            "requests": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "usd": 0.0,
                        },
                    )
                    row["requests"] += line.calls
                    row["input_tokens"] += line.input_tokens
                    row["usd"] += line.usd or 0.0
            retrieved = await retriever.retrieve([task.query], catalog, k=20)
            candidates = [match.card for match in retrieved.matches]
            for cards in (candidates, list(catalog)):
                payload = agent_payload(task.query, function_cards(cards), catalog_name=selected.source)
                tokens = len(encoding.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=())) + 120
                row = counts.setdefault(
                    "agent",
                    {
                        "provider": "openai",
                        "model": AGENT_MODEL,
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "usd": 0.0,
                    },
                )
                row["requests"] += 1
                row["input_tokens"] += tokens
                row["output_tokens"] += AGENT_MAX_OUTPUT_TOKENS
                row["usd"] += openai_usd(model=AGENT_MODEL, input_tokens=tokens, output_tokens=AGENT_MAX_OUTPUT_TOKENS)
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
    )


class DirectRunner:
    """Run every catalog and arm in fixed order, sharing search but recording each strategy's search overhead."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        jev_model: DecisionModel,
        agent: CachedAgent,
        guard: SpendGuard,
    ) -> None:
        self.retriever = SharedRetrieval(retriever)
        self.jev_model, self.agent, self.guard = jev_model, agent, guard
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
        """Write requests incrementally and stop on budget exhaustion, a crash, or more than 5% errors in an arm."""
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = _manifest(catalogs, run_id=run_id, pilot=pilot, estimate=estimate)
        manifest["jev"] = {
            "model": self.jev_model.model_id,
            "limits": self.jev_model.limits.model_dump(mode="json"),
            "max_detail": "FULL",
            "abstention": {"reserved_option": True, "threshold": 0},
            "expected_rounds": 1,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        completed = False
        errors = dict.fromkeys(DIRECT_ARMS, 0)
        counts = dict.fromkeys(DIRECT_ARMS, 0)
        stop_reason: str | None = None
        try:
            with (run_dir / "run.jsonl").open("w") as output:
                for selected in catalogs:
                    for arm in DIRECT_ARMS:
                        for task, variant, catalog, phase in selected.requests():
                            self.guard.context = {
                                "catalog": selected.source,
                                "arm": arm,
                                "task": task.id,
                                "variant": variant,
                                "phase": phase,
                            }
                            record = await self._request(
                                selected, arm=arm, task=task, variant=variant, catalog=catalog, phase=phase
                            )
                            output.write(json.dumps(record) + "\n")
                            output.flush()
                            counts[arm] += 1
                            errors[arm] += record["error"] is not None
                            if errors[arm] / estimate.requests_per_arm > 0.05:
                                raise ProviderFailure(f"more than 5% errored requests in {arm}")
                completed = True
        except (ProviderFailure, SpendLimit) as error:
            stop_reason = str(error)
        except Exception as error:
            stop_reason = f"crash: {type(error).__name__}"
        finally:
            manifest |= {
                "completed": completed,
                "stop_reason": stop_reason,
                "counts": counts,
                "errors": errors,
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
    ) -> dict[str, Any]:
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
            "detail": [],
            "decision_key": None,
            "historical_usage": None,
            "provider_calls": [],
        }
        before: int | None = None
        try:
            if arm in {"hybrid@20", "hybrid@20+jev", "agent@20"}:
                retrieval, seconds, usd = await self._retrieval(task, catalog)
                cards = [match.card for match in retrieval.matches]
                record |= {"search_seconds": seconds, "search_usd": usd}
            else:
                cards = list(catalog)
            record["candidates"] = [card.id for card in cards]
            before = len(self.guard.calls)
            if arm == "hybrid@20":
                record |= {"ranked": record["candidates"], "pick": cards[0].id if cards else None, "abstained": False}
            elif arm in {"hybrid@20+jev", "jev-all"}:
                pipeline = ToolSearchPipeline(
                    CatalogOrderRetriever(),
                    k=max(1, len(cards)),
                    decider=ChoiceDecider(self.jev_model, abstention=Abstention(), max_detail=DetailLevel.FULL),
                )
                result = await pipeline.search([task.query], ToolCatalog(cards), context=task.query)
                assert result.decision is not None
                decision = result.decision
                record |= {
                    "ranked": result.ids,
                    "pick": None if result.abstained or not result.ids else result.ids[0],
                    "abstained": result.abstained,
                    "decision_key": str(decision.key),
                    "detail": [exchange.detail.name for exchange in decision.exchanges],
                    "state_cut": decision.state_cut,
                }
                record["replayed"] = len(self.guard.calls) == before and bool(decision.exchanges)
                record["historical_usage"] = asdict(decision.usage)
            else:
                answer, replayed = await self.agent.ask(task.query, function_cards(cards), catalog_name=selected.source)
                record |= {
                    "pick": answer.pick,
                    "abstained": answer.pick is None,
                    "extra_calls": answer.extra_calls,
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
        finally:
            if before is not None:
                own = self.guard.calls[before:]
                record["provider_calls"] = own
                record["decision_seconds"] = sum(call["seconds"] for call in own) if own else None
        return record


def _manifest(
    catalogs: Sequence[DirectCatalog],
    *,
    run_id: str,
    pilot: bool,
    estimate: DirectEstimate,
) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10).stdout.strip()

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
        "arms": list(DIRECT_ARMS),
        "order": "catalog, arm, positives, negatives",
        "concurrency": 1,
        "negative_seed": 0,
        "negative_allocation": "largest remainder, stable source order",
        "prompt_versions": {"jev": PROMPT_VERSION, "agent": AGENT_PROMPT_VERSION},
        "agent": {
            "model": AGENT_MODEL,
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
