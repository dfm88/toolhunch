"""An opt-in five-order experiment over the existing decision benchmark."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import anyio
import httpx2

from toolhunch import BM25Retriever, DetailLevel, OpenAIEmbedder
from toolhunch.decision import DecisionError, DecisionUsage
from toolhunch_bench import BENCH_DIR
from toolhunch_bench.datasets.toolret import TOOLRET_SUBTASKS, load_toolret, read_task_file
from toolhunch_bench.deciders import DECIDERS as REGISTRY
from toolhunch_bench.deciders import DeciderName, decision_model
from toolhunch_bench.decision import (
    CacheOnlyRetrieval,
    SharedRetrieval,
    build_decision_arms,
    estimate_decisions,
    run_decisions,
)
from toolhunch_bench.direct_cost import (
    P1_CAP_USD,
    GuardedEmbedder,
    ProviderCall,
    SpendGuard,
    attempt_failure,
    openai_usd,
)
from toolhunch_bench.embedding_cache import (
    EMBEDDING_CACHE_PATH,
    CachedEmbedder,
    TruncatingEmbedder,
)
from toolhunch_bench.ledger import Budget, LedgerEntry, append_ledger, f2a_spend, read_ledger
from toolhunch_bench.retrieval import build_arms

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from toolhunch.decision import DecisionModel, DecisionRequest, DecisionResponse, ModelLimits, QuestionKind
    from toolhunch_bench.decision import DecisionArm

ORDER_SEEDS = (0, 1, 2, 3, 4)
MAX_INPUT_TOKENS = 8191
ORDER_DETAILS = {"jev": DetailLevel.BRIEF, "logprob": DetailLevel.FULL}
"""P1's deciders and their published configurations: the default of `order_experiment`."""
P1_TARGET_USD = 5.0
P1_BUDGET = Budget("P1:", P1_TARGET_USD, P1_CAP_USD)
"""The budget P1's order runs were charged to: the default of `order_experiment`."""


class OrderDecisionModel:
    """Account for each physical decision attempt, without local replay or hidden retries."""

    def __init__(self, inner: DecisionModel, *, decider: str, guard: SpendGuard) -> None:
        self._inner, self._decider, self._guard = inner, decider, guard
        self._lock = anyio.Lock()

    @property
    def model_id(self) -> str:
        """The configured model and endpoint identity."""
        return self._inner.model_id

    @property
    def limits(self) -> ModelLimits:
        """Declared limits used by the unchanged planner."""
        return self._inner.limits

    @property
    def question_kinds(self) -> frozenset[QuestionKind]:
        """The wrapped model's supported questions."""
        return self._inner.question_kinds

    @property
    def prompt_version(self) -> str | None:
        """The wrapped model's prompt version."""
        return self._inner.prompt_version

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Keep an unknown-usage failure's prudent reserve charged to the cumulative budget."""
        async with self._lock:
            context = self._guard.context
            self._guard.context = dict(context) | {"request": asdict(request)}
            try:
                return await self._ask(request, **options)
            finally:
                self._guard.context = context

    async def _ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        spec = REGISTRY[DeciderName(self._decider)]
        provider = spec.provider
        if spec.provider == "openai":
            # ASCII JSON bytes bound BPE tokens, including escaping and the fixed letter prompt.
            upper = openai_usd(
                model=spec.model, input_tokens=len(json.dumps(asdict(request)).encode()) + 4096, output_tokens=1
            )
        elif spec.billing == "local":
            upper = 0.0
        else:
            if self.limits.max_request_tokens is None:
                raise ValueError(f"a request token cap is required for the {self._decider} spend reservation")
            upper = self.limits.estimate_usd(DecisionUsage(1, self.limits.max_request_tokens, 0)) or 0.0
        self._guard.before(upper)
        started = time.perf_counter()
        try:
            response = await self._inner.ask(request, **options)
        except Exception as error:
            status, reached = attempt_failure(error)
            self._guard.record(
                ProviderCall(
                    provider=provider,
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
            raise DecisionError(f"order decision failed: {type(error).__name__}") from None
        if upper > 0 and response.usage.input_tokens <= 0:
            # A priced reply without input tokens cannot be accounted for: its reservation is charged, never $0.
            self._guard.record(
                ProviderCall(
                    provider=provider,
                    model=self.model_id,
                    input_tokens=None,
                    output_tokens=None,
                    cache_read_tokens=None,
                    seconds=time.perf_counter() - started,
                    usd=None,
                    list_usd=None,
                    budget_charge_usd=upper,
                    error="priced reply without input tokens",
                )
            )
            raise DecisionError("order decision failed: priced reply without input tokens")
        usage = response.raw.get("usage", {})
        details: Any = cast("dict[str, Any]", usage).get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
        cached = cast("dict[str, Any]", details).get("cached_tokens") if isinstance(details, dict) else None
        if not isinstance(cached, int) or not 0 <= cached <= response.usage.input_tokens:
            cached = None
        if spec.provider == "openai":
            usd = openai_usd(
                model=spec.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=cached or 0,
            )
            list_usd = openai_usd(
                model=spec.model, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
            )
        else:
            usd = list_usd = 0.0 if spec.billing == "local" else self.limits.estimate_usd(response.usage) or 0.0
        self._guard.record(
            ProviderCall(
                provider=provider,
                model=self.model_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=cached,
                seconds=time.perf_counter() - started,
                usd=usd,
                list_usd=list_usd,
                budget_charge_usd=usd,
            )
        )
        return response


def order_ledger_entries(
    guard: SpendGuard, *, run_id: str, pilot: bool, budget: Budget = P1_BUDGET
) -> list[LedgerEntry]:
    """Build ledger charges from real attempts, retaining failed-usage reserves."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for call in guard.calls:
        grouped.setdefault((call["provider"], call["model"]), []).append(call)
    entries: list[LedgerEntry] = []
    for (provider, model), calls in grouped.items():
        if provider == "local":
            continue  # a local model is not paid for: its calls stay in the run's calls.jsonl
        verified = sum(call["usd"] or 0 for call in calls)
        charge = sum(call["budget_charge_usd"] for call in calls)
        entries.append(
            LedgerEntry(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                provider=provider,
                model=model,
                input_tokens=sum(call["input_tokens"] or 0 for call in calls),
                output_tokens=sum(call["output_tokens"] or 0 for call in calls),
                usd=charge,
                purpose=f"{budget.prefix} order sensitivity " + ("pilot" if pilot else "full"),
                note=f"Verified usage ${verified:.9f}; uncertain failed-attempt reserves "
                f"${charge - verified:.9f}; {len(calls)} physical attempts; "
                "no local replays; decision cache usage unmeasured.",
            )
        )
    return entries


def _configuration(arms: list[DecisionArm]) -> dict[str, Any]:
    fields = ("model_id", "limits", "prompt_version", "max_detail", "reserved_option", "k")
    return {arm.name: {key: arm.config[key] for key in fields} for arm in arms if arm.model is not None}


async def order_experiment(
    task_file: Path,
    *,
    cache_dir: Path,
    runs_dir: Path,
    ledger_path: Path,
    dry_run: bool,
    pilot: bool,
    pilot_run: Path | None,
    estimate_out: Path | None,
    echo: Callable[[str], None],
    embedding_cache_path: Path = EMBEDDING_CACHE_PATH,
    deciders: Mapping[str, DetailLevel] = ORDER_DETAILS,
    budget: Budget = P1_BUDGET,
) -> Path | None:
    """Estimate both workloads freely; paid full runs require a matching, automatically validated pilot."""
    import tiktoken
    from dotenv import load_dotenv

    from toolhunch_bench.order_report import pilot_gates

    required = [
        cache_dir / "corpus.parquet",
        *(cache_dir / f"{name}-{kind}.parquet" for name in TOOLRET_SUBTASKS for kind in ("qrels", "queries")),
    ]
    if not all(path.is_file() for path in required):
        raise ValueError("order sensitivity needs the complete local ToolRet cache; downloads are disabled")

    def offline_only(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("order sensitivity never downloads datasets")

    with httpx2.Client(transport=httpx2.MockTransport(offline_only)) as client:
        data = load_toolret(cache_dir=cache_dir, client=client)
    tasks = read_task_file(task_file, data)
    if len(tasks) != 200:
        raise ValueError("order sensitivity requires the 200-task held-out task file")
    published = json.loads((BENCH_DIR / "results/2026-09-toolret-decision/summary.json").read_text())
    original = next(run["manifest"] for run in published["runs"] if run["role"] == "main" and run["split"] == "heldout")
    if (
        hashlib.sha256(task_file.read_bytes()).hexdigest() != original["tasks"]["sha256"]
        or data.catalog.fingerprint != original["dataset"]["catalog_fingerprint"]
    ):
        raise ValueError("order mode requires the published F2a held-out tasks and catalog")
    selected = tasks[:10] if pilot else tasks
    prior = f2a_spend(read_ledger(ledger_path), purpose_prefix=budget.prefix).usd
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ("-order-pilot" if pilot else "-order")
    run_dir = runs_dir / run_id

    def persist(call: dict[str, Any]) -> None:
        with (run_dir / "calls.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(call) + "\n")

    guard = SpendGuard(prior_usd=prior, cap_usd=budget.cap_usd, sink=persist)
    names = list(deciders)
    if any(REGISTRY[DeciderName(name)].billing != "local" and name not in ORDER_DETAILS for name in names):
        # Building a hosted decider beyond P1's pair can read its account from .env (Clef); keys stay at call time.
        load_dotenv(BENCH_DIR.parent / ".env", override=False)
    adapters = {name: decision_model(DeciderName(name), max_retries=0) for name in names}
    inner = OpenAIEmbedder("text-embedding-3-small", max_input_bytes=None, batch_size=512)
    cache = CachedEmbedder(
        GuardedEmbedder(
            TruncatingEmbedder(inner, max_tokens=MAX_INPUT_TOKENS, encoding=tiktoken.get_encoding("cl100k_base")),
            guard=guard,
        ),
        path=embedding_cache_path,
        chunk_size=512,
    )
    try:
        [hybrid] = build_arms(["hybrid"], embedder=cache, raw_text=data.raw_text)
        free = CacheOnlyRetrieval(hybrid.retriever, stand_in=BM25Retriever(), embeddings=cache)
        estimate_arms = build_decision_arms(names, ks=[20], models=adapters, max_detail=deciders)
        estimates: dict[str, Any] = {}
        for phase, phase_tasks in (("pilot", tasks[:10]), ("full", tasks)):
            lines = await estimate_decisions(
                estimate_arms,
                data,
                phase_tasks,
                retriever=SharedRetrieval(free),
                sources=["plain"],
                model_queries=None,
                negatives=True,
                repeat=1,
                embeddings=cache,
                order_seeds=ORDER_SEEDS,
            )
            if free.stand_in_queries:
                raise ValueError(
                    "order estimates require cached hybrid query and corpus embeddings; "
                    "lexical stand-ins cannot estimate the K20 protocol reliably; nothing spent"
                )
            estimates[phase] = {
                "tasks": len(phase_tasks),
                "searches": len(phase_tasks) * 20,
                "lines": [asdict(line) for line in lines],
                "usd": sum(line.usd or 0 for line in lines),
            }
        estimates |= {
            "prior_p1_usd": prior,
            "target_usd": budget.target_usd,
            "cap_usd": budget.cap_usd,
            "full_plus_prior_usd": prior + estimates["full"]["usd"],
            "pilot_full_plus_prior_usd": prior + estimates["pilot"]["usd"] + estimates["full"]["usd"],
            "retrieval_stand_in_queries": len(free.stand_in_queries),
            "retrieval_queries": len(free.searched),
            "order_seeds": list(ORDER_SEEDS),
            "configuration": _configuration(estimate_arms),
            "assumptions": "All five decisions billed; heuristic input tokens; no cache savings; "
            "uniform stand-ins select logprob finalists; embeddings priced once.",
        }
        echo(
            f"Order sensitivity estimates: first-ten pilot ${estimates['pilot']['usd']:.6f}; "
            f"full ${estimates['full']['usd']:.6f}"
        )
        estimate = estimates["pilot" if pilot else "full"]
        echo(
            f"Prior {budget.prefix.rstrip(':')} ledger charge ${prior:.9f}; selected plus prior "
            f"${prior + estimate['usd']:.6f}; target ${budget.target_usd:.2f}; hard cap ${budget.cap_usd:.2f}"
        )
        echo(f"Prior + new pilot + full (no pilot replay): ${estimates['pilot_full_plus_prior_usd']:.6f}")
        echo(
            f"Retrieval stand-ins: {len(free.stand_in_queries)}/{len(free.searched)} query lists; "
            "all decisions estimated offline, zero provider calls."
        )
        if estimate_out is not None:
            estimate_out.parent.mkdir(parents=True, exist_ok=True)
            estimate_out.write_text(json.dumps(estimates, indent=2) + "\n")
        if prior + estimate["usd"] > budget.cap_usd:
            raise ValueError(
                f"selected estimate plus prior {budget.prefix.rstrip(':')} charge exceeds the hard cap; nothing spent"
            )
        if dry_run:
            return None
        if not pilot:
            if pilot_run is None:
                raise ValueError("full order run requires --pilot-run with a validated first-ten pilot")
            gates = pilot_gates(pilot_run)
            previous = json.loads((pilot_run / "manifest.json").read_text())
            if (
                not gates["passed"]
                or previous.get("pilot") is not True
                or previous["tasks"]["sha256"] != original["tasks"]["sha256"]
                or previous.get("task_ids") != [task.id for task in tasks[:10]]
            ):
                raise ValueError("pilot gates or first-ten-task identities do not match")
            if previous.get("order_configuration") != _configuration(estimate_arms) or (
                previous["dataset"]["catalog_fingerprint"] != data.catalog.fingerprint
            ):
                raise ValueError("pilot configuration or catalog does not match the full run")
        if free.stand_in_queries:
            raise ValueError("paid order mode requires cached hybrid query and corpus embeddings; no stand-ins allowed")
        load_dotenv(BENCH_DIR.parent / ".env", override=False)
        models = {name: OrderDecisionModel(adapter, decider=name, guard=guard) for name, adapter in adapters.items()}
        arms = build_decision_arms(names, ks=[20], models=models, max_detail=deciders)
        extra = {
            "experiment": "order-sensitivity-v1",
            "pilot": pilot,
            "completed": False,
            "estimate": estimate,
            "prior_p1_usd": prior,
            "budget": {"prefix": budget.prefix, "prior_usd": prior, "cap_usd": budget.cap_usd},
            "target_usd": budget.target_usd,
            "cap_usd": budget.cap_usd,
            "order_configuration": _configuration(estimate_arms),
            "provider_retries": 0,
            "pilot_run": None if pilot_run is None else pilot_run.name,
            "scheduling": "arm-major, task, positive then negative, order seeds 0-4; physical asks serialized",
            "task_mix": {"positive": len(selected), "negative": len(selected), "negative_share": 0.5},
        }
        await run_decisions(
            arms,
            data,
            selected,
            retriever=SharedRetrieval(hybrid.retriever, config=hybrid.config),
            split="heldout",
            sources=["plain"],
            model_queries=None,
            negatives=True,
            repeat=1,
            out_dir=runs_dir,
            task_file=task_file,
            model_queries_file=None,
            run_id=run_id,
            order_seeds=ORDER_SEEDS,
            manifest_extra=extra,
            before_search=lambda context: setattr(guard, "context", context),
        )
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) | {"completed": True}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        gates = pilot_gates(run_dir)
        echo("Pilot gates: " + json.dumps(gates))
        if not gates["passed"]:
            raise ValueError("order run failed automatic gates; full progression is prohibited")
        return run_dir
    finally:
        for entry in order_ledger_entries(guard, run_id=run_id, pilot=pilot, budget=budget):
            append_ledger(entry, path=ledger_path)
        cache.close()
        for adapter in adapters.values():
            if (close := getattr(adapter, "aclose", None)) is not None:
                await close()
        await inner.aclose()
