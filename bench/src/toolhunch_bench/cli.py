"""Command line of the benchmark harness, installed as `toolhunch-bench`."""

# No `from __future__ import annotations`: Typer reads the command signatures at runtime.

import asyncio
import functools
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic_ai
import tiktoken
import typer
from dotenv import load_dotenv
from genai_prices import Usage, calc_price
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.usage import RunUsage

from toolhunch import BM25Retriever, DetailLevel, Embedder, OpenAIEmbedder, default_search_text
from toolhunch.decision import JevWireModel, OpenAILogprobModel, clm, jev
from toolhunch_bench import BENCH_DIR
from toolhunch_bench.charts import build_decision_charts
from toolhunch_bench.checks import check_bm25, check_tokens
from toolhunch_bench.datasets.model_queries import (
    WRITER_MODEL,
    WRITER_SETTINGS,
    estimate_writer_cost,
    load_model_queries,
    model_queries_path,
    save_model_queries,
    search_tools_definition,
    write_queries,
    writer_block,
)
from toolhunch_bench.datasets.toolret import (
    TOOLRET_DATASET,
    TOOLRET_REVISION,
    ToolRetData,
    ToolRetTask,
    load_toolret,
    read_task_file,
    sample_tasks,
    write_task_file,
)
from toolhunch_bench.decision import (
    CLM_DEPLOYMENT,
    CLM_MODEL,
    DECIDERS,
    JEV_MODEL,
    LOGPROB_MODEL,
    QUERY_SOURCES,
    CacheOnlyRetrieval,
    DecisionArm,
    EstimateLine,
    QuerySource,
    SharedRetrieval,
    build_decision_arms,
    clm_usd,
    estimate_decisions,
    jev_usd,
    run_decisions,
    warm_up_clm,
)
from toolhunch_bench.decision_cache import DECISION_CACHE_PATH, CachedDecisionModel
from toolhunch_bench.decision_report import build_decision_report
from toolhunch_bench.direct import DirectRunner, direct_catalogs, estimate_direct
from toolhunch_bench.direct_agent import CachedAgent
from toolhunch_bench.direct_cost import (
    AGENT_MODEL,
    P1_CAP_USD,
    GuardedDecisionModel,
    GuardedEmbedder,
    SpendGuard,
)
from toolhunch_bench.direct_report import build_direct_report
from toolhunch_bench.embedding_cache import (
    EMBEDDING_CACHE_PATH,
    CachedEmbedder,
    TruncatingEmbedder,
    estimate_embedding_cost,
)
from toolhunch_bench.ledger import F2A_CAP_EUR, LEDGER_PATH, LedgerEntry, append_ledger, f2a_spend, read_ledger
from toolhunch_bench.report import build_report
from toolhunch_bench.retrieval import (
    ARM_NAMES,
    PAID_ARMS,
    RUNS_DIR,
    WARM_UP_QUERY,
    Arm,
    QueryMode,
    build_arms,
    query_text,
    run_arms,
)

if TYPE_CHECKING:
    from toolhunch.retrieval import EmbeddingKind

TOOLRET_CACHE = BENCH_DIR / "runs" / "data" / "toolret" / TOOLRET_REVISION
MODES: tuple[QueryMode, ...] = ("plain", "instructed")
MAX_INPUT_TOKENS = 8191  # text-embedding-3-* context length (OpenAI cookbook, Embedding_long_inputs)
DECISION_EMBEDDING_MODEL = "text-embedding-3-small"  # the hybrid retriever's, as in the F1 pilot
_FALLBACK_EXAMPLES = 5  # what the writer answered instead of searching, shown after a queries run

app = typer.Typer(help="toolhunch benchmark harness.", no_args_is_help=True, add_completion=False)
toolret_app = typer.Typer(help=f"ToolRet ({TOOLRET_DATASET}) data.", no_args_is_help=True)
check_app = typer.Typer(help="Implementation checks on the real catalog.", no_args_is_help=True)
queries_app = typer.Typer(help="Search queries written by a model.", no_args_is_help=True)
app.add_typer(toolret_app, name="toolret")
app.add_typer(check_app, name="check")
app.add_typer(queries_app, name="queries")

TaskFile = Annotated[Path, typer.Option("--tasks", help="Task file.", exists=True, dir_okay=False)]
ChecksFile = Annotated[Path | None, typer.Option("--out", help="JSON file to add the result to.")]


@toolret_app.command()
def fetch() -> None:
    """Download the pinned revision and print how the tools were mapped."""
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    tools = len(data.catalog)
    typer.echo(
        f"ToolRet @ {TOOLRET_REVISION[:7]}: {tools:,} tools, {len(data.tasks):,} tasks "
        f"in {len({task.subtask for task in data.tasks})} subtasks (cache: {TOOLRET_CACHE})"
    )
    labels = {"name": "a name", "description": "a description", "properties": "at least one parameter name"}
    for field, label in labels.items():
        share = data.mapping_stats[field]
        typer.echo(f"cards with {label}: {round(share * tools):,} ({share:.3%})")


@toolret_app.command()
def sample(
    out: Annotated[Path, typer.Option(help="Task file to write.")],
    n: Annotated[int, typer.Option(help="Number of tasks.")] = 50,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    exclude: Annotated[
        Path | None,
        typer.Option(help="Task file whose ids are left out of the sample.", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Write a stratified sample of task ids, leaving out those of the `--exclude` task file."""
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    left_out = read_task_file(exclude, data) if exclude is not None else []
    tasks = sample_tasks(data.tasks, n=n, seed=seed, exclude={task.id for task in left_out})
    write_task_file(out, tasks, seed=seed)
    typer.echo(f"{len(tasks)} task ids written to {out}")


@queries_app.command("write")
def write_model_queries(
    tasks: TaskFile,
    model: Annotated[str, typer.Option(help="OpenAI model that writes the queries.")] = WRITER_MODEL,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the estimate and the cap check, and stop.")] = False,
) -> None:
    """Freeze what an agent would search for on each task beside the task file. Paid: the estimate comes first."""
    pydantic_ai.BANNER_ENABLED = False  # the harness owns its output
    selected = read_task_file(tasks, load_toolret(cache_dir=TOOLRET_CACHE))
    definition = search_tools_definition()
    estimate = estimate_writer_cost(selected, model=model, definition=definition)
    spend = f2a_spend(read_ledger(LEDGER_PATH))
    total = spend.usd + estimate.usd
    typer.echo(
        f"Estimate for {model} on {estimate.tasks} tasks (tiktoken o200k_base): {estimate.input_tokens:,} input + "
        f"{estimate.output_tokens:,} output tokens (upper bound), ${estimate.usd:.4f}"
    )
    typer.echo(f"F2a total so far: ${spend.usd:.4f} (Modal credits, apart: ${spend.modal_usd:.4f})")
    typer.echo(f"Cap check: ${spend.usd:.4f} + ${estimate.usd:.4f} = ${total:.4f} against the ${F2A_CAP_EUR:.2f} cap")
    if total > F2A_CAP_EUR:
        typer.echo(
            f"Over the cap: nothing was spent. F2a total ${spend.usd:.4f}, estimate ${estimate.usd:.4f}.", err=True
        )
        raise typer.Exit(code=2)
    if dry_run:
        return
    load_dotenv(BENCH_DIR.parent / ".env", override=False)  # the key comes from the environment, never printed
    writer = OpenAIChatModel(model, settings=OpenAIChatModelSettings(**WRITER_SETTINGS))
    started = datetime.now(UTC)
    usage = RunUsage()
    answers: dict[str, str] = {}
    try:
        written, _ = asyncio.run(write_queries(selected, model=writer, usage=usage, answers=answers))
    finally:
        if usage.requests:  # a failed run still paid for the tasks it finished
            entry = _writer_ledger_entry(usage, model=model, run_id=started.strftime("%Y%m%dT%H%M%SZ"), tasks=tasks)
            append_ledger(entry, path=LEDGER_PATH)
            typer.echo(
                f"Billed: {usage.requests} requests, {usage.input_tokens:,} input + {usage.output_tokens:,} output "
                f"tokens, ${entry.usd:.4f}"
            )
    path = model_queries_path(tasks)
    save_model_queries(path, written, writer=writer_block(model=model, definition=definition, run_date=started.date()))
    fallbacks = sum(item.fallback for item in written)
    typer.echo(f"Queries for {len(written)} tasks written to {path}")
    typer.echo(f"Fallbacks (no search in the first response): {fallbacks} of {len(written)}")
    fallen = list(answers.items())
    shown = min(len(fallen), _FALLBACK_EXAMPLES)
    for task_id, text in (fallen[(2 * i + 1) * len(fallen) // (2 * shown)] for i in range(shown)):  # evenly spread
        words = " ".join(text.split())
        typer.echo(f"  {task_id}: {words[:200]}{'...' if len(words) > 200 else ''}")


@app.command()
def retrieval(
    tasks: TaskFile,
    arms: Annotated[str, typer.Option(help=f"Comma-separated, from: {', '.join(ARM_NAMES)}.")],
    embedding_model: Annotated[str, typer.Option(help="OpenAI embedding model.")] = "text-embedding-3-small",
    k: Annotated[int, typer.Option(help="Results kept per search.")] = 50,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the embedding cost estimate and stop.")] = False,
) -> None:
    """Run retrieval arms on the tasks of a task file; paid arms print their estimate first."""
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    selected = read_task_file(tasks, data)
    names = [name.strip() for name in arms.split(",") if name.strip()]
    paid = [name for name in names if name in PAID_ARMS]
    inner = OpenAIEmbedder(embedding_model, max_input_bytes=None)  # TruncatingEmbedder cuts exactly, by tokens
    cache: CachedEmbedder | None = None
    if paid:
        encoding = tiktoken.get_encoding("cl100k_base")
        cache = CachedEmbedder(TruncatingEmbedder(inner, max_tokens=MAX_INPUT_TOKENS, encoding=encoding))
    embedder: Embedder = cache if cache is not None else inner
    built = build_arms(names, embedder=embedder, raw_text=data.raw_text)
    if cache is None:
        if dry_run:
            typer.echo("no paid arms: nothing to estimate")
            return
    else:
        _print_estimate(built, data, selected, cache=cache, model=embedding_model)
        if dry_run:
            cache.close()
            return
        load_dotenv(BENCH_DIR.parent / ".env", override=False)  # keys come from the environment, never printed
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        run_dir = asyncio.run(_run(built, data, selected, inner=inner, task_file=tasks, k=k, run_id=run_id))
    finally:
        if cache is not None:
            if cache.billed_tokens:  # a failed run still paid for what it embedded
                entry = _ledger_entry(cache.billed_tokens, model=embedding_model, run_id=run_id, arms=paid, tasks=tasks)
                append_ledger(entry)
            cache.close()
    typer.echo(f"run written to {run_dir}")


@app.command()
def decision(
    tasks: TaskFile,
    split: Annotated[Literal["dev", "heldout"], typer.Option(help="The split the task file belongs to.")],
    deciders: Annotated[str, typer.Option(help=f"Comma-separated, from: {', '.join(DECIDERS)}.")],
    k: Annotated[str, typer.Option("--k", help="Comma-separated candidate counts, such as 20,50.")],
    sources: Annotated[str, typer.Option(help=f"Comma-separated query sources, from: {', '.join(QUERY_SOURCES)}.")],
    no_negatives: Annotated[
        bool, typer.Option("--no-negatives", help="Skip the searches with the gold tools taken out.")
    ] = False,
    no_reserved: Annotated[
        bool, typer.Option("--no-reserved", help='Ask without the reserved "none of these" option.')
    ] = False,
    max_detail: Annotated[
        str, typer.Option(help="Most detail per decider, such as jev=full,logprob=brief; full when left out.")
    ] = "",
    repeat: Annotated[int, typer.Option(min=1, help="How many times each search is made.")] = 1,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N tasks of the task file.")] = None,
    bypass_cache: Annotated[
        bool, typer.Option("--bypass-cache", help="Ask the models even when the decision cache has the answer.")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the estimate and the cap check, and stop.")] = False,
    order_sensitivity: Annotated[
        bool, typer.Option("--order-sensitivity", help="Five fresh candidate orders on held-out K20 plain.")
    ] = False,
    pilot: Annotated[
        bool, typer.Option("--pilot", help="Order mode: first ten held-out tasks, all five orders.")
    ] = False,
    pilot_run: Annotated[
        Path | None,
        typer.Option(
            "--pilot-run", help="Order mode: validated pilot required before full.", exists=True, file_okay=False
        ),
    ] = None,
    estimate_out: Annotated[
        Path | None, typer.Option("--estimate-out", help="Order mode: write pilot/full estimates JSON.")
    ] = None,
) -> None:
    """Run decision arms on the tasks of a task file. Paid: the estimate and the cap check come first."""
    names = _listed(deciders, allowed=DECIDERS, option="--deciders")
    ks = _counts(k)
    if not (chosen := _listed(sources, allowed=QUERY_SOURCES, option="--sources")):
        raise typer.BadParameter("give one or more query sources", param_hint="--sources")
    source_list: list[QuerySource] = [source for name in chosen for source in QUERY_SOURCES if source == name]
    details = _max_detail(max_detail)
    negatives, reserved = not no_negatives, not no_reserved
    if order_sensitivity:
        from toolhunch_bench.order import ORDER_DETAILS, order_experiment

        if (
            split != "heldout"
            or names != ["jev", "logprob"]
            or ks != [20]
            or source_list != ["plain"]
            or not negatives
            or not reserved
            or repeat != 1
            or limit is not None
            or any(level != ORDER_DETAILS[name] for name, level in details.items())
        ):
            raise typer.BadParameter(
                "order mode requires heldout, jev,logprob, K20, plain, positives/negatives, "
                "reserved option, Jev BRIEF/logprob FULL, repeat 1 and no --limit"
            )
        try:
            result = asyncio.run(
                order_experiment(
                    tasks,
                    cache_dir=TOOLRET_CACHE,
                    runs_dir=RUNS_DIR,
                    ledger_path=LEDGER_PATH,
                    dry_run=dry_run,
                    pilot=pilot,
                    pilot_run=pilot_run,
                    estimate_out=estimate_out,
                    echo=typer.echo,
                )
            )
        except Exception as error:
            # Provider errors can carry sensitive headers; only configuration ValueErrors are printable.
            typer.echo(
                str(error) if isinstance(error, ValueError) else f"Order run stopped: {type(error).__name__}", err=True
            )
            raise typer.Exit(code=2) from None
        if result is not None:
            typer.echo(f"run written to {result}")
        return
    if pilot or pilot_run is not None or estimate_out is not None:
        raise typer.BadParameter("--pilot, --pilot-run and --estimate-out require --order-sensitivity")
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    selected = read_task_file(tasks, data)[:limit]
    queries_file = model_queries_path(tasks) if "model" in source_list else None
    written = None if queries_file is None else load_model_queries(queries_file, selected)
    load_dotenv(BENCH_DIR.parent / ".env", override=False)  # CLM_BASE_URL now, the keys at call time, never printed
    clm_base_url = os.environ.get("CLM_BASE_URL", "")
    if "clm" in names and not clm_base_url:
        typer.echo(
            "The clm decider needs CLM_BASE_URL, the root of the server bench/deploy/clm_modal.py runs.", err=True
        )
        raise typer.Exit(code=1)
    adapters = _decision_models(names, clm_base_url=clm_base_url)  # nothing is contacted before the first ask
    inner = OpenAIEmbedder(DECISION_EMBEDDING_MODEL, max_input_bytes=None)  # TruncatingEmbedder cuts exactly, by tokens
    encoding = tiktoken.get_encoding("cl100k_base")
    cache = CachedEmbedder(
        TruncatingEmbedder(inner, max_tokens=MAX_INPUT_TOKENS, encoding=encoding), path=EMBEDDING_CACHE_PATH
    )
    try:
        [hybrid] = build_arms(["hybrid"], embedder=cache, raw_text=data.raw_text)
        estimate_arms = build_decision_arms(names, ks=ks, models=adapters, max_detail=details, reserved_option=reserved)
        free = CacheOnlyRetrieval(hybrid.retriever, stand_in=BM25Retriever(), embeddings=cache)
        lines = asyncio.run(
            estimate_decisions(
                estimate_arms,
                data,
                selected,
                retriever=SharedRetrieval(free),
                sources=source_list,
                model_queries=written,
                negatives=negatives,
                repeat=repeat,
                embeddings=cache,
            )
        )
        searches = len(names) * len(ks) * len(source_list) * len(selected) * (2 if negatives else 1) * repeat
        _print_decision_estimate(lines, searches=searches, retrieval=free)
        _check_the_cap(sum(line.usd for line in lines if line.usd is not None))
        if dry_run:
            return

        models = {
            name: CachedDecisionModel(adapter, path=DECISION_CACHE_PATH, bypass=bypass_cache)
            for name, adapter in adapters.items()
        }
        arms = build_decision_arms(names, ks=ks, models=models, max_detail=details, reserved_option=reserved)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        settings = [", ".join(names) or "retrieval only", "k " + "/".join(map(str, ks)), "/".join(source_list)]
        settings += [] if negatives else ["no negatives"]
        settings += [] if reserved else ["no reserved option"]
        settings += [
            f"{name} {details[name].name.lower()}"
            for name in names
            if details.get(name, DetailLevel.FULL) < DetailLevel.FULL
        ]
        settings += [f"repeat {repeat}"] if repeat > 1 else []
        settings += ["cache bypassed"] if bypass_cache else []
        settings += [f"first {limit} tasks"] if limit is not None else []
        purpose = f"F2a: decision {split} ({'; '.join(settings)}) on {tasks.name}"
        run = functools.partial(
            run_decisions,
            arms,
            data,
            selected,
            retriever=SharedRetrieval(hybrid.retriever, config=hybrid.config),
            split=split,
            sources=source_list,
            model_queries=written,
            negatives=negatives,
            repeat=repeat,
            out_dir=RUNS_DIR,
            task_file=tasks,
            model_queries_file=queries_file,
            run_id=run_id,
            before_arm=_clm_warmer(adapters.get("clm"), base_url=clm_base_url),
        )
        try:
            asyncio.run(_decide(run, adapters=adapters, inner=inner))
        finally:
            records = _run_records(RUNS_DIR / run_id)  # a failed run still paid for what it asked
            for entry in _decision_ledger_entries(
                models, arms=arms, records=records, embeddings=cache, run_id=run_id, purpose=purpose
            ):
                append_ledger(entry, path=LEDGER_PATH)
                paid = " in Modal credits" if entry.provider == "modal" else ""
                typer.echo(
                    f"Billed: {entry.provider} {entry.model}, {entry.input_tokens:,} input + "
                    f"{entry.output_tokens:,} output tokens, ${entry.usd:.4f}{paid}"
                )
            for model in models.values():
                model.close()
    finally:
        cache.close()
    typer.echo(f"run written to {RUNS_DIR / run_id}")
    for record in records:
        if record["record"] == "arm":
            typer.echo(
                f"  {record['arm']}: {record['searches']:,} searches, {record['errors']:,} errors, "
                f"{record['wall_seconds']:,.0f} s"
            )
    spend = f2a_spend(read_ledger(LEDGER_PATH))
    typer.echo(f"F2a total now: ${spend.usd:.4f} (Modal credits, apart: ${spend.modal_usd:.4f})")


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory under bench/runs.", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option(help="Directory for summary.json and README.md.")],
) -> None:
    """Compute the metrics and tables of a run."""
    build_report(run_dir, out_dir=out)
    typer.echo(f"report written to {out}")


@app.command("decision-report")
def decision_report(
    dev: Annotated[
        list[Path],
        typer.Option("--dev", help="A dev run directory; repeat for each run.", exists=True, file_okay=False),
    ],
    out: Annotated[Path, typer.Option(help="Directory for summary.json and README.md.")],
    heldout: Annotated[
        Path | None, typer.Option("--heldout", help="The held-out run directory.", exists=True, file_okay=False)
    ] = None,
    heldout_repeats: Annotated[
        Path | None,
        typer.Option(
            "--heldout-repeats",
            help="A held-out run that repeats its searches, for run-to-run variation.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
) -> None:
    """Choose abstention thresholds on dev runs, apply them to a held-out run, and write the report."""
    build_decision_report(dev, heldout, out_dir=out, heldout_repeats=heldout_repeats)
    typer.echo(f"report written to {out}")


@app.command("decision-charts")
def decision_charts(
    summary: Annotated[Path, typer.Argument(help="Generated decision summary JSON.", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option(help="Directory for the two SVG figures.")],
) -> None:
    """Draw held-out ranking-cost and coverage-accuracy charts without calling a provider."""
    for path in build_decision_charts(summary, out_dir=out):
        typer.echo(f"chart written to {path}")


@app.command("order-report")
def order_report(
    run_dir: Annotated[Path, typer.Argument(help="Recorded five-order decision run.", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option(help="Directory for generated summary and table.")] = BENCH_DIR
    / "results/2026-09-toolret-order",
    noise_reference_run: Annotated[
        Path, typer.Option(help="Recorded same-order F2a repeats for matched-task noise diagnostics.")
    ] = RUNS_DIR / "20260929T162701Z",
) -> None:
    """Generate order diagnostics and pilot gates from recorded evidence, without provider calls."""
    from toolhunch_bench.order_report import build_order_report

    build_order_report(run_dir, out_dir=out, noise_reference_run=noise_reference_run)
    typer.echo(f"report written to {out}")


@app.command()
def direct(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Estimate pilot and full runs; no paid calls.")] = False,
    pilot: Annotated[
        bool, typer.Option("--pilot", help="Ten positives and five negatives per source catalog.")
    ] = False,
    estimate_out: Annotated[Path | None, typer.Option("--estimate-out", help="Write both estimates as JSON.")] = None,
) -> None:
    """Compare five direct-choice strategies; every paid attempt shares the P1 spend guard."""
    from dataclasses import asdict

    from openai import AsyncOpenAI
    from pydantic_ai.providers.openai import OpenAIProvider

    pydantic_ai.BANNER_ENABLED = False
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    full, small = direct_catalogs(data), direct_catalogs(data, pilot=True)
    spend = f2a_spend(read_ledger(LEDGER_PATH), purpose_prefix="P1:")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ("-direct-pilot" if pilot else "-direct")

    def persist_call(call: dict[str, Any]) -> None:
        with (RUNS_DIR / run_id / "calls.jsonl").open("a") as journal:
            journal.write(json.dumps(call) + "\n")

    guard = SpendGuard(prior_usd=spend.usd, sink=persist_call)
    inner = OpenAIEmbedder(DECISION_EMBEDDING_MODEL, batch_size=512, max_input_bytes=None)
    cache = CachedEmbedder(
        GuardedEmbedder(
            TruncatingEmbedder(inner, max_tokens=MAX_INPUT_TOKENS, encoding=tiktoken.get_encoding("cl100k_base")),
            guard=guard,
        ),
        path=EMBEDDING_CACHE_PATH,
        chunk_size=512,
    )
    adapter = jev(model=JEV_MODEL, max_retries=0)
    models: CachedDecisionModel | None = None
    agent: CachedAgent | None = None
    client: AsyncOpenAI | None = None
    paid_cleanup_done = False
    try:
        [hybrid] = build_arms(["hybrid"], embedder=cache, raw_text=data.raw_text)
        free = CacheOnlyRetrieval(hybrid.retriever, stand_in=BM25Retriever(), embeddings=cache)
        full_estimate = asyncio.run(
            estimate_direct(full, retriever=SharedRetrieval(free), jev_model=adapter, embeddings=cache)
        )
        pilot_estimate = asyncio.run(
            estimate_direct(small, retriever=SharedRetrieval(free), jev_model=adapter, embeddings=cache)
        )
        estimate = pilot_estimate if pilot else full_estimate
        estimates = {
            "full": asdict(full_estimate),
            "pilot": asdict(pilot_estimate),
            "prior_p1_usd": spend.usd,
            "full_plus_prior_usd": spend.usd + full_estimate.usd,
            "assumptions": "No prompt-cache or replay savings; configured maximum output tokens per agent request. "
            "Short missing-dense lexical stand-ins use longest applicable FULL cards; Jev planner limits/detail "
            "retained. Other lexical stand-ins and token framing are approximate, not a guaranteed upper bound.",
        }
        typer.echo(
            "Conservative estimate (no cache savings, bounded output): "
            f"pilot ${pilot_estimate.usd:.4f}; full ${full_estimate.usd:.4f}"
        )
        typer.echo(
            f"P1 total so far: ${spend.usd:.4f}; selected estimate plus prior: "
            f"${spend.usd + estimate.usd:.4f}; cap ${P1_CAP_USD:.2f}"
        )
        if estimate_out is not None:
            estimate_out.parent.mkdir(parents=True, exist_ok=True)
            estimate_out.write_text(json.dumps(estimates, indent=2) + "\n")
        if spend.usd + estimate.usd > P1_CAP_USD:
            typer.echo("Over the P1 cap: nothing was spent.", err=True)
            raise typer.Exit(code=2)
        if dry_run:
            return
        load_dotenv(BENCH_DIR.parent / ".env", override=False)
        if any(not os.environ.get(name) for name in ("OPENAI_API_KEY", "TYPESAFE_API_KEY")):
            typer.echo("Paid direct choice requires OPENAI_API_KEY and TYPESAFE_API_KEY.", err=True)
            raise typer.Exit(code=2)
        client = AsyncOpenAI(base_url="https://api.openai.com/v1", max_retries=0, timeout=60)
        model = OpenAIChatModel(AGENT_MODEL, provider=OpenAIProvider(openai_client=client))
        agent = CachedAgent(model, path=BENCH_DIR / "runs" / "cache" / "agent.sqlite", guard=guard)
        models = CachedDecisionModel(
            GuardedDecisionModel(adapter, guard=guard),
            path=BENCH_DIR / "runs" / "cache" / "direct-decisions.sqlite",
        )
        runner = DirectRunner(retriever=hybrid.retriever, jev_model=models, agent=agent, guard=guard)

        async def execute() -> Path:
            nonlocal paid_cleanup_done
            try:
                return await runner.run(
                    small if pilot else full, out_dir=RUNS_DIR, run_id=run_id, pilot=pilot, estimate=estimate
                )
            finally:
                await inner.aclose()
                await adapter.aclose()
                await client.close()
                paid_cleanup_done = True

        run_dir = asyncio.run(execute())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest |= {
            "full_estimate": asdict(full_estimate),
            "full_catalogs": [
                {"source": c.source, "positives": len(c.positives), "negatives": len(c.negatives)} for c in full
            ],
            "prior_p1_usd": spend.usd,
            "purpose": "P1: direct choice " + ("pilot" if pilot else "full"),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        typer.echo(f"run written to {run_dir}; verified/guarded cost ${guard.run_usd:.4f}")
        if not manifest["completed"]:
            typer.echo(f"Stopped: {manifest['stop_reason']}", err=True)
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as error:
        typer.echo(f"Direct run failed: {type(error).__name__}", err=True)
        raise typer.Exit(code=2) from None
    finally:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for call in guard.calls:
            grouped.setdefault((call["provider"], call["model"]), []).append(call)
        for (provider, billed_model), calls in grouped.items():
            verified = sum(call["usd"] or 0 for call in calls)
            charged = sum(call["budget_charge_usd"] for call in calls)
            append_ledger(
                LedgerEntry(
                    timestamp=datetime.now(UTC),
                    run_id=run_id,
                    provider=provider,
                    model=billed_model,
                    input_tokens=sum(call["input_tokens"] or 0 for call in calls),
                    output_tokens=sum(call["output_tokens"] or 0 for call in calls),
                    usd=charged,
                    purpose="P1: direct choice " + ("pilot" if pilot else "full"),
                    note=f"Verified usage ${verified:.6f}; "
                    f"uncertain failed-attempt reserves ${charged - verified:.6f}; "
                    f"cache-read tokens {sum(call['cache_read_tokens'] or 0 for call in calls)}; "
                    f"{len(calls)} provider attempts; local replays excluded.",
                ),
                path=LEDGER_PATH,
            )
        cache.close()
        if models is not None:
            models.close()
        if agent is not None:
            agent.close()
        if not paid_cleanup_done:
            asyncio.run(inner.aclose())
            asyncio.run(adapter.aclose())
            if client is not None:
                asyncio.run(client.close())


@app.command("direct-report")
def direct_report(
    run_dir: Annotated[Path, typer.Argument(help="Recorded direct-choice run.", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option(help="Directory for generated summary and public table.")],
) -> None:
    """Report direct-choice outcomes and observed provider cache costs without paid calls."""
    build_direct_report(run_dir, out_dir=out)
    typer.echo(f"report written to {out}")


@check_app.command("bm25")
def bm25_check(tasks: TaskFile, out: ChecksFile = None) -> None:
    """Our BM25 against bm25s on every card, for every task query."""
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    _report_check("bm25", asyncio.run(check_bm25(data, read_task_file(tasks, data))), out=out)


@check_app.command("tokens")
def tokens_check(out: ChecksFile = None) -> None:
    """The token heuristic against tiktoken on every card text."""
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    _report_check("tokens", check_tokens([default_search_text(card) for card in data.catalog]), out=out)


async def _run(
    arms: Sequence[Arm],
    data: ToolRetData,
    tasks: Sequence[ToolRetTask],
    *,
    inner: OpenAIEmbedder,
    task_file: Path,
    k: int,
    run_id: str,
) -> Path:
    try:
        return await run_arms(arms, data, tasks, out_dir=RUNS_DIR, task_file=task_file, k_max=k, run_id=run_id)
    finally:
        await inner.aclose()


def _print_estimate(
    arms: Sequence[Arm], data: ToolRetData, tasks: Sequence[ToolRetTask], *, cache: CachedEmbedder, model: str
) -> None:
    documents = [default_search_text(card) for card in data.catalog]
    queries = [WARM_UP_QUERY] + [
        query_text(task, mode=mode, arm_kind=arm.query_kind)
        for arm in arms
        if arm.name in PAID_ARMS
        for mode in MODES
        for task in tasks
    ]
    typer.echo(f"Embedding estimate for {model} (tiktoken cl100k_base), texts not yet in the cache:")
    total = 0.0
    batches: tuple[tuple[EmbeddingKind, list[str]], ...] = (("document", documents), ("query", queries))
    for kind, texts in batches:
        estimate = estimate_embedding_cost(texts, model=model, cache=cache, kind=kind, max_tokens=MAX_INPUT_TOKENS)
        total += estimate.usd
        typer.echo(
            f"  {kind} texts: {estimate.texts:,}, {estimate.tokens:,} tokens, ${estimate.usd:.4f}; "
            f"longest {estimate.max_tokens_per_text:,} tokens, {estimate.truncated} cut to {MAX_INPUT_TOKENS:,}"
        )
    typer.echo(f"  total: ${total:.4f}")


def _ledger_entry(tokens: int, *, model: str, run_id: str, arms: Sequence[str], tasks: Path) -> LedgerEntry:
    price = calc_price(Usage(input_tokens=tokens), model_ref=model, provider_id="openai")
    return LedgerEntry(
        timestamp=datetime.now(UTC),
        run_id=run_id,
        provider="openai",
        model=model,
        input_tokens=tokens,
        usd=float(price.total_price),
        purpose=f"ToolRet retrieval ({', '.join(arms)}) on {tasks.name}",
    )


def _writer_ledger_entry(usage: RunUsage, *, model: str, run_id: str, tasks: Path) -> LedgerEntry:
    price = calc_price(
        Usage(
            input_tokens=usage.input_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            output_tokens=usage.output_tokens,
        ),
        model_ref=model,
        provider_id="openai",
    )
    return LedgerEntry(
        timestamp=datetime.now(UTC),
        run_id=run_id,
        provider="openai",
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        usd=float(price.total_price),
        purpose=f"F2a: model-written queries for {tasks.name}",
        note=f"{usage.requests} requests",
    )


def _listed(text: str, *, allowed: Sequence[str], option: str) -> list[str]:
    """The comma-separated names of `text`, each once and in order; all of them must be `allowed`."""
    names = list(dict.fromkeys(name.strip() for name in text.split(",") if name.strip()))
    if unknown := [name for name in names if name not in allowed]:
        raise typer.BadParameter(f"unknown {', '.join(unknown)}; choose from {', '.join(allowed)}", param_hint=option)
    return names


def _counts(text: str) -> list[int]:
    """The comma-separated candidate counts of `--k`, each once and in order."""
    try:
        counts = list(dict.fromkeys(int(item) for item in text.split(",") if item.strip()))
    except ValueError:
        raise typer.BadParameter(f"{text!r} is not a list of whole numbers", param_hint="--k") from None
    if not counts or min(counts) < 1:
        raise typer.BadParameter("give one or more counts of at least 1, such as 20,50", param_hint="--k")
    return counts


def _max_detail(text: str) -> dict[str, DetailLevel]:
    """`--max-detail`: `<decider>=<level>` pairs, such as `jev=full,logprob=brief`."""
    details: dict[str, DetailLevel] = {}
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        name, _, level = (piece.strip() for piece in item.partition("="))
        if name not in DECIDERS or level.upper() not in DetailLevel.__members__:
            levels = ", ".join(member.name.lower() for member in DetailLevel)
            raise typer.BadParameter(
                f"{item!r} is not <decider>=<level>, the decider one of {', '.join(DECIDERS)} and the level one of "
                f"{levels}",
                param_hint="--max-detail",
            )
        details[name] = DetailLevel[level.upper()]
    return details


def _decision_models(names: Sequence[str], *, clm_base_url: str) -> dict[str, JevWireModel | OpenAILogprobModel]:
    """The model behind each decider. Nothing is contacted here: each creates its client on its first ask."""
    models: dict[str, JevWireModel | OpenAILogprobModel] = {}
    for name in names:
        if name == "jev":
            models[name] = jev(JEV_MODEL)
        elif name == "clm":
            models[name] = clm(clm_base_url, model=CLM_MODEL)
        else:
            models[name] = OpenAILogprobModel(LOGPROB_MODEL)
    return models


def _print_decision_estimate(lines: Sequence[EstimateLine], *, searches: int, retrieval: CacheOnlyRetrieval) -> None:
    typer.echo(
        f"Estimate for {searches:,} decider searches, planned with stand-in models that ask nothing "
        "(heuristic token counts; every ask is counted, the ones the decision cache would answer too):"
    )
    for line in lines:
        cost = "Modal credits" if line.usd is None else f"${line.usd:.4f}"
        typer.echo(f"  {line.provider} {line.model}: {cost}; {line.note}")
    typer.echo(
        f"  retrieval: {len(retrieval.stand_in_queries):,} of {len(retrieval.searched):,} query lists have no cached "
        "embedding yet, so the estimate takes their candidates from BM25 alone"
    )


def _check_the_cap(estimate: float) -> None:
    """Print the F2a total and the cap check; exit with code 2, having spent nothing, when the estimate passes it."""
    spend = f2a_spend(read_ledger(LEDGER_PATH))
    total = spend.usd + estimate
    typer.echo(f"Estimate counted in the cap (Modal credits apart): ${estimate:.4f}")
    typer.echo(f"F2a total so far: ${spend.usd:.4f} (Modal credits, apart: ${spend.modal_usd:.4f})")
    typer.echo(f"Cap check: ${spend.usd:.4f} + ${estimate:.4f} = ${total:.4f} against the ${F2A_CAP_EUR:.2f} cap")
    if total > F2A_CAP_EUR:
        typer.echo(f"Over the cap: nothing was spent. F2a total ${spend.usd:.4f}, estimate ${estimate:.4f}.", err=True)
        raise typer.Exit(code=2)


def _clm_warmer(
    model: JevWireModel | OpenAILogprobModel | None, *, base_url: str
) -> Callable[[DecisionArm], Awaitable[None]] | None:
    """A `before_arm` hook that wakes CLM before each CLM arm: the server may have scaled to zero during the others.

    It warms `model` itself, not a cache in front of it: an answer replayed from the cache would wake nothing. `None`
    without a CLM model.
    """
    if model is None:
        return None

    async def before_arm(arm: DecisionArm) -> None:
        if arm.decider_name == "clm":
            waited = await warm_up_clm(model, base_url=base_url)
            typer.echo(f"{arm.name}: CLM answered after {waited:,.0f} s of warm-up")

    return before_arm


async def _decide(
    run: Callable[[], Awaitable[Path]],
    *,
    adapters: Mapping[str, JevWireModel | OpenAILogprobModel],
    inner: OpenAIEmbedder,
) -> Path:
    """Await `run`; the models' and the embedder's clients are closed either way."""
    try:
        return await run()
    finally:
        for adapter in adapters.values():
            await adapter.aclose()
        await inner.aclose()


def _run_records(run_dir: Path) -> list[dict[str, Any]]:
    """The records a run has written so far; none when it never started."""
    path = run_dir / "run.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _decision_ledger_entries(
    models: Mapping[str, CachedDecisionModel],
    *,
    arms: Sequence[DecisionArm],
    records: Sequence[Mapping[str, Any]],
    embeddings: CachedEmbedder,
    run_id: str,
    purpose: str,
) -> list[LedgerEntry]:
    """One ledger line per provider a decision run paid: what the caches billed, and CLM's wall time in Modal credits.

    A decision cache bills only the asks the model answered, so each line's note also counts the asks that failed: the
    provider may have billed some of them. CLM is billed by time: the wall seconds of its finished arms at the
    deployment's GPU price, in Modal credits, which the cap leaves out.
    """
    decider_of = {arm.name: arm.decider_name for arm in arms}
    wall: dict[str, float] = {}
    for record in records:
        if record["record"] == "arm" and (name := decider_of.get(record["arm"])) is not None:
            wall[name] = wall.get(name, 0.0) + record["wall_seconds"]
    timestamp = datetime.now(UTC)
    entries: list[LedgerEntry] = []
    for name, model in models.items():
        billed = model.billed
        asks = (
            f"asks: {model.misses:,} answered by the model, {model.hits:,} by the cache, {model.failures:,} failed, "
            "which the provider may have billed"
        )
        if name == "clm":
            if name not in wall and not model.misses and not model.failures:
                continue
            seconds = wall.get(name, 0.0)
            provider, billed_model, usd = "modal", CLM_MODEL, clm_usd(seconds)
            price = float(CLM_DEPLOYMENT["usd_per_gpu_hour"])
            asks = f"Modal credits: {seconds:,.0f} s of the CLM arms' wall time at ${price:.2f}/h; {asks}"
        elif not model.misses and not model.failures:
            continue
        elif name == "jev":
            provider, billed_model, usd = "typesafe", JEV_MODEL, jev_usd(billed.input_tokens)
        else:
            usage = Usage(input_tokens=billed.input_tokens, output_tokens=billed.output_tokens)
            price = calc_price(usage, model_ref=LOGPROB_MODEL, provider_id="openai")
            provider, billed_model, usd = "openai", LOGPROB_MODEL, float(price.total_price)
        entries.append(
            LedgerEntry(
                timestamp=timestamp,
                run_id=run_id,
                provider=provider,
                model=billed_model,
                input_tokens=billed.input_tokens,
                output_tokens=billed.output_tokens,
                usd=usd,
                purpose=purpose,
                note=asks,
            )
        )
    if embeddings.billed_tokens:
        usage = Usage(input_tokens=embeddings.billed_tokens)
        price = calc_price(usage, model_ref=DECISION_EMBEDDING_MODEL, provider_id="openai")
        entries.append(
            LedgerEntry(
                timestamp=timestamp,
                run_id=run_id,
                provider="openai",
                model=DECISION_EMBEDDING_MODEL,
                input_tokens=embeddings.billed_tokens,
                usd=float(price.total_price),
                purpose=purpose,
                note="the texts the hybrid retrieval embedded that the embedding cache lacked",
            )
        )
    return entries


def _report_check(name: str, result: dict[str, Any], *, out: Path | None) -> None:
    typer.echo(json.dumps(result, indent=2))
    if out is not None:
        existing: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(existing | {name: result}, indent=2) + "\n")
