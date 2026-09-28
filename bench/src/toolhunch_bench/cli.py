"""Command line of the benchmark harness, installed as `toolhunch-bench`."""

# No `from __future__ import annotations`: Typer reads the command signatures at runtime.

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import tiktoken
import typer
from dotenv import load_dotenv
from genai_prices import Usage, calc_price

from toolhunch import Embedder, OpenAIEmbedder, default_search_text
from toolhunch_bench import BENCH_DIR
from toolhunch_bench.checks import check_bm25, check_tokens
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
from toolhunch_bench.embedding_cache import CachedEmbedder, TruncatingEmbedder, estimate_embedding_cost
from toolhunch_bench.ledger import LedgerEntry, append_ledger
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

app = typer.Typer(help="toolhunch benchmark harness.", no_args_is_help=True, add_completion=False)
toolret_app = typer.Typer(help=f"ToolRet ({TOOLRET_DATASET}) data.", no_args_is_help=True)
check_app = typer.Typer(help="Implementation checks on the real catalog.", no_args_is_help=True)
app.add_typer(toolret_app, name="toolret")
app.add_typer(check_app, name="check")

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
) -> None:
    """Write a stratified sample of task ids."""
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    tasks = sample_tasks(data.tasks, n=n, seed=seed)
    write_task_file(out, tasks, seed=seed)
    typer.echo(f"{len(tasks)} task ids written to {out}")


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
    inner = OpenAIEmbedder(embedding_model)
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
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory under bench/runs.", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option(help="Directory for summary.json and README.md.")],
) -> None:
    """Compute the metrics and tables of a run."""
    build_report(run_dir, out_dir=out)
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


def _report_check(name: str, result: dict[str, Any], *, out: Path | None) -> None:
    typer.echo(json.dumps(result, indent=2))
    if out is not None:
        existing: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(existing | {name: result}, indent=2) + "\n")
