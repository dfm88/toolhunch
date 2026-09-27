"""Command line of the benchmark harness, installed as `toolhunch-bench`."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tiktoken
from dotenv import load_dotenv
from genai_prices import Usage, calc_price

from toolhunch import OpenAIEmbedder, default_search_text
from toolhunch_bench import BENCH_DIR
from toolhunch_bench.checks import check_bm25, check_tokens
from toolhunch_bench.datasets.toolret import (
    TOOLRET_DATASET,
    TOOLRET_REVISION,
    load_toolret,
    read_task_file,
    sample_tasks,
    write_task_file,
)
from toolhunch_bench.embedding_cache import CachedEmbedder, TruncatingEmbedder, estimate_embedding_cost
from toolhunch_bench.ledger import LedgerEntry, append_ledger
from toolhunch_bench.report import build_report
from toolhunch_bench.retrieval import ARM_NAMES, PAID_ARMS, RUNS_DIR, WARM_UP_QUERY, build_arms, query_text, run_arms

if TYPE_CHECKING:
    from collections.abc import Sequence

    from toolhunch import Embedder
    from toolhunch.retrieval import EmbeddingKind
    from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask
    from toolhunch_bench.retrieval import Arm, QueryMode

TOOLRET_CACHE = BENCH_DIR / "runs" / "data" / "toolret" / TOOLRET_REVISION
MODES: tuple[QueryMode, ...] = ("plain", "instructed")
MAX_INPUT_TOKENS = 8191  # text-embedding-3-* reject inputs over 8,192 tokens; cut one short of it


def main(argv: Sequence[str] | None = None) -> int:
    """Parse `argv` and run the command; returns the exit code."""
    args = _parser().parse_args(argv)
    match args.command:
        case "toolret":
            return _toolret(args)
        case "retrieval":
            return _retrieval(args)
        case "report":
            build_report(args.run_dir, out_dir=args.out)
            print(f"report written to {args.out}")
            return 0
        case _:
            return _check(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolhunch-bench", description="toolhunch benchmark harness")
    commands = parser.add_subparsers(dest="command", required=True)

    toolret = commands.add_parser("toolret", help=f"ToolRet ({TOOLRET_DATASET}) data")
    toolret_actions = toolret.add_subparsers(dest="action", required=True)
    toolret_actions.add_parser("fetch", help="download the pinned revision and print how the tools were mapped")
    sample = toolret_actions.add_parser("sample", help="write a stratified sample of task ids")
    sample.add_argument("--n", type=int, default=50, help="number of tasks (default: 50)")
    sample.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    sample.add_argument("--out", type=Path, required=True, help="task file to write")

    retrieval = commands.add_parser("retrieval", help="run retrieval arms on the tasks of a task file")
    retrieval.add_argument("--tasks", type=Path, required=True, help="task file")
    retrieval.add_argument("--arms", required=True, help=f"comma-separated, from: {', '.join(ARM_NAMES)}")
    retrieval.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
    retrieval.add_argument("--k", type=int, default=50, help="results kept per search (default: 50)")
    retrieval.add_argument("--dry-run", action="store_true", help="print the embedding cost estimate and stop")

    report = commands.add_parser("report", help="compute metrics and tables for a run")
    report.add_argument("run_dir", type=Path, help="run directory under bench/runs")
    report.add_argument("--out", type=Path, required=True, help="directory for summary.json and README.md")

    check = commands.add_parser("check", help="implementation checks on the real catalog")
    checks = check.add_subparsers(dest="check", required=True)
    bm25 = checks.add_parser("bm25", help="our BM25 against bm25s on every card, for every task query")
    bm25.add_argument("--tasks", type=Path, required=True, help="task file")
    bm25.add_argument("--out", type=Path, help="JSON file to add the result to")
    tokens = checks.add_parser("tokens", help="the token heuristic against tiktoken on every card text")
    tokens.add_argument("--out", type=Path, help="JSON file to add the result to")
    return parser


def _toolret(args: argparse.Namespace) -> int:
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    if args.action == "fetch":
        tools = len(data.catalog)
        print(
            f"ToolRet @ {TOOLRET_REVISION[:7]}: {tools:,} tools, {len(data.tasks):,} tasks "
            f"in {len({task.subtask for task in data.tasks})} subtasks (cache: {TOOLRET_CACHE})"
        )
        labels = {"name": "a name", "description": "a description", "properties": "at least one parameter name"}
        for field, label in labels.items():
            share = data.mapping_stats[field]
            print(f"cards with {label}: {round(share * tools):,} ({share:.3%})")
        return 0
    tasks = sample_tasks(data.tasks, n=args.n, seed=args.seed)
    write_task_file(args.out, tasks, seed=args.seed)
    print(f"{len(tasks)} task ids written to {args.out}")
    return 0


def _retrieval(args: argparse.Namespace) -> int:
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    tasks = read_task_file(args.tasks, data)
    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    paid = [name for name in names if name in PAID_ARMS]
    model: str = args.embedding_model
    inner = OpenAIEmbedder(model)
    cache: CachedEmbedder | None = None
    if paid:
        encoding = tiktoken.get_encoding("cl100k_base")
        cache = CachedEmbedder(TruncatingEmbedder(inner, max_tokens=MAX_INPUT_TOKENS, encoding=encoding))
    embedder: Embedder = cache if cache is not None else inner
    arms = build_arms(names, embedder=embedder, raw_text=data.raw_text)
    if cache is None:
        if args.dry_run:
            print("no paid arms: nothing to estimate")
            return 0
    else:
        _print_estimate(arms, data, tasks, cache=cache, model=model)
        if args.dry_run:
            cache.close()
            return 0
        load_dotenv(BENCH_DIR.parent / ".env", override=False)  # keys come from the environment, never printed
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        run_dir = asyncio.run(_run(arms, data, tasks, inner=inner, task_file=args.tasks, k=args.k, run_id=run_id))
    finally:
        if cache is not None:
            if cache.billed_tokens:  # a failed run still paid for what it embedded
                append_ledger(
                    _ledger_entry(cache.billed_tokens, model=model, run_id=run_id, arms=paid, tasks=args.tasks)
                )
            cache.close()
    print(f"run written to {run_dir}")
    return 0


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
    print(f"Embedding estimate for {model} (tiktoken cl100k_base), texts not yet in the cache:")
    total = 0.0
    batches: tuple[tuple[EmbeddingKind, list[str]], ...] = (("document", documents), ("query", queries))
    for kind, texts in batches:
        estimate = estimate_embedding_cost(texts, model=model, cache=cache, kind=kind, max_tokens=MAX_INPUT_TOKENS)
        total += estimate.usd
        print(
            f"  {kind} texts: {estimate.texts:,}, {estimate.tokens:,} tokens, ${estimate.usd:.4f}; "
            f"longest {estimate.max_tokens_per_text:,} tokens, {estimate.truncated} cut to {MAX_INPUT_TOKENS:,}"
        )
    print(f"  total: ${total:.4f}")


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


def _check(args: argparse.Namespace) -> int:
    data = load_toolret(cache_dir=TOOLRET_CACHE)
    if args.check == "bm25":
        result = asyncio.run(check_bm25(data, read_task_file(args.tasks, data)))
    else:
        result = check_tokens([default_search_text(card) for card in data.catalog])
    print(json.dumps(result, indent=2))
    if args.out is not None:
        out: Path = args.out
        existing: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(existing | {args.check: result}, indent=2) + "\n")
    return 0
