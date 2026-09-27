"""Command line of the benchmark harness, installed as `toolhunch-bench`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from toolhunch_bench.datasets.toolret import TOOLRET_DATASET, TOOLRET_REVISION, load_toolret, sample_tasks

if TYPE_CHECKING:
    from collections.abc import Sequence

BENCH_DIR = Path(__file__).resolve().parents[2]
TOOLRET_CACHE = BENCH_DIR / "runs" / "data" / "toolret" / TOOLRET_REVISION


def main(argv: Sequence[str] | None = None) -> int:
    """Parse `argv` and run the command; returns the exit code."""
    parser = argparse.ArgumentParser(prog="toolhunch-bench", description="toolhunch benchmark harness")
    commands = parser.add_subparsers(dest="command", required=True)
    toolret = commands.add_parser("toolret", help=f"ToolRet ({TOOLRET_DATASET}) data")
    toolret_commands = toolret.add_subparsers(dest="action", required=True)
    toolret_commands.add_parser("fetch", help="download the pinned revision and print how the tools were mapped")
    sample = toolret_commands.add_parser("sample", help="write a stratified sample of task ids")
    sample.add_argument("--n", type=int, default=50, help="number of tasks (default: 50)")
    sample.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    sample.add_argument("--out", type=Path, required=True, help="task file to write")
    args = parser.parse_args(argv)

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
    task_file = {
        "dataset": TOOLRET_DATASET,
        "revision": TOOLRET_REVISION,
        "seed": args.seed,
        "n": args.n,
        "ids": [task.id for task in tasks],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(task_file, indent=2) + "\n")
    print(f"{len(tasks)} task ids written to {args.out}")
    return 0
