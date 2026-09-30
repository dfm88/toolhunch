"""Retrieval-only runs: every arm answers every task through `ToolSearchPipeline`, with full provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Literal

from toolhunch import BM25Retriever, DenseRetriever, HybridRetriever, RetrievalUsage, ToolSearchPipeline
from toolhunch.retrieval import TextAnalyzer, s_stemmer
from toolhunch_bench import BENCH_DIR
from toolhunch_bench.baselines import BM25sToolRetRetriever, KeywordsRetriever
from toolhunch_bench.datasets.toolret import TOOLRET_CORPUS_SHA256, TOOLRET_DATASET, TOOLRET_REVISION

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from toolhunch import Embedder, Retriever, SearchResult, ToolCard
    from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask

__all__ = [
    "ARM_NAMES",
    "PAID_ARMS",
    "QUERY_FORMATS",
    "RUNS_DIR",
    "Arm",
    "QueryKind",
    "QueryMode",
    "build_arms",
    "query_text",
    "repo_path",
    "run_arms",
    "run_provenance",
]

type QueryMode = Literal["plain", "instructed"]
"""`plain`: the query alone, what an agent has; `instructed`: prefixed with ToolRet's task instruction."""
type QueryKind = Literal["lexical", "dense"]
"""Which instructed format an arm gets: ToolRet's BM25 one or its embedding-model one."""

QUERY_FORMATS: dict[str, str] = {
    "plain": "{query}",
    "instructed/lexical": "{instruction} {query}",
    "instructed/dense": "Instruct: {instruction}\nQuery: {query}",
}
ARM_NAMES = ("keywords", "bm25", "bm25-stem", "bm25-namedesc", "bm25-raw", "bm25s-toolret", "dense", "hybrid")
PAID_ARMS = frozenset({"dense", "hybrid"})
RUNS_DIR = BENCH_DIR / "runs"
WARM_UP_QUERY = "warm-up search"
_CARD_TEXT = "name, description and parameter names"


def query_text(task: ToolRetTask, *, mode: QueryMode, arm_kind: QueryKind) -> str:
    """The text an arm searches with, in ToolRet's formats (`toolret/eval.py`) when instructed."""
    if mode == "plain":
        return task.query
    if arm_kind == "lexical":
        return f"{task.instruction} {task.query}"
    return f"Instruct: {task.instruction}\nQuery: {task.query}"


@dataclass(frozen=True, slots=True)
class Arm:
    """A named retriever, its configuration as recorded in the manifest, and its query format."""

    name: str
    retriever: Retriever
    config: Mapping[str, Any]
    query_kind: QueryKind = "lexical"


def build_arms(names: Sequence[str], *, embedder: Embedder, raw_text: Mapping[str, str]) -> list[Arm]:
    """Build the arms named in `names`, in that order; `embedder` is called only by paid arms.

    Raises:
        ValueError: A name is not one of `ARM_NAMES`.
    """
    if unknown := sorted(set(names) - set(ARM_NAMES)):
        raise ValueError(f"unknown arms {unknown}; choose from {', '.join(ARM_NAMES)}")
    return [_arm(name, embedder=embedder, raw_text=raw_text) for name in names]


def _arm(name: str, *, embedder: Embedder, raw_text: Mapping[str, str]) -> Arm:
    bm25 = {"retriever": "BM25Retriever", "k1": 1.2, "b": 0.75, "stop_words": "Lucene English", "stemmer": None}
    match name:
        case "keywords":
            config = {"retriever": "KeywordsRetriever", "copies": "pydantic-ai-slim 2.50.0 keywords_search_fn"}
            return Arm(name, KeywordsRetriever(seed=0), config | {"seed": 0, "text": "name and description"})
        case "bm25":
            return Arm(name, BM25Retriever(), bm25 | {"text": _CARD_TEXT})
        case "bm25-stem":
            analyzer = TextAnalyzer(stemmer=s_stemmer)
            return Arm(name, BM25Retriever(analyzer=analyzer), bm25 | {"stemmer": "s_stemmer", "text": _CARD_TEXT})
        case "bm25-namedesc":
            return Arm(name, BM25Retriever(search_text=_name_and_description), bm25 | {"text": "name and description"})
        case "bm25-raw":

            def raw(card: ToolCard) -> str:
                return raw_text[card.id]

            return Arm(name, BM25Retriever(search_text=raw), bm25 | {"text": "raw corpus JSON"})
        case "bm25s-toolret":
            config = {
                "retriever": "bm25s",
                "bm25s": _version("bm25s"),
                "k1": 1.5,
                "b": 0.75,
                "method": "lucene",
                "stop_words": "bm25s English, documents and query",
                "text": "raw corpus JSON",
                "replays": "toolret/eval.py eval_bm25 @ d181f1c",
            }
            return Arm(name, BM25sToolRetRetriever(raw_text=raw_text), config)
        case "dense":
            config = {
                "retriever": "DenseRetriever",
                "embedding_model": embedder.model_id,
                "embedder": repr(embedder),
                "text": _CARD_TEXT,
            }
            return Arm(name, DenseRetriever(embedder), config, query_kind="dense")
        case _:
            config = {
                "retriever": "HybridRetriever(bm25, dense)",
                "rrf_k": 60,
                "depth": 100,
                "embedding_model": embedder.model_id,
                "embedder": repr(embedder),
                "text": _CARD_TEXT,
                "instructed_format": "lexical for both halves",
            }
            return Arm(name, HybridRetriever([BM25Retriever(), DenseRetriever(embedder)]), config)


def _name_and_description(card: ToolCard) -> str:
    return f"{card.name}\n{card.description}"


async def run_arms(
    arms: Sequence[Arm],
    data: ToolRetData,
    tasks: Sequence[ToolRetTask],
    *,
    modes: Sequence[QueryMode] = ("plain", "instructed"),
    out_dir: Path,
    task_file: Path,
    k_max: int = 50,
    run_id: str | None = None,
) -> Path:
    """Run every arm on every task in every mode; returns the run directory `out_dir/<run_id>`.

    The directory gets `manifest.json` (provenance, written first) and `run.jsonl`: per arm one
    `index` record (a warm-up search that builds the index, timed apart), then one `search` record
    per mode and task with the query text, the top `k_max` ids, the relevant ids, seconds and usage.

    Args:
        arms: What to run, from [`build_arms`][toolhunch_bench.retrieval.build_arms].
        data: The loaded dataset; its catalog is searched.
        tasks: The tasks, usually read from `task_file`.
        modes: Query modes to run.
        out_dir: Parent of the run directory, usually `bench/runs`.
        task_file: The task file, hashed into the manifest.
        k_max: Results kept per search.
        run_id: Directory name; a UTC timestamp by default.
    """
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True)
    manifest = _manifest(arms, data, tasks, modes=modes, k_max=k_max, task_file=task_file, run_id=run_id)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (run_dir / "run.jsonl").open("w", encoding="utf-8") as out:
        for arm in arms:
            pipeline = ToolSearchPipeline(arm.retriever, k=k_max)
            warm_up = await pipeline.search([WARM_UP_QUERY], data.catalog)
            out.write(json.dumps({"record": "index", "arm": arm.name, **_cost(warm_up)}) + "\n")
            for mode in modes:
                for task in tasks:
                    query = query_text(task, mode=mode, arm_kind=arm.query_kind)
                    result = await pipeline.search([query], data.catalog)
                    record = {
                        "record": "search",
                        "arm": arm.name,
                        "mode": mode,
                        "task": task.id,
                        "query": query,
                        "relevant": sorted(task.relevant),
                        "ids": result.ids,
                    }
                    out.write(json.dumps(record | _cost(result)) + "\n")
            out.flush()
    return run_dir


def _cost(result: SearchResult) -> dict[str, Any]:
    retrieval = [stage.usage for stage in result.trace if isinstance(stage.usage, RetrievalUsage)]
    return {
        "seconds": sum(stage.seconds for stage in result.trace),
        "usage": {
            "index_tokens": sum(usage.index_tokens for usage in retrieval),
            "query_tokens": sum(usage.query_tokens for usage in retrieval),
        },
    }


def _manifest(
    arms: Sequence[Arm],
    data: ToolRetData,
    tasks: Sequence[ToolRetTask],
    *,
    modes: Sequence[QueryMode],
    k_max: int,
    task_file: Path,
    run_id: str,
) -> dict[str, Any]:
    manifest = run_provenance(data, tasks, task_file=task_file, run_id=run_id)
    manifest["seeds"]["keywords_order"] = 0
    return manifest | {
        "modes": list(modes),
        "query_formats": QUERY_FORMATS,
        "k_max": k_max,
        "embedding_model": next(
            (arm.config["embedding_model"] for arm in arms if "embedding_model" in arm.config), None
        ),
        "arms": {arm.name: {"query_kind": arm.query_kind, **arm.config} for arm in arms},
    }


def run_provenance(data: ToolRetData, tasks: Sequence[ToolRetTask], *, task_file: Path, run_id: str) -> dict[str, Any]:
    """The fields every run manifest starts with: the run, the code, the machine, the dataset and the tasks."""
    packages = ("toolhunch", "toolhunch-bench", "pydantic-ai-slim", "bm25s", "tiktoken", "genai-prices")
    return {
        "run_id": run_id,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_state(),
        "versions": {"python": platform.python_version()} | {package: _version(package) for package in packages},
        "platform": {"system": platform.platform(), "machine": platform.machine()},
        "dataset": {
            "name": TOOLRET_DATASET,
            "revision": TOOLRET_REVISION,
            "corpus_sha256": TOOLRET_CORPUS_SHA256,
            "tools": len(data.catalog),
            "catalog_fingerprint": data.catalog.fingerprint,
            "mapping_stats": dict(data.mapping_stats),
        },
        "tasks": {
            "file": repo_path(task_file),
            "sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
            "count": len(tasks),
        },
        "seeds": {"sample": json.loads(task_file.read_text()).get("seed")},
    }


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=BENCH_DIR, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def repo_path(path: Path) -> str:
    """`path` relative to the repository, so no local directory leaks into committed results."""
    try:
        return path.resolve().relative_to(BENCH_DIR.parent).as_posix()
    except ValueError:
        return path.name


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
