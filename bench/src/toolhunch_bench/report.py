"""Reports of retrieval runs: `summary.json` and `README.md`, one table per query mode."""

from __future__ import annotations

import json
from statistics import mean
from typing import TYPE_CHECKING, Any

from genai_prices import Usage, calc_price

from toolhunch_bench.metrics import bootstrap_ci, ndcg_at_k, percentile, precision_at_1, recall_at_k

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["build_report"]

RECALL_KS = (1, 5, 10, 20, 50)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0


def build_report(run_dir: Path, *, out_dir: Path) -> None:
    """Compute the metrics of the run in `run_dir` and write `summary.json` and `README.md` to `out_dir`."""
    manifest: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text())
    records = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines() if line]
    indexes = {record["arm"]: record for record in records if record["record"] == "index"}
    searches = [record for record in records if record["record"] == "search"]
    results = [
        _row(mode, arm, lines, index=indexes[arm], searches=searches, model=manifest["embedding_model"])
        for mode in manifest["modes"]
        for arm in manifest["arms"]
        if (lines := [r for r in searches if (r["mode"], r["arm"]) == (mode, arm)])
    ]
    bootstrap = {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "level": 0.95}
    summary = {"run_id": manifest["run_id"], "bootstrap": bootstrap, "manifest": manifest, "results": results}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "README.md").write_text(_markdown(summary))


def _row(
    mode: str,
    arm: str,
    lines: list[dict[str, Any]],
    *,
    index: dict[str, Any],
    searches: list[dict[str, Any]],
    model: str | None,
) -> dict[str, Any]:
    per_task: dict[str, list[float]] = {
        **{f"recall@{k}": [recall_at_k(r["ids"], set(r["relevant"]), k=k) for r in lines] for k in RECALL_KS},
        "ndcg@10": [ndcg_at_k(r["ids"], set(r["relevant"]), k=10) for r in lines],
        "p@1": [precision_at_1(r["ids"], set(r["relevant"])) for r in lines],
    }
    seconds = [r["seconds"] for r in lines]
    # Tokens are the arm's, both modes together: the index is built once for them.
    tokens = sum(_tokens(r) for r in [index, *(r for r in searches if r["arm"] == arm)])
    return {
        "mode": mode,
        "arm": arm,
        "tasks": len(lines),
        "metrics": {
            name: {
                "mean": mean(values),
                "ci95": list(bootstrap_ci(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)),
            }
            for name, values in per_task.items()
        },
        "latency_ms": {"p50": percentile(seconds, 50) * 1000, "p95": percentile(seconds, 95) * 1000},
        "index_seconds": index["seconds"],
        "embedding_tokens": tokens,
        "usd": _usd(tokens, model=model),
    }


def _tokens(record: dict[str, Any]) -> int:
    return record["usage"]["index_tokens"] + record["usage"]["query_tokens"]


def _usd(tokens: int, *, model: str | None) -> float | None:
    if not tokens:
        return 0.0
    if model is None:
        return None
    try:
        price = calc_price(Usage(input_tokens=tokens), model_ref=model.split("@")[0], provider_id="openai")
    except LookupError:
        return None
    return float(price.total_price)


def _markdown(summary: dict[str, Any]) -> str:
    manifest = summary["manifest"]
    dataset, tasks, git = manifest["dataset"], manifest["tasks"], manifest["git"]
    commit = (git["commit"] or "unknown")[:7] + (" (dirty)" if git["dirty"] else "")
    lines = [
        f"# ToolRet retrieval, run {manifest['run_id']}",
        "",
        f"{tasks['count']} tasks from `{tasks['file']}` (sha256 `{tasks['sha256'][:12]}`), `{dataset['name']}` at "
        f"`{dataset['revision'][:7]}`: {dataset['tools']:,} tools, catalog `{dataset['catalog_fingerprint'][:19]}`. "
        f"toolhunch {manifest['versions']['toolhunch']}, commit `{commit}`, Python {manifest['versions']['python']}.",
        "",
        "Query modes:",
        "",
        "- `plain`: the query text alone, which is what an agent has.",
        "- `instructed`: ToolRet's formats, `{instruction} {query}` for lexical arms and "
        "`Instruct: {instruction}\\nQuery: {query}` for `dense`; `hybrid` gets the lexical one for both halves.",
    ]
    for mode in manifest["modes"]:
        lines += [
            "",
            f"## {mode}",
            "",
            "| arm | R@1 | R@5 | R@10 | R@20 | R@50 | nDCG@10 (95% CI) | P@1 (95% CI) | p50 ms | p95 ms "
            "| index s | emb. tokens | USD |",
            "|---|" + "---:|" * 12,
        ]
        lines += [_table_row(row) for row in summary["results"] if row["mode"] == mode]
    lines += [
        "",
        "Notes:",
        "",
        "- Relevance is binary (ToolRet qrels). R@k: share of the relevant tools in the top k. nDCG@10: log2 "
        "discount. P@1: the first result is relevant. 95% CIs: percentile bootstrap over tasks (1,000 "
        "resamples, seed 0); with few tasks they are wide, and differences inside them are not findings.",
        "- Latency is per search, after one warm-up search per arm that builds the index (`index s`). "
        "`dense` and `hybrid` latency includes one embeddings API round trip for the query.",
        "- `keywords` is Pydantic AI's keyword search, copied and parity-tested. Inside an agent "
        "`ToolSearch(max_results=10)` keeps its first 10 results, so R@20 and R@50 describe the ranking, not "
        "what the model would see.",
        "- `bm25s-toolret` replays ToolRet's own BM25 baseline (bm25s defaults, raw tool JSON, English stop "
        "words) on this corpus.",
        "- Embedding tokens and USD are what the run billed per arm, for both modes; texts already in the "
        "embedding cache cost nothing.",
    ]
    return "\n".join(lines) + "\n"


def _table_row(row: dict[str, Any]) -> str:
    metrics = row["metrics"]

    def with_ci(name: str) -> str:
        low, high = metrics[name]["ci95"]
        return f"{metrics[name]['mean']:.3f} ({low:.2f}-{high:.2f})"

    usd = "n/a" if row["usd"] is None else f"{row['usd']:.4f}"
    cells = [
        row["arm"],
        *(f"{metrics[f'recall@{k}']['mean']:.3f}" for k in RECALL_KS),
        with_ci("ndcg@10"),
        with_ci("p@1"),
        f"{row['latency_ms']['p50']:.1f}",
        f"{row['latency_ms']['p95']:.1f}",
        f"{row['index_seconds']:.2f}",
        f"{row['embedding_tokens']:,}",
        usd,
    ]
    return "| " + " | ".join(cells) + " |"
