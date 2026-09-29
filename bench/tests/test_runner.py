import json
from pathlib import Path
from statistics import mean
from typing import Any

import pytest
from inline_snapshot import snapshot

from toolhunch_bench.datasets.toolret import ToolRetData
from toolhunch_bench.metrics import ndcg_at_k, precision_at_1, recall_at_k
from toolhunch_bench.report import build_report
from toolhunch_bench.retrieval import build_arms, run_arms

pytestmark = pytest.mark.anyio


async def test_run_writes_provenance_and_the_report_recomputes_the_metrics(
    tmp_path: Path, toolret_data: ToolRetData, axis_embedder: Any
) -> None:
    data = toolret_data
    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({"seed": 0, "n": 4, "ids": [t.id for t in data.tasks]}))
    embedder = axis_embedder()
    arms = build_arms(
        ["keywords", "bm25", "bm25s-toolret", "dense", "hybrid"], embedder=embedder, raw_text=data.raw_text
    )

    run_dir = await run_arms(arms, data, data.tasks, out_dir=tmp_path / "runs", task_file=task_file)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert sorted(manifest) == snapshot(
        [
            "arms",
            "dataset",
            "embedding_model",
            "git",
            "k_max",
            "modes",
            "platform",
            "query_formats",
            "run_id",
            "seeds",
            "started",
            "tasks",
            "versions",
        ]
    )
    assert sorted(manifest["arms"]) == ["bm25", "bm25s-toolret", "dense", "hybrid", "keywords"]
    records = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()]
    searches = [r for r in records if r["record"] == "search"]
    assert len(searches) == 5 * 2 * 4  # arms x query modes x tasks
    assert {(r["arm"], r["mode"], r["task"]): r["query"] for r in searches if r["task"] == "t_query_1"} == snapshot(
        {
            ("keywords", "plain", "t_query_1"): "mail my boss",
            ("keywords", "instructed", "t_query_1"): "Given a `email` task, retrieve email tools mail my boss",
            ("bm25", "plain", "t_query_1"): "mail my boss",
            ("bm25", "instructed", "t_query_1"): "Given a `email` task, retrieve email tools mail my boss",
            ("bm25s-toolret", "plain", "t_query_1"): "mail my boss",
            ("bm25s-toolret", "instructed", "t_query_1"): "Given a `email` task, retrieve email tools mail my boss",
            ("dense", "plain", "t_query_1"): "mail my boss",
            ("dense", "instructed", "t_query_1"): """\
Instruct: Given a `email` task, retrieve email tools
Query: mail my boss\
""",
            ("hybrid", "plain", "t_query_1"): "mail my boss",
            ("hybrid", "instructed", "t_query_1"): "Given a `email` task, retrieve email tools mail my boss",
        }
    )

    build_report(run_dir, out_dir=tmp_path / "report")

    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    [row] = [r for r in summary["results"] if (r["mode"], r["arm"]) == ("instructed", "bm25")]
    lines = [r for r in searches if (r["mode"], r["arm"]) == ("instructed", "bm25")]
    assert row["metrics"]["recall@10"]["mean"] == mean(recall_at_k(r["ids"], set(r["relevant"]), k=10) for r in lines)
    assert row["metrics"]["ndcg@10"]["mean"] == mean(ndcg_at_k(r["ids"], set(r["relevant"])) for r in lines)
    assert row["metrics"]["p@1"]["mean"] == mean(precision_at_1(r["ids"], set(r["relevant"])) for r in lines)
    assert (tmp_path / "report" / "README.md").read_text().count("| bm25 |") == 2  # one table per query mode
