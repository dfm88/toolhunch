import json
import re
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

import pytest
from inline_snapshot import snapshot

from toolhunch import ToolCard, ToolCatalog
from toolhunch.retrieval import EmbeddingBatch, EmbeddingKind
from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask
from toolhunch_bench.metrics import ndcg_at_k, precision_at_1, recall_at_k
from toolhunch_bench.report import build_report
from toolhunch_bench.retrieval import build_arms, run_arms

pytestmark = pytest.mark.anyio

AXES = ("weather", "email", "calendar", "diary", "image", "user")


class AxisEmbedder:
    """One axis per topic word; one billed token per word. Records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[EmbeddingKind, list[str]]] = []

    @property
    def model_id(self) -> str:
        return "axis-model"

    async def embed(self, texts: Sequence[str], /, *, kind: EmbeddingKind) -> EmbeddingBatch:
        self.calls.append((kind, list(texts)))
        words = [re.findall(r"[a-z]+", text.lower()) for text in texts]
        return EmbeddingBatch(
            vectors=tuple(tuple(float(ws.count(axis)) + 0.01 for axis in AXES) for ws in words),
            input_tokens=sum(len(ws) for ws in words),
        )


TOOLS = {
    f"web_tool_{i}": {"name": name, "description": description}
    for i, (name, description) in enumerate(
        [
            ("get_weather", "Weather forecast for a city."),
            ("weather_alerts", "Severe weather alerts."),
            ("send_email", "Send an email."),
            ("read_email", "Read the email inbox."),
            ("list_calendar", "List calendar events."),
            ("add_calendar_event", "Add an event to the calendar."),
            ("get_diary_day", "Read one day of the diary."),
            ("write_diary", "Write a diary entry."),
            ("resize_image", "Resize an image."),
            ("describe_image", "Describe an image in words."),
            ("get_user", "Get a user by id."),
            ("list_users", "List every user."),
            ("stock_price", "Stock price for a ticker."),
            ("translate", "Translate text."),
            ("http_get", "HTTP GET request."),
            ("ocr", "Read text in a picture."),
            ("timer", "Start a timer."),
            ("calculator", "Evaluate arithmetic."),
            ("news", "Latest news headlines."),
            ("maps", "Directions between places."),
        ]
    )
}
RAW_TEXT = {doc_id: json.dumps(doc) for doc_id, doc in TOOLS.items()}


def task(number: int, query: str, topic: str, *relevant: int) -> ToolRetTask:
    instruction = f"Given a `{topic}` task, retrieve {topic} tools"
    return ToolRetTask(f"t_query_{number}", "t", query, instruction, frozenset(f"web_tool_{i}" for i in relevant))


TASKS = (
    task(0, "will it rain in Rome", "weather", 0, 1),
    task(1, "mail my boss", "email", 2),
    task(2, "what did I do on Monday", "diary", 6),
    task(3, "who is user 42", "user", 10),
)
DATA = ToolRetData(
    catalog=ToolCatalog(ToolCard(id=i, name=d["name"], description=d["description"]) for i, d in TOOLS.items()),
    raw_text=RAW_TEXT,
    tasks=TASKS,
    mapping_stats={"name": 1.0, "description": 1.0, "properties": 0.0},
)


async def test_run_writes_provenance_and_the_report_recomputes_the_metrics(tmp_path: Path) -> None:
    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({"seed": 0, "n": 4, "ids": [t.id for t in TASKS]}))
    embedder = AxisEmbedder()
    arms = build_arms(["keywords", "bm25", "bm25s-toolret", "dense", "hybrid"], embedder=embedder, raw_text=RAW_TEXT)

    run_dir = await run_arms(arms, DATA, TASKS, out_dir=tmp_path / "runs", task_file=task_file)

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
