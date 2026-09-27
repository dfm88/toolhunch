import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx2
import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]
import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]
import pytest
from inline_snapshot import snapshot

from toolhunch import ToolCard
from toolhunch_bench.datasets import toolret
from toolhunch_bench.datasets.toolret import (
    TOOLRET_SUBTASKS,
    ToolRetTask,
    card_from_toolret,
    load_toolret,
    sample_tasks,
)

FIXTURES = Path(__file__).parent / "fixtures"

# Queries per subtask in ToolRet at the pinned revision (sizes only; the tasks below are synthetic).
SUBTASK_SIZES = dict(
    zip(
        TOOLRET_SUBTASKS,
        [101, 1000, 32, 22, 32, 11, 280, 174, 200, 500, 43, 55, 32, 14, 200, 33, 40, 54, 200, 550, 50, 50, 40, 23, 40,
         350, 1000, 94, 1100, 197, 38, 95, 497, 314, 500],
        strict=True,
    )
)  # fmt: skip


def property_names(card: ToolCard) -> list[str]:
    return list(card.parameters["properties"]) if card.parameters else []


def test_cards_from_every_schema_shape() -> None:
    rows = json.loads((FIXTURES / "toolret_shapes.json").read_text())
    cards = [card_from_toolret(row["id"], json.dumps(row["doc"])) for row in rows]

    assert {c.id: (c.name, c.description, property_names(c), c.source) for c in cards} == snapshot(
        {
            "alpha_tool_0": ("create_wallet", "Create a new wallet. Returns its address.", ["password"], "alpha"),
            "beta_tool_0": ("zip_lookup", "Look up ZIP codes.", ["ids", "properties", "type"], "beta"),
            "gamma_tool_0": ("get_album", "Get an album by id.", ["id", "market"], "gamma"),
            "delta_tool_0": ("task_status", "Check a task.", ["task_id", "verbose"], "delta"),
            "epsilon_tool_0": ("image_classifier", "Image classification", ["images", "return_tensors"], "epsilon"),
            "epsilon_tool_1": ("qa_model", "Answer questions.", ["question", "context"], "epsilon"),
            "epsilon_tool_2": ("summarizer", "Summarize text.", ["pretrained_model_name"], "epsilon"),
            "epsilon_tool_3": ("resnet", "Image model.", ["pretrained"], "epsilon"),
            "zeta_tool_0": ("count_chars", "Count the characters of a file.", ["file_path"], "zeta"),
            "eta_tool_0": ("Calculator", "Evaluate an expression.", ["expression"], "eta"),
            "theta_tool_0": ("book_house", "Book a house.", ["where_to", "check_in_date", "rating"], "theta"),
            "iota_tool_0": ("iota_tool_0", "Convert units of length.", [], "iota"),
            "kappa_tool_0": ("Terminal", "Run terminal commands.", [], "kappa"),
            "lambda_tool_0": ("image_crop", "", ["image"], "lambda"),
            "mu_tool_0": ("ocr", "Read the text in an image.", [], "mu"),
        }
    )


def test_malformed_text_names_the_id() -> None:
    with pytest.raises(ValueError, match="broken_tool_0"):
        card_from_toolret("broken_tool_0", "{not json")
    with pytest.raises(ValueError, match="list_tool_0"):
        card_from_toolret("list_tool_0", "[1, 2]")


def synthetic_tasks() -> list[ToolRetTask]:
    return [
        ToolRetTask(id=f"{subtask}_query_{i}", subtask=subtask, query="q", instruction="i", relevant=frozenset({"t"}))
        for subtask, size in SUBTASK_SIZES.items()
        for i in range(size)
    ]


def test_sample_is_stratified_and_reproducible() -> None:
    tasks = synthetic_tasks()

    pilot = sample_tasks(tasks, n=50, seed=0)

    assert len(pilot) == 50
    assert [t.id for t in pilot] == sorted(t.id for t in pilot)
    assert Counter(t.subtask for t in pilot) == snapshot(
        Counter(
            {
                "apibank": 1,
                "apigen": 3,
                "appbench": 1,
                "autotools-food": 1,
                "autotools-music": 1,
                "autotools-weather": 1,
                "craft-math-algebra": 2,
                "craft-tabmwp": 1,
                "craft-vqa": 2,
                "gorilla-huggingface": 2,
                "gorilla-pytorch": 1,
                "gorilla-tensor": 1,
                "gpt4tools": 1,
                "gta": 1,
                "metatool": 2,
                "mnms": 1,
                "restgpt-spotify": 1,
                "restgpt-tmdb": 1,
                "reversechain": 1,
                "rotbench": 2,
                "t-eval-dialog": 1,
                "t-eval-step": 1,
                "taskbench-daily": 1,
                "taskbench-huggingface": 1,
                "taskbench-multimedia": 1,
                "tool-be-honest": 2,
                "toolace": 3,
                "toolalpaca": 1,
                "toolbench-sam": 1,
                "toolbench": 3,
                "toolemu": 1,
                "tooleyes": 1,
                "toolink": 2,
                "toollens": 2,
                "ultratool": 2,
            }
        )
    )  # one each, the other 15 by size
    assert sample_tasks(tasks, n=50, seed=0) == pilot
    assert sample_tasks(tasks, n=50, seed=1) != pilot
    with pytest.raises(ValueError, match="35 subtasks"):
        sample_tasks(tasks, n=34, seed=0)


def parquet_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    arrow: Any = pa  # untyped
    parquet: Any = pq
    buffer = io.BytesIO()
    parquet.write_table(arrow.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


WEATHER_TEXT = json.dumps({"name": "get_weather", "description": "Weather forecast.", "parameters": {"city": "City."}})
CORPUS = parquet_bytes(
    [
        {"id": "alpha_tool_0", "text": WEATHER_TEXT, "title": ""},
        {
            "id": "alpha_tool_1",
            "text": json.dumps({"name": "send_email", "description": "Send an email."}),
            "title": "",
        },
    ]
)


def huggingface(requested: list[str]) -> httpx2.MockTransport:
    """Serves `<subtask>-<corpus|queries|qrels>/test-00000-of-00001.parquet` like the dataset repo."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        requested.append(request.url.path)
        subtask, kind = request.url.path.split("/")[-2].rsplit("-", 1)
        if kind == "corpus":
            return httpx2.Response(200, content=CORPUS)
        query_id = f"{subtask}_query_0"
        if kind == "queries":
            rows = [{"id": query_id, "text": "weather in Rome", "instruction": "Given a `weather` task"}]
        else:
            rows = [{"query-id": query_id, "corpus-id": "alpha_tool_0", "score": 1}]
        return httpx2.Response(200, content=parquet_bytes(rows))

    return httpx2.MockTransport(handler)


def test_load_rejects_a_corpus_with_the_wrong_checksum(tmp_path: Path) -> None:
    with httpx2.Client(transport=huggingface([])) as client, pytest.raises(ValueError, match="sha256"):
        load_toolret(cache_dir=tmp_path, client=client)
    assert list(tmp_path.iterdir()) == []  # nothing cached


def test_load_joins_queries_and_qrels_and_reuses_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(toolret, "TOOLRET_CORPUS_SHA256", hashlib.sha256(CORPUS).hexdigest())
    requested: list[str] = []

    with httpx2.Client(transport=huggingface(requested)) as client:
        data = load_toolret(cache_dir=tmp_path, client=client)
        again = load_toolret(cache_dir=tmp_path, client=client)

    assert len(requested) == 1 + 2 * len(TOOLRET_SUBTASKS)  # the shared corpus once; the second load is cached
    assert TOOLRET_SUBTASKS[0] in requested[0]
    assert data.tasks[0] == ToolRetTask(
        id="apibank_query_0",
        subtask="apibank",
        query="weather in Rome",
        instruction="Given a `weather` task",
        relevant=frozenset({"alpha_tool_0"}),
    )
    assert len(data.tasks) == len(TOOLRET_SUBTASKS)
    assert [card.name for card in data.catalog] == ["get_weather", "send_email"]
    assert data.raw_text["alpha_tool_0"] == WEATHER_TEXT
    assert data.mapping_stats == {"name": 1.0, "description": 1.0, "properties": 0.5}
    assert (again.catalog.fingerprint, again.tasks) == (data.catalog.fingerprint, data.tasks)
