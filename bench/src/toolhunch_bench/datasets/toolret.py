"""ToolRet (`mteb/ToolRetrieval`): download at a pinned revision, map tools to cards, sample tasks.

ToolRet (arXiv:2503.01763) pairs 7,961 queries from 35 tool-use datasets with one shared corpus of
44,453 tools. Its licence is unknown, so the repository commits task ids only and the data is
downloaded into a local cache.

Each corpus row is a JSON document whose schema depends on the source dataset. Parameter names are
read from about a dozen shapes, checked against the whole corpus on 2026-09-27:

- `parameters` as a JSON Schema object, a `name → spec` mapping, or a list of `{"name": ...}` objects;
- `required_parameters` + `optional_parameters`, `inputs`: lists of `{"name": ...}` objects;
- `doc_arguments`: a JSON Schema object;
- `api_arguments`: a list, a `name → value` mapping, or free text (`"question, context"`), of which
  only identifier-like pieces are kept;
- `additional_required_arguments` + `optional_arguments`: mappings keyed `"name (type)"`.

Free-text `parameters` (`"(image: str)"`) and nested toolkit tools yield no names.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import httpx2
import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

from toolhunch import ToolCard, ToolCatalog

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = [
    "TOOLRET_CORPUS_SHA256",
    "TOOLRET_DATASET",
    "TOOLRET_REVISION",
    "TOOLRET_SUBTASKS",
    "ToolRetData",
    "ToolRetTask",
    "card_from_toolret",
    "load_toolret",
    "sample_tasks",
]

TOOLRET_DATASET = "mteb/ToolRetrieval"
TOOLRET_REVISION = "76d45e560059754e289ea202462865a585679619"
# Every `<subtask>-corpus` file at this revision is the same file.
TOOLRET_CORPUS_SHA256 = "774aa8c5263eed9870fb6eefc47491c67d3998e7eda1d897f9b359e20be67efb"
TOOLRET_SUBTASKS: tuple[str, ...] = (
    "apibank", "apigen", "appbench", "autotools-food", "autotools-music", "autotools-weather",
    "craft-math-algebra", "craft-tabmwp", "craft-vqa", "gorilla-huggingface", "gorilla-pytorch",
    "gorilla-tensor", "gpt4tools", "gta", "metatool", "mnms", "restgpt-spotify", "restgpt-tmdb",
    "reversechain", "rotbench", "t-eval-dialog", "t-eval-step", "taskbench-daily", "taskbench-huggingface",
    "taskbench-multimedia", "tool-be-honest", "toolace", "toolalpaca", "toolbench", "toolbench-sam",
    "toolemu", "tooleyes", "toolink", "toollens", "ultratool",
)  # fmt: skip

_BASE_URL = f"https://huggingface.co/datasets/{TOOLRET_DATASET}/resolve/{TOOLRET_REVISION}"
_DESCRIPTION_FIELDS = ("description", "functionality", "func_description", "description_for_human")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TYPE_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")  # "where_to (str)" → "where_to"
_SOURCE = re.compile(r"(.+)_tool_\d+")


@dataclass(frozen=True, slots=True)
class ToolRetTask:
    """One ToolRet query and the ids of the tools that answer it."""

    id: str
    subtask: str
    query: str
    instruction: str
    relevant: frozenset[str]


@dataclass(frozen=True, slots=True)
class ToolRetData:
    """The corpus as a catalog, each tool's raw JSON text, the tasks sorted by id, and mapping coverage.

    `mapping_stats` gives the share of cards whose name, description and at least one parameter name
    were found in the raw text (keys `name`, `description`, `properties`).
    """

    catalog: ToolCatalog
    raw_text: Mapping[str, str]
    tasks: tuple[ToolRetTask, ...]
    mapping_stats: Mapping[str, float]


def card_from_toolret(doc_id: str, text: str) -> ToolCard:
    """Map one corpus row to a card; the shapes read are listed in the module docstring.

    Raises:
        ValueError: `text` is not a JSON object; the message names `doc_id`.
    """
    try:
        doc: object = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"ToolRet tool {doc_id!r}: text is not JSON ({error})") from error
    if not isinstance(doc, dict):
        raise ValueError(f"ToolRet tool {doc_id!r}: text is not a JSON object")
    fields = cast("dict[str, object]", doc)
    names = _parameter_names(fields)
    source = _SOURCE.fullmatch(doc_id)
    return ToolCard(
        id=doc_id,
        name=_text(fields.get("name")) or doc_id,
        description=next((found for field in _DESCRIPTION_FIELDS if (found := _text(fields.get(field)))), ""),
        parameters={"type": "object", "properties": {name: {} for name in names}} if names else None,
        source=source.group(1) if source else None,
    )


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parameter_names(fields: Mapping[str, object]) -> list[str]:
    names = [
        *_declared_names(fields.get("parameters")),
        *_declared_names(fields.get("required_parameters")),
        *_declared_names(fields.get("optional_parameters")),
        *_declared_names(fields.get("inputs")),
        *_declared_names(fields.get("doc_arguments")),
        *_declared_names(fields.get("api_arguments")),
        *_free_text_names(fields.get("api_arguments")),
    ]
    for field in ("additional_required_arguments", "optional_arguments"):
        names += [_TYPE_SUFFIX.sub("", key) for key in _declared_names(fields.get(field))]
    return list(dict.fromkeys(name for name in names if name))


def _declared_names(value: object) -> list[str]:
    """Names from a JSON Schema object, a `name → spec` mapping or a list of `{"name": ...}` objects."""
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        properties = mapping.get("properties")
        # A mapping may hold parameters literally named "type" and "properties": a schema's "type" is a string.
        if isinstance(mapping.get("type"), str) and isinstance(properties, dict):
            return list(cast("dict[str, object]", properties))
        return list(mapping)
    names: list[str] = []
    if isinstance(value, list):
        for item in cast("list[object]", value):
            if isinstance(item, dict) and isinstance(name := cast("dict[str, object]", item).get("name"), str):
                names.append(name)
    return names


def _free_text_names(value: object) -> list[str]:
    """Identifier-like pieces of free-text arguments: `"question, context"`, `["images", "data.csv"]`."""
    if isinstance(value, str):
        pieces = value.split(",")
    elif isinstance(value, list):
        pieces = [item for item in cast("list[object]", value) if isinstance(item, str)]
    else:
        return []
    return [piece.strip() for piece in pieces if _IDENTIFIER.fullmatch(piece.strip())]


def load_toolret(*, cache_dir: Path, client: httpx2.Client | None = None) -> ToolRetData:
    """Load ToolRet at `TOOLRET_REVISION`, downloading into `cache_dir` what is not there yet.

    The shared corpus is downloaded once and checked against `TOOLRET_CORPUS_SHA256` on every load;
    each subtask's queries and qrels are cached next to it. Files are written atomically, so an
    interrupted download leaves nothing behind. Tasks without a relevant tool are dropped.

    Args:
        cache_dir: Directory for the parquet files, for example `bench/runs/data/toolret/<revision>`.
        client: HTTP client to download with; by default a new one, closed on return.

    Raises:
        ValueError: The corpus does not match `TOOLRET_CORPUS_SHA256`.
    """
    if client is None:
        with httpx2.Client(timeout=120.0) as own_client:
            return load_toolret(cache_dir=cache_dir, client=own_client)
    corpus = _read_parquet(_cached_corpus(client, cache_dir))
    raw_text = {cast("str", row["id"]): cast("str", row["text"]) for row in corpus}
    catalog = ToolCatalog([card_from_toolret(doc_id, text) for doc_id, text in raw_text.items()])
    tasks: list[ToolRetTask] = []
    for subtask in TOOLRET_SUBTASKS:
        relevant: defaultdict[str, set[str]] = defaultdict(set)
        for row in _read_parquet(_cached_file(client, cache_dir, f"{subtask}-qrels")):
            if row["score"] > 0:
                relevant[row["query-id"]].add(row["corpus-id"])
        tasks += [
            ToolRetTask(
                id=row["id"],
                subtask=subtask,
                query=row["text"],
                instruction=row["instruction"] or "",
                relevant=frozenset(relevant[row["id"]]),
            )
            for row in _read_parquet(_cached_file(client, cache_dir, f"{subtask}-queries"))
            if relevant.get(row["id"])
        ]
    return ToolRetData(
        catalog=catalog,
        raw_text=raw_text,
        tasks=tuple(sorted(tasks, key=lambda task: task.id)),
        mapping_stats=_mapping_stats(catalog),
    )


def _cached_corpus(client: httpx2.Client, cache_dir: Path) -> Path:
    path = cache_dir / "corpus.parquet"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == TOOLRET_CORPUS_SHA256:
        return path
    content = _download(client, f"{TOOLRET_SUBTASKS[0]}-corpus")
    digest = hashlib.sha256(content).hexdigest()
    if digest != TOOLRET_CORPUS_SHA256:
        raise ValueError(f"ToolRet corpus sha256 mismatch: expected {TOOLRET_CORPUS_SHA256}, got {digest}")
    _write_atomically(path, content)
    return path


def _cached_file(client: httpx2.Client, cache_dir: Path, folder: str) -> Path:
    path = cache_dir / f"{folder}.parquet"
    if not path.exists():
        _write_atomically(path, _download(client, folder))
    return path


def _download(client: httpx2.Client, folder: str) -> bytes:
    response = client.get(f"{_BASE_URL}/{folder}/test-00000-of-00001.parquet", follow_redirects=True)
    response.raise_for_status()
    return response.content


def _write_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(content)
    partial.replace(path)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    """Rows of a parquet file: the one call into pyarrow, which ships no type information."""
    parquet: Any = pq
    return parquet.read_table(path).to_pylist()


def _mapping_stats(catalog: ToolCatalog) -> dict[str, float]:
    total = len(catalog) or 1
    return {
        "name": sum(card.name != card.id for card in catalog) / total,
        "description": sum(bool(card.description) for card in catalog) / total,
        "properties": sum(card.parameters is not None for card in catalog) / total,
    }


def sample_tasks(tasks: Sequence[ToolRetTask], *, n: int, seed: int) -> list[ToolRetTask]:
    """Draw a stratified sample of `n` tasks, sorted by id.

    Every subtask gets one task; the other `n - subtasks` places are shared in proportion to the
    tasks each subtask has left (largest remainder, ties broken by subtask name). Within a subtask,
    tasks are drawn by a `random.Random` seeded with `seed` and the subtask name.

    Raises:
        ValueError: `n` is smaller than the number of subtasks.
    """
    by_subtask: defaultdict[str, list[ToolRetTask]] = defaultdict(list)
    for task in sorted(tasks, key=lambda task: task.id):
        by_subtask[task.subtask].append(task)
    if n < len(by_subtask):
        raise ValueError(f"n={n} is less than the {len(by_subtask)} subtasks, which need one task each")
    if n >= len(tasks):
        return sorted(tasks, key=lambda task: task.id)
    left = {subtask: len(group) - 1 for subtask, group in by_subtask.items()}
    quotas = {subtask: (n - len(by_subtask)) * count / sum(left.values()) for subtask, count in left.items()}
    counts = {subtask: 1 + math.floor(quota) for subtask, quota in quotas.items()}
    by_remainder = sorted(quotas, key=lambda subtask: (math.floor(quotas[subtask]) - quotas[subtask], subtask))
    for subtask in by_remainder[: n - sum(counts.values())]:
        counts[subtask] += 1
    sample: list[ToolRetTask] = []
    for subtask, group in by_subtask.items():
        shuffled = group.copy()
        random.Random(f"{seed}/{subtask}").shuffle(shuffled)
        sample += shuffled[: counts[subtask]]
    return sorted(sample, key=lambda task: task.id)
