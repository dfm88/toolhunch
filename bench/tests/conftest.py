import hashlib
import json
import re
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import tiktoken

from toolhunch import ToolCard, ToolCatalog
from toolhunch.decision import (
    Answer,
    ChoiceQuestion,
    DecisionError,
    DecisionRefused,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    ModelLimits,
    QuestionKind,
    check_request,
    choice_answer,
)
from toolhunch.decision.planner import NONE_KEY, NONE_TEXT
from toolhunch.retrieval import EmbeddingBatch, EmbeddingKind
from toolhunch_bench import deciders
from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask

AXES = ("weather", "email", "calendar", "diary", "image", "user")


class AxisEmbedder:
    """One axis per topic word; one billed token per word. Records every call, and whether it was closed."""

    def __init__(self, *, model_id: str = "axis-model") -> None:
        self.calls: list[tuple[EmbeddingKind, list[str]]] = []
        self.closed = False
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed(self, texts: Sequence[str], /, *, kind: EmbeddingKind) -> EmbeddingBatch:
        self.calls.append((kind, list(texts)))
        words = [re.findall(r"[a-z]+", text.lower()) for text in texts]
        return EmbeddingBatch(
            vectors=tuple(tuple(float(ws.count(axis)) + 0.01 for axis in AXES) for ws in words),
            input_tokens=sum(len(ws) for ws in words),
        )

    async def aclose(self) -> None:
        self.closed = True


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


def _task(number: int, query: str, topic: str, *relevant: int) -> ToolRetTask:
    instruction = f"Given a `{topic}` task, retrieve {topic} tools"
    return ToolRetTask(f"t_query_{number}", "t", query, instruction, frozenset(f"web_tool_{i}" for i in relevant))


TASKS = (
    _task(0, "will it rain in Rome", "weather", 0, 1),
    _task(1, "mail my boss", "email", 2),
    _task(2, "what did I do on Monday", "diary", 6),
    _task(3, "who is user 42", "user", 10),
)
DATA = ToolRetData(
    catalog=ToolCatalog(ToolCard(id=i, name=d["name"], description=d["description"]) for i, d in TOOLS.items()),
    raw_text=RAW_TEXT,
    tasks=TASKS,
    mapping_stats={"name": 1.0, "description": 1.0, "properties": 0.0},
)


class FakeDecisionModel:
    """A decision model for bench tests: weighted choice answers, and a `DecisionError` on demand.

    Every option weighs 1, the tool whose option key is `favourite` 8, and the reserved option `none_weight`; the
    answer is those weights normalised. A request whose state contains `fail_on` raises `DecisionError` instead, as a
    model that answers nothing usable does, or, with `fail_status`, as a server that refuses with that HTTP status. A
    request whose state contains `refuse_on` raises `DecisionRefused`, billed at 10 input tokens.
    Like the real adapters it checks each request against its `limits` first. Each call reports 10 input tokens,
    1 output token, 0.1 s of wall time (or what `seconds` gives for the request) and 0.01 s of server time.

    Attributes:
        asks: Every request it was asked, failed ones included, in order.
        closed: Whether `aclose` was called.
    """

    def __init__(
        self,
        *,
        model_id: str = "fake@test",
        limits: ModelLimits | None = None,
        prompt_version: str | None = None,
        favourite: str | None = None,
        none_weight: float = 1.0,
        fail_on: str | None = None,
        fail_status: int | None = None,
        refuse_on: str | None = None,
        seconds: Callable[[DecisionRequest], float] | None = None,
    ) -> None:
        self.model_id = model_id
        self.limits = ModelLimits(source="test", checked=date(2026, 9, 29)) if limits is None else limits
        self.question_kinds: frozenset[QuestionKind] = frozenset({"choice"})
        self.prompt_version = prompt_version
        self.asks: list[DecisionRequest] = []
        self.closed = False
        self._favourite = favourite
        self._none_weight = none_weight
        self._fail_on = fail_on
        self._fail_status = fail_status
        self._refuse_on = refuse_on
        self._seconds = seconds

    def __repr__(self) -> str:
        return f"FakeDecisionModel({self.model_id!r})"

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        check_request(request, limits=self.limits, kinds=self.question_kinds)
        self.asks.append(request)
        if self._fail_on is not None and self._fail_on in request.state:
            if self._fail_status is not None:
                raise DecisionError(f"{self.model_id}: HTTP {self._fail_status}: refused", status=self._fail_status)
            raise DecisionError(f"{self.model_id}: no option letter among the top logprobs")
        if self._refuse_on is not None and self._refuse_on in request.state:
            usage = DecisionUsage(1, 10, 0)
            raise DecisionRefused(f"{self.model_id}: refused", names=list(request.questions), usage=usage)
        answers: dict[str, Answer] = {}
        for key, question in request.questions.items():
            assert isinstance(question, ChoiceQuestion)
            weights = {option: self._weight(option, text) for option, text in question.options.items()}
            answers[key] = choice_answer(weights, keys=list(question.options), model_id=self.model_id)
        seconds = 0.1 if self._seconds is None else self._seconds(request)
        return DecisionResponse(
            answers=answers, usage=DecisionUsage(1, 10, 1), seconds=seconds, server_seconds=0.01, raw={}
        )

    async def aclose(self) -> None:
        self.closed = True

    def _weight(self, key: str, text: str) -> float:
        if (key, text) == (NONE_KEY, NONE_TEXT):
            return self._none_weight
        return 8.0 if key == self._favourite else 1.0


class WordEncoding:
    """A stand-in for a tiktoken encoding: one token per word."""

    name = "words"

    def encode(self, text: str, *, disallowed_special: Any = ()) -> list[int]:
        return list(range(len(text.split())))


@pytest.fixture(autouse=True)
def offline_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count tokens with `WordEncoding` in every bench test.

    tiktoken downloads a vocabulary the first time it is asked for one; with sockets blocked that passes only on a
    machine that has it cached, and fails on a fresh CI runner.
    """

    def word_encoding(name: str) -> WordEncoding:
        return WordEncoding()

    monkeypatch.setattr(tiktoken, "get_encoding", word_encoding)


@pytest.fixture
def axis_embedder() -> type[AxisEmbedder]:
    """The `AxisEmbedder` class; under importlib import mode a test module cannot import it from here."""
    return AxisEmbedder


@pytest.fixture
def toolret_data() -> ToolRetData:
    """Twenty tools and four tasks shaped like ToolRet's, with the gold tool ids of each task."""
    return DATA


@pytest.fixture
def fake_decision_model() -> type[FakeDecisionModel]:
    """The `FakeDecisionModel` class; under importlib import mode a test module cannot import it from here."""
    return FakeDecisionModel


@pytest.fixture
def laya_word_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """A tokenizer file that takes one token per word and per run of punctuation, and its SHA-256.

    The registry's Laya entries read it in place of the pinned checkpoint's, which no test reads.
    """
    # A WordLevel model behind a Whitespace pre-tokenizer, written as `tokenizers` saves one: its typed constructors
    # resolve to an untyped module under pyright.
    config: dict[str, Any] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(deciders, "laya_tokenizer_path", lambda: path)
    monkeypatch.setattr(deciders, "LAYA_TOKENIZER_SHA256", digest)
    return path, digest
