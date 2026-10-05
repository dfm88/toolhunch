"""Laya's own tokenizer, a check that refuses before sending what Laya would cut, and the calibration of both.

Laya 0.3.27 fits every question into a token budget by cutting it, and reports only some of the cuts. The bench
counts each Laya request with Laya's own tokenizer, the way Laya builds the question, and refuses before sending
any request Laya would cut in silence. `LayaCalibration` measures, offline, how the planner's counts compare.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from tokenizers import Tokenizer as _HubTokenizer

from toolhunch import DetailLevel, render_within_budget
from toolhunch.decision import CandidatesDoNotFit, ChoiceQuestion, DecisionError
from toolhunch.decision.planner import (
    OPTION_OVERHEAD_TOKENS,
    build_state,
    fit_state,
    option_keys,
    plan_question,
    plan_rounds,
)
from toolhunch.retrieval.base import clean_queries
from toolhunch.tokens import HeuristicTokenizer
from toolhunch_bench import BENCH_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from toolhunch import RenderedCards, ToolCard
    from toolhunch.decision import DecisionModel, DecisionRequest, DecisionResponse, ModelLimits, QuestionKind
    from toolhunch.decision.planner import PlannedQuestion, RoundPlan
    from toolhunch.tokens import Tokenizer

__all__ = [
    "FREE_HEAD",
    "LAYA_REVISION",
    "LAYA_TOKENIZER_SHA256",
    "LAYA_VERSION",
    "OPTION_CAP",
    "LayaCalibration",
    "LayaCheckedModel",
    "LayaHead",
    "LayaTokenizer",
    "PlanConfig",
    "RecordedSearch",
    "laya_head",
    "laya_tokenizer_path",
    "recorded_searches",
]

LAYA_VERSION = "0.3.27"
"""The laya package the local server runs, whose way of building a question `laya_head` mirrors."""
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
"""The pinned commit of `convaiinnovations/laya` on the Hugging Face Hub, whose English checkpoint the bench runs."""
LAYA_TOKENIZER_SHA256 = "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"
"""The SHA-256 of `tokenizer/tokenizer.json` at `LAYA_REVISION`."""
OPTION_CAP = 48  # laya 0.3.27 common.py build_head: each " <option>" cut at 48 tokens
FREE_HEAD = 16  # build_head: below 16 tokens left, every option is cut

_HUB_CACHE = BENCH_DIR / "models" / "hf"
_MASK = "[MASK]"  # the checkpoint's mask token: Laya replaces it with a space in every text it encodes
_CHOICE_PREFIX = "choice question: "  # build_head's "%s question: %s" for a choice question
# What the English checkpoint reads when a request leaves them out (its rl_agent_config.json).
_DEFAULT_HEAD_MAX_LEN = 192
_DEFAULT_MAX_LEN = 512
_FINALISTS_PER_CHUNK = 2  # ChoiceDecider's default, which the bench's deciders keep
_REFUSAL_EXAMPLES = 5


def laya_tokenizer_path(hub_cache: Path = _HUB_CACHE) -> Path:
    """Where the pinned checkpoint's `tokenizer.json` sits in the Hugging Face hub cache `hub_cache`.

    Raises:
        FileNotFoundError: It is not there. `bench/deploy/laya_local.sh` downloads it with the checkpoint.
    """
    path = hub_cache / "models--convaiinnovations--laya" / "snapshots" / LAYA_REVISION / "tokenizer" / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Laya's tokenizer is not at {path}: start bench/deploy/laya_local.sh once to download the checkpoint"
        )
    return path


class LayaTokenizer:
    """Laya's own tokenizer: a [`Tokenizer`][toolhunch.tokens.Tokenizer] that counts what Laya counts.

    It reads `tokenizer.json` of the pinned checkpoint and counts a text as Laya encodes each part of a question,
    without special tokens. A file from another revision would plan and check with the wrong vocabulary in silence,
    so a file is refused unless its SHA-256 is the expected one.

    Attributes:
        sha256: The SHA-256 of the file it read.
    """

    def __init__(self, path: Path | None = None, *, sha256: str = LAYA_TOKENIZER_SHA256) -> None:
        """Read the tokenizer at `path`, by default the pinned checkpoint's in `bench/models/hf`.

        Raises:
            FileNotFoundError: `path` is `None` and the pinned checkpoint has not been downloaded.
            ValueError: The file's SHA-256 is not `sha256`.
        """
        path = laya_tokenizer_path() if path is None else path
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha256:
            raise ValueError(
                f"{path} has sha256 {digest}, expected {sha256}: not Laya's tokenizer at {LAYA_REVISION[:7]}"
            )
        self.sha256 = digest
        self._tokenizer = _HubTokenizer.from_buffer(data)

    def __repr__(self) -> str:
        return f"LayaTokenizer(revision={LAYA_REVISION[:7]!r})"

    def count(self, text: str, /) -> int:
        """Return the number of tokens Laya encodes `text` into, special tokens left out."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class LayaHead:
    """How Laya builds the head of one choice question: its instructions and options, before the state.

    Attributes:
        instruction_tokens: The tokens of `"choice question: <instructions>"`, uncut.
        option_tokens: Per option, in option order: 1 for its `[MASK]` and the tokens of `" <key>: <text>"`, or of
            `" <key>"` for an empty text, uncut.
        tokens: The length of the head Laya builds, `[CLS]`, instructions, `[SEP]`, options and `[SEP]`, after any
            cut it makes. With no `refusal` nothing is cut.
        refusal: Why Laya would cut part of the question; `None` when it keeps all of it.
    """

    instruction_tokens: int
    option_tokens: tuple[int, ...]
    tokens: int
    refusal: str | None


def laya_head(question: ChoiceQuestion, *, tokenizer: LayaTokenizer, head_max_len: int) -> LayaHead:
    """Count `question` as Laya builds its head within `head_max_len` tokens, and say whether Laya would cut it.

    The first refusal found is given, in this order: an option over `OPTION_CAP` tokens; the options, one `[MASK]`
    each, leaving fewer than `FREE_HEAD` tokens of `head_max_len`, below which Laya cuts every option; the
    instructions over what the options leave.
    """
    # Mirrors laya 0.3.27 common.py. render_options writes "<key>: <text>", or the key alone for an empty text.
    # build_head replaces [MASK] with a space, encodes "choice question: <instructions>" and each " <option>" cut at
    # 48 tokens behind a [MASK]; with fewer than 16 tokens left it cuts each option to max(4, (head_max_len - 16) // n),
    # then it cuts the instructions to max(8, what the options leave).
    instruction_tokens = tokenizer.count(_CHOICE_PREFIX + question.instructions.replace(_MASK, " "))
    rendered = [key if text == "" else f"{key}: {text}" for key, text in question.options.items()]
    option_tokens = tuple(1 + tokenizer.count(" " + option.replace(_MASK, " ")) for option in rendered)

    left = head_max_len - sum(option_tokens)
    over = [
        (key, tokens - 1)
        for key, tokens in zip(question.options, option_tokens, strict=True)
        if tokens > 1 + OPTION_CAP
    ]
    refusal: str | None = None
    if over:
        refusal = f"option {over[0][0]!r} takes {over[0][1]} tokens, over {OPTION_CAP}"
    elif left < FREE_HEAD:
        refusal = (
            f"the options take {sum(option_tokens)} of head_max_len {head_max_len}, leaving fewer than {FREE_HEAD}"
        )
    elif instruction_tokens > left:
        refusal = f"the instructions take {instruction_tokens} tokens, more than the {left} the options leave"

    kept = [min(tokens, 1 + OPTION_CAP) for tokens in option_tokens]
    if head_max_len - sum(kept) < FREE_HEAD:
        per_option = max(4, (head_max_len - FREE_HEAD) // len(kept))
        kept = [min(tokens, per_option) for tokens in kept]
    instructions_kept = min(instruction_tokens, max(8, head_max_len - sum(kept)))
    return LayaHead(
        instruction_tokens=instruction_tokens,
        option_tokens=option_tokens,
        tokens=1 + instructions_kept + 1 + sum(kept) + 1,
        refusal=refusal,
    )


class LayaCheckedModel:
    """A Laya model that refuses, before sending, any request Laya would cut in silence.

    Laya cuts an option over 48 tokens, every option when they leave too little of the head, and instructions longer
    than what the options leave, and reports it only when two options end up identical, so `strict` cannot catch
    the rest. Each choice question of a request is counted with Laya's own tokenizer first (`laya_head`), and a
    request Laya would cut fails loudly, as a server refusal would, instead of being answered about a question Laya
    never read in full. The wrapped model's `model_id`, `limits`, `question_kinds` and `prompt_version` are passed
    through.

    Attributes:
        refused: The requests refused before sending.
    """

    def __init__(self, inner: DecisionModel, *, tokenizer: LayaTokenizer, head_max_len: int) -> None:
        """Check every request for `inner`, a Laya model whose questions get `head_max_len` tokens."""
        self._inner = inner
        self._tokenizer = tokenizer
        self._head_max_len = head_max_len
        self.refused = 0

    @property
    def model_id(self) -> str:
        """The wrapped model's identity."""
        return self._inner.model_id

    @property
    def limits(self) -> ModelLimits:
        """The wrapped model's limits."""
        return self._inner.limits

    @property
    def question_kinds(self) -> frozenset[QuestionKind]:
        """The wrapped model's question kinds."""
        return self._inner.question_kinds

    @property
    def prompt_version(self) -> str | None:
        """The wrapped model's prompt version."""
        return self._inner.prompt_version

    @property
    def head_max_len(self) -> int:
        """The head budget each question is checked against."""
        return self._head_max_len

    def __repr__(self) -> str:
        return f"LayaCheckedModel({self._inner!r}, head_max_len={self._head_max_len})"

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Ask the wrapped model, once every choice question of `request` has passed `laya_head`.

        Raises:
            DecisionError: Laya would cut a question; nothing is sent. Or the wrapped model failed.
        """
        for question in request.questions.values():
            if not isinstance(question, ChoiceQuestion):
                continue
            head = laya_head(question, tokenizer=self._tokenizer, head_max_len=self._head_max_len)
            if head.refusal is not None:
                self.refused += 1
                raise DecisionError(f"{self.model_id}: refused before sending: {head.refusal}")
        return await self._inner.ask(request, **options)

    async def aclose(self) -> None:
        """Close the wrapped model's client, if it has one."""
        if (close := getattr(self._inner, "aclose", None)) is not None:
            await close()


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanConfig:
    """One way the bench plans Laya's questions: a registry entry's limits and head budget, and a range of detail.

    Attributes:
        name: What the calibration calls it, such as `laya-wide floor BRIEF`.
        limits: The limits the model declares, which the planner keeps within.
        head_max_len: The head budget Laya builds each question within.
        max_detail: The most detail a card is shown at.
        min_detail: The floor: a question that fits only below it is split.
    """

    name: str
    limits: ModelLimits
    head_max_len: int
    max_detail: DetailLevel
    min_detail: DetailLevel


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordedSearch:
    """One search of a decision run: what a decider was given to rank.

    Attributes:
        task: The task id.
        variant: `positive`, or `negative` with the gold tools taken out.
        k: The candidates retrieved.
        context: The request the state is written from.
        queries: The search queries.
        cards: The candidates, in retrieval order.
    """

    task: str
    variant: str
    k: int
    context: str | None
    queries: tuple[str, ...]
    cards: tuple[ToolCard, ...]


def recorded_searches(run_dir: Path, *, cards: Mapping[str, ToolCard], source: str = "plain") -> list[RecordedSearch]:
    """The first-repeat searches from `source` of the retrieval-only arms of the decision run in `run_dir`.

    Every decider of a run ranks the candidates of its retrieval-only arm, so each search is read once.
    """
    searches: list[RecordedSearch] = []
    for line in (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record") != "search" or record["decider"] is not None or record["source"] != source:
            continue
        if record.get("repeat", 0) != 0:
            continue
        searches.append(
            RecordedSearch(
                task=record["task"],
                variant=record["variant"],
                k=record["k"],
                context=record["context"],
                queries=tuple(record["queries"]),
                cards=tuple(cards[card_id] for card_id in record["candidates"]),
            )
        )
    return searches


class LayaCalibration:
    """Laya's own counts against the planner's: on the catalog, on recorded searches and on recorded probes.

    Nothing is asked of any model.
    """

    def __init__(self, tokenizer: LayaTokenizer) -> None:
        """Count with `tokenizer`, the pinned checkpoint's."""
        self._tokenizer = tokenizer
        self._heuristic = HeuristicTokenizer()

    def catalog(self, cards: Sequence[ToolCard], *, sample_every: int = 20) -> dict[str, Any]:
        """How many name-level options Laya would cut, and how the planner's counts compare on a sample.

        A card's key is its name, so its option at name level is `" <name>: <name>"`; a key alone is `" <name>"`.
        The sample takes every `sample_every`-th card. At each level it compares Laya's option, `1 + " <key>:
        <text>"` with its `[MASK]`, with what the planner charges, `key + 4 + text`, counted by the heuristic and by
        Laya's tokenizer: a positive shortfall is an undercount.
        """
        count = self._tokenizer.count
        sample = cards[::sample_every]
        levels: dict[str, Any] = {}
        for level in (DetailLevel.NAME, DetailLevel.BRIEF, DetailLevel.FULL):
            options: list[int] = []
            heuristic: list[int] = []
            laya: list[int] = []
            for card in sample:
                text = card.render(level)
                tokens = 1 + count(f" {card.name}: {text}")
                options.append(tokens)
                heuristic.append(tokens - self._planned_option(card.name, text, tokenizer=self._heuristic))
                laya.append(tokens - self._planned_option(card.name, text, tokenizer=self._tokenizer))
            levels[level.name] = {
                "median_option_tokens": statistics.median(options),
                "over_cap": sum(tokens - 1 > OPTION_CAP for tokens in options),
                "heuristic_shortfall": _shortfalls(heuristic),
                "laya_shortfall": _shortfalls(laya),
            }
        return {
            "cards": len(cards),
            "name_options_over_cap": sum(count(f" {card.name}: {card.name}") > OPTION_CAP for card in cards),
            "keys_alone_over_cap": sum(count(f" {card.name}") > OPTION_CAP for card in cards),
            "sample": {"every": sample_every, "cards": len(sample), "levels": levels},
        }

    def searches(self, searches: Sequence[RecordedSearch]) -> dict[str, dict[str, int]]:
        """Per k, the searches with a name-level option over `OPTION_CAP`, and with a key alone over it."""
        over: dict[str, tuple[bool, bool]] = {}  # per card id: each card is counted once, however often retrieved
        by_k: dict[int, dict[str, int]] = {}
        for search in searches:
            counts = by_k.setdefault(search.k, {"searches": 0, "name_option_over_cap": 0, "key_alone_over_cap": 0})
            for card in search.cards:
                if card.id not in over:
                    over[card.id] = self._over_cap(card)
            flags = [over[card.id] for card in search.cards]
            counts["searches"] += 1
            counts["name_option_over_cap"] += any(name for name, _ in flags)
            counts["key_alone_over_cap"] += any(key for _, key in flags)
        return {f"K{k}": counts for k, counts in sorted(by_k.items())}

    def plans(self, searches: Sequence[RecordedSearch], *, config: PlanConfig) -> dict[str, Any]:
        """Plan every search as the bench's decider does under `config`, and check every question with `laya_head`.

        Each search is planned with Laya's tokenizer and the reserved option, as the bench asks: one question, or
        round one's questions and a final over the finalists the planner checks against, the cards of each chunk
        whose options cost the most, planned as the decider plans its final. Reported: the questions `laya_head`
        refuses; the largest shortfall of the planner's count, per question `sum(option_tokens) + max(16,
        instruction_tokens)` against its estimate without the state, and per option `option_tokens - 1` against
        `key + 4 + text`; and the searches the planner refuses with `CandidatesDoNotFit`, by kind: `card` when one
        card cannot be shown, `candidates` when the candidates as a whole do not fit, `final` when the final
        question the decider would build does not fit; in all and per k.
        """
        count = self._tokenizer.count
        rounds: Counter[str] = Counter()
        refusals: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        do_not_fit: dict[str, Counter[str]] = {}
        card_ids: set[str] = set()
        question_shortfalls: list[int] = []
        option_shortfalls: list[int] = []
        for search in searches:
            state, _ = fit_state(
                build_state(search.context, clean_queries(search.queries)),
                limits=config.limits,
                tokenizer=self._tokenizer,
            )
            try:
                plan = plan_rounds(
                    state,
                    search.cards,
                    reserved=True,
                    limits=config.limits,
                    tokenizer=self._tokenizer,
                    max_detail=config.max_detail,
                    min_detail=config.min_detail,
                    finalists_per_chunk=_FINALISTS_PER_CHUNK,
                )
            except CandidatesDoNotFit as error:
                if error.card_id is not None:
                    card_ids.add(error.card_id)
                kind = "candidates" if error.card_id is None else "card"
                do_not_fit.setdefault(f"K{search.k}", Counter())[kind] += 1
                continue
            questions = self._questions(state, search.cards, plan, config=config)
            if questions is None:
                do_not_fit.setdefault(f"K{search.k}", Counter())["final"] += 1
                continue
            chunks = len(plan.chunks)
            rounds["one question" if plan.single is not None else f"{chunks} chunk{'s' * (chunks != 1)}"] += 1
            state_tokens = count(state)
            for asked, planned in questions:
                head = laya_head(planned.question, tokenizer=self._tokenizer, head_max_len=config.head_max_len)
                needed = sum(head.option_tokens) + max(FREE_HEAD, head.instruction_tokens)
                question_shortfalls.append(needed - (planned.estimated_input_tokens - state_tokens))
                option_shortfalls.extend(
                    tokens - 1 - (count(key) + OPTION_OVERHEAD_TOKENS + count(text))
                    for (key, text), tokens in zip(planned.question.options.items(), head.option_tokens, strict=True)
                )
                if head.refusal is None:
                    continue
                refusals[_refusal_kind(head.refusal)] += 1
                if len(examples) < _REFUSAL_EXAMPLES:
                    examples.append(
                        {"task": search.task, "variant": search.variant, "k": search.k, "question": asked}
                        | {"options": len(planned.question.options), "refusal": head.refusal}
                    )
        return {
            "name": config.name,
            "head_max_len": config.head_max_len,
            "max_question_tokens": config.limits.max_question_tokens,
            "max_option_tokens": config.limits.max_option_tokens,
            "max_state_plus_question_tokens": config.limits.max_state_plus_question_tokens,
            "max_detail": config.max_detail.name,
            "min_detail": config.min_detail.name,
            "searches": len(searches),
            "plans": dict(sorted(rounds.items())),
            "candidates_do_not_fit": {
                kind: sum(counts[kind] for counts in do_not_fit.values()) for kind in ("card", "candidates", "final")
            },
            "candidates_do_not_fit_by_k": {k: dict(sorted(counts.items())) for k, counts in sorted(do_not_fit.items())},
            "card_ids": sorted(card_ids),
            "questions": len(question_shortfalls),
            "refused": sum(refusals.values()),
            "refusals": dict(sorted(refusals.items())),
            "refusal_examples": examples,
            "question_shortfall": _shortfalls(question_shortfalls),
            "option_shortfall": _shortfalls(option_shortfalls),
        }

    def probes(self, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Laya's own count of each recorded probe, `usage.input_tokens`, against this module's.

        This module's count is the head, then the state cut to what the head leaves of `max_len`, then a closing
        `[SEP]`, as laya 0.3.27 `build_sequence` builds a sequence. A probe is skipped, by label, when its reply has
        no `usage` or its request is not one choice question.
        """
        rows: list[dict[str, Any]] = []
        skipped: list[str] = []
        for record in records:
            request = record["request"]
            body: Mapping[str, Any] = record.get("body") or {}
            usage: Mapping[str, Any] | None = body.get("usage")
            questions = list(request["questions"].values())
            if usage is None or len(questions) != 1 or questions[0].get("type") != "choice":
                skipped.append(record["label"])
                continue
            wire = questions[0]
            question = ChoiceQuestion(
                instructions=wire["instructions"], options={key: text or "" for key, text in wire["criteria"].items()}
            )
            max_len = request.get("max_len", _DEFAULT_MAX_LEN)
            head = laya_head(
                question, tokenizer=self._tokenizer, head_max_len=request.get("head_max_len", _DEFAULT_HEAD_MAX_LEN)
            )
            state_tokens = self._tokenizer.count(str(request["state"]).replace(_MASK, " "))
            kept = min(state_tokens, max(0, max_len - head.tokens - 1))
            input_tokens = min(max_len, head.tokens + kept + 1)
            rows.append(
                {
                    "label": record["label"],
                    "input_tokens": input_tokens,
                    "laya_input_tokens": usage["input_tokens"],
                    "state_tokens": state_tokens,
                    "laya_state_tokens": usage.get("state_tokens"),
                    "refusal": head.refusal,
                    "match": input_tokens == usage["input_tokens"] and state_tokens == usage.get("state_tokens"),
                }
            )
        return {"matched": sum(row["match"] for row in rows), "probes": len(rows), "skipped": skipped, "rows": rows}

    def _planned_option(self, key: str, text: str, *, tokenizer: Tokenizer) -> int:
        """What the planner charges for one option: its key, its framing and its text."""
        return tokenizer.count(key) + OPTION_OVERHEAD_TOKENS + tokenizer.count(text)

    def _over_cap(self, card: ToolCard) -> tuple[bool, bool]:
        """Whether the card's name-level option is over `OPTION_CAP`, and whether its key alone is."""
        name, key = (
            self._tokenizer.count(text) > OPTION_CAP for text in (f" {card.name}: {card.name}", f" {card.name}")
        )
        return name, key

    def _questions(
        self, state: str, cards: Sequence[ToolCard], plan: RoundPlan, *, config: PlanConfig
    ) -> list[tuple[str, PlannedQuestion]] | None:
        """Every question the decider asks under `plan`, the final at its worst case; `None` when that does not fit."""
        if plan.single is not None:
            return [("single", plan.single)]
        asked = [("round one", chunk.question) for chunk in plan.chunks if chunk.question is not None]
        final = plan_question(
            state,
            self._worst_finalists(cards, plan, config=config),
            reserved=True,
            limits=config.limits,
            tokenizer=self._tokenizer,
            max_detail=config.max_detail,
            min_detail=config.min_detail,
        )
        return None if final is None else [*asked, ("final", final)]

    def _worst_finalists(self, cards: Sequence[ToolCard], plan: RoundPlan, *, config: PlanConfig) -> list[ToolCard]:
        """The finalists `plan_rounds` plans the final against: in each chunk, the costliest options at the floor.

        A card is charged under the key it has among all the candidates, as `plan_rounds` charges it; the
        finalists keep candidate order, as the decider's final does.
        """
        keys = option_keys(cards, reserved=True)
        chunk_count = len(plan.chunks)
        kept: set[str] = set()
        for start, chunk in enumerate(plan.chunks):
            costs = [
                self._floor_cost(card, key, config=config)
                for card, key in zip(chunk.cards, keys[start::chunk_count], strict=True)
            ]
            ranked = sorted(zip(costs, chunk.cards, strict=True), key=lambda entry: entry[0], reverse=True)
            kept.update(card.id for _, card in ranked[: plan.finalists_per_chunk])
        return [card for card in cards if card.id in kept]

    def _floor_cost(self, card: ToolCard, key: str, *, config: PlanConfig) -> int:
        """What the planner charges for the card's option at the floor, under its own caps, as `plan_rounds` ranks."""
        limits = config.limits
        fixed = self._tokenizer.count(key) + OPTION_OVERHEAD_TOKENS
        caps = [
            cap
            for cap in (
                limits.max_text_tokens,
                None if limits.max_option_tokens is None else limits.max_option_tokens - fixed,
            )
            if cap is not None
        ]
        cap = min(caps) if caps else None
        if cap is not None and self._tokenizer.count(card.name) > cap:
            return fixed  # shown by its key alone
        rendered = render_within_budget(
            [card],
            max_tokens=None,
            tokenizer=self._tokenizer,
            max_tokens_per_text=cap,
            max_detail=config.min_detail,
            min_detail=config.min_detail,
        )
        return fixed + cast("RenderedCards", rendered).tokens  # without a total budget the level always fits


def _shortfalls(values: Sequence[int]) -> dict[str, int | None]:
    """The largest shortfall, and how many are positive: undercounts."""
    return {"largest": max(values, default=None), "undercounted": sum(value > 0 for value in values)}


def _refusal_kind(refusal: str) -> str:
    """Which of `laya_head`'s three refusals `refusal` is."""
    if refusal.startswith("option "):
        return "option over the cap"
    if refusal.startswith("the options "):
        return "options over the head"
    return "instructions over what the options leave"
