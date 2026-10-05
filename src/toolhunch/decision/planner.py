"""Plan choice questions within a decision model's declared limits: pure functions, no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from toolhunch.cards import DetailLevel, render_within_budget
from toolhunch.decision.base import ChoiceQuestion

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from toolhunch.cards import ToolCard
    from toolhunch.decision.base import ModelLimits
    from toolhunch.tokens import Tokenizer

__all__ = [
    "INSTRUCTIONS",
    "NONE_KEY",
    "NONE_TEXT",
    "OPTION_OVERHEAD_TOKENS",
    "PROMPT_VERSION",
    "CandidatesDoNotFit",
    "Chunk",
    "PlannedQuestion",
    "RoundPlan",
    "build_state",
    "fit_state",
    "option_keys",
    "plan_question",
    "plan_rounds",
]

PROMPT_VERSION = "tool-choice-v1"
"""The version of the wording of a planned question: its instructions, the reserved option's text and the state layout.

A threshold holds for one prompt version, so change `INSTRUCTIONS`, `NONE_TEXT` or the layout `build_state` writes
only together with this constant.
"""

INSTRUCTIONS = "Which tool should the assistant call next to fulfil the request?"
"""What every planned question asks. Nothing taken from a card is ever added to it."""

NONE_KEY = "none"
"""The key of the reserved option, the one that stands for "no tool fits"."""

NONE_TEXT = "None of these tools can fulfil the request."
"""The text of the reserved option."""

OPTION_OVERHEAD_TOKENS = 4
"""Tokens set aside for the framing of each option, beyond its key and its text: JSON quotes, a colon, a comma."""

_REQUEST_PREFIX = "Request: "
_QUERIES_PREFIX = "Search queries: "
_QUERIES_SEPARATOR = " | "
_ELLIPSIS = " … "
_STATE_SHARE = 4  # the state may take a quarter of what a state and its longest question may take together


class CandidatesDoNotFit(ValueError):
    """The candidates cannot be asked about within the model's declared limits, however they are planned.

    Attributes:
        card_id: The id of the card no question can show, when one card is the cause. `None` when the
            candidates as a whole do not fit: more than two rounds hold, not even a pair of them, or not even
            one finalist from each group in the final question.
    """

    def __init__(self, message: str, /, *, card_id: str | None = None) -> None:
        super().__init__(message)
        self.card_id = card_id


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannedQuestion:
    """A choice question and the way back from each of its options to a card.

    Attributes:
        question: The question to ask; the reserved option, when there is one, comes last.
        card_ids: Option key to the id of the card it stands for. The reserved option has no entry.
        detail: The level of detail the cards are rendered at. A card over its per-text or per-option cap is
            rendered lower, or as its key alone. `NAME` when every card is its key alone.
        estimated_input_tokens: What the question counts against the model's token budget, by the tokenizer
            it was planned with: the state, the instructions, and every option's key, text and overhead.
    """

    question: ChoiceQuestion
    card_ids: Mapping[str, str]
    detail: DetailLevel
    estimated_input_tokens: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Chunk:
    """One share of the candidates in the first of two rounds.

    Attributes:
        cards: The share, in retrieval order.
        question: The question that ranks it, asked without the reserved option. `None` for a share of one
            card, which needs no question.
    """

    cards: tuple[ToolCard, ...]
    question: PlannedQuestion | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundPlan:
    """How to ask about a set of candidates: one question, or two rounds.

    Attributes:
        single: The one question over every candidate, when it fits; then `chunks` is empty.
        chunks: The first round: the candidates dealt round-robin, so each share is a similar mix.
        finalists_per_chunk: How many candidates each chunk keeps for the final question, which the caller
            builds once the first round has answered; planned so that the final fits whichever they are.
            0 when there are no chunks.
    """

    single: PlannedQuestion | None
    chunks: tuple[Chunk, ...]
    finalists_per_chunk: int


def build_state(context: str | None, queries: Sequence[str]) -> str:
    """Write the state every question is asked about: the request, then the search queries.

    The layout is a `Request: <context>` line and a `Search queries: <q1> | <q2>` line. A part that would be
    empty is left out: a missing or blank `context`, or no queries. A query is dropped when it is blank or
    repeats the context, compared with whitespace collapsed and case folded. With nothing left the state is
    an empty string.
    """
    text = "" if context is None else context
    context_key = _normalise(text)
    lines: list[str] = []
    if context_key:
        lines.append(_REQUEST_PREFIX + text)
    kept = [query for query in queries if (key := _normalise(query)) and key != context_key]
    if kept:
        lines.append(_QUERIES_PREFIX + _QUERIES_SEPARATOR.join(kept))
    return "\n".join(lines)


def fit_state(state: str, *, limits: ModelLimits, tokenizer: Tokenizer) -> tuple[str, bool]:
    """Cut a state that is longer than the model's state budget; return it and whether it was cut.

    The budget is the smaller of the values `limits` declares: `max_text_tokens` less the tokens of
    `INSTRUCTIONS`, and a quarter of `max_state_plus_question_tokens`. With neither declared there is no
    budget, and a state within it comes back as it is.

    The queries line, the last line that starts with `Search queries: ` (or the whole state, when it starts
    with that), is never cut. What comes before it is cut in the middle: its beginning and its end are kept
    around `" … "`, the same number of characters on each side, the most that fits.

    Raises:
        ValueError: The state does not fit even with everything before the queries line cut down to
            `" … "`: the queries, which are never cut, take the budget. The message says "queries". With no
            queries line, the budget is too small for `" … "` itself.
    """
    budgets: list[int] = []
    if limits.max_text_tokens is not None:
        budgets.append(limits.max_text_tokens - tokenizer.count(INSTRUCTIONS))
    if limits.max_state_plus_question_tokens is not None:
        budgets.append(limits.max_state_plus_question_tokens // _STATE_SHARE)
    if not budgets:
        return state, False
    budget = min(budgets)
    if tokenizer.count(state) <= budget:
        return state, False

    split = 0 if state.startswith(_QUERIES_PREFIX) else state.rfind("\n" + _QUERIES_PREFIX)
    head, tail = (state, "") if split < 0 else (state[:split], state[split:])

    def cut(kept: int) -> str:
        # `len(head) - kept`, not `-kept`: the latter would keep the whole head for 0.
        return head[:kept] + _ELLIPSIS + head[len(head) - kept :] + tail

    if tokenizer.count(cut(0)) > budget:
        if tail:
            raise ValueError(
                f"the search queries take {tokenizer.count(tail)} tokens, which leaves no room in the state "
                f"budget of {budget}: queries are never cut, so shorten them"
            )
        raise ValueError(f"the state budget of {budget} tokens leaves no room for the request")
    # Binary search for the most characters to keep on each side; `low` always fits. The most it may keep
    # leaves at least one character out, or the cut would only add the ellipsis.
    low, high = 0, (len(head) - 1) // 2
    while low < high:
        middle = (low + high + 1) // 2
        if tokenizer.count(cut(middle)) <= budget:
            low = middle
        else:
            high = middle - 1
    return cut(low), True


def option_keys(cards: Sequence[ToolCard], *, reserved: bool) -> list[str]:
    """Name each card's option: the tool's name, made unique with ` #2`, ` #3` for the repeats.

    Cards are named in order, and the first tool to use a name keeps it. With `reserved`, `NONE_KEY` counts
    as taken from the start, so a tool that is itself called `none` gets `none #2`; the reserved key is not
    in the result. A suffix is never put on a key that is already in use: a tool called `search #2` and a
    second `search` do not collide.
    """
    taken = {NONE_KEY} if reserved else set[str]()
    keys: list[str] = []
    for card in cards:
        key, number = card.name, 1
        while key in taken:
            number += 1
            key = f"{card.name} #{number}"
        taken.add(key)
        keys.append(key)
    return keys


def plan_question(
    state: str,
    cards: Sequence[ToolCard],
    *,
    reserved: bool,
    limits: ModelLimits,
    tokenizer: Tokenizer,
    max_detail: DetailLevel,
    min_detail: DetailLevel = DetailLevel.NAME,
) -> PlannedQuestion | None:
    """Fit one choice question over `cards` within `limits`; `None` when it cannot be done.

    The options follow the order of `cards`, keyed by `option_keys`, and the reserved option comes last when
    `reserved`. The cards are rendered at the most detailed level, from `max_detail` down to `min_detail`, at
    which the question fits its token budget. That budget is the smallest of the declared remainders:
    `limits.max_state_plus_question_tokens` and `limits.max_request_tokens` less the state, the instructions,
    and each option's key and `OPTION_OVERHEAD_TOKENS`, the reserved option's key, text and overhead
    included; and `limits.max_question_tokens` less the same without the state. There is no budget when none
    of them is declared. The state is counted as given: `fit_state` is what makes it fit its own budget.

    A card whose text is over its own cap at that level drops alone to a lower one, below `min_detail` if it
    must. Its cap is the smaller of `limits.max_text_tokens` and what `limits.max_option_tokens` leaves after
    its key and `OPTION_OVERHEAD_TOKENS`. A card whose name is over what `limits.max_option_tokens` leaves is
    shown by its key alone, with an empty text.

    Returns `None` when the options, the reserved one included, are more than `limits.max_options_per_choice`,
    or when no level down to `min_detail` fits.

    Raises:
        CandidatesDoNotFit: A card's name alone is over `limits.max_text_tokens`, or its key and
            `OPTION_OVERHEAD_TOKENS` alone are over `limits.max_option_tokens`; `card_id` names the card. No
            question could show it, so this is raised whatever the token budget, though not for a question
            with too many options.
        ValueError: There are fewer than two options in all, or `min_detail` is above `max_detail`.
    """
    option_count = len(cards) + reserved
    if option_count < 2:
        raise ValueError(f"a choice question needs at least two options, got {option_count}")
    if limits.max_options_per_choice is not None and option_count > limits.max_options_per_choice:
        return None
    keys = option_keys(cards, reserved=reserved)
    key_tokens = [tokenizer.count(key) for key in keys]
    # The cards shown with a text, by index, and each one's cap on it; the others are their key alone.
    shown: list[int] = []
    caps: list[int] = []
    for index, card in enumerate(cards):
        if limits.max_text_tokens is not None and tokenizer.count(card.name) > limits.max_text_tokens:
            raise CandidatesDoNotFit(
                f"card {card.id!r}: its name alone is over max_tokens_per_text={limits.max_text_tokens}",
                card_id=card.id,
            )
        if limits.max_option_tokens is None:
            shown.append(index)
            continue
        key_alone = key_tokens[index] + OPTION_OVERHEAD_TOKENS
        if key_alone > limits.max_option_tokens:
            raise CandidatesDoNotFit(
                f"card {card.id!r}: its option key alone takes {key_alone} tokens, over "
                f"max_option_tokens={limits.max_option_tokens}",
                card_id=card.id,
            )
        cap = limits.max_option_tokens - key_alone
        if limits.max_text_tokens is not None:
            cap = min(cap, limits.max_text_tokens)
        if tokenizer.count(card.render(DetailLevel.NAME)) <= cap:
            shown.append(index)
            caps.append(cap)

    question_fixed = tokenizer.count(INSTRUCTIONS) + sum(key_tokens) + OPTION_OVERHEAD_TOKENS * len(keys)
    if reserved:
        question_fixed += tokenizer.count(NONE_KEY) + tokenizer.count(NONE_TEXT) + OPTION_OVERHEAD_TOKENS
    fixed = tokenizer.count(state) + question_fixed
    remainders = [
        tokens - fixed
        for tokens in (limits.max_state_plus_question_tokens, limits.max_request_tokens)
        if tokens is not None
    ]
    if limits.max_question_tokens is not None:
        remainders.append(limits.max_question_tokens - question_fixed)
    # A negative remainder goes on as it is: then nothing fits.
    rendered = render_within_budget(
        [cards[index] for index in shown],
        max_tokens=min(remainders) if remainders else None,
        tokenizer=tokenizer,
        max_tokens_per_text=limits.max_text_tokens if limits.max_option_tokens is None else caps,
        max_detail=max_detail,
        min_detail=min_detail,
    )
    if rendered is None:
        return None
    texts = [""] * len(cards)
    for index, text in zip(shown, rendered.texts, strict=True):
        texts[index] = text
    options = dict(zip(keys, texts, strict=True))
    if reserved:
        options[NONE_KEY] = NONE_TEXT
    return PlannedQuestion(
        question=ChoiceQuestion(instructions=INSTRUCTIONS, options=options),
        card_ids={key: card.id for key, card in zip(keys, cards, strict=True)},
        detail=rendered.detail if shown else DetailLevel.NAME,
        estimated_input_tokens=fixed + rendered.tokens,
    )


def plan_rounds(
    state: str,
    cards: Sequence[ToolCard],
    *,
    reserved: bool,
    limits: ModelLimits,
    tokenizer: Tokenizer,
    max_detail: DetailLevel,
    finalists_per_chunk: int,
    min_detail: DetailLevel = DetailLevel.NAME,
) -> RoundPlan:
    """Plan how to ask about `cards`: one question when everything fits, otherwise two rounds.

    One question over every card, and the reserved option when `reserved`, is tried first. When it does
    not fit, the cards are dealt round-robin into chunks: the card at position `i` goes to chunk `i % m`,
    where `m` starts at `ceil(K / cap)` for `K` cards and `cap` options per question (`K` when
    `limits.max_options_per_choice` is not declared). Each chunk of two or more cards is planned as a
    question of its own, without the reserved option; when one does not fit, `m` grows by one and the
    cards are dealt again. A chunk of one card needs no question. Every question is planned with
    `min_detail` as its floor (see `plan_question`), so a question that fits only below it is split.

    The final question is built by the caller once the first round has answered: the top
    `finalists_per_chunk` of every chunk, fewer when the option cap leaves no room for that many beside
    the reserved option, and the reserved option. Which cards those are is not known yet, so the final is
    planned here against its worst case, at `min_detail`: the longest that many cards of every chunk, by
    the tokens of their name and of their text at `min_detail`. When they do not fit, every chunk keeps
    one. `RoundPlan.finalists_per_chunk` is the number in effect.

    Raises:
        CandidatesDoNotFit: A card cannot be shown in any question (see `plan_question`), and `card_id`
            names it. Or, with `card_id` `None`: there are more candidates than two rounds can hold, and the
            message gives the most; two candidates do not fit the token budget together; or not even one
            finalist from each chunk fits the final question. All of it is raised before anything is asked.
        ValueError: There are fewer than two options in all, or `min_detail` is above `max_detail`.
    """

    def ask(members: Sequence[ToolCard], *, with_reserved: bool, top: DetailLevel) -> PlannedQuestion | None:
        return plan_question(
            state,
            members,
            reserved=with_reserved,
            limits=limits,
            tokenizer=tokenizer,
            max_detail=top,
            min_detail=min_detail,
        )

    single = ask(cards, with_reserved=reserved, top=max_detail)
    if single is not None:
        return RoundPlan(single=single, chunks=(), finalists_per_chunk=0)

    cap = len(cards) if limits.max_options_per_choice is None else limits.max_options_per_choice

    def finalists(chunk_count: int) -> int:
        kept = min(finalists_per_chunk, (cap - reserved) // chunk_count)
        if kept < 1:
            raise CandidatesDoNotFit(f"at most {cap * (cap - reserved)} candidates fit two rounds for this model")
        return kept

    chunk_count = math.ceil(len(cards) / cap)
    finalists(chunk_count)  # too many candidates for two rounds: say so before planning a chunk
    while True:
        shares = [tuple(cards[start::chunk_count]) for start in range(chunk_count)]
        chunks = tuple(
            Chunk(cards=share, question=ask(share, with_reserved=False, top=max_detail) if len(share) > 1 else None)
            for share in shares
        )
        if all(chunk.question is not None or len(chunk.cards) < 2 for chunk in chunks):
            break
        # The first chunk is the largest. Once it is a pair there is no smaller question to ask.
        if len(chunks[0].cards) <= 2:
            raise CandidatesDoNotFit("a single option does not fit the model's token budget")
        chunk_count += 1

    def length(card: ToolCard) -> int:
        return tokenizer.count(card.name) + tokenizer.count(card.render(min_detail))

    for kept in sorted({finalists(chunk_count), 1}, reverse=True):
        # `sorted` is stable, so cards of equal length are taken in chunk order.
        worst = [card for chunk in chunks for card in sorted(chunk.cards, key=length, reverse=True)[:kept]]
        if ask(worst, with_reserved=reserved, top=min_detail) is not None:
            return RoundPlan(single=None, chunks=chunks, finalists_per_chunk=kept)
    raise CandidatesDoNotFit(
        f"{len(cards)} candidates do not fit two rounds for {limits.source!r}: even one finalist from each of "
        f"the {chunk_count} groups does not fit the final question"
    )


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()
