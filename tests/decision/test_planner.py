from collections.abc import Sequence
from datetime import date

import pytest

from toolhunch import DetailLevel, ToolCard
from toolhunch.decision import CLM_LIMITS, JEV_LIMITS, LOGPROB_LIMITS, CandidatesDoNotFit, ModelLimits
from toolhunch.decision.planner import (
    INSTRUCTIONS,
    NONE_KEY,
    NONE_TEXT,
    OPTION_OVERHEAD_TOKENS,
    PROMPT_VERSION,
    PlannedQuestion,
    RoundPlan,
    build_state,
    fit_state,
    option_keys,
    plan_question,
    plan_rounds,
)


class Words:
    """A token is a word, so the budgets in these tests can be worked out by hand."""

    def count(self, text: str, /) -> int:
        return len(text.split())


WORDS = Words()


class Chars:
    """A token is a character, so a cut can be worked out to the character."""

    def count(self, text: str, /) -> int:
        return len(text)


CHARS = Chars()


def card(name: str, *, id: str = "", description: str = "") -> ToolCard:
    return ToolCard(name=name, description=description, id=id)


def limits(
    *, options: int | None = None, request: int | None = None, state: int | None = None, text: int | None = None
) -> ModelLimits:
    """Declare only the limits a test is about."""
    return ModelLimits(
        max_options_per_choice=options,
        max_request_tokens=request,
        max_state_plus_question_tokens=state,
        max_text_tokens=text,
        source="test",
        checked=date(2026, 9, 28),
    )


STATE = build_state("Book a table", [])  # 4 words
CARDS_50 = [
    card(f"tool{i:02d}", id=f"c{i:02d}", description=f"Handles case {i}. It also does a few other things.")
    for i in range(50)
]


def question(
    cards: Sequence[ToolCard], model: ModelLimits, *, reserved: bool = True, max_detail: DetailLevel = DetailLevel.FULL
) -> PlannedQuestion | None:
    return plan_question(STATE, cards, reserved=reserved, limits=model, tokenizer=WORDS, max_detail=max_detail)


def rounds(cards: Sequence[ToolCard], model: ModelLimits, *, reserved: bool = True) -> RoundPlan:
    return plan_rounds(
        STATE,
        cards,
        reserved=reserved,
        limits=model,
        tokenizer=WORDS,
        max_detail=DetailLevel.FULL,
        finalists_per_chunk=2,
    )


def test_the_prompt_contract_is_pinned() -> None:
    # A threshold holds for one prompt version: change any of this wording only together with the version.
    assert PROMPT_VERSION == "tool-choice-v1"
    assert INSTRUCTIONS == "Which tool should the assistant call next to fulfil the request?"
    assert (NONE_KEY, NONE_TEXT) == ("none", "None of these tools can fulfil the request.")
    assert OPTION_OVERHEAD_TOKENS == 4


def test_build_state() -> None:
    assert (
        build_state("Book a table", ["book a  table", "restaurant reservation"])
        == "Request: Book a table\nSearch queries: restaurant reservation"
    )
    assert build_state(None, ["a", "b"]) == "Search queries: a | b"
    assert build_state("Book a table", []) == "Request: Book a table"
    # An empty part is left out: a blank context, blank queries, queries that only repeat the context.
    assert build_state("  \n", ["  ", "a"]) == "Search queries: a"
    assert build_state("Book a table", ["BOOK A  TABLE", " "]) == "Request: Book a table"
    assert build_state(None, []) == ""


def test_fit_state_cuts_the_context_and_keeps_the_queries() -> None:
    model = ModelLimits(max_text_tokens=60, source="t", checked=date(2026, 9, 28))
    state = build_state("word " * 500 + "tail-marker", ["restaurant booking"])
    text, cut = fit_state(state, limits=model, tokenizer=WORDS)
    assert cut
    assert " … " in text
    assert text.startswith("Request: word")
    assert text.endswith("tail-marker\nSearch queries: restaurant booking")
    assert WORDS.count(text) <= 60 - WORDS.count(INSTRUCTIONS)
    assert fit_state("Request: short", limits=model, tokenizer=WORDS) == ("Request: short", False)
    with pytest.raises(ValueError, match="queries"):
        fit_state(build_state(None, ["q " * 200]), limits=model, tokenizer=WORDS)


@pytest.mark.parametrize(
    ("model", "budget"),
    [
        pytest.param(limits(text=60), 49, id="text-minus-instructions"),
        pytest.param(limits(state=200), 50, id="a-quarter-of-state-plus-question"),
        pytest.param(limits(text=60, state=100), 25, id="the-smaller-is-the-state-limit"),
        pytest.param(limits(text=30, state=1_000), 19, id="the-smaller-is-the-text-limit"),
        pytest.param(CLM_LIMITS, 2048 - 11, id="clm"),
        pytest.param(JEV_LIMITS, 32_000 // 4, id="jev"),
    ],
)
def test_the_state_budget_is_the_smaller_declared_value(model: ModelLimits, budget: int) -> None:
    state = build_state("word " * 20_000, ["restaurant booking"])
    text, cut = fit_state(state, limits=model, tokenizer=WORDS)
    assert cut
    assert text.endswith("\nSearch queries: restaurant booking")
    # The cut keeps the most that fits: one more character on each side would break the budget, and a
    # character adds at most one word to each side.
    assert budget - 1 <= WORDS.count(text) <= budget


def test_the_cut_keeps_equal_shares_and_the_most_that_fits() -> None:
    state = build_state("x" * 200, ["q"])  # "Request: " and 200 x, then a queries line of 18 characters
    model = limits(text=len(INSTRUCTIONS) + 63)  # a state budget of 63 characters
    text, cut = fit_state(state, limits=model, tokenizer=CHARS)
    # 63 - 18 for the queries line - 3 for " … " leaves 42 characters: 21 from each end of the request.
    assert cut
    assert text == "Request: " + "x" * 12 + " … " + "x" * 21 + "\nSearch queries: q"
    assert len(text) == 63
    # Whatever the budget, from room for " … " and the queries line up to the whole state less a character.
    request = "Request: " + "x" * 200
    for budget in range(21, len(state)):
        kept = (budget - 21) // 2
        expected = request[:kept] + " … " + (request[-kept:] if kept else "") + "\nSearch queries: q"
        assert fit_state(state, limits=limits(text=len(INSTRUCTIONS) + budget), tokenizer=CHARS) == (expected, True)


def test_a_state_is_left_alone_within_the_budget_or_without_one() -> None:
    state = build_state("word " * 500, ["restaurant booking"])
    assert fit_state(state, limits=limits(), tokenizer=WORDS) == (state, False)
    # Only the text cap and the state-plus-question cap set a state budget.
    assert fit_state(state, limits=limits(request=10, options=5), tokenizer=WORDS) == (state, False)
    at_the_budget = limits(text=WORDS.count(state) + WORDS.count(INSTRUCTIONS))
    assert fit_state(state, limits=at_the_budget, tokenizer=WORDS) == (state, False)
    one_under = limits(text=WORDS.count(state) + WORDS.count(INSTRUCTIONS) - 1)
    assert fit_state(state, limits=one_under, tokenizer=WORDS)[1]


def test_a_state_without_a_queries_line_is_cut_whole() -> None:
    text, cut = fit_state("Request: " + "word " * 500, limits=limits(text=60), tokenizer=WORDS)
    assert cut
    assert text.startswith("Request: word")
    assert " … " in text
    assert 48 <= WORDS.count(text) <= 49  # the budget is 60 - 11 tokens


def test_the_last_queries_line_is_the_one_kept() -> None:
    # A pasted context may hold a line that looks like ours; the real queries line is the last one.
    state = build_state("intro\nSearch queries: from the pasted text\n" + "word " * 500, ["real query"])
    text, cut = fit_state(state, limits=limits(text=60), tokenizer=WORDS)
    assert cut
    assert text.startswith("Request: intro")
    assert text.endswith("\nSearch queries: real query")


def test_the_queries_are_never_cut() -> None:
    model = limits(text=60)  # a state budget of 49 tokens
    # "Search queries:" and 47 words take exactly the budget: no room is left for " … ", so no cut fits.
    with pytest.raises(ValueError, match="queries"):
        fit_state(build_state("word " * 100, [" ".join(["q"] * 47)]), limits=model, tokenizer=WORDS)
    # One word fewer leaves room for " … " alone: the whole request goes, the queries stay.
    queries = " ".join(["q"] * 46)
    state = build_state("word " * 100, [queries])
    assert fit_state(state, limits=model, tokenizer=WORDS) == (f" … \nSearch queries: {queries}", True)


def test_option_keys_are_unique_names_with_the_reserved_key_first() -> None:
    cards = [card("search", id="a.search"), card("search", id="b.search"), card("none"), card("get")]
    assert option_keys(cards, reserved=True) == ["search", "search #2", "none #2", "get"]
    assert option_keys(cards, reserved=False) == ["search", "search #2", "none", "get"]
    # A suffix never lands on a key that is taken, whichever tool came first.
    assert option_keys([card("search"), card("search"), card("search #2")], reserved=False) == [
        "search",
        "search #2",
        "search #2 #2",
    ]
    assert option_keys([card("search #2"), card("search"), card("search"), card("search")], reserved=False) == [
        "search #2",
        "search",
        "search #3",
        "search #4",
    ]


def test_duplicate_names_map_back_to_their_cards() -> None:
    cards = [
        card("search", id="web.search", description="Search the web."),
        card("none", id="x.none"),
        card("search", id="docs.search", description="Search the docs."),
    ]
    planned = question(cards, JEV_LIMITS)
    assert planned is not None
    assert planned.card_ids == {"search": "web.search", "none #2": "x.none", "search #2": "docs.search"}
    assert list(planned.question.options) == ["search", "none #2", "search #2", NONE_KEY]
    # The text is the card's own, named by the tool and not by the key.
    assert planned.question.options["search #2"] == cards[2].render(DetailLevel.FULL) == "search: Search the docs."
    assert planned.question.instructions == INSTRUCTIONS


def test_one_question_when_everything_fits() -> None:
    plan = plan_rounds(
        STATE,
        CARDS_50,
        reserved=True,
        limits=JEV_LIMITS,
        tokenizer=WORDS,
        max_detail=DetailLevel.FULL,
        finalists_per_chunk=2,
    )
    assert plan.single is not None
    assert plan.chunks == ()
    assert plan.finalists_per_chunk == 0  # no chunks, so no finalists to keep from them
    assert plan.single.detail is DetailLevel.FULL
    options = plan.single.question.options
    assert len(options) == 51
    assert list(options)[-1] == NONE_KEY
    assert options[NONE_KEY] == NONE_TEXT
    assert NONE_KEY not in plan.single.card_ids
    assert set(plan.single.card_ids.values()) == {c.id for c in CARDS_50}


# Five one-word tools, which take 23 words each at FULL, 3 at BRIEF and 1 at NAME.
FIVE = [card(f"t{i}", description="Short one. " + "long " * 20) for i in range(5)]
# What a question of them counts beside their texts: the state (4 words), the instructions (11), each key
# with its overhead, and the reserved option (key, an 8-word text, overhead).
FIXED_WITH_RESERVED = 4 + 11 + 5 * (1 + OPTION_OVERHEAD_TOKENS) + (1 + 8 + OPTION_OVERHEAD_TOKENS)


@pytest.mark.parametrize(
    ("budget", "detail", "estimated"),
    [
        pytest.param(
            FIXED_WITH_RESERVED + 5 * 23, DetailLevel.FULL, FIXED_WITH_RESERVED + 5 * 23, id="full-at-the-cap"
        ),
        pytest.param(FIXED_WITH_RESERVED + 5 * 23 - 1, DetailLevel.BRIEF, FIXED_WITH_RESERVED + 5 * 3, id="brief"),
        pytest.param(
            FIXED_WITH_RESERVED + 5 * 3, DetailLevel.BRIEF, FIXED_WITH_RESERVED + 5 * 3, id="brief-at-the-cap"
        ),
        pytest.param(FIXED_WITH_RESERVED + 5 * 3 - 1, DetailLevel.NAME, FIXED_WITH_RESERVED + 5, id="name"),
        pytest.param(FIXED_WITH_RESERVED + 5, DetailLevel.NAME, FIXED_WITH_RESERVED + 5, id="name-at-the-cap"),
        pytest.param(FIXED_WITH_RESERVED + 5 - 1, None, None, id="nothing-fits"),
    ],
)
def test_a_tight_budget_lowers_the_detail(budget: int, detail: DetailLevel | None, estimated: int | None) -> None:
    planned = question(FIVE, limits(state=budget))
    if detail is None:
        assert planned is None
        return
    assert planned is not None
    assert (planned.detail, planned.estimated_input_tokens) == (detail, estimated)
    assert planned.estimated_input_tokens <= budget


def test_the_estimate_counts_the_reserved_option_only_when_it_is_asked_for() -> None:
    fixed = 4 + 11 + 5 * (1 + OPTION_OVERHEAD_TOKENS)
    planned = question(FIVE, limits(state=fixed + 5 * 23), reserved=False)
    assert planned is not None
    assert (planned.detail, planned.estimated_input_tokens) == (DetailLevel.FULL, fixed + 5 * 23)
    assert NONE_KEY not in planned.question.options
    lower = question(FIVE, limits(state=fixed + 5 * 23 - 1), reserved=False)
    assert lower is not None
    assert lower.detail is DetailLevel.BRIEF


def test_the_question_budget_is_the_smaller_of_the_state_and_request_limits() -> None:
    # 100 tokens are enough for BRIEF (68) and not for FULL (168), whichever limit declares them.
    for model in (limits(state=100, request=1_000), limits(state=1_000, request=100)):
        planned = question(FIVE, model)
        assert planned is not None
        assert planned.detail is DetailLevel.BRIEF
    unlimited = question(FIVE, limits(), max_detail=DetailLevel.BRIEF)  # no total limit: max_detail is the cap
    assert unlimited is not None
    assert unlimited.detail is DetailLevel.BRIEF


def test_clm_limits_keep_every_text_under_the_cap() -> None:
    long = card("archive", id="srv/archive", description="Search the archive. " + "word " * 3_000)
    cards = [card(f"t{i:02d}", id=f"c{i:02d}", description="Does one thing. Then a second.") for i in range(19)]
    cards.insert(7, long)
    assert WORDS.count(long.render(DetailLevel.FULL)) > 2048

    plan = plan_rounds(
        STATE,
        cards,
        reserved=True,
        limits=CLM_LIMITS,
        tokenizer=WORDS,
        max_detail=DetailLevel.FULL,
        finalists_per_chunk=2,
    )

    assert plan.single is not None
    options = plan.single.question.options
    assert plan.single.detail is DetailLevel.FULL
    assert all(WORDS.count(text) <= 2048 for text in options.values())
    texts = {plan.single.card_ids[key]: text for key, text in options.items() if key != NONE_KEY}
    assert texts.pop(long.id) == long.render(DetailLevel.BRIEF)  # the long card alone dropped a level
    assert texts == {c.id: c.render(DetailLevel.FULL) for c in cards if c is not long}


def test_a_question_that_does_not_fit_is_none() -> None:
    four = CARDS_50[:4]
    assert question(four, limits(options=4)) is None  # four tools and the reserved option make five
    assert question(four, limits(options=5)) is not None
    assert question(four, limits(options=4), reserved=False) is not None
    with pytest.raises(ValueError, match="two options"):
        question(four[:1], limits(), reserved=False)
    for model in (limits(), limits(state=1)):  # with a budget nothing fits in, too: no round can help
        with pytest.raises(ValueError, match="two options"):
            rounds([], model)


def test_a_name_over_the_text_cap_is_an_error_and_not_a_misfit() -> None:
    long = card("a name of five words", id="srv/long")
    cards = [long, card("ping"), card("get")]
    # Whatever the total budget says: this card can never be shown to the model as it is.
    for state_budget in (None, 1):
        with pytest.raises(CandidatesDoNotFit, match="srv/long") as raised:
            question(cards, limits(text=4, state=state_budget))
        assert raised.value.card_id == "srv/long"
    with pytest.raises(CandidatesDoNotFit, match="srv/long") as raised:
        rounds(cards, limits(options=2, text=4), reserved=False)  # it lands in a chunk of two
    assert raised.value.card_id == "srv/long"


def test_two_rounds_deal_candidates_round_robin() -> None:
    plan = plan_rounds(
        STATE,
        CARDS_50,
        reserved=True,
        limits=LOGPROB_LIMITS,
        tokenizer=WORDS,
        max_detail=DetailLevel.FULL,
        finalists_per_chunk=2,
    )
    assert plan.single is None
    assert [len(chunk.cards) for chunk in plan.chunks] == [17, 17, 16]
    assert [c.id for c in plan.chunks[0].cards][:3] == ["c00", "c03", "c06"]
    assert plan.finalists_per_chunk == 2
    assert all(chunk.question and NONE_KEY not in chunk.question.question.options for chunk in plan.chunks)
    twenty = plan_rounds(
        STATE,
        CARDS_50[:20],
        reserved=True,
        limits=LOGPROB_LIMITS,
        tokenizer=WORDS,
        max_detail=DetailLevel.FULL,
        finalists_per_chunk=2,
    )
    assert ([len(c.cards) for c in twenty.chunks], twenty.finalists_per_chunk) == ([20], 2)
    # The finalists and the reserved option must fit one question: 3 chunks x 6 and "none" make 19 of 20.
    greedy = plan_rounds(
        STATE,
        CARDS_50,
        reserved=True,
        limits=LOGPROB_LIMITS,
        tokenizer=WORDS,
        max_detail=DetailLevel.FULL,
        finalists_per_chunk=10,
    )
    assert greedy.finalists_per_chunk == 6


def test_one_card_chunk_needs_no_question() -> None:
    plan = rounds(CARDS_50[:3], limits(options=2), reserved=False)
    assert plan.single is None
    assert [[c.id for c in chunk.cards] for chunk in plan.chunks] == [["c00", "c02"], ["c01"]]
    assert plan.chunks[0].question is not None
    assert plan.chunks[1].question is None
    assert plan.finalists_per_chunk == 1  # the final question holds two options: one finalist per chunk


def test_more_candidates_than_two_rounds_hold() -> None:
    model = limits(options=3)
    # The final question keeps a seat for the reserved option: 3 chunks of 3 with it, 3 x 3 without.
    for reserved, most in ((True, 6), (False, 9)):
        assert rounds(CARDS_50[:most], model, reserved=reserved).single is None
        with pytest.raises(ValueError, match=f"at most {most} candidates fit two rounds"):
            rounds(CARDS_50[: most + 1], model, reserved=reserved)
    with pytest.raises(ValueError, match="at most 6"):
        rounds(CARDS_50[:10], model)
    # It is said before any chunk is planned, so before a name over the per-text cap is met in one.
    with pytest.raises(ValueError, match="at most 6"):
        rounds([card("a name of five words"), *CARDS_50[:6]], limits(options=3, text=4))


@pytest.mark.parametrize(
    ("budget", "sizes", "finalists"),
    [
        # Fourteen names take 4 + 11 + 14 x (1 + 4) + 14 = 99 tokens. Twenty options are allowed, yet the chunks
        # hold fourteen: more than the ceil(40 / 20) = 2 the cap alone gives. Two finalists from each of the
        # three and the reserved option take 4 + 11 + 6 x 6 + 13 = 64.
        pytest.param(99, [14, 13, 13], 2, id="fourteen-names"),
        # Five names take 45 tokens, six 51, seven 57: the chunks hold five or six, and then not even one finalist
        # from each fits the final question.
        pytest.param(45, [5] * 8, None, id="five-names"),
        pytest.param(51, [6, 6, 6, 6, 6, 5, 5], None, id="six-names"),  # grown by one at a time: 7 chunks, not 8
    ],
)
def test_the_token_budget_adds_chunks(budget: int, sizes: list[int], finalists: int | None) -> None:
    model = limits(options=20, state=budget)
    if finalists is None:
        # These plans used to come back, and their final question (16 and 14 finalists) failed at runtime.
        with pytest.raises(CandidatesDoNotFit, match=f"each of the {len(sizes)} groups"):
            rounds(CARDS_50[:40], model)
        return
    plan = rounds(CARDS_50[:40], model)
    assert plan.single is None
    assert [len(chunk.cards) for chunk in plan.chunks] == sizes
    assert plan.finalists_per_chunk == finalists
    for chunk in plan.chunks:
        assert chunk.question is not None
        assert chunk.question.detail is DetailLevel.NAME
        assert chunk.question.estimated_input_tokens <= budget
    assert [c.id for c in plan.chunks[1].cards][:2] == ["c01", f"c{1 + len(sizes):02d}"]  # still round-robin


def test_without_an_option_cap_the_candidates_are_the_cap() -> None:
    # Ten candidates and no cap: the chunks are only as many as the token budget needs, and the final question
    # may hold all ten candidates less the reserved option, spread over the chunks. Two-word names take 8
    # tokens an option: ten in one question take 4 + 11 + 80 = 95, five take 55, and eight finalists with the
    # reserved option 4 + 11 + 64 + 13 = 92, the budget.
    # Two-word names: with one-word names at 45 the final question (52 tokens, then 76) failed at runtime.
    ten = [card(f"tool {i:02d}", id=f"c{i:02d}") for i in range(10)]
    for per_chunk, kept in ((2, 2), (10, 4)):
        plan = plan_rounds(
            STATE,
            ten,
            reserved=True,
            limits=limits(state=92),
            tokenizer=WORDS,
            max_detail=DetailLevel.FULL,
            finalists_per_chunk=per_chunk,
        )
        assert [len(chunk.cards) for chunk in plan.chunks] == [5, 5]
        assert plan.finalists_per_chunk == kept  # min(per_chunk, (10 - 1) // 2)


def test_growing_the_chunks_can_leave_no_room_for_finalists() -> None:
    # Two chunks of three would do, with one finalist each beside the reserved option.
    assert len(rounds(CARDS_50[:6], limits(options=3)).chunks) == 2
    # A budget for a pair of names (4 + 11 + 2 x 5 + 2 = 27) and no more needs three chunks, and then the
    # final question holds no finalist per chunk.
    with pytest.raises(ValueError, match="at most 6 candidates fit two rounds"):
        rounds(CARDS_50[:6], limits(options=3, state=27))


def test_two_options_that_do_not_fit_are_an_error() -> None:
    with pytest.raises(ValueError, match="a single option does not fit the model's token budget"):
        rounds(CARDS_50[:6], limits(options=3, state=26))


LAYA_LIKE = ModelLimits(max_question_tokens=60, max_option_tokens=12, source="test", checked=date(2026, 10, 5))


def test_the_option_cap_counts_the_key_and_falls_back_to_the_key_alone() -> None:
    cards = [
        card("short", description="One thing."),
        card("verbose", description="w " * 30),
        card("a b c d e f g", id="long-name"),  # key 7 + overhead 4 + name 7 > 12
    ]
    # The question takes 11 + (1 + 4) x 2 + (7 + 4) + 13 for the reserved option = 45 tokens beside the texts:
    # 15 are left for them, and FULL takes 3 + 1.
    planned = question(cards, LAYA_LIKE)
    assert planned is not None
    assert planned.question.options["a b c d e f g"] == ""  # the key alone, with an empty text
    assert planned.question.options["verbose"] == "verbose"  # its own text is over the cap: name
    assert planned.question.options["short"] == "short: One thing."
    assert planned.detail is DetailLevel.FULL  # the others keep the question's level
    # The per-question budget leaves the state out: a 200-word state changes nothing.
    in_a_long_state = plan_question(
        build_state("w " * 200, []),
        cards,
        reserved=True,
        limits=LAYA_LIKE,
        tokenizer=WORDS,
        max_detail=DetailLevel.FULL,
    )
    assert in_a_long_state is not None
    assert in_a_long_state.question == planned.question
    with pytest.raises(CandidatesDoNotFit, match="'too-long': its option key alone takes 13 tokens") as raised:
        question([*cards, card("k " * 9, id="too-long")], LAYA_LIKE)  # key 9 + 4 > 12
    assert raised.value.card_id == "too-long"
    assert isinstance(raised.value, ValueError)


def test_the_floor_splits_into_groups_instead_of_dropping_to_names() -> None:
    # One question over CARDS_50 takes 4 + 11 + 50 x (1 + 4) + 13 = 278 tokens beside the texts: 50 names make
    # it 328, 50 BRIEF texts (4 words each) 478.
    model = limits(state=328)
    single = rounds(CARDS_50, model).single
    assert single is not None
    assert single.detail is DetailLevel.NAME
    plan = plan_rounds(
        STATE,
        CARDS_50,
        reserved=True,
        limits=model,
        tokenizer=WORDS,
        max_detail=DetailLevel.FULL,
        finalists_per_chunk=2,
        min_detail=DetailLevel.BRIEF,
    )
    assert plan.single is None
    # One chunk of 50 at BRIEF takes 465; two of 25 take 240 each.
    assert [len(chunk.cards) for chunk in plan.chunks] == [25, 25]
    assert all(c.question.detail >= DetailLevel.BRIEF for c in plan.chunks if c.question)
    assert plan.finalists_per_chunk == 2


def test_the_final_is_planned_against_its_worst_case() -> None:
    # No option cap. A one-word name takes 1 + 4 + 1 = 6 tokens an option, a three-word name 10. The eight cards
    # do not fit one chunk (4 + 11 + 56 = 71); two groups, [s0, s2, l0, s4] and [s1, s3, l1, s5], take 43 each.
    cards = [card(name) for name in ("s0", "s1", "s2", "s3", "l0 b c", "l1 b c", "s4", "s5")]
    # The worst final holds each group's longest: two of each take 4 + 11 + 2 x 16 + 13 = 60, one of each 48.
    # The first two of each would take 52, the first one 40: those budgets would wrongly pass.
    assert rounds(cards, limits(state=52)).finalists_per_chunk == 1
    with pytest.raises(CandidatesDoNotFit, match="even one finalist from each of the 2 groups") as raised:
        rounds(cards, limits(state=47))
    assert raised.value.card_id is None
    # Under the option cap the cost is the option's as planned, without the state. "x0 a b c d" is its key alone,
    # 5 + 4 = 9 tokens; "y0 a b c" is its key and name, 4 + 4 + 4 = 12. Groups [x0, y0] and [x1, y1] take
    # 11 + 9 + 12 = 32 each, all four 53. A final with the y cards takes 11 + 2 x 12 + 13 = 48, with the x
    # cards 42: ranked by their names' length instead, the x cards would pass at 47.
    capped = [card("x0 a b c d"), card("x1 a b c d"), card("y0 a b c"), card("y1 a b c")]
    assert rounds(capped, LAYA_LIKE.model_copy(update={"max_question_tokens": 48})).finalists_per_chunk == 1
    with pytest.raises(CandidatesDoNotFit, match="even one finalist from each of the 2 groups"):
        rounds(capped, LAYA_LIKE.model_copy(update={"max_question_tokens": 47}))
