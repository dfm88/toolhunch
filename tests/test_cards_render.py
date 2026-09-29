import pytest
from inline_snapshot import snapshot

from toolhunch import DetailLevel, RenderedCards, ToolCard, render_within_budget

WEATHER = ToolCard(
    name="get_weather",
    description="Get the forecast for a city.\nUses the Open-Meteo API, updated hourly.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}},
)


class WordCounter:
    def count(self, text: str, /) -> int:
        return len(text.split())


def detail_for(cards: list[ToolCard], *, max_tokens: int) -> DetailLevel | None:
    rendered = render_within_budget(cards, max_tokens=max_tokens, tokenizer=WordCounter())
    return None if rendered is None else rendered.detail


def test_render_levels() -> None:
    assert [WEATHER.render(level) for level in DetailLevel] == snapshot(
        [
            "get_weather",
            "get_weather: Get the forecast for a city. (params: city, days)",
            """\
get_weather: Get the forecast for a city.
Uses the Open-Meteo API, updated hourly.
parameters: {"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}\
""",
        ]
    )


def test_render_degrades_gracefully_without_description_or_parameters() -> None:
    bare = ToolCard(name="ping")
    assert {bare.render(level) for level in DetailLevel} == {"ping"}
    assert ToolCard(name="t", parameters={"properties": {"a": {}}}).render(DetailLevel.BRIEF) == "t (params: a)"


def test_budget_picks_the_most_detailed_level_that_fits() -> None:
    cards = [WEATHER, ToolCard(name="ping")]
    full, brief = (
        sum(len(card.render(level).split()) for card in cards) for level in (DetailLevel.FULL, DetailLevel.BRIEF)
    )

    assert render_within_budget(cards, max_tokens=full, tokenizer=WordCounter()) == RenderedCards(
        detail=DetailLevel.FULL, texts=tuple(card.render(DetailLevel.FULL) for card in cards), tokens=full
    )
    assert detail_for(cards, max_tokens=full - 1) == DetailLevel.BRIEF
    assert detail_for(cards, max_tokens=brief - 1) == DetailLevel.NAME
    assert detail_for(cards, max_tokens=1) is None
    assert render_within_budget([], max_tokens=0, tokenizer=WordCounter()) == RenderedCards(DetailLevel.FULL, (), 0)


def words(count: int) -> str:
    return " ".join(["word"] * count)


def archive_card(description: str) -> ToolCard:
    """BRIEF keeps only the first sentence of `description`, FULL all of it."""
    parameters = {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}
    return ToolCard(name="search_archive", description=description, parameters=parameters)


def test_max_detail_caps_the_level() -> None:
    cards = [WEATHER, ToolCard(name="ping")]
    brief = sum(len(card.render(DetailLevel.BRIEF).split()) for card in cards)

    unlimited = render_within_budget(cards, max_tokens=None, tokenizer=WordCounter())
    assert unlimited is not None
    assert unlimited.detail is DetailLevel.FULL  # no total limit: the first level tried wins

    for max_tokens in (None, brief + 1_000):  # a roomy budget does not lift the cap either
        capped = render_within_budget(
            cards, max_tokens=max_tokens, tokenizer=WordCounter(), max_detail=DetailLevel.BRIEF
        )
        assert capped is not None
        assert capped.detail is DetailLevel.BRIEF
        assert capped.texts == tuple(card.render(DetailLevel.BRIEF) for card in cards)

    # The walk down still starts at `max_detail`.
    tight = render_within_budget(cards, max_tokens=brief - 1, tokenizer=WordCounter(), max_detail=DetailLevel.BRIEF)
    assert tight is not None
    assert tight.detail is DetailLevel.NAME


@pytest.mark.parametrize(
    ("description", "dropped_to"),
    [
        pytest.param("Search the archive. " + words(57), DetailLevel.BRIEF, id="brief-fits"),
        pytest.param(words(60), DetailLevel.NAME, id="only-the-name-fits"),
    ],
)
def test_a_card_over_the_per_text_cap_drops_alone(description: str, dropped_to: DetailLevel) -> None:
    long = archive_card(description)
    cards = [WEATHER, long, ToolCard(name="ping")]

    rendered = render_within_budget(cards, max_tokens=None, tokenizer=WordCounter(), max_tokens_per_text=20)

    assert rendered is not None
    assert rendered.detail is DetailLevel.FULL
    assert rendered.reduced == (1,)
    assert rendered.texts == (WEATHER.render(DetailLevel.FULL), long.render(dropped_to), "ping")
    assert all(len(text.split()) <= 20 for text in rendered.texts)
    assert rendered.tokens == sum(len(text.split()) for text in rendered.texts)


@pytest.mark.parametrize(
    ("description", "long_level", "reduced"),
    [
        pytest.param("Search the archive. " + words(57), DetailLevel.BRIEF, (), id="fits-again-at-brief"),
        pytest.param(words(60), DetailLevel.NAME, (1,), id="still-reduced-at-brief"),
    ],
)
def test_the_total_budget_is_checked_on_the_reduced_texts(
    description: str, long_level: DetailLevel, reduced: tuple[int, ...]
) -> None:
    long = archive_card(description)
    cards = [WEATHER, long, ToolCard(name="ping")]
    at_full = render_within_budget(cards, max_tokens=None, tokenizer=WordCounter(), max_tokens_per_text=20)
    assert at_full is not None
    assert at_full.reduced == (1,)

    # What the reduced texts add up to is enough for FULL.
    fitting = render_within_budget(cards, max_tokens=at_full.tokens, tokenizer=WordCounter(), max_tokens_per_text=20)
    assert fitting == at_full

    # One token less and every card drops to BRIEF, the long one to `long_level` if BRIEF is still over
    # the cap. `reduced` lists only the cards rendered below `detail`, so a card that fits again is not in it.
    lower = render_within_budget(cards, max_tokens=at_full.tokens - 1, tokenizer=WordCounter(), max_tokens_per_text=20)
    texts = (WEATHER.render(DetailLevel.BRIEF), long.render(long_level), "ping")
    assert lower == RenderedCards(
        detail=DetailLevel.BRIEF, texts=texts, tokens=sum(len(text.split()) for text in texts), reduced=reduced
    )


def test_a_name_over_the_per_text_cap_is_an_error() -> None:
    cards = [
        ToolCard(name="ping"),
        ToolCard(name="a name of five words", id="srv-b/long"),
        ToolCard(name="another name of five words", id="srv-a/long"),
    ]

    # The card is named by id, and the first offender in input order is the one reported.
    with pytest.raises(ValueError, match="srv-b/long") as raised:
        render_within_budget(cards, max_tokens=None, tokenizer=WordCounter(), max_tokens_per_text=4)
    assert "srv-a/long" not in str(raised.value)

    # A name of exactly the cap is fine.
    at_the_cap = render_within_budget(cards, max_tokens=None, tokenizer=WordCounter(), max_tokens_per_text=5)
    assert at_the_cap is not None
    assert at_the_cap.reduced == ()
