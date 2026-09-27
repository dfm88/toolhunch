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
