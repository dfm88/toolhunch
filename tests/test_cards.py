from typing import Any

import pytest
from inline_snapshot import snapshot

from toolhunch import ToolCard, ToolCatalog, default_search_text


def test_search_text_uses_name_description_and_property_names() -> None:
    card = ToolCard(
        name="get_weather",
        description="Forecast for a city.",
        parameters={"type": "object", "properties": {"city": {"description": "not indexed"}, "days": {}}},
    )
    assert default_search_text(card) == "get_weather\nForecast for a city.\ncity days"


@pytest.mark.parametrize("parameters", [None, {"type": "string"}, {"properties": ["x"]}, {"properties": {}}])
def test_search_text_tolerates_odd_schemas(parameters: dict[str, Any] | None) -> None:
    assert default_search_text(ToolCard(name="t", parameters=parameters)) == "t"


def test_catalog_is_canonical_and_fingerprint_tracks_content() -> None:
    a, b = ToolCard(name="a"), ToolCard(name="b", description="x")
    catalog = ToolCatalog([b, a])

    assert [card.id for card in catalog] == ["a", "b"]
    assert catalog["b"] is b
    assert "a" in catalog
    assert catalog.fingerprint == ToolCatalog([a, b]).fingerprint
    assert catalog.fingerprint != ToolCatalog([a, ToolCard(name="b", description="y")]).fingerprint
    assert catalog.fingerprint != ToolCatalog([a, ToolCard(name="b", description="x", id="b2")]).fingerprint


def test_fingerprint_is_stable_across_releases() -> None:
    # Keys index caches and benchmark provenance: a change here invalidates recorded results.
    card = ToolCard(name="get_weather", description="Forecast.", parameters={"type": "object"}, tags=("w",))
    assert ToolCatalog([card]).fingerprint == snapshot(
        "sha256:ec1faa29dad605ba706fa22f935e2eba987043d2e1f8620cd9381f8a4b3fecd6"
    )


def test_ids_must_be_unique_but_names_may_repeat() -> None:
    with pytest.raises(ValueError, match="'search'"):
        ToolCatalog([ToolCard(name="search"), ToolCard(name="search")])
    catalog = ToolCatalog([ToolCard(name="search", id="x1"), ToolCard(name="search", id="x2")])
    assert len(catalog) == 2


def test_invalid_cards_are_rejected() -> None:
    with pytest.raises(ValueError, match="name"):
        ToolCard(name="")
    with pytest.raises(ValueError, match="'bad'"):
        ToolCatalog([ToolCard(name="bad", parameters={"default": object()})])
