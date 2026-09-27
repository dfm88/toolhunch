"""Framework-free tool descriptions: cards, catalogs and the text retrievers index.

Card text is untrusted — third-party servers write it. Nothing here interprets it; code that puts
it in front of a model must fence it as data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

__all__ = ["SearchText", "ToolCard", "ToolCatalog", "default_search_text"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCard:
    """One tool (or any selectable item), described without reference to a framework.

    Attributes:
        name: The name the model calls the tool by.
        description: Free-text description.
        parameters: JSON Schema of the input, if any. Treat it as immutable once the card exists.
        tags: Free labels, for example a category.
        source: Where the tool comes from, for example an MCP server name.
        id: Unique key within a catalog; defaults to `name`. Set it when names can collide, as they
            do across MCP servers or in pooled benchmark corpora.
    """

    name: str
    description: str = ""
    parameters: Mapping[str, Any] | None = None
    tags: tuple[str, ...] = ()
    source: str | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolCard.name must not be empty")
        if not self.id:
            object.__setattr__(self, "id", self.name)


type SearchText = Callable[[ToolCard], str]
"""Turns a card into the text a retriever indexes."""


def default_search_text(card: ToolCard) -> str:
    """Name, description and top-level input property names, one per line.

    Property descriptions, enum values and output schemas are left out: ratel measured this mix as
    the best for BM25 (its ADR-0023); the toolhunch benchmark re-checks it.
    """
    property_names = ""
    properties = None if card.parameters is None else card.parameters.get("properties")
    if isinstance(properties, Mapping):
        property_names = " ".join(str(key) for key in cast("Mapping[object, object]", properties))
    return "\n".join(part for part in (card.name, card.description, property_names) if part)


class ToolCatalog:
    """An immutable set of cards with a stable fingerprint.

    Cards are kept sorted by `id`, so equal catalogs iterate, and break ranking ties, the same way
    whatever order they were built from. The fingerprint is the SHA-256 of the canonical JSON of
    every card field; it keys index caches and benchmark provenance.
    """

    __slots__ = ("_by_id", "cards", "fingerprint")

    cards: tuple[ToolCard, ...]
    fingerprint: str

    def __init__(self, cards: Iterable[ToolCard]) -> None:
        """Build a catalog; raises `ValueError` on a duplicate id or non-JSON parameters."""
        self.cards = tuple(sorted(cards, key=lambda card: card.id))
        self._by_id: dict[str, ToolCard] = {}
        for card in self.cards:
            if card.id in self._by_id:
                raise ValueError(f"duplicate card id {card.id!r}")
            self._by_id[card.id] = card
        self.fingerprint = _fingerprint(self.cards)

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self) -> Iterator[ToolCard]:
        return iter(self.cards)

    def __contains__(self, card_id: object) -> bool:
        return card_id in self._by_id

    def __getitem__(self, card_id: str) -> ToolCard:
        return self._by_id[card_id]

    def __repr__(self) -> str:
        return f"ToolCatalog({len(self.cards)} cards, {self.fingerprint[:19]})"


def _fingerprint(cards: tuple[ToolCard, ...]) -> str:
    digest = hashlib.sha256()
    for card in cards:
        record = {
            "id": card.id,
            "name": card.name,
            "description": card.description,
            "parameters": None if card.parameters is None else dict(card.parameters),
            "tags": list(card.tags),
            "source": card.source,
        }
        try:
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"card {card.id!r}: parameters are not JSON-serialisable ({error})") from error
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"
