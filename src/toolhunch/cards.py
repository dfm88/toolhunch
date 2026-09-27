"""Framework-free tool descriptions: cards, catalogs and the text retrievers index.

Card text is untrusted — third-party servers write it. Nothing here interprets it; code that puts
it in front of a model must fence it as data.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from toolhunch.tokens import Tokenizer

__all__ = [
    "DetailLevel",
    "RenderedCards",
    "SearchText",
    "ToolCard",
    "ToolCatalog",
    "default_search_text",
    "render_within_budget",
]

_FIRST_SENTENCE = re.compile(r".*?[.!?](?=\s|$)")


class DetailLevel(IntEnum):
    """How much of a card to render, from least to most."""

    NAME = 0
    """The name only."""
    BRIEF = 1
    """Name, first sentence of the description, input property names."""
    FULL = 2
    """Name, full description, input JSON Schema."""


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

    def render(self, detail: DetailLevel) -> str:
        """Render the card as plain text at the given level of detail.

        The text is data written by whoever published the tool; fence it before showing it to a model.
        """
        if detail is DetailLevel.NAME:
            return self.name
        if detail is DetailLevel.BRIEF:
            text = self.name
            if summary := _first_sentence(self.description):
                text += f": {summary}"
            if names := _property_names(self):
                text += f" (params: {', '.join(names)})"
            return text
        description = self.description.strip()
        text = f"{self.name}: {description}" if description else self.name
        if self.parameters is not None:
            text += "\nparameters: " + json.dumps(dict(self.parameters), separators=(",", ":"), ensure_ascii=False)
        return text


type SearchText = Callable[[ToolCard], str]
"""Turns a card into the text a retriever indexes."""


def default_search_text(card: ToolCard) -> str:
    """Name, description and top-level input property names, one per line.

    Property descriptions, enum values and output schemas are left out: ratel measured this mix as
    the best for BM25 (its ADR-0023); the toolhunch benchmark re-checks it.
    """
    property_names = " ".join(_property_names(card))
    return "\n".join(part for part in (card.name, card.description, property_names) if part)


@dataclass(frozen=True, slots=True)
class RenderedCards:
    """Cards rendered at one level of detail, with their total token count."""

    detail: DetailLevel
    texts: tuple[str, ...]
    tokens: int


def render_within_budget(cards: Sequence[ToolCard], *, max_tokens: int, tokenizer: Tokenizer) -> RenderedCards | None:
    """Render every card at the most detailed level whose total fits in `max_tokens`.

    All cards share one level. `tokens` is the sum of per-text counts; separators and any framing
    are the caller's to budget. Returns `None` when even names alone do not fit.
    """
    for detail in sorted(DetailLevel, reverse=True):
        texts = tuple(card.render(detail) for card in cards)
        tokens = sum(tokenizer.count(text) for text in texts)
        if tokens <= max_tokens:
            return RenderedCards(detail=detail, texts=texts, tokens=tokens)
    return None


def _property_names(card: ToolCard) -> list[str]:
    properties = None if card.parameters is None else card.parameters.get("properties")
    if not isinstance(properties, Mapping):
        return []
    return [str(key) for key in cast("Mapping[object, object]", properties)]


def _first_sentence(description: str) -> str:
    collapsed = " ".join(description.split())
    match = _FIRST_SENTENCE.match(collapsed)
    return match.group(0) if match else collapsed


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
