"""BM25 over tool cards, with a pluggable text analyzer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ENGLISH_STOP_WORDS", "Analyzer", "TextAnalyzer", "s_stemmer"]

# Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET (lucene/analysis/common, checked 2026-09-27).
ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",
        "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there",
        "these", "they", "this", "to", "was", "will", "with",
    }
)  # fmt: skip

# camelCase and PascalCase boundaries: "getDiary" → "get Diary", "HTTPServer" → "HTTP Server".
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Runs of Unicode letters and digits; "_" and punctuation separate words.
_WORD = re.compile(r"[^\W_]+")


class Analyzer(Protocol):
    """Turns text into the terms BM25 indexes and matches."""

    def analyze(self, text: str, /) -> list[str]:
        """Return the terms of `text`, in order, repeats included."""
        ...


def s_stemmer(token: str) -> str:
    """Harman's S-stemmer (1991): strip English plural endings, nothing else.

    Only the first matching rule applies: `-ies` → `-y` (not after `e`/`a`), `-es` → `-e` (not after
    `a`/`e`/`o`), `-s` → `` (not after `u`/`s`). Tokens of three characters or fewer are unchanged.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if token.endswith("s") and not token.endswith(("us", "ss")):
        return token[:-1]
    return token


@dataclass(frozen=True, slots=True, kw_only=True)
class TextAnalyzer:
    """Default analyzer: identifier-aware splitting, lowercase, stop words, optional stemming.

    Tool names carry much of the signal (`get_diary_day`, `listUserRepos`), so identifiers are split
    on `snake_case` and `camelCase` before matching. Letters and digits stay together (`base64`).

    Attributes:
        stop_words: Lowercase terms to drop.
        stemmer: Applied to each remaining term, for example [`s_stemmer`][toolhunch.retrieval.s_stemmer].
    """

    stop_words: frozenset[str] = ENGLISH_STOP_WORDS
    stemmer: Callable[[str], str] | None = None

    def analyze(self, text: str, /) -> list[str]:
        """Split, lowercase, drop stop words, then stem."""
        words = _WORD.findall(_CAMEL_BOUNDARY.sub(" ", text))
        terms = [lowered for word in words if (lowered := word.lower()) not in self.stop_words]
        if self.stemmer is None:
            return terms
        return [self.stemmer(term) for term in terms]
