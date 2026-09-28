"""BM25 over tool cards, with a pluggable text analyzer."""

from __future__ import annotations

import functools
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from toolhunch.cards import default_search_text
from toolhunch.retrieval.base import (
    FUSION_DEPTH,
    IndexCache,
    Retrieval,
    ScoredCard,
    check_k,
    clean_queries,
    fuse_query_rankings,
    top_k,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from toolhunch.cards import SearchText, ToolCard, ToolCatalog

__all__ = ["ENGLISH_STOP_WORDS", "Analyzer", "BM25Retriever", "TextAnalyzer", "s_stemmer"]

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


@functools.cache
def _word_pattern() -> re.Pattern[str]:
    r"""Runs of Unicode letters and digits with their combining marks; `_` and punctuation separate words.

    `re`'s `\w` leaves combining marks (categories Mn, Mc, Me) out, which would cut Devanagari, Burmese
    or Tamil words at every vowel sign, so the marks of the running Unicode database (about 2,500) are
    added. Built once, on first use (about 50 ms).
    """
    ranges: list[list[int]] = []
    for code in range(sys.maxunicode + 1):
        if unicodedata.category(chr(code)).startswith("M"):
            if ranges and ranges[-1][1] == code - 1:
                ranges[-1][1] = code
            else:
                ranges.append([code, code])
    marks = "".join(f"\\U{start:08x}-\\U{end:08x}" for start, end in ranges)
    return re.compile(rf"[^\W_](?:[^\W_]|[{marks}])*")


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
    """Default analyzer: identifier-aware splitting, Unicode normalisation, case folding, stop words, stemming.

    Tool names carry much of the signal (`get_diary_day`, `listUserRepos`), so identifiers are split
    on `snake_case` and `camelCase` before matching. Letters and digits stay together (`base64`). Text
    is NFKC-normalised and case-folded, so a decomposed and a precomposed "caffè", full-width letters
    and "Straße"/"STRASSE" match; words keep their combining marks (Devanagari, Burmese, Tamil).

    Language: the defaults are **English** — Lucene's English stop words and, if enabled, an English
    plural stemmer. Splitting is Unicode-aware, so text in other languages still matches word for
    word, without stop-word removal or stemming. For a catalog written in another language, pass
    that language's stop words and stemmer, for example Snowball's (`pip install snowballstemmer`,
    36 languages, the algorithms behind Lucene's language analyzers)::

        TextAnalyzer(stop_words=italian_stop_words, stemmer=snowballstemmer.stemmer("italian").stemWord)

    or implement [`Analyzer`][toolhunch.retrieval.Analyzer] from scratch. No analyzer bridges
    languages: English cards and Italian queries share few words, which is what multilingual
    embeddings are for.

    Attributes:
        stop_words: Lowercase terms to drop.
        stemmer: Applied to each remaining term, for example [`s_stemmer`][toolhunch.retrieval.s_stemmer].
    """

    stop_words: frozenset[str] = ENGLISH_STOP_WORDS
    stemmer: Callable[[str], str] | None = None

    def analyze(self, text: str, /) -> list[str]:
        """Normalise (NFKC), split, case-fold, drop stop words, then stem."""
        words = _word_pattern().findall(_CAMEL_BOUNDARY.sub(" ", unicodedata.normalize("NFKC", text)))
        terms = [folded for word in words if (folded := word.casefold()) not in self.stop_words]
        if self.stemmer is None:
            return terms
        return [self.stemmer(term) for term in terms]


@dataclass(frozen=True, slots=True)
class _Index:
    cards: tuple[ToolCard, ...]
    postings: dict[str, list[tuple[int, int]]]  # term → [(card position, term frequency)]
    idf: dict[str, float]
    norms: list[float]  # k1 · (1 - b + b · |d| / avgdl), per card


class BM25Retriever:
    """Okapi BM25, Lucene variant, over each card's search text.

    `score(q, d) = Σ idf(t) · tf / (tf + k1 · (1 - b + b · |d| / avgdl))` over the unique terms of
    the query, with `idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))` — the formula of `bm25s`
    `method="lucene"`, which the benchmark checks this implementation against. Only cards with a
    positive score are returned. The index is built once per catalog fingerprint.
    """

    def __init__(
        self,
        *,
        analyzer: Analyzer | None = None,
        search_text: SearchText = default_search_text,
        k1: float = 1.2,
        b: float = 0.75,
        cache_size: int = 8,
    ) -> None:
        """Configure the analyzer, the indexed text and the BM25 parameters (Lucene defaults)."""
        self._analyzer = analyzer if analyzer is not None else TextAnalyzer()
        self._search_text = search_text
        self._k1 = k1
        self._b = b
        self._indexes: IndexCache[_Index] = IndexCache(max_entries=cache_size)

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """Rank `catalog` for each query and fuse; see [`Retriever.retrieve`][toolhunch.Retriever.retrieve]."""
        check_k(k)
        cleaned = clean_queries(queries)
        if not cleaned or not len(catalog):
            return Retrieval()
        index = self._index(catalog)
        depth = max(k, FUSION_DEPTH)
        rankings = [ranking for query in cleaned if (ranking := self._rank(index, query, depth=depth))]
        if not rankings:
            return Retrieval()
        return Retrieval(matches=tuple(fuse_query_rankings(rankings, k=k)))

    def _rank(self, index: _Index, query: str, *, depth: int) -> list[ScoredCard]:
        scores: dict[int, float] = {}
        for term in set(self._analyzer.analyze(query)):
            idf = index.idf.get(term)
            if idf is None:
                continue
            for position, tf in index.postings[term]:
                scores[position] = scores.get(position, 0.0) + idf * tf / (tf + index.norms[position])
        return top_k(
            (ScoredCard(index.cards[position], score) for position, score in scores.items() if score > 0), depth
        )

    def _index(self, catalog: ToolCatalog) -> _Index:
        if (cached := self._indexes.get(catalog.fingerprint)) is not None:
            return cached
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths: list[int] = []
        for position, card in enumerate(catalog):
            terms = self._analyzer.analyze(self._search_text(card))
            lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                postings.setdefault(term, []).append((position, tf))
        n = len(lengths)
        avg_length = (sum(lengths) / n) or 1.0
        index = _Index(
            cards=catalog.cards,
            postings=postings,
            idf={term: math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5)) for term, docs in postings.items()},
            norms=[self._k1 * (1 - self._b + self._b * length / avg_length) for length in lengths],
        )
        self._indexes.put(catalog.fingerprint, index)
        return index
