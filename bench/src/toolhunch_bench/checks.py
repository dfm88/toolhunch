"""Checks run next to a benchmark: our BM25 against bm25s, and the token heuristic against tiktoken."""

from __future__ import annotations

import heapq
import random
from typing import TYPE_CHECKING, Any

import bm25s  # pyright: ignore[reportMissingTypeStubs]
import tiktoken

from toolhunch import BM25Retriever, HeuristicTokenizer, default_search_text
from toolhunch.retrieval import TextAnalyzer
from toolhunch_bench.metrics import percentile
from toolhunch_bench.retrieval import query_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask
    from toolhunch_bench.retrieval import QueryMode

__all__ = ["check_bm25", "check_tokens"]


async def check_bm25(
    data: ToolRetData, tasks: Sequence[ToolRetTask], *, modes: Sequence[QueryMode] = ("plain", "instructed")
) -> dict[str, Any]:
    """Score every card for every task query with our BM25 and with bm25s (Lucene, same tokens, same k1/b).

    Returns the largest relative score difference over cards either side scores, and the mean overlap
    of the two top-10 lists. Query terms are deduplicated, since bm25s counts repeats.
    """
    analyzer = TextAnalyzer()
    ours = BM25Retriever(analyzer=analyzer)
    library: Any = bm25s
    reference = library.BM25(k1=1.2, b=0.75, method="lucene")
    reference.index([analyzer.analyze(default_search_text(card)) for card in data.catalog], show_progress=False)
    cards = data.catalog.cards
    largest = 0.0
    overlaps: list[float] = []
    for mode in modes:
        for task in tasks:
            terms = list(dict.fromkeys(analyzer.analyze(query_text(task, mode=mode, arm_kind="lexical"))))
            if not terms:
                continue
            expected = [float(score) for score in reference.get_scores(terms)]
            result = await ours.retrieve([" ".join(terms)], data.catalog, k=len(cards))
            scores = {match.card.id: match.score for match in result.matches}
            for card, want in zip(cards, expected, strict=True):
                got = scores.get(card.id, 0.0)
                if want or got:
                    largest = max(largest, abs(got - want) / max(want, 1e-12))
            top = heapq.nlargest(10, (i for i, score in enumerate(expected) if score > 0), key=expected.__getitem__)
            reference_top = {cards[i].id for i in top}
            our_top = {match.card.id for match in result.matches[:10]}
            overlaps.append(len(reference_top & our_top) / max(len(reference_top), 1))
    return {
        "queries": len(overlaps),
        "max_relative_difference": largest,
        "mean_top10_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
        "bm25s": library.__version__,
    }


def check_tokens(
    texts: Sequence[str],
    *,
    encodings: Sequence[str] = ("cl100k_base", "o200k_base"),
    groups: int = 20_000,
    group_size: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare `HeuristicTokenizer` with tiktoken encodings on `texts`.

    Per encoding: the share of texts the heuristic undercounts, how many random groups of
    `group_size` texts it undercounts in total, and the median, p99 and minimum ratio of heuristic
    to exact counts.
    """
    heuristic = HeuristicTokenizer()
    estimated = [heuristic.count(text) for text in texts]
    report: dict[str, Any] = {"texts": len(texts), "heuristic": repr(heuristic)}
    for name in encodings:
        encoding = tiktoken.get_encoding(name)
        exact = [len(encoding.encode(text, disallowed_special=())) for text in texts]
        ratios = [ours / theirs for ours, theirs in zip(estimated, exact, strict=True) if theirs]
        rng = random.Random(seed)
        group_undercounts = 0
        for _ in range(groups):
            members = rng.sample(range(len(texts)), group_size)
            group_undercounts += sum(estimated[i] for i in members) < sum(exact[i] for i in members)
        report[name] = {
            "single_undercount_share": sum(o < e for o, e in zip(estimated, exact, strict=True)) / len(texts),
            "group_undercounts": group_undercounts,
            "groups": groups,
            "group_size": group_size,
            "median_ratio": percentile(ratios, 50),
            "p99_ratio": percentile(ratios, 99),
            "min_ratio": min(ratios),
        }
    return report
