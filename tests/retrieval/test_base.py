import pytest

from toolhunch import ScoredCard, ToolCard
from toolhunch.retrieval import reciprocal_rank_fusion


def ranking(ids: str) -> list[ScoredCard]:
    return [ScoredCard(ToolCard(name=card_id), 1.0) for card_id in ids]


@pytest.mark.parametrize(
    ("rankings", "weights", "k", "expected"),
    [
        # b: 1/62 + 1/61 > a: 1/61 + 1/63 > c: 1/63 + 1/62
        (["abc", "bca"], None, 3, ["b", "a", "c"]),
        # a: 2/61 + 1/63 > b: 2/62 + 1/61 — weights flip the winner
        (["abc", "bca"], [2.0, 1.0], 1, ["a"]),
        # equal fused scores fall back to id order, so results are deterministic
        (["b", "a"], None, 2, ["a", "b"]),
        # a card missing from one ranking still gets the other's contribution
        (["ab", "c"], None, 3, ["a", "c", "b"]),
    ],
)
def test_reciprocal_rank_fusion(rankings: list[str], weights: list[float] | None, k: int, expected: list[str]) -> None:
    fused = reciprocal_rank_fusion([ranking(ids) for ids in rankings], k=k, weights=weights)
    assert [scored.card.id for scored in fused] == expected


def test_rrf_score_is_the_weighted_sum_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([ranking("ab"), ranking("b")], k=2, rrf_k=60)
    assert {scored.card.id: scored.score for scored in fused} == pytest.approx({"b": 1 / 62 + 1 / 61, "a": 1 / 61})
