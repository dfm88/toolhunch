from datetime import date

import pytest
from pydantic import ValidationError

from toolhunch.decision import (
    BinaryQuestion,
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    DecisionUsage,
    ModelLimits,
    QuestionKind,
    ScoreQuestion,
    check_request,
    choice_answer,
)

LIMITS = ModelLimits(
    max_options_per_choice=3, max_questions_per_request=1, score_levels=(2, 3), source="test", checked=date(2026, 9, 28)
)


def test_questions_and_requests_validate_their_shape() -> None:
    with pytest.raises(ValueError, match="two options"):
        ChoiceQuestion(instructions="q", options={"a": "x"})
    with pytest.raises(ValueError, match="two levels"):
        ScoreQuestion(instructions="q", levels=("low",))
    with pytest.raises(ValueError, match="question"):
        DecisionRequest(state="s", questions={})


def test_choice_answer_follows_option_order_and_renormalises() -> None:
    answer = choice_answer({"c": 0.33, "a": 0.33, "b": 0.33}, keys=["a", "b", "c"], model_id="m@h")
    assert list(answer.probabilities) == ["a", "b", "c"]
    assert answer.probabilities["a"] == pytest.approx(1 / 3)
    assert answer.choice == "a"  # ties go to the earlier option
    assert choice_answer({"b": 0.99, "a": 0.0}, keys=["a", "b"], model_id="m@h").probabilities == {"a": 0.0, "b": 1.0}


@pytest.mark.parametrize(
    "probabilities",
    [
        {"a": 0.5},
        {"a": 0.5, "b": 0.3, "z": 0.2},
        {"a": float("nan"), "b": 1.0},
        {"a": -0.1, "b": 1.1},
        {"a": "0.5", "b": 0.5},
        {"a": True, "b": 0.0},
        {"a": 0.0, "b": 0.0},
        {"a": 10**400, "b": 1.0},  # a JSON integer too large for a float
        {"a": 1e308, "b": 1e308},  # the sum overflows
    ],
)
def test_choice_answer_rejects_unusable_distributions(probabilities: dict[str, object]) -> None:
    with pytest.raises(DecisionError, match="m@h"):
        choice_answer(probabilities, keys=["a", "b"], model_id="m@h")


def test_requests_are_checked_against_declared_limits() -> None:
    four = ChoiceQuestion(instructions="q", options=dict.fromkeys("abcd", ""))
    binary = BinaryQuestion(instructions="q")
    for questions, reason in (
        ({"q": four}, "4 options"),
        ({"q1": binary, "q2": binary}, "2 questions"),
        ({"q": ScoreQuestion(instructions="q", levels=("1", "2", "3", "4"))}, "4 levels"),
        ({"q": binary}, "binary question"),
    ):
        with pytest.raises(ValueError, match=reason):
            check_request(
                DecisionRequest(state="s", questions=questions), limits=LIMITS, kinds=frozenset({"choice", "score"})
            )
    two = ChoiceQuestion(instructions="q", options={"a": "", "b": ""})
    check_request(DecisionRequest(state="s", questions={"q": two}), limits=LIMITS, kinds=frozenset({"choice"}))


def test_declared_bounds_are_inclusive() -> None:
    kinds: frozenset[QuestionKind] = frozenset({"choice", "score"})
    at_cap = ChoiceQuestion(instructions="q", options=dict.fromkeys("abc", ""))  # LIMITS allow 3 options
    fewest = ScoreQuestion(instructions="q", levels=("1", "2"))  # and 2 to 3 levels
    most = ScoreQuestion(instructions="q", levels=("1", "2", "3"))
    for question in (at_cap, fewest, most):
        check_request(DecisionRequest(state="s", questions={"q": question}), limits=LIMITS, kinds=kinds)
    with pytest.raises(ValueError, match="2 levels"):
        check_request(
            DecisionRequest(state="s", questions={"q": fewest}),
            limits=LIMITS.model_copy(update={"score_levels": (3, 5)}),
            kinds=kinds,
        )


def test_question_kind_is_a_class_constant() -> None:
    assert [question.kind for question in (ChoiceQuestion, BinaryQuestion, ScoreQuestion)] == [
        "choice",
        "binary",
        "score",
    ]
    with pytest.raises(TypeError, match="kind"):
        BinaryQuestion(instructions="q", kind="score")  # pyright: ignore[reportCallIssue]


def test_limits_are_frozen_declared_data() -> None:
    jev_like = ModelLimits(
        price_input_per_mtok=0.042, price_output_per_mtok=0.0, source="docs", checked=date(2026, 9, 27)
    )
    assert jev_like.estimate_usd(DecisionUsage(requests=1, input_tokens=1_000_000, output_tokens=5)) == pytest.approx(
        0.042
    )
    assert LIMITS.estimate_usd(DecisionUsage(input_tokens=10)) is None
    assert LIMITS.model_copy(update={"max_options_per_choice": 10}).max_options_per_choice == 10
    with pytest.raises(ValidationError):
        ModelLimits(source="x", checked=date(2026, 9, 28), unknown=1)  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValidationError):
        ModelLimits(max_options_per_choice=1, source="x", checked=date(2026, 9, 28))


def test_usd_estimate_prices_output_tokens_and_treats_a_missing_output_price_as_zero() -> None:
    usage = DecisionUsage(input_tokens=500_000, output_tokens=250_000)
    priced = ModelLimits(price_input_per_mtok=1.0, price_output_per_mtok=4.0, source="test", checked=date(2026, 9, 28))
    assert priced.estimate_usd(usage) == pytest.approx(1.5)  # 0.5 M x $1 + 0.25 M x $4
    assert priced.model_copy(update={"price_output_per_mtok": None}).estimate_usd(usage) == pytest.approx(0.5)


def test_usage_adds_up() -> None:
    assert DecisionUsage(1, 10, 2) + DecisionUsage(2, 5, 0) == DecisionUsage(3, 15, 2)
