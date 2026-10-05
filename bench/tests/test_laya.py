"""Laya in the bench: the check that refuses before sending what Laya would cut, and the two Laya entries."""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from toolhunch import DetailLevel, ScoredCard
from toolhunch.decision import Abstention, ChoiceQuestion, DecisionError, DecisionRequest, laya
from toolhunch_bench.datasets.toolret import ToolRetData
from toolhunch_bench.deciders import DECIDERS, DeciderName, choice_decider, decision_model
from toolhunch_bench.decision_cache import CachedDecisionModel
from toolhunch_bench.laya import LayaCheckedModel, LayaTokenizer, laya_head

pytestmark = pytest.mark.anyio

# The request of the recorded Laya reply `laya_tool_choice.json`, which answers exactly these options.
RECORDED = Path(__file__).parents[2] / "tests" / "decision" / "fixtures" / "laya_tool_choice.json"
STATE = "Request: Book me a table for two at an Italian place tonight at 8."
INSTRUCTIONS = "Which tool should the assistant call next to fulfil the request?"
OPTIONS = {
    "get_weather": "get_weather: Get the current weather and the forecast for a city.",
    "book_restaurant": "book_restaurant: Reserve a table at a restaurant for a given date, time and party size.",
    "order_food_delivery": "order_food_delivery: Order food for home delivery from a nearby restaurant.",
    "send_email": "send_email: Send an email to a recipient with a subject and a body.",
    "none": "None of these tools can fulfil the request.",
}
QUESTION = ChoiceQuestion(instructions=INSTRUCTIONS, options=OPTIONS)


def request(options: dict[str, str], *, instructions: str = INSTRUCTIONS) -> DecisionRequest:
    return DecisionRequest(state=STATE, questions={"tool": ChoiceQuestion(instructions=instructions, options=options)})


def words(count: int) -> str:
    return " ".join(["word"] * count)


async def test_laya_check_refuses_what_build_head_would_cut(laya_word_tokenizer: tuple[Path, str]) -> None:
    path, digest = laya_word_tokenizer
    tokenizer = LayaTokenizer(path, sha256=digest)
    recorded: dict[str, Any] = json.loads(RECORDED.read_text(encoding="utf-8"))
    calls: list[httpx2.Request] = []

    def handler(sent: httpx2.Request) -> httpx2.Response:
        calls.append(sent)
        return httpx2.Response(recorded["status"], headers=recorded["headers"], json=recorded["body"])

    inner = laya(http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    checked = LayaCheckedModel(inner, tokenizer=tokenizer, head_max_len=192)

    # " key: word ..." takes the key, the colon and 48 words: 50 tokens, which Laya would cut at 48.
    with pytest.raises(DecisionError, match=r"refused before sending: option 'key' takes 50 tokens, over 48"):
        await checked.ask(request({"key": words(48), "none": ""}))
    # Nine options of [MASK], key, colon and 19 words: 9 x 22 = 198 tokens, more than the 192 - 16 Laya keeps whole.
    with pytest.raises(DecisionError, match=r"the options take 198 of head_max_len 192, leaving fewer than 16"):
        await checked.ask(request({f"key{index}": words(19) for index in range(9)}))
    # Two key-only options take 4 tokens; "choice question: " and 186 words take 189, over the 188 they leave.
    with pytest.raises(DecisionError, match=r"the instructions take 189 tokens, more than the 188 the options leave"):
        await checked.ask(request({"a": "", "b": ""}, instructions=words(186)))
    assert calls == []
    assert checked.refused == 3
    assert checked.model_id == inner.model_id

    answered = await checked.ask(request(OPTIONS))
    assert len(calls) == 1
    assert answered.answers["tool"]

    # [CLS], the instructions, [SEP], each option behind its [MASK], [SEP], counted by hand one token per word or
    # run of punctuation: "choice question: Which tool ... request?" is 15; " get_weather: get_weather: Get the
    # current weather and the forecast for a city." is 15, and so on.
    head = laya_head(QUESTION, tokenizer=tokenizer, head_max_len=192)
    assert head.instruction_tokens == 15
    assert head.option_tokens == (16, 21, 15, 18, 12)
    assert head.tokens == 1 + 15 + 1 + (16 + 21 + 15 + 18 + 12) + 1
    assert head.refusal is None

    # At each limit nothing is cut: an option of exactly 48 tokens; eight options of 22 that leave exactly 16, and
    # instructions of exactly 16 ("choice question: " and 13 words).
    def at_limit(options: dict[str, str], instructions: str) -> str | None:
        question = ChoiceQuestion(instructions=instructions, options=options)
        return laya_head(question, tokenizer=tokenizer, head_max_len=192).refusal

    assert at_limit({"key": words(46), "none": ""}, INSTRUCTIONS) is None
    assert at_limit({f"key{index}": words(19) for index in range(8)}, words(13)) is None

    # A tokenizer file from another revision is refused, naming the hash it expected.
    with pytest.raises(ValueError, match="sha256"):
        LayaTokenizer(path, sha256="0" * 64)


async def test_laya_wide_has_its_own_cache_and_tokenizer(
    tmp_path: Path, laya_word_tokenizer: tuple[Path, str], fake_decision_model: Any, toolret_data: ToolRetData
) -> None:
    # Both entries build offline, against one server and one checkpoint: only the budgets they send differ.
    narrow, wide = decision_model(DeciderName.LAYA), decision_model(DeciderName.LAYA_WIDE)
    assert narrow.model_id == wide.model_id
    assert "head_max_len=192" in repr(narrow)
    assert "head_max_len=512" in repr(wide)

    # One identical request through one cache file: each configuration asks its own, and repeats replay.
    inner = fake_decision_model(model_id=wide.model_id, limits=wide.limits)
    for name in (DeciderName.LAYA, DeciderName.LAYA_WIDE, DeciderName.LAYA_WIDE):
        cached = CachedDecisionModel(
            inner, path=tmp_path / "decisions.sqlite", namespace=DECIDERS[name].cache_namespace
        )
        await cached.ask(request(OPTIONS))
        cached.close()
    assert len(inner.asks) == 2

    # The Laya entries plan with Laya's tokenizer, which the threshold key records; a name outside the registry plans
    # with the heuristic, and its key does not mention one.
    candidates = [ScoredCard(card, 1.0) for card in list(toolret_data.catalog)[:3]]
    abstention = Abstention()
    decision = await choice_decider(
        "laya-wide", inner, abstention=abstention, max_detail=DetailLevel.BRIEF, min_detail=DetailLevel.BRIEF
    ).decide(STATE, candidates)
    assert decision.shape["tokenizer"] == "LayaTokenizer"
    assert decision.shape["min_detail"] == "BRIEF"
    heuristic = await choice_decider("fake", inner, abstention=abstention, max_detail=DetailLevel.BRIEF).decide(
        STATE, candidates
    )
    assert "tokenizer" not in heuristic.shape
