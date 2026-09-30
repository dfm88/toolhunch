"""Rank two demo tools with a local Jev-compatible server, without executing them."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import anyio
import httpx2

from toolhunch import Abstention, ChoiceDecider, JevWireModel, ModelLimits, ScoredCard, ToolCard

if TYPE_CHECKING:
    from toolhunch import Decision, DecisionModel

LAYA_LIMITS = ModelLimits(
    max_options_per_choice=100,
    max_state_plus_question_tokens=512,
    max_questions_per_request=64,
    source=(
        "https://huggingface.co/convaiinnovations/laya/resolve/"
        "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/rl_agent_config.json; "
        "https://github.com/NandhaKishorM/laya/blob/6d942c92081fbc139e736bbd9ac0023223c29b7f/laya/serve.py"
    ),
    checked=date(2026, 9, 30),
)


async def rank_demo_tools(model: DecisionModel) -> Decision:
    """Rank weather and e-mail cards, with a reserved 'none' option."""
    cards = (
        ToolCard(name="get_weather", description="Get the current weather in a city."),
        ToolCard(name="send_email", description="Send an e-mail message to a recipient."),
    )
    return await ChoiceDecider(model, abstention=Abstention()).decide(
        "What's the weather in Milan?", tuple(ScoredCard(card, 0.0) for card in cards)
    )


async def main() -> None:
    """Ask a local Laya English server and print the selected card's name."""
    async with httpx2.AsyncClient(trust_env=False) as client:
        model = JevWireModel(
            "english", base_url="http://127.0.0.1:8000/v1", api_key_env=None, limits=LAYA_LIMITS, http_client=client
        )
        decision = await rank_demo_tools(model)
        print("none" if decision.abstained else decision.ranked[0].card.name)


if __name__ == "__main__":
    anyio.run(main)
