"""Run a tool-search agent with small, simulated tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

from toolhunch import Abstention, BM25Retriever, ChoiceDecider, ToolSearchPipeline, jev
from toolhunch.integrations.pydantic_ai import reveal_strategy

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from toolhunch import DecisionModel


def build_agent(model: Model | str, *, decision_model: DecisionModel) -> Agent[None, str]:
    """Build an agent that reveals the decider's pick after searching deferred tools."""
    pipeline = ToolSearchPipeline(
        BM25Retriever(), decider=ChoiceDecider(decision_model, abstention=Abstention()), k=20, top_n=1
    )
    agent = Agent(model, capabilities=[ToolSearch(strategy=reveal_strategy(pipeline))])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:
        """Get a simulated weather forecast for a city."""
        return f"{city}: sunny, 18 degrees Celsius (demo)."

    @agent.tool_plain(defer_loading=True)
    def send_email(to: str, body: str) -> str:
        """Simulate sending an email message."""
        return f"Demo email to {to}: {body}"

    return agent


async def main() -> None:
    """Load credentials and run the example against the two paid providers."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    decision_model = jev("jev-1.13.0")
    try:
        agent = build_agent("openai:gpt-4.1-mini-2025-04-14", decision_model=decision_model)
        result = await agent.run("What's the weather in Milan?")
        print(result.output)
    finally:
        await decision_model.aclose()


if __name__ == "__main__":
    anyio.run(main)
