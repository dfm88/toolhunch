# toolhunch

Toolhunch ranks an agent's tools with retrieval and an optional decision model that can say “none of these”.

## How it works

In the ToolRet experiment: **44,453 tools → hybrid search → 20 candidates → a decider ranks the candidates or abstains**.
The pipeline ranks tools; the integration decides how to reveal them to the agent. The core is framework-free.

## What we measured

On 200 held-out ToolRet requests, using the request itself as the search query:

| Configuration | A relevant tool ranked first (95% interval) | Decision cost / 1,000 searches |
|---|---:|---:|
| Hybrid retrieval | 22.0% (16.5 to 27.5%) | — |
| Hybrid + Jev | 32.0% (25.5 to 38.5%) | $0.0538 |
| Hybrid + GPT-4.1 mini logprobs | 33.0% (26.5 to 39.5%) | $0.5220 |
| Retrieval ceiling: a relevant tool anywhere in the 20 candidates | 59.0% | — |

“A relevant tool” means one of the dataset's labelled tools. Some requests need several tools;
these results measure selection, not task completion. Ranking ignores abstention. Costs cover the
decision stage only, excluding retrieval, embedding preparation and the downstream agent.
The measured Jev cards use BRIEF text; the logprob cards use FULL text, so this is a comparison of
configurations, not models given identical information.

There are three ways to read the same predictions. The following counts use **400 searches:
200 positives and 200 negatives (50% negatives)**. Negatives remove the labelled tools from the
candidates; other tools might still be useful, so the labels are an approximation.

| Reading, on the same 50/50 mix | Decider | Correct | Wrong | Abstained |
|---|---|---:|---:|---:|
| Ranking: answer always | Jev | 64 | 336 | 0 |
| Ranking: answer always | GPT-4.1 mini logprobs | 66 | 334 | 0 |
| With a “none” option | Jev | 64 | 265 | 71 |
| With a “none” option | GPT-4.1 mini logprobs | 61 | 272 | 67 |
| With an abstention threshold | Jev | 20 | 29 | 351 |
| With an abstention threshold | GPT-4.1 mini logprobs | 52 | 215 | 133 |

The threshold was chosen on dev data and frozen before held-out evaluation. It cuts wrong picks,
but also loses useful ones: the counts above show why an abstention threshold is not a guarantee
of a reliable answer. “Answer always” reuses the ranking produced with “none” available; it does
not predict what the model would have returned if that option had been absent.

![Ranking precision against decision cost](docs/assets/decision-precision-cost.svg)

The [technical page](docs/experiments/toolret.md) explains the protocol, intervals, latency,
thresholds and limitations. All figures come from the generated
[summary](bench/results/2026-09-toolret-decision/summary.json).

In a separate [direct-choice test](docs/experiments/toolret.md#direct-choice-on-small-catalogs), Jev selected a relevant tool from the whole catalog on **74.1%** of 290 positive requests across four catalogs of 40–101 tools, versus **54.5%** for hybrid retrieval; abstentions count as misses.

## Pydantic AI example

Install the checkout with `uv sync --all-packages`, set `OPENAI_API_KEY` and `TYPESAFE_API_KEY`
in your environment or local `.env`, then run `uv run python examples/pydantic_ai_tool_search.py`.
This calls paid APIs. The demo tools return simulated data. The example uses BM25 retrieval;
the benchmark above uses hybrid lexical and embedding retrieval.

The following block is the complete [example file](examples/pydantic_ai_tool_search.py).
An offline test runs it with a `FunctionModel` and a fake decision model, and checks that this block stays identical.

```python
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
```

`Abstention()` adds the “none” option with no probability threshold. If the decider abstains,
the strategy reveals no candidate tool. Add your own threshold only after evaluating the same
model, prompt version and payload shape; the benchmark's thresholds are not portable defaults.

## Other deciders

`JevWireModel` supports Jev-compatible servers through `base_url`, with explicit `ModelLimits`.
`OpenAILogprobModel` supports OpenAI-compatible Chat Completions endpoints returning `top_logprobs`.
Declare the option cap supported by your endpoint; the planner can split a choice into rounds.
The measurements above tested `jev-1.13.0` on TypeSafe and
`gpt-4.1-mini-2025-04-14` on OpenAI.

Smoke-tested on 2026-09-30 with laya-serve 0.3.22 and rizzo-flow 0.1.0 on a two-tool example; not benchmarked.
The [generated smoke summary](bench/results/2026-09-jev-compatible-smoke/summary.json) records the model revisions,
declared limits, sources and outcomes. With a local Laya English server running at `127.0.0.1:8000`, this
[executable example](examples/jev_compatible_server.py) ranks cards without executing either tool:

```python
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
```

For rizzo-flow, use `model="rizzo-latest"`, your local API root and its declared limits from the summary:
26 options including “none”, and `max_state_plus_question_tokens=8192` for the tested `--ctx 8192` setting.
Laya English uses a 512-token window. Toolhunch estimates tokens while planning; server templates and tokenizers
determine the actual fit. These smoke tests do not establish accuracy or large-catalog support.
The local live test opts in through `TOOLHUNCH_LOCAL_BASE_URL`; set `TOOLHUNCH_LOCAL_SERVER=rizzo-flow` for that
server's recorded configuration. It skips without a URL and is excluded from the default offline suite.

## Status and roadmap

Pre-alpha: the retrieval pipeline, decision adapters and Pydantic AI reveal integration are implemented.
The direct-choice comparison now measures single-turn provider prompt-cache reads and billed cost.
Multi-turn agent evaluation follows. The OpenAI Decisions API is a future integration when available.

## Reproduce

Run offline checks with `uv run pytest -q`. Regenerate charts from published results without API calls:

```shell
uv run toolhunch-bench decision-charts bench/results/2026-09-toolret-decision/summary.json --out docs/assets
```

The [technical page](docs/experiments/toolret.md#reproduce) gives report regeneration and paid rerun commands.
For the main held-out Jev/logprob configurations, the summary implies about $2 in decision API calls
($1.62 before rounding); embedding setup, query writing, dev runs and repeated runs cost extra.
Every paid command prints an estimate first. See [AGENTS.md](AGENTS.md) for development conventions.

MIT. Not affiliated with TypeSafe, OpenAI or Pydantic.
