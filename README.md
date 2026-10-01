<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dfm88/toolhunch/main/assets/brand/toolhunch-logo-white.png">
    <img src="https://raw.githubusercontent.com/dfm88/toolhunch/main/assets/brand/toolhunch-logo.png" alt="toolhunch" width="420">
  </picture>
</p>

<p align="center"><b>Your agent has too many tools and calls the wrong one.<br>
toolhunch searches them, lets a fast decision model pick, and can say “none”.</b></p>

<p align="center">
  <a href="https://pypi.org/project/toolhunch/"><img src="https://img.shields.io/pypi/v/toolhunch" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="alpha">
  <img src="https://img.shields.io/badge/python-3.12%2B-3776AB" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license">
  <img src="https://img.shields.io/badge/Pydantic%20AI-integration-E92063" alt="Pydantic AI integration">
  <img src="https://img.shields.io/badge/Jev--compatible-Laya-2F6DB5" alt="Jev-compatible: Laya">
  <img src="https://img.shields.io/badge/ToolRet-44%2C453%20tools-2E8B57" alt="Benchmarked on ToolRet, 44,453 tools">
</p>

![Which tool should the agent call? GPT-6 Luna agent 80%, Jev 74%, GPT-4.1 mini agent 61%, search only 54%](https://raw.githubusercontent.com/dfm88/toolhunch/main/docs/assets/direct-choice.svg)

On catalogs of 40–101 tools, a GPT-6 Luna agent given every tool as a function picked a relevant tool for
**80%** of 290 requests, Jev **74%** and a GPT-4.1 mini agent **61%**. Jev answered in 0.32 s against Luna's
1.45 s, and Luna never answered “none”. [Results and limits below](#results).

## Why search tools at all?

An agent usually receives every tool definition on every call. That is fine for a handful of tools, but:

- **Context and cost.** Every definition takes context-window space and input tokens on every call.
- **Hard limits.** OpenAI rejected our request with 200 functions (`array_above_max_length`).
- **Caching helps, within limits.** Prompt caching makes a fixed tool list cheaper, but the list still
  fills the context, and changing it breaks the cache.

What we measured: at 40–101 tools, both GPT agents did about as well with every tool as with 20 searched ones,
and caching made the whole catalog cheap (GPT-6 Luna: $0.07 against $0.18 per 1,000 requests). Search earns
its place as catalogs grow: hundreds of MCP tools, or ToolRet's 44,453.

## How it works

```text
request ──► search (BM25 + embeddings) ──► 20 candidates ──► decider ──► one tool, or “none”
```

- **Search** ranks tools by keywords (BM25) and by meaning (embeddings), merged with reciprocal rank fusion.
- **A decider** reads the request and the candidates, then picks one or answers “none”. It can be a System-1
  decision model such as [TypeSafe's Jev](https://docs.typesafe.ai), a Jev-compatible server
  ([Laya](https://github.com/NandhaKishorM/laya), [rizzo-flow](https://github.com/Rizzo-AI-Academy/rizzo-flow)),
  or an LLM that ranks the options with its token probabilities (logprobs) or names one as structured output.
- **Two ways to use a decider:** after search, on the 20 candidates, at any catalog size; or instead of search,
  reading the whole catalog, when it is small.

The pipeline ranks tools; an integration decides how they reach the agent. The core is framework-free.

## Results

The data is [ToolRet](https://arxiv.org/abs/2503.01763) (Shi et al., Findings of ACL 2025): requests labelled
with the tools that solve them, over a 44,453-tool corpus. “Relevant” means one of the labelled tools. These
tests measure tool selection on single-turn requests, not task completion. Intervals are 95% bootstrap intervals
over tasks; every figure is generated from a published summary.

### 1. Small catalogs: read everything, or search first?

Four ToolRet catalogs of 40–101 tools, with the same 290 requests for every strategy (chart above):

| Strategy | Relevant tool picked (95% CI) | Cost / 1,000 requests | Median latency |
|---|---:|---:|---:|
| GPT-6 Luna agent, whole catalog | **79.7%** (74.8–84.1) | $0.07 (99.6% cached) | 1.45 s |
| GPT-6 Luna agent, 20 searched tools | 77.6% (72.4–82.1) | $0.18 | 1.24 s |
| Jev reads the whole catalog | 74.1% (69.0–79.0) | $0.28 | 0.32 s |
| Search, then Jev picks among 20 | 71.4% (65.9–76.6) | $0.08 | 0.45 s |
| GPT-4.1 mini agent, 20 searched tools | 61.7% (55.9–66.9) | $0.68 | 0.94 s |
| GPT-4.1 mini agent, whole catalog | 60.7% (55.2–66.2) | $0.79 (92% cached) | 0.86 s |
| Search only, top result | 54.5% (48.6–60.0) | ≈ $0 | 0.16 s |

The agent makes one function-calling request; its first call is scored, never executed. GPT-6 Luna runs with
reasoning off, in a later run on the same requests. Costs and latency are for warm requests, after the first of
each catalog, and include search where a strategy searches first.

### 2. 44,453 tools: search, then decide

![Search only 22%, Jev 32%, GPT-4.1 mini 33%, GPT-6 Luna 34%; ceiling 59%](https://raw.githubusercontent.com/dfm88/toolhunch/main/docs/assets/rerank-44k.svg)

On 200 held-out requests, search alone puts a relevant tool first 22% of the time. A decider over the 20
candidates raises that to 32% (Jev), 33% (GPT-4.1 mini logprobs) or 34% (GPT-6 Luna): the same precision
within the intervals, with Jev the cheapest and fastest ($0.05 per 1,000 searches and 0.25 s, against $0.12
and 1.06 s for Luna, $0.52 and 1.31 s for GPT-4.1 mini). Search is the limit: a relevant tool is among the
20 candidates only 59% of the time. With 50 candidates the ceiling is 71.5%, and Jev reaches 37.0%.

Why GPT-4.1 mini for logprobs? The decider reads the probability of every option letter, and GPT-4.1 mini is
the newest OpenAI model we found that returns 20 of them: GPT-5-mini refuses logprobs, GPT-5.4-mini returns at
most 5, and GPT-6 Luna at most 5, only with reasoning off. So Luna answers the same prompt with one letter as
structured output: no probabilities, hence no confidence threshold and no averaging below.

### 3. When no tool fits

To test “none”, we remove the labelled tools from some requests and check who notices. The same answer can be
read in three ways:

| Reading | The decider… |
|---|---|
| Answer always | must pick a tool |
| Allow “none” | may answer “none of these” |
| Confidence threshold | answers only when its probability is at least 0.95 (chosen on dev data), otherwise “none” |

![Outcomes with and without a right tool in the catalog](https://raw.githubusercontent.com/dfm88/toolhunch/main/docs/assets/direct-none-option.svg)

On small catalogs, with “none” allowed: when no right tool existed, Jev said “none” 45% of the time and
the GPT-4.1 mini agent 41%, so both usually picked something anyway. The GPT-6 Luna agent never said it: it
called a tool for all 145 requests without a right one. When a right tool existed, the GPT-4.1 mini agent
said “none” to 27% of requests, Jev to 12%.

A threshold is no cure either. At 44,453 tools, on 200 requests with a relevant tool and 200 without:

| Jev, 20 candidates | Right | Wrong | “None” |
|---|---:|---:|---:|
| Answer always | 64 | 336 | 0 |
| Allow “none” | 64 | 265 | 71 |
| Threshold 0.95 | 20 | 29 | 351 |

Without a relevant tool any pick is wrong, so “answer always” is wrong at least 200 times by design.
The threshold removed about 9 in 10 wrong picks, and 2 in 3 right ones. GPT-4.1 mini gives 66/334/0, 61/272/67
and 52/215/133 for the same readings; GPT-6 Luna, allowed “none”, gives 60/257/83.

### 4. Does the order of the candidates matter?

A common criticism of decision models is that shuffling the options changes the answer. We asked the same
200 questions with the 20 candidates in five orders: search order and four shuffles.

![Pairwise agreement: Jev 96% same order, 76.5% shuffled; GPT-4.1 mini 98% and 60.5%; GPT-6 Luna 92.1% and 62.9%](https://raw.githubusercontent.com/dfm88/toolhunch/main/docs/assets/order-sensitivity.svg)

- **All three change their pick.** Two orders agree on the top tool 76.5% of the time for Jev, 60.5%
  for GPT-4.1 mini logprobs and 62.9% for GPT-6 Luna, against 92–98% when the same order is repeated.
- **GPT-4.1 mini's precision moved most.** A relevant tool came first 30.5% → 31.2% (mean of the shuffles)
  for Jev, 34.3% → 33.0% for Luna, and 33.0% → 29.0% for GPT-4.1 mini, lower in all four shuffles.
- **Position bias.** GPT-4.1 mini and Luna picked one of the first three slots 23% of the time, Jev 18.2%;
  a uniform pick gives 15%.
- **Averaging five orders did not help** (31.5% Jev, 31.0% GPT-4.1 mini) and costs five decisions per search.

The same-order repeats come from a separate run, so this is an observational comparison. Luna's figures use the
198 tasks it answered in every order (it returned an empty answer on 3 of 1,000 asks).

### 5. Cost and latency

![Precision against median latency and cost: Luna most precise on small catalogs but slowest; at 44,453 tools all deciders tie and Jev is fastest and cheapest](https://raw.githubusercontent.com/dfm88/toolhunch/main/docs/assets/cost-latency.svg)

On small catalogs the GPT-6 Luna agent was the most precise and, with its prompt cache, cheap, but took
1.45 s against Jev's 0.32 s. At 44,453 tools the deciders tie on precision, and Jev is both the fastest
and the cheapest.

For the agents given the whole catalog, OpenAI served most input tokens from its prompt cache on warm
requests: 92% for GPT-4.1 mini ($0.79 per 1,000 instead of $2.43 at list price) and 99.6% for GPT-6 Luna
($0.07 instead of $0.62). With 20 searched tools the list changes on every request, and nothing was cached
($0.68 and $0.18). Jev's provider-side caching was not measured.
Multi-turn agent loops, where caching and tool reveal interact, come next.

The [technical page](https://github.com/dfm88/toolhunch/blob/main/docs/experiments/toolret.md) has the protocol, every table, per-catalog results,
latency, thresholds and limitations.

## Quickstart

```shell
pip install toolhunch                  # or: uv add toolhunch
pip install "toolhunch[pydantic-ai]"   # with the Pydantic AI integration
```

Set `TYPESAFE_API_KEY` for Jev (and `OPENAI_API_KEY` for the agent below) in your environment:

```python
import anyio

from toolhunch import Abstention, BM25Retriever, ChoiceDecider, ToolCard, ToolCatalog, ToolSearchPipeline, jev

catalog = ToolCatalog(
    [
        ToolCard(name="get_weather", description="Get the current weather in a city."),
        ToolCard(name="send_email", description="Send an e-mail message to a recipient."),
        # ... the rest of your tools
    ]
)
pipeline = ToolSearchPipeline(
    BM25Retriever(),  # 1. search: keep 20 candidates
    decider=ChoiceDecider(jev("jev-1.13.0"), abstention=Abstention()),  # 2. pick one, or "none"
    k=20,
    top_n=1,
)
result = anyio.run(pipeline.search, ["What's the weather in Milan?"], catalog)
print("none" if result.abstained else result.names[0])
```

BM25 needs no key; the benchmark uses `HybridRetriever` with BM25 and OpenAI embeddings. `Abstention()` adds
“none” without a probability threshold: a threshold belongs to one model, prompt and payload shape, so
calibrate your own.

### With Pydantic AI

The same pipeline plugs into Pydantic AI's tool search: deferred tools stay hidden until the decider picks one.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

from toolhunch.integrations.pydantic_ai import reveal_strategy

agent = Agent("openai:gpt-4.1-mini", capabilities=[ToolSearch(strategy=reveal_strategy(pipeline))])


@agent.tool_plain(defer_loading=True)  # hidden until the search reveals it
def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"{city}: sunny, 18 °C (demo)"


print(agent.run_sync("What's the weather in Milan?").output)
```

A complete script, run offline by the tests, is in [`examples/pydantic_ai_tool_search.py`](https://github.com/dfm88/toolhunch/blob/main/examples/pydantic_ai_tool_search.py).

### With a Jev-compatible server

Swap the decision model; everything else stays. For a local Laya server (`laya-serve`, English checkpoint):

```python
from datetime import date

from toolhunch import JevWireModel, ModelLimits

laya = JevWireModel(
    "english",
    base_url="http://127.0.0.1:8000/v1",
    api_key_env=None,  # local server, no key
    limits=ModelLimits(  # declared, with their source and date
        max_options_per_choice=100,
        max_state_plus_question_tokens=512,
        max_questions_per_request=64,
        source="laya-serve 0.3.22",
        checked=date(2026, 9, 30),
    ),
)
pipeline = ToolSearchPipeline(BM25Retriever(), decider=ChoiceDecider(laya, abstention=Abstention()), k=20, top_n=1)
```

For rizzo-flow, use `model="rizzo-latest"`, its URL, 26 options including “none”
and `max_state_plus_question_tokens=8192` at `--ctx 8192`.
Both servers were smoke-tested on 2026-09-30 with a two-tool example and were not benchmarked
([protocol](https://github.com/dfm88/toolhunch/blob/main/docs/experiments/toolret.md#compatible-server-smoke-tests),
[summary](https://github.com/dfm88/toolhunch/blob/main/bench/results/2026-09-jev-compatible-smoke/summary.json));
a small token window can make the planner lower card detail or split a choice.
`OpenAILogprobModel` covers OpenAI-compatible endpoints that return `top_logprobs`.

## Status and roadmap

Alpha: search, deciders and the Pydantic AI integration work; the API may change.

- **Multi-turn agents:** measure whole agent loops, where prompt caching and tool reveal interact.
- **OpenAI Decisions API:** announced at [DevDay 2026](https://openai.com/index/devday-2026-recap/) in
  limited preview, it answers questions with predefined answers in about 150 ms and is built on GPT-6 Luna.
  It is the same kind of model as Jev, and the next decider to benchmark against the Luna numbers above
  once it is available.

## Reproduce

The benchmark is not on PyPI. Clone the repository, run the offline checks and regenerate the figures from the
published summaries, without API calls:

```shell
git clone https://github.com/dfm88/toolhunch && cd toolhunch && uv sync --all-packages
uv run pytest -q
uv run toolhunch-bench readme-charts
```

The [technical page](https://github.com/dfm88/toolhunch/blob/main/docs/experiments/toolret.md#reproduce) gives report regeneration and paid rerun commands:
about $2 for the main 44,453-tool decision calls, $1 for the small-catalog run and under $1 for the GPT-6 Luna runs.
Every paid command prints an estimate first. See [AGENTS.md](https://github.com/dfm88/toolhunch/blob/main/AGENTS.md) for development conventions.

## License

MIT.
