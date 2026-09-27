# toolhunch

**Tool search for LLM agents, ranked by System-1 decision models — with a benchmark that measures
prompt caching, not just tokens.**

> Status: pre-alpha, design phase. Nothing to install yet.

Agents with many tools send every schema on every request and choose worse as the catalog grows.
Existing fixes retrieve tools by keyword or embedding similarity and report token savings. toolhunch
adds two things:

- **A decision stage that can say "none of these".** After a cheap retriever narrows the catalog, a
  System-1 model — [Jev](https://docs.typesafe.ai) or any Jev-compatible endpoint — a reranker or an
  LLM picks the tool with a probability, and abstains when nothing fits. Each model declares its
  limits (options per question, tokens per request, rate, price) and a planner respects them.
- **Honest cost accounting.** With prompt caching, fewer tokens is not the same as cheaper. The
  benchmark separates *how tools are ranked* from *how they reach the model* (reveal, preselect,
  proxy) and measures cache hits on Anthropic, OpenAI and fallback paths.

First integration: [Pydantic AI](https://ai.pydantic.dev) (`ToolSearch(strategy=...)`,
preselection, proxy toolset). The core is framework-free.

## Development

```shell
uv sync --all-packages
uv run pytest
```

See [AGENTS.md](AGENTS.md) for conventions.

## License

MIT. Not affiliated with TypeSafe, Pydantic or any project listed in the landscape.
