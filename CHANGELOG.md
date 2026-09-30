# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Repository scaffold: uv workspace (library + benchmark), tooling and CI.
- `ToolCard` and `ToolCatalog`: immutable tool cards with an `id` separate from the name, a canonical
  order and a SHA-256 fingerprint; rendering at three detail levels within a token budget.
- `HeuristicTokenizer`: a conservative, dependency-free token estimate behind a `Tokenizer` protocol;
  non-ASCII characters weigh their UTF-8 length minus one, so Burmese, Georgian or emoji are not undercounted.
- Retrievers behind one async `Retriever` protocol: `BM25Retriever` (Lucene BM25, checked against bm25s,
  with an identifier-aware, Unicode-aware analyzer: NFKC, case folding, words kept whole across
  combining marks), `DenseRetriever` with an OpenAI-compatible `OpenAIEmbedder` (over-long inputs are cut
  to a declared byte limit; failures raise `EmbeddingError` with the key redacted), and `HybridRetriever`
  (reciprocal rank fusion). Several queries are fused the same way; queries without a letter or digit
  are ignored.
- `ToolSearchPipeline`: a retriever-only pipeline with a per-stage trace.
- Pydantic AI integration (`toolhunch[pydantic-ai]`): `catalog_from_tool_defs` and `reveal_strategy`
  for `ToolSearch(strategy=...)`.
- Benchmark harness (`bench/`, not published): ToolRet at a pinned revision, stratified task sampling,
  Pydantic AI's keyword search and ToolRet's BM25 as baselines, metrics with bootstrap confidence
  intervals, an embedding cache, a cost ledger, and a 50-task retrieval pilot with its cost report
  (`bench/results/2026-09-toolret-pilot/`).
- Decision stage: canonical decision types and the `DecisionModel` protocol; `ModelLimits` declared with
  source and date; `JevWireModel` for TypeSafe's Jev (`jev()`) and Jev-compatible servers such as CLM
  (`clm()`); `OpenAILogprobModel`, an LLM's option-letter logprobs as a vendor-free decider;
  `ChoiceDecider`, which plans questions within each model's limits and can abstain.
  `ToolSearchPipeline` takes `decider=` and `top_n=`, and `search()` takes `context=`. The reveal
  strategy passes the user prompt and reveals nothing when the decider abstains.
- ToolRet decision results: abstention thresholds chosen on dev only, then held-out numbers on 200 tasks
  (precision@1, abstention, latency, cost per 1,000 searches, run-to-run variation)
  (`bench/results/2026-09-toolret-decision/`).
- ToolRet direct-choice results and public write-up: five strategies on four common source catalogs,
  separate MetaTool results, single-turn provider prompt-cache reads, and billed/list cost accounting
  that separates new calls, local replays and historical pilot cold measurements
  (`bench/results/2026-09-toolret-direct/`).
