# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Repository scaffold: uv workspace (library + benchmark), tooling and CI.
- `ToolCard` and `ToolCatalog`: immutable tool cards with an `id` separate from the name, a canonical
  order and a SHA-256 fingerprint; rendering at three detail levels within a token budget.
- `HeuristicTokenizer`: a conservative, dependency-free token estimate behind a `Tokenizer` protocol.
- Retrievers behind one async `Retriever` protocol: `BM25Retriever` (Lucene BM25 with an
  identifier-aware analyzer, checked against bm25s), `DenseRetriever` with an OpenAI-compatible
  `OpenAIEmbedder`, and `HybridRetriever` (reciprocal rank fusion). Several queries are fused the same way.
- `ToolSearchPipeline`: a retriever-only pipeline with a per-stage trace.
- Pydantic AI integration (`toolhunch[pydantic-ai]`): `catalog_from_tool_defs` and `reveal_strategy`
  for `ToolSearch(strategy=...)`.
- Benchmark harness (`bench/`, not published): ToolRet at a pinned revision, stratified task sampling,
  Pydantic AI's keyword search and ToolRet's BM25 as baselines, metrics with bootstrap confidence
  intervals, an embedding cache, a cost ledger, and a 50-task retrieval pilot with its cost report
  (`bench/results/2026-09-toolret-pilot/`).
