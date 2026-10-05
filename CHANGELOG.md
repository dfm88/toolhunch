# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `strands_decider()`: AWS Strands Labs' Strands Decider 2B served locally (`strands-decider serve`), with
  `STRANDS_LIMITS` (255 options, a 4,096-token window).
- `laya()`: a local Laya server (`laya-serve`, English checkpoint), with `LAYA_LIMITS` (100 options, a 192-token
  question, a 512-token question with its state, a 48-token option). `head_max_len` and `max_len` are sent with
  every request and the limits follow them. By default (`strict=True`) a reply raises `DecisionError` when Laya cut
  the state, collapsed options into one, answered with another checkpoint than the one asked for, or carries no
  `routing`.
- `rizzo_flow()`: a local rizzo-flow server (`rizzo serve`), with `RIZZO_FLOW_LIMITS` (26 options and an 8,192-token
  context for each question with its state, less the server's template). The default model id names the weights,
  `rizzo-flow-4b-q8_0`.
- `clef()`: Cloudflare's Clef and Clef-flash on Workers AI, with `CLEF_LIMITS` and `CLEF_FLASH_LIMITS`. The account
  ID goes into the URL only, never into `model_id` or `repr`.
- `JevWireModel(path=..., response_root=...)`: a Jev-shaped body at another endpoint path, and an answer wrapped in
  an envelope member such as Workers AI's `result`.
- Benchmark: Strands Decider 2B, Clef, Clef-flash and CLM on a Mac in the held-out re-ranking, direct-choice and
  candidate-order tests (`bench/results/2026-10-toolret-{decision,direct,order}-p2/`). Deciders are registry
  entries keyed by name (`--deciders`); local models run from `bench/deploy/strands_local.sh` and
  `bench/deploy/clm_local/` with their files in the git-ignored `bench/models/`.
- `toolhunch-bench compare-runs`: one model's searches on two deployments, with top-card agreement, probability
  differences, P@1 per cell and answer-or-abstain agreement.
- `toolhunch-bench cost-latency-chart`: precision against the decision's latency and cost for every published
  decider, local ones drawn hollow with their machine in the caption.
- `DecisionError.status`: the HTTP status a server refused the call with, after any retries; `None` for every other
  failure.
- `ModelLimits.max_question_tokens` and `ModelLimits.max_option_tokens`: the tokens one question may take (its
  instructions and every option) and the tokens one option may take as the server renders it (key, framing and
  text). The planner keeps every question within them.
- `ChoiceDecider(min_detail=...)`: a floor under the detail the cards are shown at. A question that fits only below
  it is split into groups instead of dropping descriptions. `render_within_budget`, `plan_question` and
  `plan_rounds` take it too, and `render_within_budget` accepts one text cap per card.
- `CandidatesDoNotFit`, a `ValueError` for candidates no question can hold within a model's limits. Its `card_id` is
  the card that cannot be shown, or `None` when the list as a whole does not fit.
- `Decision.sequential_seconds`: the decision's time against a server that answers one request at a time, every
  call's time added. `Decision.seconds` keeps counting the calls of round one as simultaneous.

### Changed

- `JevWireModel`'s `repr` shows `model_id` instead of `base_url`.
- `JevWireModel(path=...)` gets a leading slash when it lacks one.
- Benchmark reports state each decider's caveats (an unpinned version, a model run locally and the machine) and a
  local model's cost as "local"; the direct-choice report gives each run's search time when runs differ, and the
  decision latency alone; `decision-report --heldout` reads one run per decider.
- Benchmark runs stop when a provider refuses the key, the payment or the permission (HTTP 401, 402, 403), or after
  three failed attempts in a row at one provider with HTTP 429, a server fault or no reply; the manifest says why.
  A figure takes each decider's color, and whether it is shown, from its registry entry.
- When a model declares `max_option_tokens` and a tool's option at name level (its key, the framing and the name as
  text) is over it, the option is sent as its key alone with no text, since at that level the text only repeats the
  key. A key that does not fit raises `CandidatesDoNotFit` with the card's `card_id`, as does a name over
  `max_text_tokens`, which raised a plain `ValueError` before.
- When the candidates need two rounds, `plan_rounds` checks before the first call that a final question fits, at the
  worst case of the finalists the first round can keep, and keeps one finalist per group when more do not fit. A list
  that cannot be asked about raises `CandidatesDoNotFit` before any call, where it used to fail after the first
  round's calls were paid for. Code that caught `ValueError` still catches it.
- `Decision.shape`, and so `ThresholdKey.payload_shape`, gains `min_detail`, `budgets` (the limits the questions were
  planned under) and `tokenizer` (its class name), each only when it differs from the default: above `NAME`, when
  the model declares `max_question_tokens` or `max_option_tokens`, and when the tokenizer is not a
  `HeuristicTokenizer`. Every existing key is unchanged.

- Development status: alpha, from pre-alpha.

## [0.1.0] - 2026-09-30

First release on PyPI. Pre-alpha: the API may change.

### Added

- GPT-6 Luna results, reasoning off: agent arms in the direct-choice test (`direct --luna`, merged with
  `direct-report --add`) and, since Luna returns at most 5 logprobs, a structured-output letter decider on the
  44,453-tool held-out searches, with five candidate orders (`decision --deciders luna --orders`) and
  `luna-report` (`bench/results/2026-09-toolret-luna/`).
- `toolhunch-bench readme-charts`: the README figures (direct choice, “none” outcomes, re-ranking at
  44,453 tools, candidate order, precision against latency and cost), drawn from the published summaries
  without provider calls, as SVG or PNG.
- Jev-compatible local-server example and smoke-test write-up for laya-serve and rizzo-flow,
  with declared limits and model provenance; the two-tool example is not a catalog benchmark
  (`bench/results/2026-09-jev-compatible-smoke/`).
- `Exchange.option_card_ids`: the actual option-key-to-card-ID mapping for a decision question,
  including multi-round decisions with duplicate tool names; defaults to an empty mapping for
  manually constructed exchanges.
- ToolRet candidate-order sensitivity results at K=20: five fresh orders, abstention and position
  diagnostics, matched same-order run-noise comparison, and costed probability averaging
  (`bench/results/2026-09-toolret-order/`).
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

[Unreleased]: https://github.com/dfm88/toolhunch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dfm88/toolhunch/releases/tag/v0.1.0
