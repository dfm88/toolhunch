# ToolRet: does a decision stage improve tool retrieval?

A relevant tool appeared first on 22.0% of held-out requests with hybrid retrieval, 32.0%
with Jev and 33.0% with a GPT-4.1 mini logprob decision stage, at 20 candidates.
These are ranking measurements. Abstention changes which requests get a tool, and the observed
thresholds sacrifice many useful picks as well as preventing wrong ones.

This page reads the generated
[F2a summary](https://github.com/dfm88/toolhunch/blob/main/bench/results/2026-09-toolret-decision/summary.json)
and the benchmark implementation. It describes configuration performance, not end-to-end task success.

## Data and protocol

The catalog is `mteb/ToolRetrieval` at revision `76d45e560059754e289ea202462865a585679619`:
44,453 tools, catalog fingerprint
`sha256:7d56b70f34151288b0351d0764064e2d9852b9c2f1ef4e5ed076f1edb83debbf`.
The dev sample has 50 tasks; the separate held-out sample has 200. The main held-out run
is `20260929T153251Z`, recorded at toolhunch commit `57fd48a`, with Pydantic AI 2.50.0.

Hybrid retrieval fuses BM25 and `text-embedding-3-small` rankings with reciprocal rank fusion.
The decider receives either 20 or 50 candidates. The held-out Jev configuration uses BRIEF
card text, including only the description's first sentence; the logprob configuration uses FULL.
Parameter names are included. The planner can reduce detail when model limits require it.
This information difference limits any attribution of the result to the models alone.

The measured decision models are `jev-1.13.0@api.typesafe.ai`, with planner prompt version
`tool-choice-v1`, and `gpt-4.1-mini-2025-04-14@api.openai.com`, with
`tool-choice-v1+letters-v1`. Jev answers a choice question; the OpenAI adapter asks for one
option letter and ranks the probabilities returned in its token logprobs. The latter must split
larger choices into rounds to respect its declared option cap.

`plain` uses the task request as the query. `model` uses frozen queries written by
`gpt-5.4-mini-2026-03-17`; if the writer did not search, retrieval falls back to the original
request. That happened on 81 of the 200 held-out tasks. `model (searched)` reports the remaining
119 tasks separately. The headline tables below use `plain`.

Every positive has a paired negative formed by removing its labelled relevant tools from the
retrieval candidates. The pooled outcome tables therefore contain **200 positives and
200 negatives: 50% negatives**. A different, unlabelled tool might still serve a negative;
the test approximates “no suitable tool” rather than establishing it through tool execution.

Intervals are 95% percentile bootstrap intervals from 2,000 resamples, seed 0. The unit is a
task: its positive, negative and any repetitions are sampled together. The delta against hybrid
uses paired picks from the same tasks.

## Ranking a relevant tool first

P@1 considers positives and asks whether the first ranked card is a labelled relevant tool,
even when the decider abstains. The retrieval ceiling is the share of positives with any
relevant tool among the candidates. Several tools can be relevant to a request; choosing one
does not establish that it is the first executable step or that the task was completed.

| Candidates | Configuration | P@1 (95% CI) | Paired delta vs hybrid (95% CI) | Retrieval ceiling |
|---:|---|---:|---:|---:|
| 20 | Hybrid | 22.0% (16.5 to 27.5%) | — | 59.0% |
| 20 | Hybrid + Jev | 32.0% (25.5 to 38.5%) | +10.0 pp (4.0 to 15.5) | 59.0% |
| 20 | Hybrid + logprobs | 33.0% (26.5 to 39.5%) | +11.0 pp (5.0 to 17.0) | 59.0% |
| 50 | Hybrid | 22.0% (16.5 to 27.5%) | — | 71.5% |
| 50 | Hybrid + Jev | 37.0% (30.0 to 43.5%) | +15.0 pp (8.5 to 21.5) | 71.5% |
| 50 | Hybrid + logprobs | 34.5% (28.0 to 41.0%) | +12.5 pp (6.0 to 19.0) | 71.5% |

![Ranking precision and decision cost](../assets/decision-precision-cost.svg)

## Three readings of the same predictions

The following table uses **400 searches per row, split 50/50 positives and negatives**.
Correct means a labelled relevant tool was picked; wrong means a different tool was picked.
An abstention produces no tool pick and is shown separately.

“Answer always” ignores the reserved option and uses the top card of the existing ranking.
It is a free reinterpretation of a response generated with “none” available. It is not a
prediction of a forced-choice prompt. The dev ablation without “none” is a separate experiment;
for the logprob adapter it also changes the number of decision rounds.

With a “none” option, a tie between the best card and that option also abstains. With an
abstention threshold, the best card must additionally clear the threshold chosen on dev.

| K | Decider | Reading (50% negatives) | Correct | Wrong | Abstained | Coverage | Accuracy of picks | Wrong / searches (95% CI) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 20 | Hybrid | Answer always | 44 | 356 | 0 | 100.0% | 11.0% | 89.0% (86.2 to 91.8%) |
| 20 | Jev | Answer always | 64 | 336 | 0 | 100.0% | 16.0% | 84.0% (80.8 to 87.2%) |
| 20 | Jev | With a “none” option | 64 | 265 | 71 | 82.2% | 19.5% | 66.2% (60.8 to 71.8%) |
| 20 | Jev | With an abstention threshold | 20 | 29 | 351 | 12.2% | 40.8% | 7.2% (4.2 to 10.5%) |
| 20 | Logprobs | Answer always | 66 | 334 | 0 | 100.0% | 16.5% | 83.5% (80.2 to 86.8%) |
| 20 | Logprobs | With a “none” option | 61 | 272 | 67 | 83.2% | 18.3% | 68.0% (62.5 to 73.5%) |
| 20 | Logprobs | With an abstention threshold | 52 | 215 | 133 | 66.8% | 19.5% | 53.8% (47.5 to 60.0%) |
| 50 | Hybrid | Answer always | 44 | 356 | 0 | 100.0% | 11.0% | 89.0% (86.2 to 91.8%) |
| 50 | Jev | Answer always | 74 | 326 | 0 | 100.0% | 18.5% | 81.5% (78.2 to 85.0%) |
| 50 | Jev | With a “none” option | 74 | 294 | 32 | 92.0% | 20.1% | 73.5% (68.8 to 78.0%) |
| 50 | Jev | With an abstention threshold | 21 | 20 | 359 | 10.2% | 51.2% | 5.0% (2.8 to 7.5%) |
| 50 | Logprobs | Answer always | 69 | 331 | 0 | 100.0% | 17.2% | 82.8% (79.5 to 86.0%) |
| 50 | Logprobs | With a “none” option | 68 | 298 | 34 | 91.5% | 18.6% | 74.5% (69.8 to 79.2%) |
| 50 | Logprobs | With an abstention threshold | 59 | 219 | 122 | 69.5% | 21.2% | 54.8% (49.0 to 60.8%) |

Coverage is picks divided by searches. Accuracy of picks is correct divided by all picks.
These pooled values depend on the artificial negative share, so they do not estimate production
reliability for an unspecified workload.

### Thresholds and stability

A threshold belongs to one model identity, prompt version, payload shape and question kind.
It cannot be carried to a different configuration as a general confidence cutoff. The dev
search considers the recorded threshold grid and maximises `(correct - wrong) / searches`,
with abstention worth zero; ties prefer the lower threshold. Dev runs with repeated searches
are excluded from threshold selection, and duplicated searches are deduplicated. Model-query
fallbacks still contribute to both query sources.

The held-out Jev and logprob rows above use dev thresholds of 0.95. No held-out key lacks a
dev threshold. The held-out coverage/accuracy curve is a diagnostic plot, not a place to tune:

![Held-out coverage and accuracy, with dev thresholds marked](../assets/decision-coverage-accuracy.svg)

For this curve the outcome mix is again **50% negatives**. The corresponding correct, wrong and
abstained counts for every threshold are in each summary row's `risk_coverage.grid`; the table
above includes all three counts for the deployed dev threshold.

The separate held-out repetition run bypasses local decision replay. Jev's P@1 is
30.5%, 31.0%, 30.5%; logprobs gives 32.0%, 32.5%, 33.0%, at 20 candidates on plain queries.
Probability variation means one deterministic-looking API setting does not ensure identical
probabilities or threshold outcomes between runs.

## Cost and latency

| K | Decider | Decision requests / search | Decision USD / 1,000 searches | Client decision p50 / p95 |
|---:|---|---:|---:|---:|
| 20 | Jev | 1 | $0.0538 | 253 / 312 ms |
| 20 | Logprobs | 2 | $0.5220 | 1,308 / 1,922 ms |
| 50 | Jev | 1 | $0.1145 | 264 / 343 ms |
| 50 | Logprobs | 4 | $1.3460 | 1,504 / 2,492 ms |

These are decision-stage costs for plain queries, pooled over positives and negatives, using
the price metadata of the run. Hybrid retrieval adds its own work: its recorded p50 / p95 is
782 / 1,023 ms. Embedding setup, query writing and downstream agent calls are excluded from
the table. A higher candidate count also changes the logprob planner's number of rounds.

The F2a local SQLite decision cache is a replay store. Replayed answers retain their original
usage and timing in this report, so per-search figures describe the original decisions;
they are not the actual spend or elapsed time of report regeneration. Retrieval timings were
measured with an embedding cache that could already contain query vectors. F2a does not measure
provider prompt-cache savings. Single-turn provider cache accounting is the next experiment;
multi-turn agent loops remain subsequent work.

## Limits of the experiment

- Task labels establish relevant tools, not successful tool execution or complete agent tasks.
- Gold-removed negatives can leave unlabelled alternatives that would still be useful.
- The dev sample is small, and configuration-specific probabilities can vary across repeats.
- Jev and logprob card detail differs in this run. The next direct-choice comparison aligns full card text.
- Cost and latency are tied to the models, routing, caching state and date of these runs.
- Searches that raised `DecisionError` are excluded from rates, latency and costs; the published dev and
  main held-out configurations reported no errors.

## Reproduce

Install the workspace with `uv sync --all-packages`. Keep API credentials in your environment or
local `.env`; the benchmark loads them in code. No keys are needed to regenerate reports from
existing local run directories:

```shell
uv run toolhunch-bench decision-report \
  --dev bench/runs/20260929T140534Z --dev bench/runs/20260929T144900Z --dev bench/runs/20260929T145416Z \
  --dev bench/runs/20260929T145809Z --dev bench/runs/20260929T150417Z \
  --heldout bench/runs/20260929T153251Z --heldout-repeats bench/runs/20260929T162701Z \
  --out bench/results/2026-09-toolret-decision/
uv run toolhunch-bench decision-charts bench/results/2026-09-toolret-decision/summary.json --out docs/assets
```

Raw run directories are local and git-ignored. The published summary and SVGs are sufficient
to inspect the measurements; regenerating from raw exchanges requires those run directories.

To estimate a new main held-out run of the measured Jev and logprob configurations, use:

```shell
uv run toolhunch-bench decision --tasks bench/tasks/toolret-heldout-200.json --split heldout \
  --deciders jev,logprob --k 20,50 --sources plain,model --max-detail jev=brief,logprob=full \
  --bypass-cache --dry-run
```

Remove `--dry-run` to make paid calls. The frozen model-query file beside the task file must
already exist. Fetch the pinned dataset first with `uv run toolhunch-bench toolret fetch` if it
is not cached locally. The summary's per-search costs imply **$1.62, roughly $2**, for these main
held-out decision calls across both query sources and candidate counts. This estimate excludes
corpus/query embeddings, query writing, dev ablations and repetition runs; it is not a total
price for rebuilding every artifact. Each paid command prints its current estimate and ledger
cap check before sending requests.

Not affiliated with TypeSafe, OpenAI or Pydantic.
