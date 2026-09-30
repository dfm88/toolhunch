# ToolRet held-out with gpt-6-luna as a structured-output decider

200 held-out tasks, 44,453 tools, hybrid retrieval at K20, plain request. gpt-6-luna answers the logprob decider's letter prompt with reasoning off, as one letter from a JSON-schema enum: it returns at most 5 top logprobs, too few to rank 20 options. The answer has no probabilities, so there is no threshold reading and no averaging over orders. 95% intervals resample tasks (2,000 resamples, seed 0).

| run | role | commit |
|---|---|---|
| `20260930T171259Z` | orders | `4c98060 (dirty)` |
| `20260930T170022Z` | none | `67bb324 (dirty)` |
| `20260930T170024Z` | repeats | `67bb324 (dirty)` |

## First pick

| P@1 (95% CI) | hybrid P@1 | Δ vs hybrid (95% CI) | ceiling | $ / 1,000 searches | latency p50 / p95 ms |
|---:|---:|---:|---:|---:|---:|
| 0.340 (0.27 to 0.41) | 0.220 | 0.120 (0.07 to 0.18) | 0.590 | $0.119 | 1058 / 1982 |

## The "none" option

| searches | correct | wrong | abstained | positives right / wrong / none | negatives none |
|---:|---:|---:|---:|---:|---:|
| 400 | 60 | 257 | 83 | 30.0% / 54.0% / 16.0% | 25.5% |

## Candidate order

| P@1 by order 0 / 1 / 2 / 3 / 4 | shuffle P@1 | pairwise agreement | same-order agreement | slots 1-3 |
|---|---:|---:|---:|---:|
| 0.343 / 0.338 / 0.313 / 0.328 / 0.338 | 0.330 (0.27 to 0.39) | 0.629 (0.58 to 0.67) | 0.921 (0.89 to 0.95) | 23.1% (uniform 15%) |

Pairwise agreement: how often two orders of one task pick the same tool (10 pairs per task). Same-order agreement: the same over three repeats in retrieval order, the run-to-run noise to compare it with.

Failed searches, left out (Luna returned no content, with an empty refusal): first pick 0 (0 tasks); "none" 0 (0 tasks); orders 3 (2 tasks); repeats 1 (1 tasks). Order and repeat comparisons keep the 198 and 199 tasks that answered every time.

## Reproduce

```shell
T=bench/tasks/toolret-heldout-200.json
uv run toolhunch-bench decision --tasks $T --split heldout --deciders luna --k 20 --sources plain --no-reserved --no-negatives --orders
uv run toolhunch-bench decision --tasks $T --split heldout --deciders luna --k 20 --sources plain
uv run toolhunch-bench decision --tasks $T --split heldout --deciders luna --k 20 --sources plain --no-reserved --no-negatives --repeat 3 --bypass-cache
uv run toolhunch-bench luna-report --orders bench/runs/ORDERS --none bench/runs/NONE --repeats bench/runs/REPEATS
```
