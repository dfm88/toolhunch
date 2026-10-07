# ToolRet decision stage, held-out run 20261007T173440Z

Held-out run `20261007T173440Z`: 200 tasks from `bench/tasks/toolret-heldout-200.json` (sha256 `b632db8a5cb3`), `mteb/ToolRetrieval` at `76d45e5` with 44,453 tools, catalog `sha256:7d56b70f3415`; toolhunch 0.2.0, commit `eb766c7 (dirty)`, Python 3.14.3. Every abstention threshold applied here was chosen on the dev runs (`20261007T171451Z`, `20261007T171824Z`, `20261007T172046Z`); no held-out search chose one.

**GPT-6 Luna (Decisions).** OpenAI serves the current `gpt-6-luna` behind its Decisions API, in public beta, and no version can be pinned: a later run may answer differently from the one reported here. The API rounds its probabilities to two decimals.

## Runs

| split | role | run | tasks | sources | negatives | repeat | K | deciders (max detail) | reserved option | model-source fallbacks | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | main | `20261007T171451Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | luna-decisions (FULL) | on | 20 of 50 | `eb766c7` |
| dev | main | `20261007T171824Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | luna-decisions (BRIEF) | on | 20 of 50 | `eb766c7 (dirty)` |
| dev | main | `20261007T172046Z` | 10 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 3 | 20 | luna-decisions (FULL) | on | - | `eb766c7 (dirty)` |
| heldout | main | `20261007T173440Z` | 200 (`bench/tasks/toolret-heldout-200.json`, sha256 `b632db8a5cb3`) | plain, model | yes | 1 | 20, 50 | luna-decisions (BRIEF) | on | 81 of 200 | `eb766c7 (dirty)` |

| decider | model | prompt version | declared limits |
|---|---|---|---|
| luna-decisions | `gpt-6-luna@api.openai.com` | `tool-choice-v1+decisions-v1` | developers.openai.com/api/docs/guides/decisions: $0.10 per M input tokens, no output charge, no caps stated; openai-python v3.26.0 types; probe 2026-10-07: a 255-option choice and 64 questions in one request accepted, more not tried; a choice needs 2 options (checked 2026-10-07) |

Model-written queries of `20261007T171451Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261007T171824Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261007T173440Z`: `bench/tasks/toolret-heldout-200.model-queries.json` (sha256 `d8df10441cfb`), written by `gpt-5.4-mini-2026-03-17`.

## Held-out

`plain` searches with the ToolRet request, `model` with the queries a model wrote for the task, and `model (searched)` keeps the tasks where that model did search: on 81 of 200 tasks it answered without searching, and the request itself was used.

### P@1

P@1 measures a relevant tool first on positive requests, ignoring abstention; it does not measure task completion.

| arm | source | positives | P@1 (95% CI) | hybrid P@1 | Δ vs hybrid (95% CI) | ceiling |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 200 | 0.220 (0.17 to 0.28) | 0.220 | - | 0.590 |
| hybrid@20 | model | 200 | 0.215 (0.16 to 0.27) | 0.215 | - | 0.600 |
| hybrid@20 | model (searched) | 119 | 0.202 (0.13 to 0.28) | 0.202 | - | 0.647 |
| hybrid+luna-decisions@20 | plain | 199 | 0.317 (0.25 to 0.38) | 0.221 | +0.095 (0.05 to 0.15) | 0.593 |
| hybrid+luna-decisions@20 | model | 200 | 0.320 (0.26 to 0.39) | 0.215 | +0.105 (0.06 to 0.15) | 0.600 |
| hybrid+luna-decisions@20 | model (searched) | 119 | 0.311 (0.23 to 0.39) | 0.202 | +0.109 (0.06 to 0.18) | 0.647 |
| hybrid@50 | plain | 200 | 0.220 (0.17 to 0.28) | 0.220 | - | 0.715 |
| hybrid@50 | model | 200 | 0.215 (0.16 to 0.27) | 0.215 | - | 0.720 |
| hybrid@50 | model (searched) | 119 | 0.202 (0.13 to 0.28) | 0.202 | - | 0.748 |
| hybrid+luna-decisions@50 | plain | 200 | 0.350 (0.28 to 0.41) | 0.220 | +0.130 (0.07 to 0.19) | 0.715 |
| hybrid+luna-decisions@50 | model | 200 | 0.360 (0.29 to 0.43) | 0.215 | +0.145 (0.09 to 0.20) | 0.720 |
| hybrid+luna-decisions@50 | model (searched) | 119 | 0.353 (0.27 to 0.44) | 0.202 | +0.151 (0.08 to 0.23) | 0.748 |

### Abstention

Positive and negative requests are pooled below; the negative share is shown beside every row.

| arm | source | rule | τ | searches | errors | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate (95% CI) | abstention precision | abstention recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid@20 | plain | answer always | - | 400 | 0 | 1.000 | 0.110 | 44 | 356 | 0 | 50% | 0.890 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@20 | model | answer always | - | 400 | 0 | 1.000 | 0.107 | 43 | 357 | 0 | 50% | 0.892 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.101 | 24 | 214 | 0 | 50% | 0.899 (0.86 to 0.93) | n/a | 0.000 |
| hybrid+luna-decisions@20 | plain | answer always | - | 397 | 3 | 1.000 | 0.159 | 63 | 334 | 0 | 50% | 0.841 (0.81 to 0.87) | n/a | 0.000 |
| hybrid+luna-decisions@20 | plain | reserved | - | 397 | 3 | 0.804 | 0.185 | 59 | 260 | 78 | 50% | 0.655 (0.60 to 0.71) | 0.897 | 0.251 |
| hybrid+luna-decisions@20 | plain | with an abstention threshold | 0.95 | 397 | 3 | 0.169 | 0.373 | 25 | 42 | 330 | 50% | 0.106 (0.07 to 0.15) | 0.752 | 0.889 |
| hybrid+luna-decisions@20 | model | answer always | - | 399 | 1 | 1.000 | 0.160 | 64 | 335 | 0 | 50% | 0.840 (0.81 to 0.87) | n/a | 0.000 |
| hybrid+luna-decisions@20 | model | reserved | - | 399 | 1 | 0.810 | 0.192 | 62 | 261 | 76 | 50% | 0.654 (0.59 to 0.71) | 0.908 | 0.247 |
| hybrid+luna-decisions@20 | model | with an abstention threshold | 0.95 | 399 | 1 | 0.168 | 0.448 | 30 | 37 | 332 | 50% | 0.093 (0.06 to 0.13) | 0.747 | 0.889 |
| hybrid+luna-decisions@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.155 | 37 | 201 | 0 | 50% | 0.845 (0.80 to 0.89) | n/a | 0.000 |
| hybrid+luna-decisions@20 | model (searched) | reserved | - | 238 | 0 | 0.891 | 0.170 | 36 | 176 | 26 | 50% | 0.739 (0.67 to 0.80) | 0.885 | 0.143 |
| hybrid+luna-decisions@20 | model (searched) | with an abstention threshold | 0.95 | 238 | 0 | 0.176 | 0.405 | 17 | 25 | 196 | 50% | 0.105 (0.06 to 0.16) | 0.704 | 0.857 |
| hybrid@50 | plain | answer always | - | 400 | 0 | 1.000 | 0.110 | 44 | 356 | 0 | 50% | 0.890 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@50 | model | answer always | - | 400 | 0 | 1.000 | 0.107 | 43 | 357 | 0 | 50% | 0.892 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@50 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.101 | 24 | 214 | 0 | 50% | 0.899 (0.86 to 0.93) | n/a | 0.000 |
| hybrid+luna-decisions@50 | plain | answer always | - | 400 | 0 | 1.000 | 0.175 | 70 | 330 | 0 | 50% | 0.825 (0.79 to 0.86) | n/a | 0.000 |
| hybrid+luna-decisions@50 | plain | reserved | - | 400 | 0 | 0.850 | 0.197 | 67 | 273 | 60 | 50% | 0.682 (0.63 to 0.74) | 0.883 | 0.206 |
| hybrid+luna-decisions@50 | plain | with an abstention threshold | 0.95 | 400 | 0 | 0.215 | 0.384 | 33 | 53 | 314 | 50% | 0.133 (0.09 to 0.17) | 0.688 | 0.840 |
| hybrid+luna-decisions@50 | model | answer always | - | 400 | 0 | 1.000 | 0.180 | 72 | 328 | 0 | 50% | 0.820 (0.79 to 0.85) | n/a | 0.000 |
| hybrid+luna-decisions@50 | model | reserved | - | 400 | 0 | 0.860 | 0.203 | 70 | 274 | 56 | 50% | 0.685 (0.63 to 0.74) | 0.875 | 0.191 |
| hybrid+luna-decisions@50 | model | with an abstention threshold | 0.95 | 400 | 0 | 0.195 | 0.397 | 31 | 47 | 322 | 50% | 0.117 (0.08 to 0.16) | 0.677 | 0.852 |
| hybrid+luna-decisions@50 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.176 | 42 | 196 | 0 | 50% | 0.824 (0.78 to 0.87) | n/a | 0.000 |
| hybrid+luna-decisions@50 | model (searched) | reserved | - | 238 | 0 | 0.924 | 0.186 | 41 | 179 | 18 | 50% | 0.752 (0.69 to 0.81) | 0.833 | 0.101 |
| hybrid+luna-decisions@50 | model (searched) | with an abstention threshold | 0.95 | 238 | 0 | 0.197 | 0.362 | 17 | 30 | 191 | 50% | 0.126 (0.08 to 0.18) | 0.649 | 0.832 |

### Latency

| arm | source | decision p50 ms | p95 | latency basis | server p50 ms | p95 | retrieval p50 ms | p95 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| hybrid@20 | plain | - | - | - | - | - | 582 | 596 |
| hybrid@20 | model | - | - | - | - | - | 585 | 1,743 |
| hybrid@20 | model (searched) | - | - | - | - | - | 1,164 | 1,774 |
| hybrid+luna-decisions@20 | plain | 253 | 544 | critical path | 171 | 437 | 582 | 596 |
| hybrid+luna-decisions@20 | model | 261 | 622 | critical path | 179 | 522 | 585 | 1,743 |
| hybrid+luna-decisions@20 | model (searched) | 267 | 684 | critical path | 183 | 579 | 1,164 | 1,774 |
| hybrid@50 | plain | - | - | - | - | - | 582 | 596 |
| hybrid@50 | model | - | - | - | - | - | 585 | 1,743 |
| hybrid@50 | model (searched) | - | - | - | - | - | 1,164 | 1,774 |
| hybrid+luna-decisions@50 | plain | 261 | 609 | critical path | 181 | 529 | 582 | 596 |
| hybrid+luna-decisions@50 | model | 262 | 817 | critical path | 181 | 681 | 585 | 1,743 |
| hybrid+luna-decisions@50 | model (searched) | 268 | 826 | critical path | 182 | 700 | 1,164 | 1,774 |

### Cost per 1,000 searches

| arm | source | asks / search | input tokens / search | USD | CLM busy USD | CLM wall USD |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 0.00 | 0 | 0.0000 | - | - |
| hybrid@20 | model | 0.00 | 0 | 0.0000 | - | - |
| hybrid@20 | model (searched) | 0.00 | 0 | 0.0000 | - | - |
| hybrid+luna-decisions@20 | plain | 1.00 | 885 | 0.0885 | - | - |
| hybrid+luna-decisions@20 | model | 1.00 | 879 | 0.0879 | - | - |
| hybrid+luna-decisions@20 | model (searched) | 1.00 | 885 | 0.0885 | - | - |
| hybrid@50 | plain | 0.00 | 0 | 0.0000 | - | - |
| hybrid@50 | model | 0.00 | 0 | 0.0000 | - | - |
| hybrid@50 | model (searched) | 0.00 | 0 | 0.0000 | - | - |
| hybrid+luna-decisions@50 | plain | 1.00 | 2,015 | 0.2015 | - | - |
| hybrid+luna-decisions@50 | model | 1.00 | 1,985 | 0.1985 | - | - |
| hybrid+luna-decisions@50 | model (searched) | 1.00 | 1,975 | 0.1975 | - | - |

Per arm, all sources together: the summed seconds of its decisions on its latency basis, which CLM's wall figure prices, next to the arm's own clock time and the asks the decision cache replayed (hits) or sent (misses) during it.

| arm | searches | errors | not applicable | decision s | latency basis | arm clock s | cache hits | cache misses |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| hybrid@20 | 800 | 0 | - | - | - | 266.8 | - | - |
| hybrid+luna-decisions@20 | 800 | 4 | - | 246.4 | critical path | 164.2 | 287 | 509 |
| hybrid@50 | 800 | 0 | - | - | - | 0.0 | - | - |
| hybrid+luna-decisions@50 | 800 | 0 | - | 282.3 | critical path | 197.8 | 252 | 548 |

## Dev

### Thresholds

| threshold key | decider | K | payload | searches (positives + negatives) | τ | U at τ | U at 0.00 | coverage at τ | wrong-tool rate at τ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `gpt-6-luna@api.openai.com\|tool-choice-v1+decisions-v1\|db6d5d57dd5971bd\|choice` | luna-decisions | 20 | reserved option, FULL, questions [[21]] | 200 (100 + 100) | 0.95 | -0.105 | -0.485 | 0.235 | 0.170 |
| `gpt-6-luna@api.openai.com\|tool-choice-v1+decisions-v1\|5be950b6c011df2d\|choice` | luna-decisions | 50 | reserved option, FULL, questions [[51]] | 200 (100 + 100) | 0.95 | -0.135 | -0.495 | 0.265 | 0.200 |
| `gpt-6-luna@api.openai.com\|tool-choice-v1+decisions-v1\|55db837d3c1774fe\|choice` | luna-decisions | 20 | reserved option, BRIEF, questions [[21]] | 200 (100 + 100) | 0.95 | -0.155 | -0.515 | 0.255 | 0.205 |
| `gpt-6-luna@api.openai.com\|tool-choice-v1+decisions-v1\|38dc60cd902e7d70\|choice` | luna-decisions | 50 | reserved option, BRIEF, questions [[51]] | 200 (100 + 100) | 0.95 | -0.140 | -0.560 | 0.280 | 0.210 |

### P@1 by dev run

P@1 on positives (their count in brackets), one column per dev run that made each search once; the ablations' settings are in the runs table.

| arm | source | `20261007T171451Z` | `20261007T171824Z` |
|---|---|---:|---:|
| hybrid@20 | plain | 0.200 (50) | 0.200 (50) |
| hybrid@20 | model | 0.200 (50) | 0.200 (50) |
| hybrid@20 | model (searched) | 0.200 (30) | 0.200 (30) |
| hybrid+luna-decisions@20 | plain | 0.260 (50) | 0.320 (50) |
| hybrid+luna-decisions@20 | model | 0.320 (50) | 0.300 (50) |
| hybrid+luna-decisions@20 | model (searched) | 0.367 (30) | 0.267 (30) |
| hybrid@50 | plain | 0.200 (50) | 0.200 (50) |
| hybrid@50 | model | 0.200 (50) | 0.200 (50) |
| hybrid@50 | model (searched) | 0.200 (30) | 0.200 (30) |
| hybrid+luna-decisions@50 | plain | 0.320 (50) | 0.300 (50) |
| hybrid+luna-decisions@50 | model | 0.360 (50) | 0.320 (50) |
| hybrid+luna-decisions@50 | model (searched) | 0.367 (30) | 0.300 (30) |

### Determinism

Largest change of one probability (a card's or the reserved option's) between repeats of the same search, over every repeated search of the arm.

| run | arm | searches repeated | largest abs Δp | at | within 0.01 |
|---|---|---:|---:|---|---|
| `20261007T172046Z` | hybrid+luna-decisions@20 | 20 | 0.0000 | apibank_query_38 (plain, positive) | yes |

### Errors

No dev search failed, and none was not applicable.

## Token heuristic against Jev's count

For every exchange with Jev, the input tokens Jev reported against the heuristic's estimate that the planner budgets with: a least-squares line (reported = intercept + slope x estimate), and the ratios reported / estimate. A ratio under 1 means the heuristic counted more than Jev billed.

| split | Jev exchanges | intercept | slope | median reported / estimated | share under 1 |
|---|---:|---:|---:|---:|---:|
| dev | 0 | n/a | n/a | n/a | n/a |
| heldout | 0 | n/a | n/a | n/a | n/a |

## Notes

- **Outcomes.** A search is answered when the decider names a tool, correct when that tool is a gold one, and wrong otherwise. A negative is the same search with its gold tools removed from the candidates, so every answer to it is wrong.
- **P@1** counts positives only and ignores abstention: is the first card of the decider's ranking a gold tool? `hybrid P@1` asks the same of retrieval order on the same searches, and `ceiling` is the share of them with a gold tool among the K candidates. `Δ vs hybrid` is the decider's P@1 minus retrieval's own P@1 on the same positives. Its interval resamples tasks with both answers drawn together, so it says whether the decision helped even where the two separate intervals overlap.
- **Answer always.** This reinterprets the same ranking obtained with the none option available: its first card is chosen, ignoring abstention. It does not show what the model would answer without that option. The dev ablation tested that setting separately; for logprob it also changed the number of rounds.
- **Rules.** `reserved`: the decider abstains when its reserved "none of these" option is at least as likely as its best card. `with an abstention threshold`: it also abstains when that card's probability is below τ, the threshold dev chose for the search's threshold key (model, prompt version, payload shape, question kind). Probabilities compare only within one question, so a τ never crosses keys.
- **Rates.** coverage: answered / searches. Selective accuracy: correct / answered. Wrong-tool rate: wrong / searches. Abstention precision: abstentions on searches without a gold tool among the candidates / abstentions. Abstention recall: those abstentions / searches without a gold tool among the candidates.
- **Choosing τ.** Per threshold key, on dev: of τ = 0.00, 0.05, ..., 0.95, the one with the highest U = (correct - wrong) / searches, where abstaining counts 0; of equal U, the lower τ wins. It uses the dev runs that made each search once, both query sources, positives and negatives; a search that several of those runs made counts once. Fallback searches of the `model` source (its writer made no search, so the request was used) repeat their `plain` search and count twice in U(τ) and in the searches counts of the thresholds table. The held-out risk-coverage tables are there to read, never to choose.
- **Intervals.** 95% percentile bootstrap, 2,000 resamples with seed 0, resampling tasks: a task's positive search, its negative and their repeats are drawn together.
- **Errors.** A search whose decider raised `DecisionError`, whose candidates hold a card no question can show, or whose final question did not fit after round one, counts as an error and stays out of every rate, latency and cost figure. A search whose candidates two rounds cannot hold is not applicable: it is counted apart, and its cell reports no P@1 and no rule, in every table.
- **Latency.** Decision: the decider's calls as the client timed them, network included; a call the decision cache replays keeps the time the original call took. Its basis is the critical path (within a round only the slowest call counts, as round one's calls run at the same time), or the sum of calls for a model asked one request at a time, such as a local server. Server: the same by the server's own clock, where it reports one. The per-arm decision seconds use the same basis. Retrieval: the first hybrid retrieval of the search's queries, measured live; a query whose vector the embedding cache already held skips the embeddings call, so it reads faster than a cold one.
- **Cost.** Jev: reported input tokens at the price its declared limits give. Logprob: the reported usage, priced by genai-prices. CLM is paid in GPU time, at the list price in the run's manifest, two ways. Busy prices the server's own seconds per search, as if the GPU never sat idle: a lower bound. Wall prices the summed seconds of the arm's decision calls as the client saw them, as if one container served the searches one after another. It leaves out the cold starts, the idle gaps between arms and the scale-down window, which Modal also bills. Modal's CPU and memory charges come on top of both and are not included. Every arm also embeds its queries for hybrid retrieval, at the same cost for each arm, which the F1 retrieval results report.
- **Negatives** stand in for a catalog without the gold tools: the gold ids are dropped from a deeper retrieval, so the other tools' order can move slightly from what a smaller catalog would give.

## Risk-coverage over the threshold grid

### Dev: `gpt-6-luna@api.openai.com|tool-choice-v1+decisions-v1|db6d5d57dd5971bd|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.775 | 0.187 | 29 | 126 | 45 | 50% | 0.630 | -0.485 |
| 0.05 | 0.775 | 0.187 | 29 | 126 | 45 | 50% | 0.630 | -0.485 |
| 0.10 | 0.775 | 0.187 | 29 | 126 | 45 | 50% | 0.630 | -0.485 |
| 0.15 | 0.775 | 0.187 | 29 | 126 | 45 | 50% | 0.630 | -0.485 |
| 0.20 | 0.775 | 0.187 | 29 | 126 | 45 | 50% | 0.630 | -0.485 |
| 0.25 | 0.765 | 0.190 | 29 | 124 | 47 | 50% | 0.620 | -0.475 |
| 0.30 | 0.755 | 0.192 | 29 | 122 | 49 | 50% | 0.610 | -0.465 |
| 0.35 | 0.730 | 0.192 | 28 | 118 | 54 | 50% | 0.590 | -0.450 |
| 0.40 | 0.685 | 0.204 | 28 | 109 | 63 | 50% | 0.545 | -0.405 |
| 0.45 | 0.675 | 0.207 | 28 | 107 | 65 | 50% | 0.535 | -0.395 |
| 0.50 | 0.625 | 0.208 | 26 | 99 | 75 | 50% | 0.495 | -0.365 |
| 0.55 | 0.550 | 0.236 | 26 | 84 | 90 | 50% | 0.420 | -0.290 |
| 0.60 | 0.530 | 0.226 | 24 | 82 | 94 | 50% | 0.410 | -0.290 |
| 0.65 | 0.470 | 0.245 | 23 | 71 | 106 | 50% | 0.355 | -0.240 |
| 0.70 | 0.420 | 0.250 | 21 | 63 | 116 | 50% | 0.315 | -0.210 |
| 0.75 | 0.405 | 0.247 | 20 | 61 | 119 | 50% | 0.305 | -0.205 |
| 0.80 | 0.385 | 0.234 | 18 | 59 | 123 | 50% | 0.295 | -0.205 |
| 0.85 | 0.340 | 0.265 | 18 | 50 | 132 | 50% | 0.250 | -0.160 |
| 0.90 | 0.290 | 0.276 | 16 | 42 | 142 | 50% | 0.210 | -0.130 |
| **0.95** | 0.235 | 0.277 | 13 | 34 | 153 | 50% | 0.170 | -0.105 |

### Dev: `gpt-6-luna@api.openai.com|tool-choice-v1+decisions-v1|5be950b6c011df2d|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.835 | 0.204 | 34 | 133 | 33 | 50% | 0.665 | -0.495 |
| 0.05 | 0.835 | 0.204 | 34 | 133 | 33 | 50% | 0.665 | -0.495 |
| 0.10 | 0.835 | 0.204 | 34 | 133 | 33 | 50% | 0.665 | -0.495 |
| 0.15 | 0.835 | 0.204 | 34 | 133 | 33 | 50% | 0.665 | -0.495 |
| 0.20 | 0.815 | 0.190 | 31 | 132 | 37 | 50% | 0.660 | -0.505 |
| 0.25 | 0.795 | 0.189 | 30 | 129 | 41 | 50% | 0.645 | -0.495 |
| 0.30 | 0.760 | 0.184 | 28 | 124 | 48 | 50% | 0.620 | -0.480 |
| 0.35 | 0.695 | 0.180 | 25 | 114 | 61 | 50% | 0.570 | -0.445 |
| 0.40 | 0.675 | 0.178 | 24 | 111 | 65 | 50% | 0.555 | -0.435 |
| 0.45 | 0.635 | 0.189 | 24 | 103 | 73 | 50% | 0.515 | -0.395 |
| 0.50 | 0.605 | 0.198 | 24 | 97 | 79 | 50% | 0.485 | -0.365 |
| 0.55 | 0.525 | 0.210 | 22 | 83 | 95 | 50% | 0.415 | -0.305 |
| 0.60 | 0.500 | 0.220 | 22 | 78 | 100 | 50% | 0.390 | -0.280 |
| 0.65 | 0.470 | 0.223 | 21 | 73 | 106 | 50% | 0.365 | -0.260 |
| 0.70 | 0.425 | 0.235 | 20 | 65 | 115 | 50% | 0.325 | -0.225 |
| 0.75 | 0.405 | 0.235 | 19 | 62 | 119 | 50% | 0.310 | -0.215 |
| 0.80 | 0.380 | 0.250 | 19 | 57 | 124 | 50% | 0.285 | -0.190 |
| 0.85 | 0.365 | 0.247 | 18 | 55 | 127 | 50% | 0.275 | -0.185 |
| 0.90 | 0.305 | 0.230 | 14 | 47 | 139 | 50% | 0.235 | -0.165 |
| **0.95** | 0.265 | 0.245 | 13 | 40 | 147 | 50% | 0.200 | -0.135 |

### Dev: `gpt-6-luna@api.openai.com|tool-choice-v1+decisions-v1|55db837d3c1774fe|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.825 | 0.188 | 31 | 134 | 35 | 50% | 0.670 | -0.515 |
| 0.05 | 0.825 | 0.188 | 31 | 134 | 35 | 50% | 0.670 | -0.515 |
| 0.10 | 0.825 | 0.188 | 31 | 134 | 35 | 50% | 0.670 | -0.515 |
| 0.15 | 0.825 | 0.188 | 31 | 134 | 35 | 50% | 0.670 | -0.515 |
| 0.20 | 0.825 | 0.188 | 31 | 134 | 35 | 50% | 0.670 | -0.515 |
| 0.25 | 0.795 | 0.195 | 31 | 128 | 41 | 50% | 0.640 | -0.485 |
| 0.30 | 0.760 | 0.204 | 31 | 121 | 48 | 50% | 0.605 | -0.450 |
| 0.35 | 0.760 | 0.204 | 31 | 121 | 48 | 50% | 0.605 | -0.450 |
| 0.40 | 0.715 | 0.217 | 31 | 112 | 57 | 50% | 0.560 | -0.405 |
| 0.45 | 0.665 | 0.233 | 31 | 102 | 67 | 50% | 0.510 | -0.355 |
| 0.50 | 0.610 | 0.221 | 27 | 95 | 78 | 50% | 0.475 | -0.340 |
| 0.55 | 0.555 | 0.207 | 23 | 88 | 89 | 50% | 0.440 | -0.325 |
| 0.60 | 0.540 | 0.204 | 22 | 86 | 92 | 50% | 0.430 | -0.320 |
| 0.65 | 0.525 | 0.210 | 22 | 83 | 95 | 50% | 0.415 | -0.305 |
| 0.70 | 0.465 | 0.183 | 17 | 76 | 107 | 50% | 0.380 | -0.295 |
| 0.75 | 0.440 | 0.182 | 16 | 72 | 112 | 50% | 0.360 | -0.280 |
| 0.80 | 0.390 | 0.179 | 14 | 64 | 122 | 50% | 0.320 | -0.250 |
| 0.85 | 0.350 | 0.186 | 13 | 57 | 130 | 50% | 0.285 | -0.220 |
| 0.90 | 0.280 | 0.196 | 11 | 45 | 144 | 50% | 0.225 | -0.170 |
| **0.95** | 0.255 | 0.196 | 10 | 41 | 149 | 50% | 0.205 | -0.155 |

### Dev: `gpt-6-luna@api.openai.com|tool-choice-v1+decisions-v1|38dc60cd902e7d70|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.850 | 0.171 | 29 | 141 | 30 | 50% | 0.705 | -0.560 |
| 0.05 | 0.850 | 0.171 | 29 | 141 | 30 | 50% | 0.705 | -0.560 |
| 0.10 | 0.850 | 0.171 | 29 | 141 | 30 | 50% | 0.705 | -0.560 |
| 0.15 | 0.850 | 0.171 | 29 | 141 | 30 | 50% | 0.705 | -0.560 |
| 0.20 | 0.850 | 0.171 | 29 | 141 | 30 | 50% | 0.705 | -0.560 |
| 0.25 | 0.825 | 0.176 | 29 | 136 | 35 | 50% | 0.680 | -0.535 |
| 0.30 | 0.780 | 0.179 | 28 | 128 | 44 | 50% | 0.640 | -0.500 |
| 0.35 | 0.765 | 0.183 | 28 | 125 | 47 | 50% | 0.625 | -0.485 |
| 0.40 | 0.715 | 0.196 | 28 | 115 | 57 | 50% | 0.575 | -0.435 |
| 0.45 | 0.690 | 0.203 | 28 | 110 | 62 | 50% | 0.550 | -0.410 |
| 0.50 | 0.675 | 0.207 | 28 | 107 | 65 | 50% | 0.535 | -0.395 |
| 0.55 | 0.640 | 0.219 | 28 | 100 | 72 | 50% | 0.500 | -0.360 |
| 0.60 | 0.625 | 0.224 | 28 | 97 | 75 | 50% | 0.485 | -0.345 |
| 0.65 | 0.550 | 0.236 | 26 | 84 | 90 | 50% | 0.420 | -0.290 |
| 0.70 | 0.520 | 0.221 | 23 | 81 | 96 | 50% | 0.405 | -0.290 |
| 0.75 | 0.505 | 0.228 | 23 | 78 | 99 | 50% | 0.390 | -0.275 |
| 0.80 | 0.470 | 0.213 | 20 | 74 | 106 | 50% | 0.370 | -0.270 |
| 0.85 | 0.380 | 0.211 | 16 | 60 | 124 | 50% | 0.300 | -0.220 |
| 0.90 | 0.320 | 0.234 | 15 | 49 | 136 | 50% | 0.245 | -0.170 |
| **0.95** | 0.280 | 0.250 | 14 | 42 | 144 | 50% | 0.210 | -0.140 |

### Held-out: hybrid+luna-decisions@20, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.807 | 0.188 | 121 | 521 | 154 | 50% | 0.655 | -0.503 |
| 0.05 | 0.807 | 0.188 | 121 | 521 | 154 | 50% | 0.655 | -0.503 |
| 0.10 | 0.807 | 0.188 | 121 | 521 | 154 | 50% | 0.655 | -0.503 |
| 0.15 | 0.807 | 0.188 | 121 | 521 | 154 | 50% | 0.655 | -0.503 |
| 0.20 | 0.807 | 0.188 | 121 | 521 | 154 | 50% | 0.655 | -0.503 |
| 0.25 | 0.796 | 0.189 | 120 | 514 | 162 | 50% | 0.646 | -0.495 |
| 0.30 | 0.779 | 0.194 | 120 | 500 | 176 | 50% | 0.628 | -0.477 |
| 0.35 | 0.722 | 0.207 | 119 | 456 | 221 | 50% | 0.573 | -0.423 |
| 0.40 | 0.681 | 0.214 | 116 | 426 | 254 | 50% | 0.535 | -0.389 |
| 0.45 | 0.617 | 0.230 | 113 | 378 | 305 | 50% | 0.475 | -0.333 |
| 0.50 | 0.562 | 0.248 | 111 | 336 | 349 | 50% | 0.422 | -0.283 |
| 0.55 | 0.514 | 0.267 | 109 | 300 | 387 | 50% | 0.377 | -0.240 |
| 0.60 | 0.481 | 0.282 | 108 | 275 | 413 | 50% | 0.345 | -0.210 |
| 0.65 | 0.443 | 0.283 | 100 | 253 | 443 | 50% | 0.318 | -0.192 |
| 0.70 | 0.403 | 0.287 | 92 | 229 | 475 | 50% | 0.288 | -0.172 |
| 0.75 | 0.378 | 0.289 | 87 | 214 | 495 | 50% | 0.269 | -0.160 |
| 0.80 | 0.339 | 0.307 | 83 | 187 | 526 | 50% | 0.235 | -0.131 |
| 0.85 | 0.289 | 0.322 | 74 | 156 | 566 | 50% | 0.196 | -0.103 |
| 0.90 | 0.230 | 0.339 | 62 | 121 | 613 | 50% | 0.152 | -0.074 |
| **0.95** | 0.168 | 0.410 | 55 | 79 | 662 | 50% | 0.099 | -0.030 |

### Held-out: hybrid+luna-decisions@50, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.855 | 0.200 | 137 | 547 | 116 | 50% | 0.684 | -0.512 |
| 0.05 | 0.855 | 0.200 | 137 | 547 | 116 | 50% | 0.684 | -0.512 |
| 0.10 | 0.855 | 0.200 | 137 | 547 | 116 | 50% | 0.684 | -0.512 |
| 0.15 | 0.855 | 0.200 | 137 | 547 | 116 | 50% | 0.684 | -0.512 |
| 0.20 | 0.846 | 0.202 | 137 | 540 | 123 | 50% | 0.675 | -0.504 |
| 0.25 | 0.839 | 0.203 | 136 | 535 | 129 | 50% | 0.669 | -0.499 |
| 0.30 | 0.796 | 0.210 | 134 | 503 | 163 | 50% | 0.629 | -0.461 |
| 0.35 | 0.755 | 0.217 | 131 | 473 | 196 | 50% | 0.591 | -0.427 |
| 0.40 | 0.719 | 0.223 | 128 | 447 | 225 | 50% | 0.559 | -0.399 |
| 0.45 | 0.659 | 0.241 | 127 | 400 | 273 | 50% | 0.500 | -0.341 |
| 0.50 | 0.608 | 0.253 | 123 | 363 | 314 | 50% | 0.454 | -0.300 |
| 0.55 | 0.555 | 0.275 | 122 | 322 | 356 | 50% | 0.403 | -0.250 |
| 0.60 | 0.525 | 0.276 | 116 | 304 | 380 | 50% | 0.380 | -0.235 |
| 0.65 | 0.487 | 0.292 | 114 | 276 | 410 | 50% | 0.345 | -0.203 |
| 0.70 | 0.434 | 0.305 | 106 | 241 | 453 | 50% | 0.301 | -0.169 |
| 0.75 | 0.385 | 0.331 | 102 | 206 | 492 | 50% | 0.258 | -0.130 |
| 0.80 | 0.344 | 0.349 | 96 | 179 | 525 | 50% | 0.224 | -0.104 |
| 0.85 | 0.306 | 0.359 | 88 | 157 | 555 | 50% | 0.196 | -0.086 |
| 0.90 | 0.264 | 0.389 | 82 | 129 | 589 | 50% | 0.161 | -0.059 |
| **0.95** | 0.205 | 0.390 | 64 | 100 | 636 | 50% | 0.125 | -0.045 |

