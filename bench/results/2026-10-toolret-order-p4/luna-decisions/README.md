# ToolRet candidate-order sensitivity

K=20, plain queries, GPT-6 Luna (Decisions) BRIEF, reserved option last. P@1 means a relevant tool first with abstention ignored; selecting it does not complete a task.

**GPT-6 Luna (Decisions).** OpenAI serves the current `gpt-6-luna` behind its Decisions API, in public beta, and no version can be pinned: a later run may answer differently from the one reported here. The API rounds its probabilities to two decimals.

Identity and task-seeded shuffles 1-4 were asked with no local decision replay. 95% intervals resample whole task clusters 2,000 times, seed 0.

Automatic gates passed: True. Excluded incomplete/error groups: 4, 4 of them refused by the model in at least one order.

| Decider | Identity P@1 | Shuffle mean | Shuffle min | Shuffle max | Five-order top-card stability | Pairwise agreement |
|---|---:|---:|---:|---:|---:|---:|
| luna-decisions | 0.318 (0.253 to 0.379) | 0.298 (0.241 to 0.354) | 0.268 (0.202 to 0.323) | 0.318 (0.263 to 0.384) | 0.303 (0.237 to 0.374) | 0.533 (0.483 to 0.581) |

The minimum and maximum are the smallest/largest population P@1 across the four shuffles; their intervals resample tasks before taking that extremum.
Selecting the maximum of noisy estimates biases it upward, and selecting the minimum biases it downward. Read these extrema alongside the same-order noise baseline, not as guaranteed gains or losses.

## Same-order noise baseline

Three same-order repeats from `20260929T162701Z`; reference git `57fd48a7ce7744bcacc08c6288790de42d03ecee`, dirty=True. Raw manifest and records SHA256 values, dataset/task identities and per-decider configurations are retained in summary.json. The reference dirty state is disclosed, not reconstructed.

Observed same-order run noise, not a causal order-effect experiment. Compare pairwise agreement; all-three-same and all-five-same are different statistics.

| Decider | Matched positive tasks | Same-order pairwise agreement | Five-order pairwise agreement | Same-order P@1 min | Same-order P@1 max |
|---|---:|---:|---:|---:|---:|
| luna-decisions | unavailable: model/config, population, repeat completeness, or identity payload differs | n/a | n/a | n/a | n/a |

Intervals resample whole matched task clusters, including all three repeats; extrema are recomputed after resampling. Comparisons are observational across runs, not a causal order-effect test or a significance test of the difference. Pairwise agreement is the comparable metric; do not compare all-three-same directly with all-five-same.

## luna-decisions

P@1 by order: 0: 0.318 (0.253 to 0.379), 1: 0.303 (0.237 to 0.364), 2: 0.268 (0.202 to 0.328), 3: 0.303 (0.242 to 0.369), 4: 0.318 (0.253 to 0.384).

| Order | Correct | Wrong | Abstained | Negative share |
|---|---:|---:|---:|---:|
| 0 | 59 | 259 | 78 | 198/396 (50%) |
| 1 | 57 | 261 | 78 | 198/396 (50%) |
| 2 | 52 | 270 | 74 | 198/396 (50%) |
| 3 | 57 | 253 | 86 | 198/396 (50%) |
| 4 | 60 | 255 | 81 | 198/396 (50%) |

Answer/abstain stability: 0.803 (0.753 to 0.851), on the stated 50/50 mix above.

Shuffled chosen slots (Positives; outer candidate slot, reserved option ignored): 1: 128, 2: 68, 3: 73, 4: 50, 5: 49, 6: 48, 7: 38, 8: 36, 9: 28, 10: 37, 11: 37, 12: 25, 13: 17, 14: 31, 15: 17, 16: 26, 17: 29, 18: 18, 19: 22, 20: 15.

Slot 1: 0.162 (0.131 to 0.192) vs 5%; slots 1-3: 0.340 (0.299 to 0.381) vs 15%; descriptive chi-square 332.1919191919192 (19 degrees of freedom, no p-value).

Shuffled chosen slots (50/50 mix; outer candidate slot, reserved option ignored): 1: 254, 2: 145, 3: 151, 4: 101, 5: 92, 6: 101, 7: 80, 8: 74, 9: 61, 10: 72, 11: 68, 12: 55, 13: 36, 14: 55, 15: 35, 16: 50, 17: 57, 18: 34, 19: 35, 20: 28.

Slot 1: 0.160 (0.133 to 0.189) vs 5%; slots 1-3: 0.347 (0.312 to 0.384) vs 15%; descriptive chi-square 689.9646464646465 (19 degrees of freedom, no p-value).

Permutation averaging: P@1 0.323 (0.258 to 0.384); correct/wrong/abstained 63/254/79, 198/396 negatives (50%). Production requires five decisions per search; measured decision cost per 1,000 searches $0.4463; 5.065656565656566 physical asks per search for the five decisions.

Average final distributions mapped to card ids; cards dropped in a logprob round get zero; identity breaks ties.
It is a heuristic with a measured five-decision cost, not a free correction or a guaranteed improvement.

Identity drift: {"comparable": false, "reason": "model, prompt, limits, or configuration differs"}

Only this date, K=20, plain queries and this configuration are covered. The chosen-slot statistic is descriptive: repeated searches are correlated and uniform-position causality is not established. Actual per-round slot/card mappings are retained in the raw exchanges. Gold-removed negatives may still admit an unlabeled relevant tool.

Verified run cost $0.176739; guarded charge $0.196400. Model, prompt, payload, catalog, task, seed, order, git and scheduling provenance are pinned in summary.json and the raw manifest.

Calls are serialized and never locally replayed. Provider cache reads use their returned discount; unreported cache usage is charged at list price.
