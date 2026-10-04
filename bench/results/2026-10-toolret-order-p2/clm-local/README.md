# ToolRet candidate-order sensitivity

K=20, plain queries, CLM (Mac bf16 MPS) BRIEF, reserved option last. P@1 means a relevant tool first with abstention ignored; selecting it does not complete a task.

**CLM (Mac bf16 MPS).** We run CLM on a Mac (transformers, bf16, MPS) instead of vLLM on CUDA. Its parity with our Modal deployment was checked on what reports publish, P@1 per cell and the answer-or-abstain decision (`bench/results/2026-10-toolret-decision-p2/clm-parity.json`); its figures stay provisional until the CLM authors confirm parity. `clm-serve` keeps the vectors of the texts it has embedded (its action cache, on by default), and our arms ask the same requests and cards more than once, so its latency here is mostly a warm-cache latency.

Identity and task-seeded shuffles 1-4 were asked with no local decision replay. 95% intervals resample whole task clusters 2,000 times, seed 0.

Automatic gates passed: True. Excluded incomplete/error groups: 0.

| Decider | Identity P@1 | Shuffle mean | Shuffle min | Shuffle max | Five-order top-card stability | Pairwise agreement |
|---|---:|---:|---:|---:|---:|---:|
| clm-local | 0.130 (0.085 to 0.175) | 0.131 (0.087 to 0.179) | 0.130 (0.085 to 0.175) | 0.135 (0.090 to 0.185) | 0.985 (0.965 to 1.000) | 0.993 (0.984 to 1.000) |

The minimum and maximum are the smallest/largest population P@1 across the four shuffles; their intervals resample tasks before taking that extremum.
Selecting the maximum of noisy estimates biases it upward, and selecting the minimum biases it downward. Read these extrema alongside the same-order noise baseline, not as guaranteed gains or losses.

## Same-order noise baseline

Three same-order repeats from `20260929T162701Z`; reference git `57fd48a7ce7744bcacc08c6288790de42d03ecee`, dirty=True. Raw manifest and records SHA256 values, dataset/task identities and per-decider configurations are retained in summary.json. The reference dirty state is disclosed, not reconstructed.

Observed same-order run noise, not a causal order-effect experiment. Compare pairwise agreement; all-three-same and all-five-same are different statistics.

| Decider | Matched positive tasks | Same-order pairwise agreement | Five-order pairwise agreement | Same-order P@1 min | Same-order P@1 max |
|---|---:|---:|---:|---:|---:|
| clm-local | unavailable: model/config, population, repeat completeness, or identity payload differs | n/a | n/a | n/a | n/a |

Intervals resample whole matched task clusters, including all three repeats; extrema are recomputed after resampling. Comparisons are observational across runs, not a causal order-effect test or a significance test of the difference. Pairwise agreement is the comparable metric; do not compare all-three-same directly with all-five-same.

## clm-local

P@1 by order: 0: 0.130 (0.085 to 0.175), 1: 0.130 (0.085 to 0.175), 2: 0.130 (0.085 to 0.175), 3: 0.130 (0.085 to 0.175), 4: 0.135 (0.090 to 0.185).

| Order | Correct | Wrong | Abstained | Negative share |
|---|---:|---:|---:|---:|
| 0 | 24 | 351 | 25 | 200/400 (50%) |
| 1 | 24 | 351 | 25 | 200/400 (50%) |
| 2 | 24 | 351 | 25 | 200/400 (50%) |
| 3 | 24 | 351 | 25 | 200/400 (50%) |
| 4 | 25 | 350 | 25 | 200/400 (50%) |

Answer/abstain stability: 1.000 (1.000 to 1.000), on the stated 50/50 mix above.

Shuffled chosen slots (Positives; outer candidate slot, reserved option ignored): 1: 39, 2: 38, 3: 37, 4: 40, 5: 47, 6: 33, 7: 53, 8: 40, 9: 39, 10: 37, 11: 38, 12: 46, 13: 33, 14: 31, 15: 41, 16: 28, 17: 53, 18: 36, 19: 41, 20: 50.

Slot 1: 0.049 (0.034 to 0.064) vs 5%; slots 1-3: 0.142 (0.120 to 0.166) vs 15%; descriptive chi-square 22.3 (19 degrees of freedom, no p-value).

Shuffled chosen slots (50/50 mix; outer candidate slot, reserved option ignored): 1: 71, 2: 67, 3: 75, 4: 77, 5: 85, 6: 68, 7: 98, 8: 77, 9: 84, 10: 76, 11: 74, 12: 91, 13: 73, 14: 69, 15: 94, 16: 67, 17: 102, 18: 78, 19: 79, 20: 95.

Slot 1: 0.044 (0.033 to 0.056) vs 5%; slots 1-3: 0.133 (0.115 to 0.152) vs 15%; descriptive chi-square 27.8 (19 degrees of freedom, no p-value).

Permutation averaging: P@1 0.130 (0.085 to 0.175); correct/wrong/abstained 24/351/25, 200/400 negatives (50%). Production requires five decisions per search; measured decision cost per 1,000 searches local; 5.0 physical asks per search for the five decisions.

Average final distributions mapped to card ids; cards dropped in a logprob round get zero; identity breaks ties.
It is a heuristic with a measured five-decision cost, not a free correction or a guaranteed improvement.

Identity drift: {"comparable": false, "reason": "model, prompt, limits, or configuration differs"}

Only this date, K=20, plain queries and this configuration are covered. The chosen-slot statistic is descriptive: repeated searches are correlated and uniform-position causality is not established. Actual per-round slot/card mappings are retained in the raw exchanges. Gold-removed negatives may still admit an unlabeled relevant tool.

Verified run cost local. Model, prompt, payload, catalog, task, seed, order, git and scheduling provenance are pinned in summary.json and the raw manifest.

Calls are serialized and never locally replayed. Provider cache reads use their returned discount; unreported cache usage is charged at list price.
