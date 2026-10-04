# ToolRet candidate-order sensitivity

K=20, plain queries, Strands Decider 2B FULL, reserved option last. P@1 means a relevant tool first with abstention ignored; selecting it does not complete a task.

Identity and task-seeded shuffles 1-4 were asked with no local decision replay. 95% intervals resample whole task clusters 2,000 times, seed 0.

Automatic gates passed: True. Excluded incomplete/error groups: 0.

| Decider | Identity P@1 | Shuffle mean | Shuffle min | Shuffle max | Five-order top-card stability | Pairwise agreement |
|---|---:|---:|---:|---:|---:|---:|
| strands | 0.285 (0.225 to 0.345) | 0.276 (0.224 to 0.330) | 0.265 (0.200 to 0.310) | 0.300 (0.245 to 0.360) | 0.240 (0.185 to 0.300) | 0.502 (0.460 to 0.546) |

The minimum and maximum are the smallest/largest population P@1 across the four shuffles; their intervals resample tasks before taking that extremum.
Selecting the maximum of noisy estimates biases it upward, and selecting the minimum biases it downward. Read these extrema alongside the same-order noise baseline, not as guaranteed gains or losses.

## Same-order noise baseline

Three same-order repeats from `20260929T162701Z`; reference git `57fd48a7ce7744bcacc08c6288790de42d03ecee`, dirty=True. Raw manifest and records SHA256 values, dataset/task identities and per-decider configurations are retained in summary.json. The reference dirty state is disclosed, not reconstructed.

Observed same-order run noise, not a causal order-effect experiment. Compare pairwise agreement; all-three-same and all-five-same are different statistics.

| Decider | Matched positive tasks | Same-order pairwise agreement | Five-order pairwise agreement | Same-order P@1 min | Same-order P@1 max |
|---|---:|---:|---:|---:|---:|
| strands | unavailable: model/config, population, repeat completeness, or identity payload differs | n/a | n/a | n/a | n/a |

Intervals resample whole matched task clusters, including all three repeats; extrema are recomputed after resampling. Comparisons are observational across runs, not a causal order-effect test or a significance test of the difference. Pairwise agreement is the comparable metric; do not compare all-three-same directly with all-five-same.

## strands

P@1 by order: 0: 0.285 (0.225 to 0.345), 1: 0.275 (0.210 to 0.335), 2: 0.265 (0.205 to 0.325), 3: 0.265 (0.205 to 0.325), 4: 0.300 (0.235 to 0.360).

| Order | Correct | Wrong | Abstained | Negative share |
|---|---:|---:|---:|---:|
| 0 | 57 | 321 | 22 | 200/400 (50%) |
| 1 | 53 | 326 | 21 | 200/400 (50%) |
| 2 | 52 | 326 | 22 | 200/400 (50%) |
| 3 | 52 | 321 | 27 | 200/400 (50%) |
| 4 | 59 | 322 | 19 | 200/400 (50%) |

Answer/abstain stability: 0.945 (0.912 to 0.973), on the stated 50/50 mix above.

Shuffled chosen slots (Positives; outer candidate slot, reserved option ignored): 1: 64, 2: 53, 3: 55, 4: 49, 5: 42, 6: 43, 7: 43, 8: 43, 9: 36, 10: 45, 11: 34, 12: 36, 13: 24, 14: 41, 15: 39, 16: 33, 17: 30, 18: 34, 19: 26, 20: 30.

Slot 1: 0.080 (0.059 to 0.102) vs 5%; slots 1-3: 0.215 (0.184 to 0.246) vs 15%; descriptive chi-square 47.85 (19 degrees of freedom, no p-value).

Shuffled chosen slots (50/50 mix; outer candidate slot, reserved option ignored): 1: 116, 2: 108, 3: 105, 4: 100, 5: 77, 6: 86, 7: 82, 8: 95, 9: 84, 10: 84, 11: 73, 12: 74, 13: 58, 14: 81, 15: 80, 16: 61, 17: 62, 18: 66, 19: 51, 20: 57.

Slot 1: 0.072 (0.055 to 0.092) vs 5%; slots 1-3: 0.206 (0.177 to 0.234) vs 15%; descriptive chi-square 77.9 (19 degrees of freedom, no p-value).

Permutation averaging: P@1 0.315 (0.250 to 0.375); correct/wrong/abstained 62/314/24, 200/400 negatives (50%). Production requires five decisions per search; measured decision cost per 1,000 searches local; 5.0 physical asks per search for the five decisions.

Average final distributions mapped to card ids; cards dropped in a logprob round get zero; identity breaks ties.
It is a heuristic with a measured five-decision cost, not a free correction or a guaranteed improvement.

Identity drift: {"comparable": false, "reason": "model, prompt, limits, or configuration differs"}

Only this date, K=20, plain queries and this configuration are covered. The chosen-slot statistic is descriptive: repeated searches are correlated and uniform-position causality is not established. Actual per-round slot/card mappings are retained in the raw exchanges. Gold-removed negatives may still admit an unlabeled relevant tool.

Verified run cost local. Model, prompt, payload, catalog, task, seed, order, git and scheduling provenance are pinned in summary.json and the raw manifest.

Calls are serialized and never locally replayed. Provider cache reads use their returned discount; unreported cache usage is charged at list price.
