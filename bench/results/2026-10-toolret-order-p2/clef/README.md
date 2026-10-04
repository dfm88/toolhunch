# ToolRet candidate-order sensitivity

K=20, plain queries, Clef BRIEF, reserved option last. P@1 means a relevant tool first with abstention ignored; selecting it does not complete a task.

**Clef.** Workers AI serves the current Clef behind `@cf/cloudflare/clef`, and no version can be pinned: a later run may answer differently from the one reported here.

Identity and task-seeded shuffles 1-4 were asked with no local decision replay. 95% intervals resample whole task clusters 2,000 times, seed 0.

Automatic gates passed: True. Excluded incomplete/error groups: 0.

| Decider | Identity P@1 | Shuffle mean | Shuffle min | Shuffle max | Five-order top-card stability | Pairwise agreement |
|---|---:|---:|---:|---:|---:|---:|
| clef | 0.310 (0.245 to 0.370) | 0.309 (0.245 to 0.371) | 0.300 (0.235 to 0.360) | 0.315 (0.250 to 0.375) | 0.965 (0.940 to 0.990) | 0.985 (0.973 to 0.996) |

The minimum and maximum are the smallest/largest population P@1 across the four shuffles; their intervals resample tasks before taking that extremum.
Selecting the maximum of noisy estimates biases it upward, and selecting the minimum biases it downward. Read these extrema alongside the same-order noise baseline, not as guaranteed gains or losses.

## Same-order noise baseline

Three same-order repeats from `20260929T162701Z`; reference git `57fd48a7ce7744bcacc08c6288790de42d03ecee`, dirty=True. Raw manifest and records SHA256 values, dataset/task identities and per-decider configurations are retained in summary.json. The reference dirty state is disclosed, not reconstructed.

Observed same-order run noise, not a causal order-effect experiment. Compare pairwise agreement; all-three-same and all-five-same are different statistics.

| Decider | Matched positive tasks | Same-order pairwise agreement | Five-order pairwise agreement | Same-order P@1 min | Same-order P@1 max |
|---|---:|---:|---:|---:|---:|
| clef | unavailable: model/config, population, repeat completeness, or identity payload differs | n/a | n/a | n/a | n/a |

Intervals resample whole matched task clusters, including all three repeats; extrema are recomputed after resampling. Comparisons are observational across runs, not a causal order-effect test or a significance test of the difference. Pairwise agreement is the comparable metric; do not compare all-three-same directly with all-five-same.

## clef

P@1 by order: 0: 0.310 (0.245 to 0.370), 1: 0.300 (0.235 to 0.360), 2: 0.310 (0.245 to 0.370), 3: 0.310 (0.245 to 0.370), 4: 0.315 (0.250 to 0.375).

| Order | Correct | Wrong | Abstained | Negative share |
|---|---:|---:|---:|---:|
| 0 | 61 | 317 | 22 | 200/400 (50%) |
| 1 | 59 | 319 | 22 | 200/400 (50%) |
| 2 | 61 | 317 | 22 | 200/400 (50%) |
| 3 | 61 | 317 | 22 | 200/400 (50%) |
| 4 | 62 | 316 | 22 | 200/400 (50%) |

Answer/abstain stability: 1.000 (1.000 to 1.000), on the stated 50/50 mix above.

Shuffled chosen slots (Positives; outer candidate slot, reserved option ignored): 1: 39, 2: 39, 3: 43, 4: 42, 5: 38, 6: 41, 7: 45, 8: 37, 9: 28, 10: 46, 11: 36, 12: 43, 13: 28, 14: 44, 15: 45, 16: 47, 17: 46, 18: 34, 19: 40, 20: 39.

Slot 1: 0.049 (0.034 to 0.064) vs 5%; slots 1-3: 0.151 (0.126 to 0.177) vs 15%; descriptive chi-square 14.15 (19 degrees of freedom, no p-value).

Shuffled chosen slots (50/50 mix; outer candidate slot, reserved option ignored): 1: 77, 2: 83, 3: 86, 4: 88, 5: 78, 6: 86, 7: 88, 8: 78, 9: 62, 10: 85, 11: 75, 12: 84, 13: 58, 14: 84, 15: 87, 16: 94, 17: 86, 18: 65, 19: 83, 20: 73.

Slot 1: 0.048 (0.036 to 0.062) vs 5%; slots 1-3: 0.154 (0.131 to 0.176) vs 15%; descriptive chi-square 21.0 (19 degrees of freedom, no p-value).

Permutation averaging: P@1 0.310 (0.245 to 0.370); correct/wrong/abstained 61/317/22, 200/400 negatives (50%). Production requires five decisions per search; measured decision cost per 1,000 searches $1.3340; 5.0 physical asks per search for the five decisions.

Average final distributions mapped to card ids; cards dropped in a logprob round get zero; identity breaks ties.
It is a heuristic with a measured five-decision cost, not a free correction or a guaranteed improvement.

Identity drift: {"comparable": false, "reason": "model, prompt, limits, or configuration differs"}

Only this date, K=20, plain queries and this configuration are covered. The chosen-slot statistic is descriptive: repeated searches are correlated and uniform-position causality is not established. Actual per-round slot/card mappings are retained in the raw exchanges. Gold-removed negatives may still admit an unlabeled relevant tool.

Verified run cost $0.533593; guarded charge $0.533593. Model, prompt, payload, catalog, task, seed, order, git and scheduling provenance are pinned in summary.json and the raw manifest.

Calls are serialized and never locally replayed. Provider cache reads use their returned discount; unreported cache usage is charged at list price.
