# ToolRet candidate-order sensitivity

K=20, plain queries, Clef-flash BRIEF, reserved option last. P@1 means a relevant tool first with abstention ignored; selecting it does not complete a task.

**Clef-flash.** Workers AI serves the current Clef-flash behind `@cf/cloudflare/clef-flash`, and no version can be pinned: a later run may answer differently from the one reported here.

Identity and task-seeded shuffles 1-4 were asked with no local decision replay. 95% intervals resample whole task clusters 2,000 times, seed 0.

Automatic gates passed: True. Excluded incomplete/error groups: 0.

| Decider | Identity P@1 | Shuffle mean | Shuffle min | Shuffle max | Five-order top-card stability | Pairwise agreement |
|---|---:|---:|---:|---:|---:|---:|
| clef-flash | 0.240 (0.180 to 0.300) | 0.239 (0.180 to 0.299) | 0.235 (0.180 to 0.295) | 0.240 (0.180 to 0.300) | 0.960 (0.930 to 0.985) | 0.980 (0.965 to 0.993) |

The minimum and maximum are the smallest/largest population P@1 across the four shuffles; their intervals resample tasks before taking that extremum.
Selecting the maximum of noisy estimates biases it upward, and selecting the minimum biases it downward. Read these extrema alongside the same-order noise baseline, not as guaranteed gains or losses.

## Same-order noise baseline

Three same-order repeats from `20260929T162701Z`; reference git `57fd48a7ce7744bcacc08c6288790de42d03ecee`, dirty=True. Raw manifest and records SHA256 values, dataset/task identities and per-decider configurations are retained in summary.json. The reference dirty state is disclosed, not reconstructed.

Observed same-order run noise, not a causal order-effect experiment. Compare pairwise agreement; all-three-same and all-five-same are different statistics.

| Decider | Matched positive tasks | Same-order pairwise agreement | Five-order pairwise agreement | Same-order P@1 min | Same-order P@1 max |
|---|---:|---:|---:|---:|---:|
| clef-flash | unavailable: model/config, population, repeat completeness, or identity payload differs | n/a | n/a | n/a | n/a |

Intervals resample whole matched task clusters, including all three repeats; extrema are recomputed after resampling. Comparisons are observational across runs, not a causal order-effect test or a significance test of the difference. Pairwise agreement is the comparable metric; do not compare all-three-same directly with all-five-same.

## clef-flash

P@1 by order: 0: 0.240 (0.180 to 0.300), 1: 0.235 (0.180 to 0.295), 2: 0.240 (0.180 to 0.300), 3: 0.240 (0.180 to 0.300), 4: 0.240 (0.180 to 0.300).

| Order | Correct | Wrong | Abstained | Negative share |
|---|---:|---:|---:|---:|
| 0 | 48 | 338 | 14 | 200/400 (50%) |
| 1 | 47 | 339 | 14 | 200/400 (50%) |
| 2 | 48 | 338 | 14 | 200/400 (50%) |
| 3 | 48 | 338 | 14 | 200/400 (50%) |
| 4 | 48 | 338 | 14 | 200/400 (50%) |

Answer/abstain stability: 1.000 (1.000 to 1.000), on the stated 50/50 mix above.

Shuffled chosen slots (Positives; outer candidate slot, reserved option ignored): 1: 37, 2: 40, 3: 41, 4: 40, 5: 41, 6: 31, 7: 41, 8: 34, 9: 41, 10: 55, 11: 38, 12: 35, 13: 35, 14: 34, 15: 47, 16: 38, 17: 49, 18: 41, 19: 35, 20: 47.

Slot 1: 0.046 (0.031 to 0.061) vs 5%; slots 1-3: 0.147 (0.122 to 0.172) vs 15%; descriptive chi-square 16.35 (19 degrees of freedom, no p-value).

Shuffled chosen slots (50/50 mix; outer candidate slot, reserved option ignored): 1: 77, 2: 81, 3: 82, 4: 83, 5: 83, 6: 68, 7: 82, 8: 72, 9: 85, 10: 105, 11: 80, 12: 75, 13: 76, 14: 67, 15: 90, 16: 72, 17: 94, 18: 70, 19: 70, 20: 88.

Slot 1: 0.048 (0.036 to 0.062) vs 5%; slots 1-3: 0.150 (0.129 to 0.172) vs 15%; descriptive chi-square 21.6 (19 degrees of freedom, no p-value).

Permutation averaging: P@1 0.240 (0.180 to 0.300); correct/wrong/abstained 48/338/14, 200/400 negatives (50%). Production requires five decisions per search; measured decision cost per 1,000 searches $0.5002; 5.0 physical asks per search for the five decisions.

Average final distributions mapped to card ids; cards dropped in a logprob round get zero; identity breaks ties.
It is a heuristic with a measured five-decision cost, not a free correction or a guaranteed improvement.

Identity drift: {"comparable": false, "reason": "model, prompt, limits, or configuration differs"}

Only this date, K=20, plain queries and this configuration are covered. The chosen-slot statistic is descriptive: repeated searches are correlated and uniform-position causality is not established. Actual per-round slot/card mappings are retained in the raw exchanges. Gold-removed negatives may still admit an unlabeled relevant tool.

Verified run cost $0.200097; guarded charge $0.200097. Model, prompt, payload, catalog, task, seed, order, git and scheduling provenance are pinned in summary.json and the raw manifest.

Calls are serialized and never locally replayed. Provider cache reads use their returned discount; unreported cache usage is charged at list price.
