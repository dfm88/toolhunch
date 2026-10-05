# ToolRet candidate-order sensitivity

K=20, plain queries, Laya wide BRIEF, reserved option last. P@1 means a relevant tool first with abstention ignored; selecting it does not complete a task.

**Laya wide.** Its authors describe the base checkpoints as a fast base to specialise, not a zero-shot decision engine; we run the English one zero-shot. Its questions are planned with Laya's own tokenizer, and an option whose name would repeat its key past 48 tokens is sent as its key alone. It runs with a 512-token option budget in a 1,024-token window, beyond the 192 and 512 the checkpoint ships with, as its model card advises for many options.

Identity and task-seeded shuffles 1-4 were asked with no local decision replay. 95% intervals resample whole task clusters 2,000 times, seed 0.

Automatic gates passed: True. Excluded incomplete/error groups: 0.

| Decider | Identity P@1 | Shuffle mean | Shuffle min | Shuffle max | Five-order top-card stability | Pairwise agreement |
|---|---:|---:|---:|---:|---:|---:|
| laya-wide | 0.200 (0.150 to 0.255) | 0.186 (0.139 to 0.233) | 0.170 (0.120 to 0.215) | 0.195 (0.155 to 0.255) | 0.260 (0.200 to 0.320) | 0.522 (0.477 to 0.567) |

The minimum and maximum are the smallest/largest population P@1 across the four shuffles; their intervals resample tasks before taking that extremum.
Selecting the maximum of noisy estimates biases it upward, and selecting the minimum biases it downward. Read these extrema alongside the same-order noise baseline, not as guaranteed gains or losses.

## Same-order noise baseline

Three same-order repeats from `20260929T162701Z`; reference git `57fd48a7ce7744bcacc08c6288790de42d03ecee`, dirty=True. Raw manifest and records SHA256 values, dataset/task identities and per-decider configurations are retained in summary.json. The reference dirty state is disclosed, not reconstructed.

Observed same-order run noise, not a causal order-effect experiment. Compare pairwise agreement; all-three-same and all-five-same are different statistics.

| Decider | Matched positive tasks | Same-order pairwise agreement | Five-order pairwise agreement | Same-order P@1 min | Same-order P@1 max |
|---|---:|---:|---:|---:|---:|
| laya-wide | unavailable: model/config, population, repeat completeness, or identity payload differs | n/a | n/a | n/a | n/a |

Intervals resample whole matched task clusters, including all three repeats; extrema are recomputed after resampling. Comparisons are observational across runs, not a causal order-effect test or a significance test of the difference. Pairwise agreement is the comparable metric; do not compare all-three-same directly with all-five-same.

## laya-wide

P@1 by order: 0: 0.200 (0.150 to 0.255), 1: 0.190 (0.140 to 0.245), 2: 0.170 (0.120 to 0.220), 3: 0.190 (0.135 to 0.240), 4: 0.195 (0.140 to 0.250).

| Order | Correct | Wrong | Abstained | Negative share |
|---|---:|---:|---:|---:|
| 0 | 39 | 301 | 60 | 200/400 (50%) |
| 1 | 36 | 303 | 61 | 200/400 (50%) |
| 2 | 33 | 297 | 70 | 200/400 (50%) |
| 3 | 37 | 302 | 61 | 200/400 (50%) |
| 4 | 38 | 303 | 59 | 200/400 (50%) |

Answer/abstain stability: 0.723 (0.662 to 0.775), on the stated 50/50 mix above.

Shuffled chosen slots (Positives; outer candidate slot, reserved option ignored): 1: 69, 2: 53, 3: 54, 4: 52, 5: 51, 6: 53, 7: 47, 8: 46, 9: 35, 10: 40, 11: 36, 12: 39, 13: 29, 14: 26, 15: 30, 16: 30, 17: 27, 18: 26, 19: 27, 20: 30.

Slot 1: 0.086 (0.068 to 0.107) vs 5%; slots 1-3: 0.220 (0.193 to 0.250) vs 15%; descriptive chi-square 72.95 (19 degrees of freedom, no p-value).

Shuffled chosen slots (50/50 mix; outer candidate slot, reserved option ignored): 1: 149, 2: 101, 3: 108, 4: 107, 5: 98, 6: 101, 7: 88, 8: 88, 9: 72, 10: 80, 11: 78, 12: 70, 13: 65, 14: 61, 15: 51, 16: 62, 17: 60, 18: 48, 19: 53, 20: 60.

Slot 1: 0.093 (0.075 to 0.112) vs 5%; slots 1-3: 0.224 (0.198 to 0.250) vs 15%; descriptive chi-square 151.0 (19 degrees of freedom, no p-value).

Permutation averaging: P@1 0.205 (0.150 to 0.260); correct/wrong/abstained 40/285/75, 200/400 negatives (50%). Production requires five decisions per search; measured decision cost per 1,000 searches local; 14.95 physical asks per search for the five decisions.

Average final distributions mapped to card ids; cards dropped in a logprob round get zero; identity breaks ties.
It is a heuristic with a measured five-decision cost, not a free correction or a guaranteed improvement.

Identity drift: {"comparable": false, "reason": "model, prompt, limits, or configuration differs"}

Only this date, K=20, plain queries and this configuration are covered. The chosen-slot statistic is descriptive: repeated searches are correlated and uniform-position causality is not established. Actual per-round slot/card mappings are retained in the raw exchanges. Gold-removed negatives may still admit an unlabeled relevant tool.

Verified run cost local. Model, prompt, payload, catalog, task, seed, order, git and scheduling provenance are pinned in summary.json and the raw manifest.

Calls are serialized and never locally replayed. Provider cache reads use their returned discount; unreported cache usage is charged at list price.
