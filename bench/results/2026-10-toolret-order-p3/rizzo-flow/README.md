# ToolRet candidate-order sensitivity

K=20, plain queries, rizzo-flow FULL, reserved option last. P@1 means a relevant tool first with abstention ignored; selecting it does not complete a task.

**rizzo-flow.** The authors' fine-tune, 4B at q8_0 on llama.cpp with Metal; quantization and hardware change its probabilities, by its authors' account.

Identity and task-seeded shuffles 1-4 were asked with no local decision replay. 95% intervals resample whole task clusters 2,000 times, seed 0.

Automatic gates passed: True. Excluded incomplete/error groups: 0.

| Decider | Identity P@1 | Shuffle mean | Shuffle min | Shuffle max | Five-order top-card stability | Pairwise agreement |
|---|---:|---:|---:|---:|---:|---:|
| rizzo-flow | 0.305 (0.240 to 0.365) | 0.265 (0.212 to 0.319) | 0.255 (0.190 to 0.300) | 0.280 (0.230 to 0.345) | 0.240 (0.185 to 0.300) | 0.469 (0.421 to 0.516) |

The minimum and maximum are the smallest/largest population P@1 across the four shuffles; their intervals resample tasks before taking that extremum.
Selecting the maximum of noisy estimates biases it upward, and selecting the minimum biases it downward. Read these extrema alongside the same-order noise baseline, not as guaranteed gains or losses.

## Same-order noise baseline

Three same-order repeats from `20260929T162701Z`; reference git `57fd48a7ce7744bcacc08c6288790de42d03ecee`, dirty=True. Raw manifest and records SHA256 values, dataset/task identities and per-decider configurations are retained in summary.json. The reference dirty state is disclosed, not reconstructed.

Observed same-order run noise, not a causal order-effect experiment. Compare pairwise agreement; all-three-same and all-five-same are different statistics.

| Decider | Matched positive tasks | Same-order pairwise agreement | Five-order pairwise agreement | Same-order P@1 min | Same-order P@1 max |
|---|---:|---:|---:|---:|---:|
| rizzo-flow | unavailable: model/config, population, repeat completeness, or identity payload differs | n/a | n/a | n/a | n/a |

Intervals resample whole matched task clusters, including all three repeats; extrema are recomputed after resampling. Comparisons are observational across runs, not a causal order-effect test or a significance test of the difference. Pairwise agreement is the comparable metric; do not compare all-three-same directly with all-five-same.

## rizzo-flow

P@1 by order: 0: 0.305 (0.240 to 0.365), 1: 0.255 (0.195 to 0.315), 2: 0.260 (0.200 to 0.320), 3: 0.265 (0.205 to 0.325), 4: 0.280 (0.220 to 0.345).

| Order | Correct | Wrong | Abstained | Negative share |
|---|---:|---:|---:|---:|
| 0 | 60 | 306 | 34 | 200/400 (50%) |
| 1 | 50 | 318 | 32 | 200/400 (50%) |
| 2 | 52 | 319 | 29 | 200/400 (50%) |
| 3 | 53 | 313 | 34 | 200/400 (50%) |
| 4 | 56 | 317 | 27 | 200/400 (50%) |

Answer/abstain stability: 0.902 (0.863 to 0.938), on the stated 50/50 mix above.

Shuffled chosen slots (Positives; outer candidate slot, reserved option ignored): 1: 109, 2: 76, 3: 77, 4: 63, 5: 47, 6: 39, 7: 39, 8: 29, 9: 39, 10: 36, 11: 45, 12: 23, 13: 23, 14: 28, 15: 31, 16: 30, 17: 24, 18: 15, 19: 15, 20: 12.

Slot 1: 0.136 (0.107 to 0.166) vs 5%; slots 1-3: 0.328 (0.291 to 0.366) vs 15%; descriptive chi-square 284.05 (19 degrees of freedom, no p-value).

Shuffled chosen slots (50/50 mix; outer candidate slot, reserved option ignored): 1: 226, 2: 156, 3: 147, 4: 125, 5: 100, 6: 84, 7: 77, 8: 68, 9: 85, 10: 72, 11: 85, 12: 52, 13: 49, 14: 52, 15: 54, 16: 56, 17: 44, 18: 26, 19: 23, 20: 19.

Slot 1: 0.141 (0.116 to 0.168) vs 5%; slots 1-3: 0.331 (0.297 to 0.365) vs 15%; descriptive chi-square 615.65 (19 degrees of freedom, no p-value).

Permutation averaging: P@1 0.305 (0.240 to 0.370); correct/wrong/abstained 61/311/28, 200/400 negatives (50%). Production requires five decisions per search; measured decision cost per 1,000 searches local; 5.0 physical asks per search for the five decisions.

Average final distributions mapped to card ids; cards dropped in a logprob round get zero; identity breaks ties.
It is a heuristic with a measured five-decision cost, not a free correction or a guaranteed improvement.

Identity drift: {"comparable": false, "reason": "model, prompt, limits, or configuration differs"}

Only this date, K=20, plain queries and this configuration are covered. The chosen-slot statistic is descriptive: repeated searches are correlated and uniform-position causality is not established. Actual per-round slot/card mappings are retained in the raw exchanges. Gold-removed negatives may still admit an unlabeled relevant tool.

Verified run cost local. Model, prompt, payload, catalog, task, seed, order, git and scheduling provenance are pinned in summary.json and the raw manifest.

Calls are serialized and never locally replayed. Provider cache reads use their returned discount; unreported cache usage is charged at list price.
