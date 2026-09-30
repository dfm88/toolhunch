# ToolRet candidate-order sensitivity

K=20, plain queries, Jev BRIEF and logprob FULL, reserved option last. P@1 means a relevant tool first with abstention ignored; selecting it does not complete a task.

Identity and task-seeded shuffles 1-4 were asked with no local decision replay. 95% intervals resample whole task clusters 2,000 times, seed 0.

Automatic gates passed: True. Excluded incomplete/error groups: 0.

| Decider | Identity P@1 | Shuffle mean | Shuffle min | Shuffle max | Five-order top-card stability | Pairwise agreement |
|---|---:|---:|---:|---:|---:|---:|
| jev | 0.305 (0.240 to 0.370) | 0.312 (0.251 to 0.371) | 0.305 (0.235 to 0.355) | 0.325 (0.265 to 0.395) | 0.575 (0.505 to 0.640) | 0.765 (0.723 to 0.804) |
| logprob | 0.330 (0.265 to 0.395) | 0.290 (0.234 to 0.346) | 0.270 (0.210 to 0.325) | 0.305 (0.255 to 0.375) | 0.390 (0.325 to 0.455) | 0.605 (0.556 to 0.651) |

The minimum and maximum are the smallest/largest population P@1 across the four shuffles; their intervals resample tasks before taking that extremum.
Selecting the maximum of noisy estimates biases it upward, and selecting the minimum biases it downward. Read these extrema alongside the same-order noise baseline, not as guaranteed gains or losses.

## Same-order noise baseline

Three same-order repeats from `20260929T162701Z`; reference git `57fd48a7ce7744bcacc08c6288790de42d03ecee`, dirty=True. Raw manifest and records SHA256 values, dataset/task identities and per-decider configurations are retained in summary.json. The reference dirty state is disclosed, not reconstructed.

Observed same-order run noise, not a causal order-effect experiment. Compare pairwise agreement; all-three-same and all-five-same are different statistics.

| Decider | Matched positive tasks | Same-order pairwise agreement | Five-order pairwise agreement | Same-order P@1 min | Same-order P@1 max |
|---|---:|---:|---:|---:|---:|
| jev | 200/200 | 0.960 (0.937 to 0.980) | 0.765 (0.723 to 0.804) | 0.305 (0.235 to 0.360) | 0.310 (0.250 to 0.375) |
| logprob | 200/200 | 0.980 (0.963 to 0.993) | 0.605 (0.556 to 0.651) | 0.320 (0.255 to 0.385) | 0.330 (0.265 to 0.395) |

jev same-order P@1 by repeat: 0: 0.305 (0.240 to 0.365), 1: 0.310 (0.245 to 0.370), 2: 0.305 (0.240 to 0.370).

logprob same-order P@1 by repeat: 0: 0.320 (0.255 to 0.385), 1: 0.325 (0.260 to 0.390), 2: 0.330 (0.265 to 0.395).

Intervals resample whole matched task clusters, including all three repeats; extrema are recomputed after resampling. Comparisons are observational across runs, not a causal order-effect test or a significance test of the difference. Pairwise agreement is the comparable metric; do not compare all-three-same directly with all-five-same.

## jev

P@1 by order: 0: 0.305 (0.240 to 0.370), 1: 0.305 (0.240 to 0.365), 2: 0.310 (0.245 to 0.375), 3: 0.310 (0.245 to 0.375), 4: 0.325 (0.260 to 0.390).

| Order | Correct | Wrong | Abstained | Negative share |
|---|---:|---:|---:|---:|
| 0 | 61 | 268 | 71 | 200/400 (50%) |
| 1 | 61 | 271 | 68 | 200/400 (50%) |
| 2 | 61 | 264 | 75 | 200/400 (50%) |
| 3 | 62 | 267 | 71 | 200/400 (50%) |
| 4 | 65 | 262 | 73 | 200/400 (50%) |

Answer/abstain stability: 0.930 (0.897 to 0.960), on the stated 50/50 mix above.

Shuffled chosen slots (Positives; outer candidate slot, reserved option ignored): 1: 58, 2: 48, 3: 40, 4: 36, 5: 44, 6: 42, 7: 39, 8: 33, 9: 23, 10: 50, 11: 50, 12: 46, 13: 36, 14: 41, 15: 33, 16: 37, 17: 42, 18: 34, 19: 35, 20: 33.

Slot 1: 0.072 (0.052 to 0.094) vs 5%; slots 1-3: 0.182 (0.155 to 0.211) vs 15%; descriptive chi-square 29.7 (19 degrees of freedom, no p-value).

Shuffled chosen slots (50/50 mix; outer candidate slot, reserved option ignored): 1: 112, 2: 94, 3: 81, 4: 79, 5: 88, 6: 81, 7: 84, 8: 81, 9: 57, 10: 85, 11: 96, 12: 85, 13: 72, 14: 88, 15: 65, 16: 75, 17: 76, 18: 64, 19: 71, 20: 66.

Slot 1: 0.070 (0.053 to 0.089) vs 5%; slots 1-3: 0.179 (0.156 to 0.203) vs 15%; descriptive chi-square 38.325 (19 degrees of freedom, no p-value).

Permutation averaging: P@1 0.315 (0.250 to 0.375); correct/wrong/abstained 63/263/74, 200/400 negatives (50%). Production requires five decisions per search; measured decision cost per 1,000 searches $0.2689; 5.0 physical asks per search for the five decisions.

Average final distributions mapped to card ids; cards dropped in a logprob round get zero; identity breaks ties.
Logprob finalists can differ across permutations: this averages different final questions, not jointly comparable logits over all twenty candidates. It is a heuristic with a measured five-decision cost, not a free correction or a guaranteed improvement.

Identity drift: {"comparable": true, "reference_run": "20260929T153251Z", "published_p_at_1": 0.32, "identity_minus_published": {"value": -0.015, "ci95": [-0.03, 0.0], "tasks": 200}, "caveat": "Identity drift is a date/run diagnostic and cannot be attributed to order."}

## logprob

P@1 by order: 0: 0.330 (0.265 to 0.395), 1: 0.295 (0.235 to 0.360), 2: 0.290 (0.225 to 0.355), 3: 0.305 (0.245 to 0.370), 4: 0.270 (0.210 to 0.330).

| Order | Correct | Wrong | Abstained | Negative share |
|---|---:|---:|---:|---:|
| 0 | 61 | 272 | 67 | 200/400 (50%) |
| 1 | 54 | 279 | 67 | 200/400 (50%) |
| 2 | 56 | 278 | 66 | 200/400 (50%) |
| 3 | 56 | 272 | 72 | 200/400 (50%) |
| 4 | 51 | 285 | 64 | 200/400 (50%) |

Answer/abstain stability: 0.830 (0.780 to 0.877), on the stated 50/50 mix above.

Shuffled chosen slots (Positives; outer candidate slot, reserved option ignored): 1: 69, 2: 56, 3: 61, 4: 48, 5: 47, 6: 43, 7: 46, 8: 39, 9: 25, 10: 43, 11: 48, 12: 33, 13: 27, 14: 37, 15: 31, 16: 32, 17: 26, 18: 27, 19: 34, 20: 28.

Slot 1: 0.086 (0.066 to 0.109) vs 5%; slots 1-3: 0.233 (0.200 to 0.265) vs 15%; descriptive chi-square 72.8 (19 degrees of freedom, no p-value).

Shuffled chosen slots (50/50 mix; outer candidate slot, reserved option ignored): 1: 138, 2: 114, 3: 129, 4: 98, 5: 102, 6: 80, 7: 92, 8: 75, 9: 58, 10: 86, 11: 84, 12: 67, 13: 61, 14: 69, 15: 59, 16: 72, 17: 56, 18: 49, 19: 60, 20: 51.

Slot 1: 0.086 (0.069 to 0.105) vs 5%; slots 1-3: 0.238 (0.211 to 0.267) vs 15%; descriptive chi-square 154.6 (19 degrees of freedom, no p-value).

Permutation averaging: P@1 0.310 (0.250 to 0.370); correct/wrong/abstained 58/271/71, 200/400 negatives (50%). Production requires five decisions per search; measured decision cost per 1,000 searches $2.5373; 10.0 physical asks per search for the five decisions.

Average final distributions mapped to card ids; cards dropped in a logprob round get zero; identity breaks ties.
Logprob finalists can differ across permutations: this averages different final questions, not jointly comparable logits over all twenty candidates. It is a heuristic with a measured five-decision cost, not a free correction or a guaranteed improvement.

Identity drift: {"comparable": true, "reference_run": "20260929T153251Z", "published_p_at_1": 0.33, "identity_minus_published": {"value": 0.0, "ci95": [-0.015, 0.015], "tasks": 200}, "caveat": "Identity drift is a date/run diagnostic and cannot be attributed to order."}

Only this date, K=20, plain queries and these two configurations are covered. The chosen-slot statistic is descriptive: repeated searches are correlated and uniform-position causality is not established. Logprob final questions contain finalists; actual per-round slot/card mappings are retained in the raw exchanges. Gold-removed negatives may still admit an unlabeled relevant tool.

Verified run cost $1.122490; guarded charge $1.122490. Model, prompt, payload, catalog, task, seed, order, git and scheduling provenance are pinned in summary.json and the raw manifest.

Calls are serialized and never locally replayed. Provider cache reads use their returned discount; unreported cache usage is charged at list price. Jev cache usage is unmeasured.

Not affiliated with TypeSafe, OpenAI or Pydantic.
