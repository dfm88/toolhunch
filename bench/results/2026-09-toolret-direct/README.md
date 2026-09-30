# ToolRet direct choice

Single-turn selection of a relevant tool from real source catalogs.

Primary pooled comparison uses the same common catalogs in every arm: webtools_spotify, webtools_tmdb, tooleyes, apibank.
Secondary all_catalogs rows include every selected catalog and are not a five-arm comparison.
Original run completed: True; original stop reason: none. Applicability does not change this historical status.

95% intervals resample tasks together, 2,000 resamples, seed 0. Abstentions and errors are misses in the positive-request headline. None-option metrics use parsed requests; errors are listed separately.

| Arm | Catalog | Relevant picks / positives | Paired Δ vs hybrid | Correct / wrong / abstained | Negatives | Errors |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | pooled | 0.545 (0.486 to 0.600) | — | 158 / 277 / 0 | 145/435 (33.3%) | 0 |
| hybrid@20 | all_catalogs | 0.524 (0.480 to 0.567) | — | 257 / 478 / 0 | 245/735 (33.3%) | 0 |
| hybrid@20 | webtools_spotify | 0.650 (0.500 to 0.800) | — | 26 / 34 / 0 | 20/60 (33.3%) | 0 |
| hybrid@20 | webtools_tmdb | 0.500 (0.370 to 0.630) | — | 27 / 54 / 0 | 27/81 (33.3%) | 0 |
| hybrid@20 | tooleyes | 0.453 (0.358 to 0.558) | — | 43 / 100 / 0 | 48/143 (33.6%) | 0 |
| hybrid@20 | apibank | 0.614 (0.515 to 0.703) | — | 62 / 89 / 0 | 50/151 (33.1%) | 0 |
| hybrid@20 | metatool_which | 0.495 (0.430 to 0.560) | — | 99 / 201 / 0 | 100/300 (33.3%) | 0 |
| hybrid@20+jev | pooled | 0.714 (0.659 to 0.766) | 0.169 (0.107 to 0.231) | 207 / 122 / 106 | 145/435 (33.3%) | 0 |
| hybrid@20+jev | all_catalogs | 0.690 (0.645 to 0.729) | 0.165 (0.118 to 0.210) | 338 / 225 / 172 | 245/735 (33.3%) | 0 |
| hybrid@20+jev | webtools_spotify | 0.900 (0.800 to 0.975) | 0.250 (0.075 to 0.425) | 36 / 11 / 13 | 20/60 (33.3%) | 0 |
| hybrid@20+jev | webtools_tmdb | 0.870 (0.778 to 0.963) | 0.370 (0.222 to 0.519) | 47 / 29 / 5 | 27/81 (33.3%) | 0 |
| hybrid@20+jev | tooleyes | 0.505 (0.400 to 0.600) | 0.053 (-0.053 to 0.147) | 48 / 40 / 55 | 48/143 (33.6%) | 0 |
| hybrid@20+jev | apibank | 0.752 (0.663 to 0.832) | 0.139 (0.040 to 0.238) | 76 / 42 / 33 | 50/151 (33.1%) | 0 |
| hybrid@20+jev | metatool_which | 0.655 (0.590 to 0.720) | 0.160 (0.095 to 0.230) | 131 / 103 / 66 | 100/300 (33.3%) | 0 |
| jev-all | pooled | 0.741 (0.690 to 0.790) | 0.197 (0.138 to 0.259) | 215 / 121 / 99 | 145/435 (33.3%) | 0 |
| jev-all | all_catalogs | 0.733 (0.692 to 0.771) | 0.208 (0.161 to 0.257) | 359 / 270 / 106 | 245/735 (33.3%) | 0 |
| jev-all | webtools_spotify | 0.975 (0.925 to 1.000) | 0.325 (0.175 to 0.475) | 39 / 12 / 9 | 20/60 (33.3%) | 0 |
| jev-all | webtools_tmdb | 0.944 (0.870 to 1.000) | 0.444 (0.296 to 0.574) | 51 / 27 / 3 | 27/81 (33.3%) | 0 |
| jev-all | tooleyes | 0.526 (0.421 to 0.632) | 0.074 (-0.032 to 0.168) | 50 / 41 / 52 | 48/143 (33.6%) | 0 |
| jev-all | apibank | 0.743 (0.663 to 0.832) | 0.129 (0.030 to 0.228) | 75 / 41 / 35 | 50/151 (33.1%) | 0 |
| jev-all | metatool_which | 0.720 (0.655 to 0.780) | 0.225 (0.145 to 0.305) | 144 / 149 / 7 | 100/300 (33.3%) | 0 |
| agent@20 | pooled | 0.617 (0.559 to 0.669) | 0.072 (0.000 to 0.145) | 179 / 145 / 111 | 145/435 (33.3%) | 0 |
| agent@20 | all_catalogs | 0.610 (0.567 to 0.653) | 0.086 (0.035 to 0.137) | 299 / 258 / 178 | 245/735 (33.3%) | 0 |
| agent@20 | webtools_spotify | 0.825 (0.700 to 0.925) | 0.175 (0.000 to 0.350) | 33 / 23 / 4 | 20/60 (33.3%) | 0 |
| agent@20 | webtools_tmdb | 0.870 (0.778 to 0.944) | 0.370 (0.241 to 0.500) | 47 / 33 / 1 | 27/81 (33.3%) | 0 |
| agent@20 | tooleyes | 0.674 (0.579 to 0.768) | 0.221 (0.126 to 0.316) | 64 / 49 / 30 | 48/143 (33.6%) | 0 |
| agent@20 | apibank | 0.347 (0.257 to 0.446) | -0.267 (-0.406 to -0.129) | 35 / 40 / 76 | 50/151 (33.1%) | 0 |
| agent@20 | metatool_which | 0.600 (0.530 to 0.665) | 0.105 (0.035 to 0.170) | 120 / 113 / 67 | 100/300 (33.3%) | 0 |
| agent-all | pooled | 0.607 (0.552 to 0.662) | 0.062 (-0.014 to 0.135) | 176 / 120 / 139 | 145/435 (33.3%) | 0 |
| agent-all | webtools_spotify | 0.800 (0.675 to 0.925) | 0.150 (0.000 to 0.300) | 32 / 21 / 7 | 20/60 (33.3%) | 0 |
| agent-all | webtools_tmdb | 0.963 (0.907 to 1.000) | 0.463 (0.333 to 0.593) | 52 / 28 / 1 | 27/81 (33.3%) | 0 |
| agent-all | tooleyes | 0.611 (0.516 to 0.705) | 0.158 (0.053 to 0.263) | 58 / 49 / 36 | 48/143 (33.6%) | 0 |
| agent-all | apibank | 0.337 (0.248 to 0.426) | -0.277 (-0.416 to -0.129) | 34 / 22 / 95 | 50/151 (33.1%) | 0 |
| agent-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | 0 rejected attempts, excluded |

## Observed provider cost

Replay responses do not contribute usage or timing. Search cost and latency are included for strategies that search. Negatives are priced separately.

| Arm | Catalog | Cold billed | Warm billed / 1,000 | Warm list / 1,000 | Warm cache share | Warm latency p50 / p95 ms | Negative billed / 1,000 |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid@20 | pooled | $0.0000 | $0.0002 | $0.0002 | n/a | 160.2 / 252.8 | $0.0000 |
| hybrid@20 | all_catalogs | $0.0000 | $0.0004 | $0.0004 | n/a | 162.5 / 253.6 | $0.0000 |
| hybrid@20 | webtools_spotify | $0.0000 | $0.0001 | $0.0001 | n/a | 154.4 / 370.6 | $0.0000 |
| hybrid@20 | webtools_tmdb | $0.0000 | $0.0002 | $0.0002 | n/a | 164.9 / 345.2 | $0.0000 |
| hybrid@20 | tooleyes | $0.0000 | $0.0003 | $0.0003 | n/a | 159.8 / 219.6 | $0.0000 |
| hybrid@20 | apibank | $0.0000 | $0.0003 | $0.0003 | n/a | 160.7 / 221.6 | $0.0000 |
| hybrid@20 | metatool_which | $0.0000 | $0.0005 | $0.0005 | n/a | 166.3 / 239.2 | $0.0000 |
| hybrid@20+jev | pooled | n/a | $0.0828 | $0.0828 | n/a | 449.2 / 630.5 | $0.0833 |
| hybrid@20+jev | all_catalogs | n/a | $0.0675 | $0.0675 | n/a | 451.8 / 629.8 | $0.0687 |
| hybrid@20+jev | webtools_spotify | n/a | $0.0603 | $0.0603 | n/a | 471.1 / 1258.0 | $0.0613 |
| hybrid@20+jev | webtools_tmdb | n/a | $0.0881 | $0.0881 | n/a | 471.4 / 701.8 | $0.0890 |
| hybrid@20+jev | tooleyes | n/a | $0.0822 | $0.0822 | n/a | 432.0 / 575.2 | $0.0827 |
| hybrid@20+jev | apibank | n/a | $0.0883 | $0.0883 | n/a | 453.3 / 597.5 | $0.0883 |
| hybrid@20+jev | metatool_which | n/a | $0.0474 | $0.0474 | n/a | 457.8 / 609.2 | $0.0467 |
| jev-all | pooled | n/a | $0.2807 | $0.2807 | n/a | 323.4 / 439.3 | $0.2752 |
| jev-all | all_catalogs | n/a | $0.2987 | $0.2987 | n/a | 325.1 / 421.8 | $0.2946 |
| jev-all | webtools_spotify | n/a | $0.1091 | $0.1091 | n/a | 306.4 / 461.3 | $0.1026 |
| jev-all | webtools_tmdb | n/a | $0.2004 | $0.2004 | n/a | 314.3 / 349.9 | $0.1929 |
| jev-all | tooleyes | n/a | $0.2958 | $0.2958 | n/a | 309.0 / 362.9 | $0.2933 |
| jev-all | apibank | n/a | $0.3621 | $0.3621 | n/a | 342.3 / 457.8 | $0.3556 |
| jev-all | metatool_which | n/a | $0.3222 | $0.3222 | n/a | 327.2 / 389.1 | $0.3202 |
| agent@20 | pooled | n/a | $0.6804 | $0.6804 | 0.0% | 936.3 / 1999.8 | $0.6934 |
| agent@20 | all_catalogs | n/a | $0.5631 | $0.5631 | 0.0% | 903.9 / 2232.5 | $0.5770 |
| agent@20 | webtools_spotify | n/a | $0.4679 | $0.4679 | 0.0% | 910.2 / 2710.5 | $0.4816 |
| agent@20 | webtools_tmdb | n/a | $0.7636 | $0.7636 | 0.0% | 854.0 / 1610.7 | $0.7769 |
| agent@20 | tooleyes | n/a | $0.6836 | $0.6836 | 0.0% | 951.7 / 2023.5 | $0.6875 |
| agent@20 | apibank | n/a | $0.7074 | $0.7074 | 0.0% | 947.3 / 1834.4 | $0.7279 |
| agent@20 | metatool_which | n/a | $0.4087 | $0.4087 | 0.0% | 863.0 / 2335.3 | $0.4096 |
| agent-all | pooled | n/a | $0.7856 | $2.4286 | 92.1% | 861.0 / 1932.3 | $1.6857 |
| agent-all | webtools_spotify | n/a | $0.3396 | $0.9130 | 86.4% | 689.4 / 846.5 | $0.8201 |
| agent-all | webtools_tmdb | n/a | $0.6362 | $1.8528 | 88.9% | 667.4 / 1069.0 | $1.5467 |
| agent-all | tooleyes | n/a | $0.8512 | $2.5855 | 91.8% | 853.6 / 2239.1 | $1.6383 |
| agent-all | apibank | n/a | $0.9435 | $3.0602 | 93.8% | 999.4 / 1632.8 | $2.0875 |
| agent-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | n/a | n/a |

## Non-applicable pairs and historical rejections

- agent-all / metatool_which: OpenAI Chat Completions rejected 200 function tools: HTTP 400 array_above_max_length, param tools (4/4 attempts); exact maximum not established. Evidence: bench/runs/20260929T221520Z-direct-pilot/run.jsonl (2026-09-30); 0 historical requests excluded, 0 rejected attempts.

| Arm | Catalog | Historical phase | Rejected provider attempts | Billed | List | Billed / 1,000 | List / 1,000 |
|---|---|---|---:|---:|---:|---:|---:|

## Caveats

- The metric is selecting a relevant tool, not completing a task; tasks can list several relevant tools.
- Negatives remove the task's labeled relevant tools; an unlabeled alternative may still serve it.
- Pooled none-option metrics state their observed positive/negative mix; errors are reported separately.
- Replay outcomes are scored, but their historical usage, latency and cache reads are excluded.
- Search overhead is attributed to each strategy that searches; physical search calls are paid once.
- Cold is the first scored positive; a replay there provides no new cold measurement.
- Caching reflects one provider's routing in one single-turn run and does not isolate a latency effect.

## Reproduce

```shell
uv run toolhunch-bench direct --dry-run
uv run toolhunch-bench direct --pilot
uv run toolhunch-bench direct
uv run toolhunch-bench direct-report bench/runs/RUN --out bench/results/2026-09-toolret-direct/
```

Verified provider usage in this run: $1.0094.
Guarded charges, including all failed/excluded attempts: $1.0094; 0 attempts have unknown usage.
Projected remaining spend: $0.0000; cumulative P1 including historical prior and all guarded charges: $1.3166.
Method: completed full run.
Separate recorded uncached budget reference for remaining workload: $0.0000. Recorded full uncached/max-output estimate retained as a conservative reference for any unfinished workload, without subtracting successful request costs. Historical retrieval approximations and token framing prevent a guaranteed upper bound; physical-call guard reservations remain independent. This report does not authorize another run.

Not affiliated with TypeSafe, OpenAI or Pydantic.
