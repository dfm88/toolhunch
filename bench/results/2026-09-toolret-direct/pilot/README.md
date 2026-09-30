# ToolRet direct choice

Single-turn selection of a relevant tool from real source catalogs.

Primary pooled comparison uses the same common catalogs in every arm: webtools_spotify, webtools_tmdb, tooleyes, apibank.
Secondary all_catalogs rows include every selected catalog and are not a five-arm comparison.
Original run completed: False; original stop reason: more than 5% errored requests in agent-all. Applicability does not change this historical status.

95% intervals resample tasks together, 2,000 resamples, seed 0. Abstentions and errors are misses in the positive-request headline. None-option metrics use parsed requests; errors are listed separately.

| Arm | Catalog | Relevant picks / positives | Paired Δ vs hybrid | Correct / wrong / abstained | Negatives | Errors |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | pooled | 0.550 (0.400 to 0.700) | — | 22 / 38 / 0 | 20/60 (33.3%) | 0 |
| hybrid@20 | all_catalogs | 0.520 (0.380 to 0.660) | — | 26 / 49 / 0 | 25/75 (33.3%) | 0 |
| hybrid@20 | webtools_spotify | 0.700 (0.400 to 1.000) | — | 7 / 8 / 0 | 5/15 (33.3%) | 0 |
| hybrid@20 | webtools_tmdb | 0.200 (0.000 to 0.500) | — | 2 / 13 / 0 | 5/15 (33.3%) | 0 |
| hybrid@20 | tooleyes | 0.800 (0.500 to 1.000) | — | 8 / 7 / 0 | 5/15 (33.3%) | 0 |
| hybrid@20 | apibank | 0.500 (0.200 to 0.800) | — | 5 / 10 / 0 | 5/15 (33.3%) | 0 |
| hybrid@20 | metatool_which | 0.400 (0.100 to 0.700) | — | 4 / 11 / 0 | 5/15 (33.3%) | 0 |
| hybrid@20+jev | pooled | 0.650 (0.500 to 0.800) | 0.100 (-0.125 to 0.300) | 26 / 21 / 13 | 20/60 (33.3%) | 0 |
| hybrid@20+jev | all_catalogs | 0.660 (0.520 to 0.780) | 0.140 (-0.040 to 0.320) | 33 / 27 / 15 | 25/75 (33.3%) | 0 |
| hybrid@20+jev | webtools_spotify | 0.700 (0.400 to 1.000) | 0.000 (-0.400 to 0.400) | 7 / 4 / 4 | 5/15 (33.3%) | 0 |
| hybrid@20+jev | webtools_tmdb | 0.800 (0.500 to 1.000) | 0.600 (0.300 to 0.900) | 8 / 6 / 1 | 5/15 (33.3%) | 0 |
| hybrid@20+jev | tooleyes | 0.400 (0.100 to 0.700) | -0.400 (-0.700 to -0.100) | 4 / 6 / 5 | 5/15 (33.3%) | 0 |
| hybrid@20+jev | apibank | 0.700 (0.400 to 1.000) | 0.200 (-0.200 to 0.600) | 7 / 5 / 3 | 5/15 (33.3%) | 0 |
| hybrid@20+jev | metatool_which | 0.700 (0.400 to 0.903) | 0.300 (-0.100 to 0.700) | 7 / 6 / 2 | 5/15 (33.3%) | 0 |
| jev-all | pooled | 0.750 (0.625 to 0.875) | 0.200 (0.000 to 0.375) | 30 / 21 / 9 | 20/60 (33.3%) | 0 |
| jev-all | all_catalogs | 0.760 (0.640 to 0.860) | 0.240 (0.060 to 0.400) | 38 / 28 / 9 | 25/75 (33.3%) | 0 |
| jev-all | webtools_spotify | 1.000 (1.000 to 1.000) | 0.300 (0.000 to 0.600) | 10 / 2 / 3 | 5/15 (33.3%) | 0 |
| jev-all | webtools_tmdb | 0.800 (0.500 to 1.000) | 0.600 (0.300 to 0.900) | 8 / 7 / 0 | 5/15 (33.3%) | 0 |
| jev-all | tooleyes | 0.500 (0.200 to 0.800) | -0.300 (-0.600 to 0.000) | 5 / 6 / 4 | 5/15 (33.3%) | 0 |
| jev-all | apibank | 0.700 (0.400 to 1.000) | 0.200 (-0.200 to 0.600) | 7 / 6 / 2 | 5/15 (33.3%) | 0 |
| jev-all | metatool_which | 0.800 (0.500 to 1.000) | 0.400 (0.000 to 0.800) | 8 / 7 / 0 | 5/15 (33.3%) | 0 |
| agent@20 | pooled | 0.675 (0.525 to 0.825) | 0.125 (-0.050 to 0.300) | 27 / 25 / 8 | 20/60 (33.3%) | 0 |
| agent@20 | all_catalogs | 0.680 (0.540 to 0.800) | 0.160 (0.000 to 0.320) | 34 / 28 / 13 | 25/75 (33.3%) | 0 |
| agent@20 | webtools_spotify | 0.800 (0.500 to 1.000) | 0.100 (-0.300 to 0.500) | 8 / 7 / 0 | 5/15 (33.3%) | 0 |
| agent@20 | webtools_tmdb | 0.700 (0.400 to 0.900) | 0.500 (0.200 to 0.800) | 7 / 8 / 0 | 5/15 (33.3%) | 0 |
| agent@20 | tooleyes | 0.700 (0.400 to 1.000) | -0.100 (-0.300 to 0.000) | 7 / 4 / 4 | 5/15 (33.3%) | 0 |
| agent@20 | apibank | 0.500 (0.200 to 0.800) | 0.000 (-0.400 to 0.400) | 5 / 6 / 4 | 5/15 (33.3%) | 0 |
| agent@20 | metatool_which | 0.700 (0.400 to 0.903) | 0.300 (-0.100 to 0.700) | 7 / 3 / 5 | 5/15 (33.3%) | 0 |
| agent-all | pooled | 0.650 (0.500 to 0.800) | 0.100 (-0.100 to 0.300) | 26 / 22 / 12 | 20/60 (33.3%) | 0 |
| agent-all | webtools_spotify | 0.700 (0.400 to 1.000) | 0.000 (-0.300 to 0.300) | 7 / 7 / 1 | 5/15 (33.3%) | 0 |
| agent-all | webtools_tmdb | 0.800 (0.500 to 1.000) | 0.600 (0.300 to 0.900) | 8 / 7 / 0 | 5/15 (33.3%) | 0 |
| agent-all | tooleyes | 0.700 (0.400 to 0.900) | -0.100 (-0.400 to 0.200) | 7 / 4 / 4 | 5/15 (33.3%) | 0 |
| agent-all | apibank | 0.400 (0.100 to 0.700) | -0.100 (-0.500 to 0.300) | 4 / 4 / 7 | 5/15 (33.3%) | 0 |
| agent-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | 4 rejected attempts, excluded |

## Observed provider cost

Replay responses do not contribute usage or timing. Search cost and latency are included for strategies that search. Negatives are priced separately.

| Arm | Catalog | Cold billed | Warm billed / 1,000 | Warm list / 1,000 | Warm cache share | Warm latency p50 / p95 ms | Negative billed / 1,000 |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid@20 | pooled | $0.0000 | $0.0003 | $0.0003 | n/a | 204.7 / 398.0 | $0.0002 |
| hybrid@20 | all_catalogs | $0.0000 | $0.0004 | $0.0004 | n/a | 197.8 / 367.0 | $0.0003 |
| hybrid@20 | webtools_spotify | $0.0000 | $0.0002 | $0.0002 | n/a | 240.5 / 959.5 | $0.0001 |
| hybrid@20 | webtools_tmdb | $0.0000 | $0.0003 | $0.0003 | n/a | 210.2 / 278.5 | $0.0002 |
| hybrid@20 | tooleyes | $0.0000 | $0.0004 | $0.0004 | n/a | 201.9 / 254.2 | $0.0003 |
| hybrid@20 | apibank | $0.0000 | $0.0004 | $0.0004 | n/a | 185.3 / 329.2 | $0.0002 |
| hybrid@20 | metatool_which | $0.0000 | $0.0008 | $0.0008 | n/a | 191.8 / 203.2 | $0.0005 |
| hybrid@20+jev | pooled | $0.0003 | $0.0833 | $0.0833 | n/a | 478.5 / 702.3 | $0.0805 |
| hybrid@20+jev | all_catalogs | $0.0004 | $0.0764 | $0.0764 | n/a | 471.5 / 690.0 | $0.0740 |
| hybrid@20+jev | webtools_spotify | $0.0001 | $0.0596 | $0.0596 | n/a | 521.2 / 1206.9 | $0.0607 |
| hybrid@20+jev | webtools_tmdb | $0.0001 | $0.1019 | $0.1019 | n/a | 480.0 / 620.9 | $0.0894 |
| hybrid@20+jev | tooleyes | $0.0001 | $0.0819 | $0.0819 | n/a | 452.3 / 506.5 | $0.0806 |
| hybrid@20+jev | apibank | $0.0001 | $0.0898 | $0.0898 | n/a | 462.6 / 645.9 | $0.0914 |
| hybrid@20+jev | metatool_which | $0.0000 | $0.0486 | $0.0486 | n/a | 445.3 / 490.5 | $0.0478 |
| jev-all | pooled | $0.0010 | $0.2419 | $0.2419 | n/a | 300.0 / 514.0 | $0.2378 |
| jev-all | all_catalogs | $0.0013 | $0.2580 | $0.2580 | n/a | 298.8 / 509.8 | $0.2541 |
| jev-all | webtools_spotify | $0.0001 | $0.1090 | $0.1090 | n/a | 269.3 / 518.9 | $0.1036 |
| jev-all | webtools_tmdb | $0.0002 | $0.2004 | $0.2004 | n/a | 269.5 / 504.3 | $0.1952 |
| jev-all | tooleyes | $0.0003 | $0.2958 | $0.2958 | n/a | 297.0 / 438.3 | $0.2935 |
| jev-all | apibank | $0.0004 | $0.3622 | $0.3622 | n/a | 351.5 / 411.9 | $0.3588 |
| jev-all | metatool_which | $0.0003 | $0.3228 | $0.3228 | n/a | 294.7 / 326.5 | $0.3194 |
| agent@20 | pooled | $0.0025 | $0.6823 | $0.6823 | 0.0% | 1009.9 / 1405.6 | $0.6676 |
| agent@20 | all_catalogs | $0.0029 | $0.6331 | $0.6331 | 0.0% | 1007.1 / 1420.9 | $0.6407 |
| agent@20 | webtools_spotify | $0.0004 | $0.4581 | $0.4581 | 0.0% | 1007.1 / 1698.9 | $0.4685 |
| agent@20 | webtools_tmdb | $0.0007 | $0.8968 | $0.8968 | 0.0% | 973.5 / 1222.9 | $0.7861 |
| agent@20 | tooleyes | $0.0007 | $0.6566 | $0.6566 | 0.0% | 1023.0 / 1105.9 | $0.6791 |
| agent@20 | apibank | $0.0007 | $0.7177 | $0.7177 | 0.0% | 1013.7 / 1310.4 | $0.7369 |
| agent@20 | metatool_which | $0.0004 | $0.4362 | $0.4362 | 0.0% | 919.3 / 1769.2 | $0.5330 |
| agent-all | pooled | $0.0083 | $0.6646 | $2.0992 | 92.9% | 847.5 / 1234.0 | $1.5977 |
| agent-all | webtools_spotify | $0.0009 | $0.2976 | $0.9120 | 92.7% | 771.2 / 1255.1 | $0.8601 |
| agent-all | webtools_tmdb | $0.0019 | $0.5525 | $1.8581 | 95.3% | 745.0 / 854.3 | $1.6239 |
| agent-all | tooleyes | $0.0025 | $0.9412 | $2.5796 | 86.7% | 966.8 / 1466.9 | $1.8666 |
| agent-all | apibank | $0.0030 | $0.8669 | $3.0472 | 96.6% | 967.2 / 1135.1 | $2.0401 |
| agent-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | n/a | n/a |

## Non-applicable pairs and historical rejections

- agent-all / metatool_which: OpenAI Chat Completions rejected 200 function tools: HTTP 400 array_above_max_length, param tools (4/4 attempts); exact maximum not established. Evidence: bench/runs/20260929T221520Z-direct-pilot/run.jsonl (2026-09-30); 4 historical requests excluded, 4 rejected attempts.

| Arm | Catalog | Historical phase | Rejected provider attempts | Billed | List | Billed / 1,000 | List / 1,000 |
|---|---|---|---:|---:|---:|---:|---:|
| agent-all | metatool_which | cold | 1 | unknown | unknown | unknown | unknown |
| agent-all | metatool_which | warm | 3 | unknown | unknown | unknown | unknown |

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

Verified provider usage in this run: $0.1365.
Guarded charges, including all failed/excluded attempts: $0.3072; 4 attempts have unknown usage.
Projected remaining spend: $1.0490; cumulative P1 including historical prior and all guarded charges: $1.3562.
Method: observed projection: per-source warm-positive and negative applicable priced rates; successful requests already covered; embedding reference; not a guaranteed upper bound.
Separate recorded uncached budget reference for remaining workload: $4.6013. Recorded full uncached/max-output estimate retained as a conservative reference for any unfinished workload, without subtracting successful request costs. Historical retrieval approximations and token framing prevent a guaranteed upper bound; physical-call guard reservations remain independent. This report does not authorize another run.

Not affiliated with TypeSafe, OpenAI or Pydantic.
