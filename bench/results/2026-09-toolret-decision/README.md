# ToolRet decision stage, held-out run 20260929T153251Z

Held-out run `20260929T153251Z`: 200 tasks from `bench/tasks/toolret-heldout-200.json` (sha256 `b632db8a5cb3`), `mteb/ToolRetrieval` at `76d45e5` with 44,453 tools, catalog `sha256:7d56b70f3415`; toolhunch 0.1.0, commit `57fd48a`, Python 3.14.3. Every abstention threshold applied here was chosen on the dev runs (`20260929T140534Z`, `20260929T144900Z`, `20260929T145416Z`, `20260929T145809Z`, `20260929T150417Z`); no held-out search chose one.

**CLM caveat.** We run CLM ourselves, from the code its authors published (`bench/deploy/clm_modal.py`). Its figures are provisional until the CLM authors confirm that this deployment matches their reference setup.

## Runs

| split | role | run | tasks | sources | negatives | repeat | K | deciders (max detail) | reserved option | model-source fallbacks | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | main | `20260929T140534Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | jev (FULL), clm (FULL), logprob (FULL) | on | 20 of 50 | `65f8f1f` |
| dev | main | `20260929T144900Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 1 | 20 | jev (FULL), clm (FULL), logprob (FULL) | off | - | `65f8f1f (dirty)` |
| dev | main | `20260929T145416Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 1 | 20 | jev (BRIEF), clm (BRIEF), logprob (BRIEF) | on | - | `65f8f1f (dirty)` |
| dev | main | `20260929T145809Z` | 10 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 3 | 20 | jev (FULL), clm (FULL), logprob (FULL) | on | - | `65f8f1f (dirty)` |
| dev | main | `20260929T150417Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | jev (BRIEF), clm (BRIEF) | on | 20 of 50 | `65f8f1f (dirty)` |
| heldout | main | `20260929T153251Z` | 200 (`bench/tasks/toolret-heldout-200.json`, sha256 `b632db8a5cb3`) | plain, model | yes | 1 | 20, 50 | jev (BRIEF), clm (BRIEF), logprob (FULL) | on | 81 of 200 | `57fd48a` |
| heldout | repeats | `20260929T162701Z` | 200 (`bench/tasks/toolret-heldout-200.json`, sha256 `b632db8a5cb3`) | plain | yes | 3 | 20 | jev (BRIEF), logprob (FULL) | on | - | `57fd48a (dirty)` |

| decider | model | prompt version | declared limits |
|---|---|---|---|
| jev | `jev-1.13.0@api.typesafe.ai` | `tool-choice-v1` | https://docs.typesafe.ai (limits and pricing) (checked 2026-09-27) |
| clm | `clm-latest@modal` | `tool-choice-v1` | Contrastive-LM/CLM @ bb42c6c: clm-serve cuts each text at 2,048 tokens by default; bench/deploy/clm_modal.py --max-model-len 2048; a 7,700-token option billed as 2,048 tokens (probe 2026-09-28) (checked 2026-09-28) |
| logprob | `gpt-4.1-mini-2025-04-14@api.openai.com` | `tool-choice-v1+letters-v1` | OpenAI API reference: top_logprobs 0-20; probe 2026-09-28: gpt-4.1-mini returns 20, gpt-5.4-mini at most 5, gpt-5-mini none (checked 2026-09-28) |

CLM runs from `bench/deploy/clm_modal.py` on one L4 GPU (vLLM 0.30.0, contrastive-lm 0.1.0), listed at $0.80 per GPU hour (checked 2026-09-28).

Model-written queries of `20260929T140534Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20260929T150417Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20260929T153251Z`: `bench/tasks/toolret-heldout-200.model-queries.json` (sha256 `d8df10441cfb`), written by `gpt-5.4-mini-2026-03-17`.

## Held-out

`plain` searches with the ToolRet request, `model` with the queries a model wrote for the task, and `model (searched)` keeps the tasks where that model did search: on 81 of 200 tasks it answered without searching, and the request itself was used.

### P@1

| arm | source | positives | P@1 (95% CI) | hybrid P@1 | ceiling |
|---|---|---:|---:|---:|---:|
| hybrid@20 | plain | 200 | 0.220 (0.17-0.28) | 0.220 | 0.590 |
| hybrid@20 | model | 200 | 0.215 (0.16-0.27) | 0.215 | 0.600 |
| hybrid@20 | model (searched) | 119 | 0.202 (0.13-0.28) | 0.202 | 0.647 |
| hybrid+jev@20 | plain | 200 | 0.320 (0.26-0.39) | 0.220 | 0.590 |
| hybrid+jev@20 | model | 200 | 0.325 (0.26-0.39) | 0.215 | 0.600 |
| hybrid+jev@20 | model (searched) | 119 | 0.328 (0.24-0.41) | 0.202 | 0.647 |
| hybrid+clm@20 | plain | 200 | 0.135 (0.09-0.18) | 0.220 | 0.590 |
| hybrid+clm@20 | model | 200 | 0.125 (0.09-0.17) | 0.215 | 0.600 |
| hybrid+clm@20 | model (searched) | 119 | 0.109 (0.06-0.17) | 0.202 | 0.647 |
| hybrid+logprob@20 | plain | 200 | 0.330 (0.27-0.40) | 0.220 | 0.590 |
| hybrid+logprob@20 | model | 200 | 0.335 (0.27-0.40) | 0.215 | 0.600 |
| hybrid+logprob@20 | model (searched) | 119 | 0.319 (0.24-0.40) | 0.202 | 0.647 |
| hybrid@50 | plain | 200 | 0.220 (0.17-0.28) | 0.220 | 0.715 |
| hybrid@50 | model | 200 | 0.215 (0.16-0.27) | 0.215 | 0.720 |
| hybrid@50 | model (searched) | 119 | 0.202 (0.13-0.28) | 0.202 | 0.748 |
| hybrid+jev@50 | plain | 200 | 0.370 (0.30-0.43) | 0.220 | 0.715 |
| hybrid+jev@50 | model | 200 | 0.350 (0.28-0.41) | 0.215 | 0.720 |
| hybrid+jev@50 | model (searched) | 119 | 0.319 (0.24-0.40) | 0.202 | 0.748 |
| hybrid+clm@50 | plain | 200 | 0.105 (0.07-0.15) | 0.220 | 0.715 |
| hybrid+clm@50 | model | 200 | 0.085 (0.05-0.12) | 0.215 | 0.720 |
| hybrid+clm@50 | model (searched) | 119 | 0.076 (0.03-0.13) | 0.202 | 0.748 |
| hybrid+logprob@50 | plain | 200 | 0.345 (0.28-0.41) | 0.220 | 0.715 |
| hybrid+logprob@50 | model | 200 | 0.350 (0.28-0.41) | 0.215 | 0.720 |
| hybrid+logprob@50 | model (searched) | 119 | 0.311 (0.23-0.39) | 0.202 | 0.748 |

### Abstention

| arm | source | rule | τ | searches | errors | coverage | selective accuracy | wrong-tool rate (95% CI) | abstention precision | abstention recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid@20 | plain | none: answers every search | - | 400 | 0 | 1.000 | 0.110 | 0.890 (0.86-0.92) | n/a | 0.000 |
| hybrid@20 | model | none: answers every search | - | 400 | 0 | 1.000 | 0.107 | 0.892 (0.86-0.92) | n/a | 0.000 |
| hybrid@20 | model (searched) | none: answers every search | - | 238 | 0 | 1.000 | 0.101 | 0.899 (0.86-0.94) | n/a | 0.000 |
| hybrid+jev@20 | plain | reserved | - | 400 | 0 | 0.823 | 0.195 | 0.662 (0.62-0.71) | 0.972 | 0.245 |
| hybrid+jev@20 | plain | reserved + dev τ | 0.95 | 400 | 0 | 0.122 | 0.408 | 0.072 (0.05-0.10) | 0.735 | 0.915 |
| hybrid+jev@20 | model | reserved | - | 400 | 0 | 0.858 | 0.190 | 0.695 (0.65-0.74) | 0.965 | 0.196 |
| hybrid+jev@20 | model | reserved + dev τ | 0.95 | 400 | 0 | 0.083 | 0.455 | 0.045 (0.03-0.07) | 0.725 | 0.950 |
| hybrid+jev@20 | model (searched) | reserved | - | 238 | 0 | 0.962 | 0.170 | 0.798 (0.75-0.85) | 0.889 | 0.050 |
| hybrid+jev@20 | model (searched) | reserved + dev τ | 0.95 | 238 | 0 | 0.063 | 0.533 | 0.029 (0.01-0.05) | 0.700 | 0.969 |
| hybrid+clm@20 | plain | reserved | - | 400 | 0 | 0.938 | 0.067 | 0.875 (0.84-0.91) | 0.760 | 0.067 |
| hybrid+clm@20 | plain | reserved + dev τ | 0.90 | 400 | 0 | 0.020 | 0.250 | 0.015 (0.01-0.03) | 0.704 | 0.979 |
| hybrid+clm@20 | model | reserved | - | 400 | 0 | 0.927 | 0.062 | 0.870 (0.84-0.90) | 0.793 | 0.082 |
| hybrid+clm@20 | model | reserved + dev τ | 0.90 | 400 | 0 | 0.025 | 0.200 | 0.020 (0.01-0.04) | 0.697 | 0.971 |
| hybrid+clm@20 | model (searched) | reserved | - | 238 | 0 | 0.966 | 0.057 | 0.912 (0.87-0.95) | 0.875 | 0.043 |
| hybrid+clm@20 | model (searched) | reserved + dev τ | 0.90 | 238 | 0 | 0.008 | 0.000 | 0.008 (0.00-0.02) | 0.674 | 0.988 |
| hybrid+logprob@20 | plain | reserved | - | 400 | 0 | 0.833 | 0.183 | 0.680 (0.63-0.72) | 0.910 | 0.216 |
| hybrid+logprob@20 | plain | reserved + dev τ | 0.95 | 400 | 0 | 0.667 | 0.195 | 0.537 (0.49-0.58) | 0.789 | 0.372 |
| hybrid+logprob@20 | model | reserved | - | 400 | 0 | 0.863 | 0.183 | 0.705 (0.66-0.75) | 0.891 | 0.175 |
| hybrid+logprob@20 | model | reserved + dev τ | 0.95 | 400 | 0 | 0.718 | 0.192 | 0.580 (0.53-0.63) | 0.796 | 0.321 |
| hybrid+logprob@20 | model (searched) | reserved | - | 238 | 0 | 0.954 | 0.159 | 0.803 (0.75-0.85) | 0.727 | 0.050 |
| hybrid+logprob@20 | model (searched) | reserved + dev τ | 0.95 | 238 | 0 | 0.790 | 0.176 | 0.651 (0.59-0.71) | 0.740 | 0.230 |
| hybrid@50 | plain | none: answers every search | - | 400 | 0 | 1.000 | 0.110 | 0.890 (0.86-0.92) | n/a | 0.000 |
| hybrid@50 | model | none: answers every search | - | 400 | 0 | 1.000 | 0.107 | 0.892 (0.86-0.92) | n/a | 0.000 |
| hybrid@50 | model (searched) | none: answers every search | - | 238 | 0 | 1.000 | 0.101 | 0.899 (0.86-0.94) | n/a | 0.000 |
| hybrid+jev@50 | plain | reserved | - | 400 | 0 | 0.920 | 0.201 | 0.735 (0.69-0.78) | 0.969 | 0.121 |
| hybrid+jev@50 | plain | reserved + dev τ | 0.95 | 400 | 0 | 0.102 | 0.512 | 0.050 (0.03-0.07) | 0.671 | 0.938 |
| hybrid+jev@50 | model | reserved | - | 400 | 0 | 0.927 | 0.189 | 0.752 (0.71-0.80) | 0.966 | 0.109 |
| hybrid+jev@50 | model | reserved + dev τ | 0.95 | 400 | 0 | 0.085 | 0.529 | 0.040 (0.02-0.06) | 0.667 | 0.953 |
| hybrid+jev@50 | model (searched) | reserved | - | 238 | 0 | 0.979 | 0.163 | 0.819 (0.77-0.87) | 1.000 | 0.034 |
| hybrid+jev@50 | model (searched) | reserved + dev τ | 0.95 | 238 | 0 | 0.071 | 0.588 | 0.029 (0.01-0.05) | 0.652 | 0.966 |
| hybrid+clm@50 | plain | reserved | - | 400 | 0 | 0.960 | 0.052 | 0.910 (0.88-0.94) | 0.625 | 0.039 |
| hybrid+clm@50 | plain | reserved + dev τ | 0.90 | 400 | 0 | 0.005 | 0.000 | 0.005 (0.00-0.01) | 0.641 | 0.992 |
| hybrid+clm@50 | model | reserved | - | 400 | 0 | 0.950 | 0.042 | 0.910 (0.88-0.94) | 0.700 | 0.055 |
| hybrid+clm@50 | model | reserved + dev τ | 0.90 | 400 | 0 | 0.005 | 0.000 | 0.005 (0.00-0.01) | 0.638 | 0.992 |
| hybrid+clm@50 | model (searched) | reserved | - | 238 | 0 | 0.983 | 0.038 | 0.945 (0.92-0.97) | 1.000 | 0.027 |
| hybrid+clm@50 | model (searched) | reserved + dev τ | 0.90 | 238 | 0 | 0.000 | n/a | 0.000 (0.00-0.00) | 0.626 | 1.000 |
| hybrid+logprob@50 | plain | reserved | - | 400 | 0 | 0.915 | 0.186 | 0.745 (0.70-0.79) | 0.882 | 0.117 |
| hybrid+logprob@50 | plain | reserved + dev τ | 0.95 | 400 | 0 | 0.695 | 0.212 | 0.547 (0.50-0.59) | 0.730 | 0.346 |
| hybrid+logprob@50 | model | reserved | - | 400 | 0 | 0.925 | 0.186 | 0.752 (0.71-0.80) | 0.900 | 0.105 |
| hybrid+logprob@50 | model | reserved + dev τ | 0.95 | 400 | 0 | 0.693 | 0.213 | 0.545 (0.50-0.59) | 0.707 | 0.340 |
| hybrid+logprob@50 | model (searched) | reserved | - | 238 | 0 | 0.971 | 0.160 | 0.815 (0.76-0.87) | 1.000 | 0.047 |
| hybrid+logprob@50 | model (searched) | reserved + dev τ | 0.95 | 238 | 0 | 0.689 | 0.183 | 0.563 (0.50-0.63) | 0.635 | 0.315 |

### Latency

| arm | source | decision p50 ms | p95 | server p50 ms | p95 | retrieval p50 ms | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid@20 | plain | - | - | - | - | 782 | 1,023 |
| hybrid@20 | model | - | - | - | - | 823 | 2,117 |
| hybrid@20 | model (searched) | - | - | - | - | 1,453 | 2,165 |
| hybrid+jev@20 | plain | 253 | 312 | 58 | 106 | 782 | 1,023 |
| hybrid+jev@20 | model | 254 | 348 | 59 | 122 | 823 | 2,117 |
| hybrid+jev@20 | model (searched) | 257 | 348 | 60 | 125 | 1,453 | 2,165 |
| hybrid+clm@20 | plain | 538 | 640 | 322 | 395 | 782 | 1,023 |
| hybrid+clm@20 | model | 470 | 611 | 250 | 355 | 823 | 2,117 |
| hybrid+clm@20 | model (searched) | 454 | 536 | 239 | 304 | 1,453 | 2,165 |
| hybrid+logprob@20 | plain | 1,308 | 1,922 | - | - | 782 | 1,023 |
| hybrid+logprob@20 | model | 1,311 | 2,061 | - | - | 823 | 2,117 |
| hybrid+logprob@20 | model (searched) | 1,308 | 1,976 | - | - | 1,453 | 2,165 |
| hybrid@50 | plain | - | - | - | - | 782 | 1,023 |
| hybrid@50 | model | - | - | - | - | 823 | 2,117 |
| hybrid@50 | model (searched) | - | - | - | - | 1,453 | 2,165 |
| hybrid+jev@50 | plain | 264 | 343 | 61 | 106 | 782 | 1,023 |
| hybrid+jev@50 | model | 264 | 342 | 64 | 103 | 823 | 2,117 |
| hybrid+jev@50 | model (searched) | 264 | 322 | 66 | 107 | 1,453 | 2,165 |
| hybrid+clm@50 | plain | 752 | 995 | 511 | 734 | 782 | 1,023 |
| hybrid+clm@50 | model | 530 | 907 | 301 | 688 | 823 | 2,117 |
| hybrid+clm@50 | model (searched) | 478 | 660 | 267 | 425 | 1,453 | 2,165 |
| hybrid+logprob@50 | plain | 1,504 | 2,492 | - | - | 782 | 1,023 |
| hybrid+logprob@50 | model | 1,457 | 2,517 | - | - | 823 | 2,117 |
| hybrid+logprob@50 | model (searched) | 1,426 | 2,329 | - | - | 1,453 | 2,165 |

### Cost per 1,000 searches

| arm | source | asks / search | input tokens / search | USD | CLM busy USD | CLM wall USD |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 0.00 | 0 | 0.0000 | - | - |
| hybrid@20 | model | 0.00 | 0 | 0.0000 | - | - |
| hybrid@20 | model (searched) | 0.00 | 0 | 0.0000 | - | - |
| hybrid+jev@20 | plain | 1.00 | 1,280 | 0.0538 | - | - |
| hybrid+jev@20 | model | 1.00 | 1,275 | 0.0536 | - | - |
| hybrid+jev@20 | model (searched) | 1.00 | 1,282 | 0.0538 | - | - |
| hybrid+clm@20 | plain | 1.00 | 391 | - | 0.0581 | 0.1257 |
| hybrid+clm@20 | model | 1.00 | 288 | - | 0.0494 | 0.1156 |
| hybrid+clm@20 | model (searched) | 1.00 | 209 | - | 0.0433 | 0.0920 |
| hybrid+logprob@20 | plain | 2.00 | 1,297 | 0.5220 | - | - |
| hybrid+logprob@20 | model | 2.00 | 1,288 | 0.5186 | - | - |
| hybrid+logprob@20 | model (searched) | 2.00 | 1,282 | 0.5161 | - | - |
| hybrid@50 | plain | 0.00 | 0 | 0.0000 | - | - |
| hybrid@50 | model | 0.00 | 0 | 0.0000 | - | - |
| hybrid@50 | model (searched) | 0.00 | 0 | 0.0000 | - | - |
| hybrid+jev@50 | plain | 1.00 | 2,727 | 0.1145 | - | - |
| hybrid+jev@50 | model | 1.00 | 2,699 | 0.1133 | - | - |
| hybrid+jev@50 | model (searched) | 1.00 | 2,693 | 0.1131 | - | - |
| hybrid+clm@50 | plain | 1.00 | 764 | - | 0.0911 | 0.1915 |
| hybrid+clm@50 | model | 1.00 | 533 | - | 0.0692 | 0.1514 |
| hybrid+clm@50 | model (searched) | 1.00 | 357 | - | 0.0521 | 0.0995 |
| hybrid+logprob@50 | plain | 4.00 | 3,349 | 1.3460 | - | - |
| hybrid+logprob@50 | model | 4.00 | 3,300 | 1.3263 | - | - |
| hybrid+logprob@50 | model (searched) | 4.00 | 3,217 | 1.2931 | - | - |

Per arm, all sources together: the summed seconds of its decisions, which CLM's wall figure prices, next to the arm's own clock time and the asks the decision cache replayed (hits) or sent (misses) during it.

| arm | searches | errors | decision s | arm clock s | cache hits | cache misses |
|---|---:|---:|---:|---:|---:|---:|
| hybrid@20 | 800 | 0 | - | 343.4 | - | - |
| hybrid+jev@20 | 800 | 0 | 210.3 | 135.4 | 289 | 511 |
| hybrid+clm@20 | 800 | 0 | 434.3 | 257.5 | 289 | 511 |
| hybrid+logprob@20 | 800 | 0 | 1112.7 | 685.6 | 615 | 985 |
| hybrid@50 | 800 | 0 | - | 0.0 | - | - |
| hybrid+jev@50 | 800 | 0 | 220.5 | 152.0 | 252 | 548 |
| hybrid+clm@50 | 800 | 0 | 617.4 | 387.6 | 252 | 548 |
| hybrid+logprob@50 | 800 | 0 | 1309.0 | 875.0 | 1,059 | 2,141 |

### Run-to-run variation

The repeats run `20260929T162701Z` makes each search 3 times with the decision cache bypassed. The tables above come from the main held-out run, which asks each search once.

| arm | source | searches | P@1 per repeat | P@1 of the main run | U at dev τ per repeat | largest abs Δp |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 400 | 0.220 / 0.220 / 0.220 | 0.220 | -0.780 / -0.780 / -0.780 | n/a |
| hybrid+jev@20 | plain | 400 | 0.305 / 0.310 / 0.305 | 0.320 | -0.020 / -0.025 / -0.015 | 0.1666 |
| hybrid+logprob@20 | plain | 400 | 0.320 / 0.325 / 0.330 | 0.330 | -0.403 / -0.410 / -0.393 | 0.9999 |

P@1 is over the positives of each repeat; U = (correct - wrong) / searches under the reserved option plus the dev τ of each search's key. `P@1 of the main run` is the single-run figure from the P@1 table. The last column is the largest change of one probability (a card's or the reserved option's) between repeats of the same search, over every repeated search of the arm and source.

## Dev

### Thresholds

| threshold key | decider | K | payload | searches (positives + negatives) | τ | U at τ | U at 0.00 | coverage at τ | wrong-tool rate at τ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `jev-1.13.0@api.typesafe.ai\|tool-choice-v1\|db6d5d57dd5971bd\|choice` | jev | 20 | reserved option, FULL, questions [[21]] | 200 (100 + 100) | 0.95 | -0.080 | -0.550 | 0.210 | 0.145 |
| `clm-latest@modal\|tool-choice-v1\|db6d5d57dd5971bd\|choice` | clm | 20 | reserved option, FULL, questions [[21]] | 200 (100 + 100) | 0.90 | 0.020 | -0.800 | 0.020 | 0.000 |
| `gpt-4.1-mini-2025-04-14@api.openai.com\|tool-choice-v1+letters-v1\|35111cf5d26f5538\|choice` | logprob | 20 | reserved option, FULL, questions [[20], [3]] | 200 (100 + 100) | 0.95 | -0.405 | -0.480 | 0.695 | 0.550 |
| `jev-1.13.0@api.typesafe.ai\|tool-choice-v1\|5be950b6c011df2d\|choice` | jev | 50 | reserved option, FULL, questions [[51]] | 200 (100 + 100) | 0.95 | -0.030 | -0.635 | 0.150 | 0.090 |
| `clm-latest@modal\|tool-choice-v1\|5be950b6c011df2d\|choice` | clm | 50 | reserved option, FULL, questions [[51]] | 200 (100 + 100) | 0.80 | 0.020 | -0.850 | 0.020 | 0.000 |
| `gpt-4.1-mini-2025-04-14@api.openai.com\|tool-choice-v1+letters-v1\|0fcf3645d5f1b6f9\|choice` | logprob | 50 | reserved option, FULL, questions [[17, 17, 16], [7]] | 200 (100 + 100) | 0.95 | -0.475 | -0.625 | 0.725 | 0.600 |
| `jev-1.13.0@api.typesafe.ai\|tool-choice-v1\|ea31f9351da8b3b9\|choice` | jev | 20 | no reserved option, FULL, questions [[20]] | 100 (50 + 50) | 0.85 | -0.080 | -0.700 | 0.280 | 0.180 |
| `clm-latest@modal\|tool-choice-v1\|ea31f9351da8b3b9\|choice` | clm | 20 | no reserved option, FULL, questions [[20]] | 100 (50 + 50) | 0.90 | 0.020 | -0.900 | 0.020 | 0.000 |
| `gpt-4.1-mini-2025-04-14@api.openai.com\|tool-choice-v1+letters-v1\|ea31f9351da8b3b9\|choice` | logprob | 20 | no reserved option, FULL, questions [[20]] | 100 (50 + 50) | 0.95 | -0.480 | -0.720 | 0.760 | 0.620 |
| `jev-1.13.0@api.typesafe.ai\|tool-choice-v1\|55db837d3c1774fe\|choice` | jev | 20 | reserved option, BRIEF, questions [[21]] | 200 (100 + 100) | 0.95 | -0.055 | -0.545 | 0.195 | 0.125 |
| `clm-latest@modal\|tool-choice-v1\|55db837d3c1774fe\|choice` | clm | 20 | reserved option, BRIEF, questions [[21]] | 200 (100 + 100) | 0.90 | 0.000 | -0.785 | 0.030 | 0.015 |
| `gpt-4.1-mini-2025-04-14@api.openai.com\|tool-choice-v1+letters-v1\|273f75b74338e7c7\|choice` | logprob | 20 | reserved option, BRIEF, questions [[20], [3]] | 100 (50 + 50) | 0.95 | -0.330 | -0.410 | 0.630 | 0.480 |
| `jev-1.13.0@api.typesafe.ai\|tool-choice-v1\|38dc60cd902e7d70\|choice` | jev | 50 | reserved option, BRIEF, questions [[51]] | 200 (100 + 100) | 0.95 | -0.035 | -0.635 | 0.155 | 0.095 |
| `clm-latest@modal\|tool-choice-v1\|38dc60cd902e7d70\|choice` | clm | 50 | reserved option, BRIEF, questions [[51]] | 200 (100 + 100) | 0.90 | 0.005 | -0.910 | 0.005 | 0.000 |

### P@1 by dev run

P@1 on positives (their count in brackets), one column per dev run that made each search once; the ablations' settings are in the runs table.

| arm | source | `20260929T140534Z` | `20260929T144900Z` | `20260929T145416Z` | `20260929T150417Z` |
|---|---|---:|---:|---:|---:|
| hybrid@20 | plain | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) |
| hybrid@20 | model | 0.200 (50) | - | - | 0.200 (50) |
| hybrid@20 | model (searched) | 0.200 (30) | - | - | 0.200 (30) |
| hybrid+jev@20 | plain | 0.300 (50) | 0.300 (50) | 0.320 (50) | 0.320 (50) |
| hybrid+jev@20 | model | 0.300 (50) | - | - | 0.280 (50) |
| hybrid+jev@20 | model (searched) | 0.300 (30) | - | - | 0.267 (30) |
| hybrid+clm@20 | plain | 0.100 (50) | 0.100 (50) | 0.120 (50) | 0.120 (50) |
| hybrid+clm@20 | model | 0.100 (50) | - | - | 0.140 (50) |
| hybrid+clm@20 | model (searched) | 0.067 (30) | - | - | 0.133 (30) |
| hybrid+logprob@20 | plain | 0.340 (50) | 0.280 (50) | 0.340 (50) | - |
| hybrid+logprob@20 | model | 0.340 (50) | - | - | - |
| hybrid+logprob@20 | model (searched) | 0.367 (30) | - | - | - |
| hybrid@50 | plain | 0.200 (50) | - | - | 0.200 (50) |
| hybrid@50 | model | 0.200 (50) | - | - | 0.200 (50) |
| hybrid@50 | model (searched) | 0.200 (30) | - | - | 0.200 (30) |
| hybrid+jev@50 | plain | 0.320 (50) | - | - | 0.320 (50) |
| hybrid+jev@50 | model | 0.320 (50) | - | - | 0.320 (50) |
| hybrid+jev@50 | model (searched) | 0.300 (30) | - | - | 0.300 (30) |
| hybrid+clm@50 | plain | 0.100 (50) | - | - | 0.100 (50) |
| hybrid+clm@50 | model | 0.100 (50) | - | - | 0.080 (50) |
| hybrid+clm@50 | model (searched) | 0.067 (30) | - | - | 0.067 (30) |
| hybrid+logprob@50 | plain | 0.280 (50) | - | - | - |
| hybrid+logprob@50 | model | 0.360 (50) | - | - | - |
| hybrid+logprob@50 | model (searched) | 0.367 (30) | - | - | - |

### Determinism

Largest change of one probability (a card's or the reserved option's) between repeats of the same search, over every repeated search of the arm.

| run | arm | searches repeated | largest abs Δp | at | within 0.01 |
|---|---|---:|---:|---|---|
| `20260929T145809Z` | hybrid+jev@20 | 20 | 0.1057 | apigen_query_491 (plain, positive) | no |
| `20260929T145809Z` | hybrid+clm@20 | 20 | 0.0000 | apibank_query_38 (plain, positive) | yes |
| `20260929T145809Z` | hybrid+logprob@20 | 20 | 0.4863 | apibank_query_38 (plain, positive) | no |

### Errors

No dev search failed.

## Token heuristic against Jev's count

For every exchange with Jev, the input tokens Jev reported against the heuristic's estimate that the planner budgets with: a least-squares line (reported = intercept + slope x estimate), and the ratios reported / estimate. A ratio under 1 means the heuristic counted more than Jev billed.

| split | Jev exchanges | intercept | slope | median reported / estimated | share under 1 |
|---|---:|---:|---:|---:|---:|
| dev | 1,060 | 328.4 | 0.800 | 0.981 | 0.581 |
| heldout | 1,600 | 267.6 | 0.853 | 1.004 | 0.482 |

## Notes

- **Outcomes.** A search is answered when the decider names a tool, correct when that tool is a gold one, and wrong otherwise. A negative is the same search with its gold tools removed from the candidates, so every answer to it is wrong.
- **P@1** counts positives only and ignores abstention: is the first card of the decider's ranking a gold tool? `hybrid P@1` asks the same of retrieval order on the same searches, and `ceiling` is the share of them with a gold tool among the K candidates.
- **Rules.** `reserved`: the decider abstains when its reserved "none of these" option is at least as likely as its best card. `reserved + dev τ`: it also abstains when that card's probability is below τ, the threshold dev chose for the search's threshold key (model, prompt version, payload shape, question kind). Probabilities compare only within one question, so a τ never crosses keys.
- **Rates.** coverage: answered / searches. Selective accuracy: correct / answered. Wrong-tool rate: wrong / searches. Abstention precision: abstentions on searches without a gold tool among the candidates / abstentions. Abstention recall: those abstentions / searches without a gold tool among the candidates.
- **Choosing τ.** Per threshold key, on dev: of τ = 0.00, 0.05, ..., 0.95, the one with the highest U = (correct - wrong) / searches, where abstaining counts 0; of equal U, the lower τ wins. It uses the dev runs that made each search once, both query sources, positives and negatives; a search that several of those runs made counts once. The held-out risk-coverage tables are there to read, never to choose.
- **Intervals.** 95% percentile bootstrap, 2,000 resamples with seed 0, resampling searches; the repeats of a search are drawn together.
- **Errors.** A search whose decider raised `DecisionError` counts as an error and stays out of every rate, latency and cost figure.
- **Latency.** Decision: the critical path of the decider's calls as the client timed them, network included; a call the decision cache replays keeps the time the original call took. Server: the same by the server's own clock, where it reports one. Retrieval: the first hybrid retrieval of the search's queries, measured live; a query whose vector the embedding cache already held skips the embeddings call, so it reads faster than a cold one.
- **Cost.** Jev: reported input tokens at the price its declared limits give. Logprob: the reported usage, priced by genai-prices. CLM is paid in GPU time, at the list price in the run's manifest, two ways. Busy prices the server's own seconds per search, as if the GPU never sat idle: a lower bound. Wall prices the summed seconds of the arm's decision calls as the client saw them, as if one container served the searches one after another. It leaves out the cold starts, the idle gaps between arms and the scale-down window, which Modal also bills. Modal's CPU and memory charges come on top of both and are not included. Every arm also embeds its queries for hybrid retrieval, at the same cost for each arm, which the F1 retrieval results report.
- **Negatives** stand in for a catalog without the gold tools: the gold ids are dropped from a deeper retrieval, so the other tools' order can move slightly from what a smaller catalog would give.

## Risk-coverage over the threshold grid

### Dev: `jev-1.13.0@api.typesafe.ai|tool-choice-v1|db6d5d57dd5971bd|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.850 | 0.176 | 0.700 | -0.550 |
| 0.05 | 0.850 | 0.176 | 0.700 | -0.550 |
| 0.10 | 0.850 | 0.176 | 0.700 | -0.550 |
| 0.15 | 0.850 | 0.176 | 0.700 | -0.550 |
| 0.20 | 0.850 | 0.176 | 0.700 | -0.550 |
| 0.25 | 0.810 | 0.185 | 0.660 | -0.510 |
| 0.30 | 0.765 | 0.190 | 0.620 | -0.475 |
| 0.35 | 0.735 | 0.197 | 0.590 | -0.445 |
| 0.40 | 0.685 | 0.212 | 0.540 | -0.395 |
| 0.45 | 0.665 | 0.211 | 0.525 | -0.385 |
| 0.50 | 0.615 | 0.211 | 0.485 | -0.355 |
| 0.55 | 0.550 | 0.209 | 0.435 | -0.320 |
| 0.60 | 0.495 | 0.232 | 0.380 | -0.265 |
| 0.65 | 0.455 | 0.253 | 0.340 | -0.225 |
| 0.70 | 0.420 | 0.274 | 0.305 | -0.190 |
| 0.75 | 0.395 | 0.291 | 0.280 | -0.165 |
| 0.80 | 0.350 | 0.314 | 0.240 | -0.130 |
| 0.85 | 0.265 | 0.321 | 0.180 | -0.095 |
| 0.90 | 0.250 | 0.320 | 0.170 | -0.090 |
| **0.95** | 0.210 | 0.310 | 0.145 | -0.080 |

### Dev: `clm-latest@modal|tool-choice-v1|db6d5d57dd5971bd|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.900 | 0.056 | 0.850 | -0.800 |
| 0.05 | 0.900 | 0.056 | 0.850 | -0.800 |
| 0.10 | 0.900 | 0.056 | 0.850 | -0.800 |
| 0.15 | 0.900 | 0.056 | 0.850 | -0.800 |
| 0.20 | 0.900 | 0.056 | 0.850 | -0.800 |
| 0.25 | 0.810 | 0.062 | 0.760 | -0.710 |
| 0.30 | 0.735 | 0.068 | 0.685 | -0.635 |
| 0.35 | 0.615 | 0.081 | 0.565 | -0.515 |
| 0.40 | 0.530 | 0.085 | 0.485 | -0.440 |
| 0.45 | 0.420 | 0.107 | 0.375 | -0.330 |
| 0.50 | 0.335 | 0.119 | 0.295 | -0.255 |
| 0.55 | 0.235 | 0.170 | 0.195 | -0.155 |
| 0.60 | 0.185 | 0.216 | 0.145 | -0.105 |
| 0.65 | 0.165 | 0.212 | 0.130 | -0.095 |
| 0.70 | 0.105 | 0.333 | 0.070 | -0.035 |
| 0.75 | 0.090 | 0.278 | 0.065 | -0.040 |
| 0.80 | 0.070 | 0.357 | 0.045 | -0.020 |
| 0.85 | 0.065 | 0.385 | 0.040 | -0.015 |
| **0.90** | 0.020 | 1.000 | 0.000 | 0.020 |
| 0.95 | 0.020 | 1.000 | 0.000 | 0.020 |

### Dev: `gpt-4.1-mini-2025-04-14@api.openai.com|tool-choice-v1+letters-v1|35111cf5d26f5538|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.05 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.10 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.15 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.20 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.25 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.30 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.35 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.40 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.45 | 0.820 | 0.207 | 0.650 | -0.480 |
| 0.50 | 0.810 | 0.210 | 0.640 | -0.470 |
| 0.55 | 0.810 | 0.210 | 0.640 | -0.470 |
| 0.60 | 0.810 | 0.210 | 0.640 | -0.470 |
| 0.65 | 0.805 | 0.211 | 0.635 | -0.465 |
| 0.70 | 0.805 | 0.211 | 0.635 | -0.465 |
| 0.75 | 0.755 | 0.212 | 0.595 | -0.435 |
| 0.80 | 0.740 | 0.216 | 0.580 | -0.420 |
| 0.85 | 0.735 | 0.211 | 0.580 | -0.425 |
| 0.90 | 0.720 | 0.201 | 0.575 | -0.430 |
| **0.95** | 0.695 | 0.209 | 0.550 | -0.405 |

### Dev: `jev-1.13.0@api.typesafe.ai|tool-choice-v1|5be950b6c011df2d|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.955 | 0.168 | 0.795 | -0.635 |
| 0.05 | 0.955 | 0.168 | 0.795 | -0.635 |
| 0.10 | 0.955 | 0.168 | 0.795 | -0.635 |
| 0.15 | 0.955 | 0.168 | 0.795 | -0.635 |
| 0.20 | 0.935 | 0.171 | 0.775 | -0.615 |
| 0.25 | 0.905 | 0.177 | 0.745 | -0.585 |
| 0.30 | 0.835 | 0.186 | 0.680 | -0.525 |
| 0.35 | 0.795 | 0.182 | 0.650 | -0.505 |
| 0.40 | 0.735 | 0.190 | 0.595 | -0.455 |
| 0.45 | 0.670 | 0.209 | 0.530 | -0.390 |
| 0.50 | 0.620 | 0.226 | 0.480 | -0.340 |
| 0.55 | 0.500 | 0.230 | 0.385 | -0.270 |
| 0.60 | 0.465 | 0.247 | 0.350 | -0.235 |
| 0.65 | 0.460 | 0.250 | 0.345 | -0.230 |
| 0.70 | 0.410 | 0.268 | 0.300 | -0.190 |
| 0.75 | 0.365 | 0.301 | 0.255 | -0.145 |
| 0.80 | 0.320 | 0.328 | 0.215 | -0.110 |
| 0.85 | 0.265 | 0.321 | 0.180 | -0.095 |
| 0.90 | 0.215 | 0.372 | 0.135 | -0.055 |
| **0.95** | 0.150 | 0.400 | 0.090 | -0.030 |

### Dev: `clm-latest@modal|tool-choice-v1|5be950b6c011df2d|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.950 | 0.053 | 0.900 | -0.850 |
| 0.05 | 0.950 | 0.053 | 0.900 | -0.850 |
| 0.10 | 0.950 | 0.053 | 0.900 | -0.850 |
| 0.15 | 0.920 | 0.054 | 0.870 | -0.820 |
| 0.20 | 0.810 | 0.062 | 0.760 | -0.710 |
| 0.25 | 0.715 | 0.070 | 0.665 | -0.615 |
| 0.30 | 0.565 | 0.080 | 0.520 | -0.475 |
| 0.35 | 0.440 | 0.102 | 0.395 | -0.350 |
| 0.40 | 0.320 | 0.125 | 0.280 | -0.240 |
| 0.45 | 0.260 | 0.115 | 0.230 | -0.200 |
| 0.50 | 0.205 | 0.146 | 0.175 | -0.145 |
| 0.55 | 0.180 | 0.167 | 0.150 | -0.120 |
| 0.60 | 0.170 | 0.176 | 0.140 | -0.110 |
| 0.65 | 0.130 | 0.192 | 0.105 | -0.080 |
| 0.70 | 0.105 | 0.238 | 0.080 | -0.055 |
| 0.75 | 0.075 | 0.333 | 0.050 | -0.025 |
| **0.80** | 0.020 | 1.000 | 0.000 | 0.020 |
| 0.85 | 0.020 | 1.000 | 0.000 | 0.020 |
| 0.90 | 0.010 | 1.000 | 0.000 | 0.010 |
| 0.95 | 0.010 | 1.000 | 0.000 | 0.010 |

### Dev: `gpt-4.1-mini-2025-04-14@api.openai.com|tool-choice-v1+letters-v1|0fcf3645d5f1b6f9|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.05 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.10 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.15 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.20 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.25 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.30 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.35 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.40 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.45 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.50 | 0.945 | 0.169 | 0.785 | -0.625 |
| 0.55 | 0.935 | 0.171 | 0.775 | -0.615 |
| 0.60 | 0.925 | 0.173 | 0.765 | -0.605 |
| 0.65 | 0.915 | 0.164 | 0.765 | -0.615 |
| 0.70 | 0.885 | 0.164 | 0.740 | -0.595 |
| 0.75 | 0.845 | 0.166 | 0.705 | -0.565 |
| 0.80 | 0.825 | 0.170 | 0.685 | -0.545 |
| 0.85 | 0.800 | 0.169 | 0.665 | -0.530 |
| 0.90 | 0.755 | 0.166 | 0.630 | -0.505 |
| **0.95** | 0.725 | 0.172 | 0.600 | -0.475 |

### Dev: `jev-1.13.0@api.typesafe.ai|tool-choice-v1|ea31f9351da8b3b9|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.150 | 0.850 | -0.700 |
| 0.05 | 1.000 | 0.150 | 0.850 | -0.700 |
| 0.10 | 1.000 | 0.150 | 0.850 | -0.700 |
| 0.15 | 1.000 | 0.150 | 0.850 | -0.700 |
| 0.20 | 1.000 | 0.150 | 0.850 | -0.700 |
| 0.25 | 0.960 | 0.156 | 0.810 | -0.660 |
| 0.30 | 0.890 | 0.157 | 0.750 | -0.610 |
| 0.35 | 0.820 | 0.171 | 0.680 | -0.540 |
| 0.40 | 0.800 | 0.175 | 0.660 | -0.520 |
| 0.45 | 0.700 | 0.171 | 0.580 | -0.460 |
| 0.50 | 0.620 | 0.194 | 0.500 | -0.380 |
| 0.55 | 0.540 | 0.222 | 0.420 | -0.300 |
| 0.60 | 0.500 | 0.240 | 0.380 | -0.260 |
| 0.65 | 0.480 | 0.229 | 0.370 | -0.260 |
| 0.70 | 0.430 | 0.256 | 0.320 | -0.210 |
| 0.75 | 0.390 | 0.282 | 0.280 | -0.170 |
| 0.80 | 0.380 | 0.289 | 0.270 | -0.160 |
| **0.85** | 0.280 | 0.357 | 0.180 | -0.080 |
| 0.90 | 0.250 | 0.280 | 0.180 | -0.110 |
| 0.95 | 0.210 | 0.286 | 0.150 | -0.090 |

### Dev: `clm-latest@modal|tool-choice-v1|ea31f9351da8b3b9|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.050 | 0.950 | -0.900 |
| 0.05 | 1.000 | 0.050 | 0.950 | -0.900 |
| 0.10 | 1.000 | 0.050 | 0.950 | -0.900 |
| 0.15 | 1.000 | 0.050 | 0.950 | -0.900 |
| 0.20 | 1.000 | 0.050 | 0.950 | -0.900 |
| 0.25 | 0.880 | 0.057 | 0.830 | -0.780 |
| 0.30 | 0.800 | 0.062 | 0.750 | -0.700 |
| 0.35 | 0.660 | 0.076 | 0.610 | -0.560 |
| 0.40 | 0.600 | 0.083 | 0.550 | -0.500 |
| 0.45 | 0.460 | 0.109 | 0.410 | -0.360 |
| 0.50 | 0.360 | 0.111 | 0.320 | -0.280 |
| 0.55 | 0.270 | 0.148 | 0.230 | -0.190 |
| 0.60 | 0.190 | 0.211 | 0.150 | -0.110 |
| 0.65 | 0.160 | 0.250 | 0.120 | -0.080 |
| 0.70 | 0.100 | 0.400 | 0.060 | -0.020 |
| 0.75 | 0.090 | 0.333 | 0.060 | -0.030 |
| 0.80 | 0.050 | 0.600 | 0.020 | 0.010 |
| 0.85 | 0.050 | 0.600 | 0.020 | 0.010 |
| **0.90** | 0.020 | 1.000 | 0.000 | 0.020 |
| 0.95 | 0.020 | 1.000 | 0.000 | 0.020 |

### Dev: `gpt-4.1-mini-2025-04-14@api.openai.com|tool-choice-v1+letters-v1|ea31f9351da8b3b9|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.140 | 0.860 | -0.720 |
| 0.05 | 1.000 | 0.140 | 0.860 | -0.720 |
| 0.10 | 1.000 | 0.140 | 0.860 | -0.720 |
| 0.15 | 1.000 | 0.140 | 0.860 | -0.720 |
| 0.20 | 1.000 | 0.140 | 0.860 | -0.720 |
| 0.25 | 1.000 | 0.140 | 0.860 | -0.720 |
| 0.30 | 1.000 | 0.140 | 0.860 | -0.720 |
| 0.35 | 1.000 | 0.140 | 0.860 | -0.720 |
| 0.40 | 0.980 | 0.143 | 0.840 | -0.700 |
| 0.45 | 0.980 | 0.143 | 0.840 | -0.700 |
| 0.50 | 0.970 | 0.144 | 0.830 | -0.690 |
| 0.55 | 0.950 | 0.147 | 0.810 | -0.670 |
| 0.60 | 0.910 | 0.154 | 0.770 | -0.630 |
| 0.65 | 0.910 | 0.154 | 0.770 | -0.630 |
| 0.70 | 0.900 | 0.156 | 0.760 | -0.620 |
| 0.75 | 0.880 | 0.159 | 0.740 | -0.600 |
| 0.80 | 0.860 | 0.163 | 0.720 | -0.580 |
| 0.85 | 0.830 | 0.169 | 0.690 | -0.550 |
| 0.90 | 0.780 | 0.179 | 0.640 | -0.500 |
| **0.95** | 0.760 | 0.184 | 0.620 | -0.480 |

### Dev: `jev-1.13.0@api.typesafe.ai|tool-choice-v1|55db837d3c1774fe|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.845 | 0.178 | 0.695 | -0.545 |
| 0.05 | 0.845 | 0.178 | 0.695 | -0.545 |
| 0.10 | 0.845 | 0.178 | 0.695 | -0.545 |
| 0.15 | 0.845 | 0.178 | 0.695 | -0.545 |
| 0.20 | 0.840 | 0.179 | 0.690 | -0.540 |
| 0.25 | 0.825 | 0.182 | 0.675 | -0.525 |
| 0.30 | 0.805 | 0.186 | 0.655 | -0.505 |
| 0.35 | 0.795 | 0.189 | 0.645 | -0.495 |
| 0.40 | 0.725 | 0.193 | 0.585 | -0.445 |
| 0.45 | 0.645 | 0.186 | 0.525 | -0.405 |
| 0.50 | 0.590 | 0.203 | 0.470 | -0.350 |
| 0.55 | 0.550 | 0.218 | 0.430 | -0.310 |
| 0.60 | 0.515 | 0.233 | 0.395 | -0.275 |
| 0.65 | 0.445 | 0.270 | 0.325 | -0.205 |
| 0.70 | 0.410 | 0.280 | 0.295 | -0.180 |
| 0.75 | 0.350 | 0.314 | 0.240 | -0.130 |
| 0.80 | 0.290 | 0.310 | 0.200 | -0.110 |
| 0.85 | 0.245 | 0.306 | 0.170 | -0.095 |
| 0.90 | 0.215 | 0.326 | 0.145 | -0.075 |
| **0.95** | 0.195 | 0.359 | 0.125 | -0.055 |

### Dev: `clm-latest@modal|tool-choice-v1|55db837d3c1774fe|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.905 | 0.066 | 0.845 | -0.785 |
| 0.05 | 0.905 | 0.066 | 0.845 | -0.785 |
| 0.10 | 0.905 | 0.066 | 0.845 | -0.785 |
| 0.15 | 0.905 | 0.066 | 0.845 | -0.785 |
| 0.20 | 0.820 | 0.073 | 0.760 | -0.700 |
| 0.25 | 0.795 | 0.069 | 0.740 | -0.685 |
| 0.30 | 0.750 | 0.073 | 0.695 | -0.640 |
| 0.35 | 0.660 | 0.083 | 0.605 | -0.550 |
| 0.40 | 0.530 | 0.094 | 0.480 | -0.430 |
| 0.45 | 0.430 | 0.116 | 0.380 | -0.330 |
| 0.50 | 0.325 | 0.154 | 0.275 | -0.225 |
| 0.55 | 0.300 | 0.167 | 0.250 | -0.200 |
| 0.60 | 0.230 | 0.196 | 0.185 | -0.140 |
| 0.65 | 0.215 | 0.209 | 0.170 | -0.125 |
| 0.70 | 0.150 | 0.233 | 0.115 | -0.080 |
| 0.75 | 0.140 | 0.250 | 0.105 | -0.070 |
| 0.80 | 0.120 | 0.292 | 0.085 | -0.050 |
| 0.85 | 0.075 | 0.467 | 0.040 | -0.005 |
| **0.90** | 0.030 | 0.500 | 0.015 | 0.000 |
| 0.95 | 0.010 | 0.500 | 0.005 | 0.000 |

### Dev: `gpt-4.1-mini-2025-04-14@api.openai.com|tool-choice-v1+letters-v1|273f75b74338e7c7|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.05 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.10 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.15 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.20 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.25 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.30 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.35 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.40 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.45 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.50 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.55 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.60 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.65 | 0.750 | 0.227 | 0.580 | -0.410 |
| 0.70 | 0.730 | 0.233 | 0.560 | -0.390 |
| 0.75 | 0.700 | 0.229 | 0.540 | -0.380 |
| 0.80 | 0.680 | 0.221 | 0.530 | -0.380 |
| 0.85 | 0.660 | 0.227 | 0.510 | -0.360 |
| 0.90 | 0.640 | 0.234 | 0.490 | -0.340 |
| **0.95** | 0.630 | 0.238 | 0.480 | -0.330 |

### Dev: `jev-1.13.0@api.typesafe.ai|tool-choice-v1|38dc60cd902e7d70|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.955 | 0.168 | 0.795 | -0.635 |
| 0.05 | 0.955 | 0.168 | 0.795 | -0.635 |
| 0.10 | 0.955 | 0.168 | 0.795 | -0.635 |
| 0.15 | 0.955 | 0.168 | 0.795 | -0.635 |
| 0.20 | 0.935 | 0.171 | 0.775 | -0.615 |
| 0.25 | 0.935 | 0.171 | 0.775 | -0.615 |
| 0.30 | 0.855 | 0.175 | 0.705 | -0.555 |
| 0.35 | 0.830 | 0.181 | 0.680 | -0.530 |
| 0.40 | 0.775 | 0.194 | 0.625 | -0.475 |
| 0.45 | 0.685 | 0.219 | 0.535 | -0.385 |
| 0.50 | 0.580 | 0.241 | 0.440 | -0.300 |
| 0.55 | 0.505 | 0.267 | 0.370 | -0.235 |
| 0.60 | 0.450 | 0.300 | 0.315 | -0.180 |
| 0.65 | 0.385 | 0.286 | 0.275 | -0.165 |
| 0.70 | 0.360 | 0.278 | 0.260 | -0.160 |
| 0.75 | 0.330 | 0.303 | 0.230 | -0.130 |
| 0.80 | 0.295 | 0.322 | 0.200 | -0.105 |
| 0.85 | 0.245 | 0.327 | 0.165 | -0.085 |
| 0.90 | 0.210 | 0.357 | 0.135 | -0.060 |
| **0.95** | 0.155 | 0.387 | 0.095 | -0.035 |

### Dev: `clm-latest@modal|tool-choice-v1|38dc60cd902e7d70|choice`

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.990 | 0.040 | 0.950 | -0.910 |
| 0.05 | 0.990 | 0.040 | 0.950 | -0.910 |
| 0.10 | 0.960 | 0.042 | 0.920 | -0.880 |
| 0.15 | 0.920 | 0.043 | 0.880 | -0.840 |
| 0.20 | 0.820 | 0.049 | 0.780 | -0.740 |
| 0.25 | 0.750 | 0.053 | 0.710 | -0.670 |
| 0.30 | 0.595 | 0.067 | 0.555 | -0.515 |
| 0.35 | 0.465 | 0.075 | 0.430 | -0.395 |
| 0.40 | 0.380 | 0.092 | 0.345 | -0.310 |
| 0.45 | 0.345 | 0.101 | 0.310 | -0.275 |
| 0.50 | 0.330 | 0.106 | 0.295 | -0.260 |
| 0.55 | 0.210 | 0.143 | 0.180 | -0.150 |
| 0.60 | 0.160 | 0.125 | 0.140 | -0.120 |
| 0.65 | 0.145 | 0.138 | 0.125 | -0.105 |
| 0.70 | 0.125 | 0.160 | 0.105 | -0.085 |
| 0.75 | 0.080 | 0.250 | 0.060 | -0.040 |
| 0.80 | 0.065 | 0.308 | 0.045 | -0.025 |
| 0.85 | 0.015 | 0.333 | 0.010 | -0.005 |
| **0.90** | 0.005 | 1.000 | 0.000 | 0.005 |
| 0.95 | 0.000 | n/a | 0.000 | 0.000 |

### Held-out: hybrid+jev@20, for reading only

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.840 | 0.192 | 0.679 | -0.517 |
| 0.05 | 0.840 | 0.192 | 0.679 | -0.517 |
| 0.10 | 0.840 | 0.192 | 0.679 | -0.517 |
| 0.15 | 0.839 | 0.192 | 0.677 | -0.516 |
| 0.20 | 0.836 | 0.193 | 0.675 | -0.514 |
| 0.25 | 0.824 | 0.194 | 0.664 | -0.504 |
| 0.30 | 0.792 | 0.200 | 0.634 | -0.475 |
| 0.35 | 0.752 | 0.201 | 0.601 | -0.450 |
| 0.40 | 0.695 | 0.214 | 0.546 | -0.398 |
| 0.45 | 0.618 | 0.225 | 0.479 | -0.340 |
| 0.50 | 0.555 | 0.239 | 0.422 | -0.290 |
| 0.55 | 0.481 | 0.260 | 0.356 | -0.231 |
| 0.60 | 0.435 | 0.261 | 0.321 | -0.207 |
| 0.65 | 0.345 | 0.293 | 0.244 | -0.142 |
| 0.70 | 0.286 | 0.297 | 0.201 | -0.116 |
| 0.75 | 0.258 | 0.316 | 0.176 | -0.095 |
| 0.80 | 0.224 | 0.346 | 0.146 | -0.069 |
| 0.85 | 0.185 | 0.385 | 0.114 | -0.043 |
| 0.90 | 0.155 | 0.395 | 0.094 | -0.033 |
| **0.95** | 0.102 | 0.427 | 0.059 | -0.015 |

### Held-out: hybrid+clm@20, for reading only

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.932 | 0.064 | 0.873 | -0.812 |
| 0.05 | 0.932 | 0.064 | 0.873 | -0.812 |
| 0.10 | 0.932 | 0.064 | 0.873 | -0.812 |
| 0.15 | 0.915 | 0.063 | 0.858 | -0.800 |
| 0.20 | 0.877 | 0.064 | 0.821 | -0.765 |
| 0.25 | 0.785 | 0.065 | 0.734 | -0.682 |
| 0.30 | 0.686 | 0.073 | 0.636 | -0.586 |
| 0.35 | 0.581 | 0.080 | 0.535 | -0.489 |
| 0.40 | 0.486 | 0.082 | 0.446 | -0.406 |
| 0.45 | 0.417 | 0.087 | 0.381 | -0.345 |
| 0.50 | 0.343 | 0.095 | 0.310 | -0.278 |
| 0.55 | 0.285 | 0.096 | 0.258 | -0.230 |
| 0.60 | 0.249 | 0.085 | 0.228 | -0.206 |
| 0.65 | 0.216 | 0.087 | 0.198 | -0.179 |
| 0.70 | 0.170 | 0.088 | 0.155 | -0.140 |
| 0.75 | 0.119 | 0.105 | 0.106 | -0.094 |
| 0.80 | 0.083 | 0.121 | 0.072 | -0.062 |
| 0.85 | 0.049 | 0.103 | 0.044 | -0.039 |
| **0.90** | 0.022 | 0.222 | 0.018 | -0.013 |
| 0.95 | 0.013 | 0.200 | 0.010 | -0.007 |

### Held-out: hybrid+logprob@20, for reading only

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.05 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.10 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.15 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.20 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.25 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.30 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.35 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.40 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.45 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.50 | 0.848 | 0.183 | 0.693 | -0.537 |
| 0.55 | 0.846 | 0.183 | 0.691 | -0.536 |
| 0.60 | 0.835 | 0.184 | 0.681 | -0.527 |
| 0.65 | 0.815 | 0.183 | 0.666 | -0.517 |
| 0.70 | 0.789 | 0.185 | 0.642 | -0.496 |
| 0.75 | 0.777 | 0.186 | 0.632 | -0.487 |
| 0.80 | 0.764 | 0.188 | 0.620 | -0.476 |
| 0.85 | 0.755 | 0.189 | 0.613 | -0.470 |
| 0.90 | 0.729 | 0.194 | 0.588 | -0.446 |
| **0.95** | 0.693 | 0.193 | 0.559 | -0.425 |

### Held-out: hybrid+jev@50, for reading only

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.924 | 0.195 | 0.744 | -0.564 |
| 0.05 | 0.924 | 0.195 | 0.744 | -0.564 |
| 0.10 | 0.924 | 0.195 | 0.744 | -0.564 |
| 0.15 | 0.924 | 0.195 | 0.744 | -0.564 |
| 0.20 | 0.920 | 0.196 | 0.740 | -0.560 |
| 0.25 | 0.870 | 0.203 | 0.694 | -0.517 |
| 0.30 | 0.811 | 0.213 | 0.639 | -0.466 |
| 0.35 | 0.771 | 0.217 | 0.604 | -0.436 |
| 0.40 | 0.704 | 0.233 | 0.540 | -0.376 |
| 0.45 | 0.630 | 0.246 | 0.475 | -0.320 |
| 0.50 | 0.564 | 0.257 | 0.419 | -0.274 |
| 0.55 | 0.468 | 0.278 | 0.338 | -0.207 |
| 0.60 | 0.411 | 0.304 | 0.286 | -0.161 |
| 0.65 | 0.356 | 0.323 | 0.241 | -0.126 |
| 0.70 | 0.309 | 0.340 | 0.204 | -0.099 |
| 0.75 | 0.250 | 0.350 | 0.163 | -0.075 |
| 0.80 | 0.200 | 0.394 | 0.121 | -0.043 |
| 0.85 | 0.163 | 0.423 | 0.094 | -0.025 |
| 0.90 | 0.139 | 0.450 | 0.076 | -0.014 |
| **0.95** | 0.094 | 0.520 | 0.045 | 0.004 |

### Held-out: hybrid+clm@50, for reading only

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.955 | 0.047 | 0.910 | -0.865 |
| 0.05 | 0.955 | 0.047 | 0.910 | -0.865 |
| 0.10 | 0.948 | 0.047 | 0.902 | -0.858 |
| 0.15 | 0.879 | 0.050 | 0.835 | -0.791 |
| 0.20 | 0.775 | 0.055 | 0.733 | -0.690 |
| 0.25 | 0.660 | 0.061 | 0.620 | -0.580 |
| 0.30 | 0.532 | 0.066 | 0.497 | -0.463 |
| 0.35 | 0.422 | 0.059 | 0.398 | -0.372 |
| 0.40 | 0.339 | 0.070 | 0.315 | -0.291 |
| 0.45 | 0.284 | 0.070 | 0.264 | -0.244 |
| 0.50 | 0.249 | 0.070 | 0.231 | -0.214 |
| 0.55 | 0.191 | 0.052 | 0.181 | -0.171 |
| 0.60 | 0.149 | 0.067 | 0.139 | -0.129 |
| 0.65 | 0.109 | 0.069 | 0.101 | -0.094 |
| 0.70 | 0.068 | 0.093 | 0.061 | -0.055 |
| 0.75 | 0.052 | 0.095 | 0.048 | -0.043 |
| 0.80 | 0.045 | 0.056 | 0.043 | -0.040 |
| 0.85 | 0.030 | 0.083 | 0.028 | -0.025 |
| **0.90** | 0.005 | 0.000 | 0.005 | -0.005 |
| 0.95 | 0.000 | n/a | 0.000 | 0.000 |

### Held-out: hybrid+logprob@50, for reading only

| τ | coverage | selective accuracy | wrong-tool rate | U |
|---:|---:|---:|---:|---:|
| 0.00 | 0.920 | 0.186 | 0.749 | -0.578 |
| 0.05 | 0.920 | 0.186 | 0.749 | -0.578 |
| 0.10 | 0.920 | 0.186 | 0.749 | -0.578 |
| 0.15 | 0.920 | 0.186 | 0.749 | -0.578 |
| 0.20 | 0.920 | 0.186 | 0.749 | -0.578 |
| 0.25 | 0.920 | 0.186 | 0.749 | -0.578 |
| 0.30 | 0.920 | 0.186 | 0.749 | -0.578 |
| 0.35 | 0.920 | 0.186 | 0.749 | -0.578 |
| 0.40 | 0.917 | 0.187 | 0.746 | -0.575 |
| 0.45 | 0.916 | 0.187 | 0.745 | -0.574 |
| 0.50 | 0.896 | 0.188 | 0.728 | -0.559 |
| 0.55 | 0.889 | 0.190 | 0.720 | -0.551 |
| 0.60 | 0.864 | 0.194 | 0.696 | -0.529 |
| 0.65 | 0.844 | 0.197 | 0.677 | -0.511 |
| 0.70 | 0.829 | 0.201 | 0.662 | -0.496 |
| 0.75 | 0.819 | 0.202 | 0.654 | -0.489 |
| 0.80 | 0.806 | 0.200 | 0.645 | -0.484 |
| 0.85 | 0.781 | 0.205 | 0.621 | -0.461 |
| 0.90 | 0.748 | 0.206 | 0.594 | -0.440 |
| **0.95** | 0.694 | 0.213 | 0.546 | -0.399 |

