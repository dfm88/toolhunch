# ToolRet decision stage, held-out run 20261004T191646Z+20261004T194433Z+20261004T200546Z+20261004T205108Z

Held-out runs `20261004T191646Z`, `20261004T194433Z`, `20261004T200546Z`, `20261004T205108Z`, one per decider: 200 tasks from `bench/tasks/toolret-heldout-200.json` (sha256 `b632db8a5cb3`), `mteb/ToolRetrieval` at `76d45e5` with 44,453 tools, catalog `sha256:7d56b70f3415`; toolhunch 0.1.0, commits `89deda9 (dirty)`, `544c81a (dirty)`, Python 3.14.3. Every abstention threshold applied here was chosen on the dev runs (`20261004T172920Z`, `20261004T173216Z`, `20261004T173356Z`, `20261004T163119Z`, `20261004T173616Z`, `20261004T173844Z`, `20261004T174644Z`, `20261004T180157Z`, `20261004T180701Z`, `20261004T180955Z`, `20261004T181450Z`, `20261004T181826Z`); no held-out search chose one.

**CLM (Mac bf16 MPS).** We run CLM on a Mac (transformers, bf16, MPS) instead of vLLM on CUDA. Its parity with our Modal deployment was checked on what reports publish, P@1 per cell and the answer-or-abstain decision (`bench/results/2026-10-toolret-decision-p2/clm-parity.json`); its figures stay provisional until the CLM authors confirm parity. `clm-serve` keeps the vectors of the texts it has embedded (its action cache, on by default), and our arms ask the same requests and cards more than once, so its latency here is mostly a warm-cache latency.

**Clef.** Workers AI serves the current Clef behind `@cf/cloudflare/clef`, and no version can be pinned: a later run may answer differently from the one reported here.

**Clef-flash.** Workers AI serves the current Clef-flash behind `@cf/cloudflare/clef-flash`, and no version can be pinned: a later run may answer differently from the one reported here.

**Local deciders.** Strands Decider 2B and CLM (Mac bf16 MPS) ran on one machine (Apple M5 Max, 128 GB, macOS 26.6.2): their cost reads “local”, and their latency is that machine's, not comparable like for like with a hosted API's.

## Runs

| split | role | run | tasks | sources | negatives | repeat | K | deciders (max detail) | reserved option | model-source fallbacks | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | main | `20261004T172920Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | strands (FULL) | on | 20 of 50 | `13bd51a (dirty)` |
| dev | main | `20261004T173216Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | strands (BRIEF) | on | 20 of 50 | `13bd51a (dirty)` |
| dev | main | `20261004T173356Z` | 10 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 3 | 20 | strands (FULL) | on | - | `13bd51a (dirty)` |
| dev | main | `20261004T163119Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | clm-local (FULL) | on | 20 of 50 | `14988c2 (dirty)` |
| dev | main | `20261004T173616Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | clm-local (BRIEF) | on | 20 of 50 | `13bd51a (dirty)` |
| dev | main | `20261004T173844Z` | 10 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 3 | 20 | clm-local (FULL) | on | - | `13bd51a (dirty)` |
| dev | main | `20261004T174644Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | clef (FULL) | on | 20 of 50 | `e066e61 (dirty)` |
| dev | main | `20261004T180157Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | clef (BRIEF) | on | 20 of 50 | `e066e61 (dirty)` |
| dev | main | `20261004T180701Z` | 10 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 3 | 20 | clef (FULL) | on | - | `e066e61 (dirty)` |
| dev | main | `20261004T180955Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | clef-flash (FULL) | on | 20 of 50 | `e066e61 (dirty)` |
| dev | main | `20261004T181450Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | clef-flash (BRIEF) | on | 20 of 50 | `e066e61 (dirty)` |
| dev | main | `20261004T181826Z` | 10 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 3 | 20 | clef-flash (FULL) | on | - | `e066e61 (dirty)` |
| heldout | main | `20261004T191646Z` | 200 (`bench/tasks/toolret-heldout-200.json`, sha256 `b632db8a5cb3`) | plain, model | yes | 1 | 20, 50 | strands (FULL) | on | 81 of 200 | `89deda9 (dirty)` |
| heldout | main | `20261004T194433Z` | 200 (`bench/tasks/toolret-heldout-200.json`, sha256 `b632db8a5cb3`) | plain, model | yes | 1 | 20, 50 | clm-local (BRIEF) | on | 81 of 200 | `544c81a (dirty)` |
| heldout | main | `20261004T200546Z` | 200 (`bench/tasks/toolret-heldout-200.json`, sha256 `b632db8a5cb3`) | plain, model | yes | 1 | 20, 50 | clef (BRIEF) | on | 81 of 200 | `544c81a (dirty)` |
| heldout | main | `20261004T205108Z` | 200 (`bench/tasks/toolret-heldout-200.json`, sha256 `b632db8a5cb3`) | plain, model | yes | 1 | 20, 50 | clef-flash (BRIEF) | on | 81 of 200 | `544c81a (dirty)` |

| decider | model | prompt version | declared limits |
|---|---|---|---|
| strands | `strands-decider-2B-hobson-v19@127.0.0.1:8000` | `tool-choice-v1` | strands-labs/strands-decider @ 75c9fd3: schema.py MAX_CHOICE_OPTIONS = 255, score levels 2-10; StrandsAgents/strands-decider-2B-hobson-v19 @ bb282d7: hobson_config.json max_length 4096; runs locally (checked 2026-10-04) |
| clm-local | `clm-latest@127.0.0.1:8700` | `tool-choice-v1` | Contrastive-LM/CLM @ bb42c6c: clm-serve cuts each text at 2,048 tokens by default; bench/deploy/clm_modal.py --max-model-len 2048; a 7,700-token option billed as 2,048 tokens (probe 2026-09-28) (checked 2026-09-28) |
| clef | `clef@api.cloudflare.com` | `tool-choice-v1` | developers.cloudflare.com/workers-ai/models/clef: 65,536-token context, 1-64 questions, $0.24 per M input tokens, no output charge; option cap: a 255-option choice accepted on 2026-10-04 (api-version 2026-10-01.epoch), more not tried (checked 2026-10-04) |
| clef-flash | `clef-flash@api.cloudflare.com` | `tool-choice-v1` | developers.cloudflare.com/workers-ai/models/clef-flash: 65,536-token context, 1-64 questions, $0.09 per M input tokens, no output charge; option cap: a 255-option choice accepted on 2026-10-04 (api-version 2026-10-01.epoch), more not tried (checked 2026-10-04) |

Model-written queries of `20261004T172920Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T173216Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T163119Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T173616Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T174644Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T180157Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T180955Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T181450Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T191646Z`: `bench/tasks/toolret-heldout-200.model-queries.json` (sha256 `d8df10441cfb`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T194433Z`: `bench/tasks/toolret-heldout-200.model-queries.json` (sha256 `d8df10441cfb`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T200546Z`: `bench/tasks/toolret-heldout-200.model-queries.json` (sha256 `d8df10441cfb`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261004T205108Z`: `bench/tasks/toolret-heldout-200.model-queries.json` (sha256 `d8df10441cfb`), written by `gpt-5.4-mini-2026-03-17`.

## Held-out

`plain` searches with the ToolRet request, `model` with the queries a model wrote for the task, and `model (searched)` keeps the tasks where that model did search: on 81 of 200 tasks it answered without searching, and the request itself was used.

### P@1

P@1 measures a relevant tool first on positive requests, ignoring abstention; it does not measure task completion.

| arm | source | positives | P@1 (95% CI) | hybrid P@1 | Δ vs hybrid (95% CI) | ceiling |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 200 | 0.220 (0.17 to 0.28) | 0.220 | - | 0.590 |
| hybrid@20 | model | 200 | 0.215 (0.16 to 0.27) | 0.215 | - | 0.600 |
| hybrid@20 | model (searched) | 119 | 0.202 (0.13 to 0.28) | 0.202 | - | 0.647 |
| hybrid+strands@20 | plain | 200 | 0.285 (0.23 to 0.34) | 0.220 | +0.065 (0.01 to 0.12) | 0.590 |
| hybrid+strands@20 | model | 200 | 0.305 (0.24 to 0.37) | 0.215 | +0.090 (0.04 to 0.15) | 0.600 |
| hybrid+strands@20 | model (searched) | 119 | 0.286 (0.21 to 0.37) | 0.202 | +0.084 (0.01 to 0.16) | 0.647 |
| hybrid@50 | plain | 200 | 0.220 (0.17 to 0.28) | 0.220 | - | 0.715 |
| hybrid@50 | model | 200 | 0.215 (0.16 to 0.27) | 0.215 | - | 0.720 |
| hybrid@50 | model (searched) | 119 | 0.202 (0.13 to 0.28) | 0.202 | - | 0.748 |
| hybrid+strands@50 | plain | 200 | 0.280 (0.22 to 0.34) | 0.220 | +0.060 (0.01 to 0.12) | 0.715 |
| hybrid+strands@50 | model | 200 | 0.275 (0.21 to 0.34) | 0.215 | +0.060 (0.01 to 0.12) | 0.720 |
| hybrid+strands@50 | model (searched) | 119 | 0.227 (0.15 to 0.30) | 0.202 | +0.025 (-0.06 to 0.10) | 0.748 |
| hybrid+clm-local@20 | plain | 200 | 0.130 (0.09 to 0.17) | 0.220 | -0.090 (-0.15 to -0.03) | 0.590 |
| hybrid+clm-local@20 | model | 200 | 0.120 (0.08 to 0.17) | 0.215 | -0.095 (-0.16 to -0.03) | 0.600 |
| hybrid+clm-local@20 | model (searched) | 119 | 0.109 (0.06 to 0.17) | 0.202 | -0.092 (-0.17 to -0.02) | 0.647 |
| hybrid+clm-local@50 | plain | 200 | 0.105 (0.07 to 0.15) | 0.220 | -0.115 (-0.17 to -0.05) | 0.715 |
| hybrid+clm-local@50 | model | 200 | 0.090 (0.06 to 0.13) | 0.215 | -0.125 (-0.19 to -0.06) | 0.720 |
| hybrid+clm-local@50 | model (searched) | 119 | 0.084 (0.03 to 0.13) | 0.202 | -0.118 (-0.20 to -0.03) | 0.748 |
| hybrid+clef@20 | plain | 200 | 0.310 (0.24 to 0.37) | 0.220 | +0.090 (0.04 to 0.14) | 0.590 |
| hybrid+clef@20 | model | 200 | 0.330 (0.27 to 0.40) | 0.215 | +0.115 (0.06 to 0.18) | 0.600 |
| hybrid+clef@20 | model (searched) | 119 | 0.311 (0.23 to 0.40) | 0.202 | +0.109 (0.03 to 0.20) | 0.647 |
| hybrid+clef@50 | plain | 200 | 0.365 (0.29 to 0.43) | 0.220 | +0.145 (0.09 to 0.21) | 0.715 |
| hybrid+clef@50 | model | 200 | 0.400 (0.34 to 0.47) | 0.215 | +0.185 (0.12 to 0.24) | 0.720 |
| hybrid+clef@50 | model (searched) | 119 | 0.387 (0.29 to 0.48) | 0.202 | +0.185 (0.10 to 0.27) | 0.748 |
| hybrid+clef-flash@20 | plain | 200 | 0.240 (0.18 to 0.30) | 0.220 | +0.020 (-0.04 to 0.07) | 0.590 |
| hybrid+clef-flash@20 | model | 200 | 0.230 (0.17 to 0.29) | 0.215 | +0.015 (-0.04 to 0.08) | 0.600 |
| hybrid+clef-flash@20 | model (searched) | 119 | 0.227 (0.15 to 0.30) | 0.202 | +0.025 (-0.05 to 0.10) | 0.647 |
| hybrid+clef-flash@50 | plain | 200 | 0.280 (0.22 to 0.34) | 0.220 | +0.060 (-0.01 to 0.12) | 0.715 |
| hybrid+clef-flash@50 | model | 200 | 0.290 (0.23 to 0.35) | 0.215 | +0.075 (0.01 to 0.14) | 0.720 |
| hybrid+clef-flash@50 | model (searched) | 119 | 0.277 (0.20 to 0.36) | 0.202 | +0.076 (-0.01 to 0.17) | 0.748 |

**Lower detail.** hybrid+strands@50 plain: 161 of 400 searches (asks FULL 239, BRIEF 157, NAME 4); hybrid+strands@50 model: 136 of 400 searches (asks FULL 264, BRIEF 128, NAME 8); hybrid+strands@50 model (searched): 64 of 238 searches (asks FULL 174, BRIEF 56, NAME 8). To fit the model's window the planner sent these searches' cards below the detail allowed, so these rows mix detail levels; their threshold keys record the detail allowed, not the detail sent.

### Abstention

Positive and negative requests are pooled below; the negative share is shown beside every row.

| arm | source | rule | τ | searches | errors | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate (95% CI) | abstention precision | abstention recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid@20 | plain | answer always | - | 400 | 0 | 1.000 | 0.110 | 44 | 356 | 0 | 50% | 0.890 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@20 | model | answer always | - | 400 | 0 | 1.000 | 0.107 | 43 | 357 | 0 | 50% | 0.892 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.101 | 24 | 214 | 0 | 50% | 0.899 (0.86 to 0.93) | n/a | 0.000 |
| hybrid+strands@20 | plain | answer always | - | 400 | 0 | 1.000 | 0.142 | 57 | 343 | 0 | 50% | 0.858 (0.83 to 0.89) | n/a | 0.000 |
| hybrid+strands@20 | plain | reserved | - | 400 | 0 | 0.945 | 0.151 | 57 | 321 | 22 | 50% | 0.802 (0.76 to 0.84) | 0.864 | 0.067 |
| hybrid+strands@20 | plain | with an abstention threshold | 0.95 | 400 | 0 | 0.005 | 1.000 | 2 | 0 | 398 | 50% | 0.000 (0.00 to 0.00) | 0.709 | 1.000 |
| hybrid+strands@20 | model | answer always | - | 400 | 0 | 1.000 | 0.152 | 61 | 339 | 0 | 50% | 0.848 (0.81 to 0.88) | n/a | 0.000 |
| hybrid+strands@20 | model | reserved | - | 400 | 0 | 0.932 | 0.161 | 60 | 313 | 27 | 50% | 0.782 (0.74 to 0.82) | 0.852 | 0.082 |
| hybrid+strands@20 | model | with an abstention threshold | 0.95 | 400 | 0 | 0.003 | 1.000 | 1 | 0 | 399 | 50% | 0.000 (0.00 to 0.00) | 0.702 | 1.000 |
| hybrid+strands@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.143 | 34 | 204 | 0 | 50% | 0.857 (0.82 to 0.89) | n/a | 0.000 |
| hybrid+strands@20 | model (searched) | reserved | - | 238 | 0 | 0.954 | 0.145 | 33 | 194 | 11 | 50% | 0.815 (0.76 to 0.87) | 0.909 | 0.062 |
| hybrid+strands@20 | model (searched) | with an abstention threshold | 0.95 | 238 | 0 | 0.000 | n/a | 0 | 0 | 238 | 50% | 0.000 (0.00 to 0.00) | 0.676 | 1.000 |
| hybrid@50 | plain | answer always | - | 400 | 0 | 1.000 | 0.110 | 44 | 356 | 0 | 50% | 0.890 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@50 | model | answer always | - | 400 | 0 | 1.000 | 0.107 | 43 | 357 | 0 | 50% | 0.892 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@50 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.101 | 24 | 214 | 0 | 50% | 0.899 (0.86 to 0.93) | n/a | 0.000 |
| hybrid+strands@50 | plain | answer always | - | 400 | 0 | 1.000 | 0.140 | 56 | 344 | 0 | 50% | 0.860 (0.83 to 0.89) | n/a | 0.000 |
| hybrid+strands@50 | plain | reserved | - | 400 | 0 | 0.978 | 0.143 | 56 | 335 | 9 | 50% | 0.838 (0.80 to 0.87) | 0.778 | 0.027 |
| hybrid+strands@50 | plain | with an abstention threshold | 0.70 | 400 | 0 | 0.028 | 0.455 | 5 | 6 | 389 | 50% | 0.015 (0.00 to 0.03) | 0.645 | 0.977 |
| hybrid+strands@50 | model | answer always | - | 400 | 0 | 1.000 | 0.138 | 55 | 345 | 0 | 50% | 0.863 (0.83 to 0.89) | n/a | 0.000 |
| hybrid+strands@50 | model | reserved | - | 400 | 0 | 0.968 | 0.140 | 54 | 333 | 13 | 50% | 0.833 (0.79 to 0.87) | 0.769 | 0.039 |
| hybrid+strands@50 | model | with an abstention threshold | 0.70 | 400 | 0 | 0.033 | 0.385 | 5 | 8 | 387 | 50% | 0.020 (0.01 to 0.04) | 0.646 | 0.977 |
| hybrid+strands@50 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.113 | 27 | 211 | 0 | 50% | 0.887 (0.85 to 0.92) | n/a | 0.000 |
| hybrid+strands@50 | model (searched) | reserved | - | 238 | 0 | 0.975 | 0.112 | 26 | 206 | 6 | 50% | 0.866 (0.82 to 0.91) | 0.833 | 0.034 |
| hybrid+strands@50 | model (searched) | with an abstention threshold | 0.70 | 238 | 0 | 0.021 | 0.200 | 1 | 4 | 233 | 50% | 0.017 (0.00 to 0.04) | 0.631 | 0.987 |
| hybrid+clm-local@20 | plain | answer always | - | 400 | 0 | 1.000 | 0.065 | 26 | 374 | 0 | 50% | 0.935 (0.91 to 0.96) | n/a | 0.000 |
| hybrid+clm-local@20 | plain | reserved | - | 400 | 0 | 0.938 | 0.064 | 24 | 351 | 25 | 50% | 0.877 (0.84 to 0.92) | 0.760 | 0.067 |
| hybrid+clm-local@20 | plain | with an abstention threshold | 0.90 | 400 | 0 | 0.020 | 0.250 | 2 | 6 | 392 | 50% | 0.015 (0.00 to 0.03) | 0.704 | 0.979 |
| hybrid+clm-local@20 | model | answer always | - | 400 | 0 | 1.000 | 0.060 | 24 | 376 | 0 | 50% | 0.940 (0.92 to 0.96) | n/a | 0.000 |
| hybrid+clm-local@20 | model | reserved | - | 400 | 0 | 0.927 | 0.059 | 22 | 349 | 29 | 50% | 0.873 (0.83 to 0.91) | 0.793 | 0.082 |
| hybrid+clm-local@20 | model | with an abstention threshold | 0.90 | 400 | 0 | 0.025 | 0.200 | 2 | 8 | 390 | 50% | 0.020 (0.01 to 0.04) | 0.697 | 0.971 |
| hybrid+clm-local@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.055 | 13 | 225 | 0 | 50% | 0.945 (0.92 to 0.97) | n/a | 0.000 |
| hybrid+clm-local@20 | model (searched) | reserved | - | 238 | 0 | 0.966 | 0.057 | 13 | 217 | 8 | 50% | 0.912 (0.87 to 0.95) | 0.875 | 0.043 |
| hybrid+clm-local@20 | model (searched) | with an abstention threshold | 0.90 | 238 | 0 | 0.008 | 0.000 | 0 | 2 | 236 | 50% | 0.008 (0.00 to 0.03) | 0.674 | 0.988 |
| hybrid+clm-local@50 | plain | answer always | - | 400 | 0 | 1.000 | 0.052 | 21 | 379 | 0 | 50% | 0.948 (0.93 to 0.97) | n/a | 0.000 |
| hybrid+clm-local@50 | plain | reserved | - | 400 | 0 | 0.960 | 0.052 | 20 | 364 | 16 | 50% | 0.910 (0.87 to 0.94) | 0.625 | 0.039 |
| hybrid+clm-local@50 | plain | with an abstention threshold | 0.90 | 400 | 0 | 0.005 | 0.000 | 0 | 2 | 398 | 50% | 0.005 (0.00 to 0.01) | 0.641 | 0.992 |
| hybrid+clm-local@50 | model | answer always | - | 400 | 0 | 1.000 | 0.045 | 18 | 382 | 0 | 50% | 0.955 (0.94 to 0.97) | n/a | 0.000 |
| hybrid+clm-local@50 | model | reserved | - | 400 | 0 | 0.950 | 0.045 | 17 | 363 | 20 | 50% | 0.907 (0.87 to 0.94) | 0.700 | 0.055 |
| hybrid+clm-local@50 | model | with an abstention threshold | 0.90 | 400 | 0 | 0.005 | 0.000 | 0 | 2 | 398 | 50% | 0.005 (0.00 to 0.01) | 0.638 | 0.992 |
| hybrid+clm-local@50 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.042 | 10 | 228 | 0 | 50% | 0.958 (0.93 to 0.98) | n/a | 0.000 |
| hybrid+clm-local@50 | model (searched) | reserved | - | 238 | 0 | 0.983 | 0.043 | 10 | 224 | 4 | 50% | 0.941 (0.91 to 0.97) | 1.000 | 0.027 |
| hybrid+clm-local@50 | model (searched) | with an abstention threshold | 0.90 | 238 | 0 | 0.000 | n/a | 0 | 0 | 238 | 50% | 0.000 (0.00 to 0.00) | 0.626 | 1.000 |
| hybrid+clef@20 | plain | answer always | - | 400 | 0 | 1.000 | 0.155 | 62 | 338 | 0 | 50% | 0.845 (0.81 to 0.88) | n/a | 0.000 |
| hybrid+clef@20 | plain | reserved | - | 400 | 0 | 0.945 | 0.161 | 61 | 317 | 22 | 50% | 0.792 (0.75 to 0.83) | 0.955 | 0.074 |
| hybrid+clef@20 | plain | with an abstention threshold | 0.95 | 400 | 0 | 0.030 | 0.500 | 6 | 6 | 388 | 50% | 0.015 (0.00 to 0.03) | 0.711 | 0.979 |
| hybrid+clef@20 | model | answer always | - | 400 | 0 | 1.000 | 0.165 | 66 | 334 | 0 | 50% | 0.835 (0.80 to 0.87) | n/a | 0.000 |
| hybrid+clef@20 | model | reserved | - | 400 | 0 | 0.955 | 0.173 | 66 | 316 | 18 | 50% | 0.790 (0.75 to 0.83) | 1.000 | 0.064 |
| hybrid+clef@20 | model | with an abstention threshold | 0.95 | 400 | 0 | 0.020 | 0.500 | 4 | 4 | 392 | 50% | 0.010 (0.00 to 0.02) | 0.704 | 0.986 |
| hybrid+clef@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.155 | 37 | 201 | 0 | 50% | 0.845 (0.80 to 0.89) | n/a | 0.000 |
| hybrid+clef@20 | model (searched) | reserved | - | 238 | 0 | 0.987 | 0.157 | 37 | 198 | 3 | 50% | 0.832 (0.79 to 0.88) | 1.000 | 0.019 |
| hybrid+clef@20 | model (searched) | with an abstention threshold | 0.95 | 238 | 0 | 0.013 | 0.333 | 1 | 2 | 235 | 50% | 0.008 (0.00 to 0.02) | 0.677 | 0.988 |
| hybrid+clef@50 | plain | answer always | - | 400 | 0 | 1.000 | 0.182 | 73 | 327 | 0 | 50% | 0.818 (0.79 to 0.85) | n/a | 0.000 |
| hybrid+clef@50 | plain | reserved | - | 400 | 0 | 0.970 | 0.188 | 73 | 315 | 12 | 50% | 0.787 (0.75 to 0.82) | 1.000 | 0.047 |
| hybrid+clef@50 | plain | with an abstention threshold | 0.95 | 400 | 0 | 0.007 | 0.667 | 2 | 1 | 397 | 50% | 0.003 (0.00 to 0.01) | 0.645 | 0.996 |
| hybrid+clef@50 | model | answer always | - | 400 | 0 | 1.000 | 0.200 | 80 | 320 | 0 | 50% | 0.800 (0.77 to 0.83) | n/a | 0.000 |
| hybrid+clef@50 | model | reserved | - | 400 | 0 | 0.980 | 0.204 | 80 | 312 | 8 | 50% | 0.780 (0.74 to 0.82) | 1.000 | 0.031 |
| hybrid+clef@50 | model | with an abstention threshold | 0.95 | 400 | 0 | 0.003 | 1.000 | 1 | 0 | 399 | 50% | 0.000 (0.00 to 0.00) | 0.642 | 1.000 |
| hybrid+clef@50 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.193 | 46 | 192 | 0 | 50% | 0.807 (0.76 to 0.85) | n/a | 0.000 |
| hybrid+clef@50 | model (searched) | reserved | - | 238 | 0 | 0.992 | 0.195 | 46 | 190 | 2 | 50% | 0.798 (0.75 to 0.84) | 1.000 | 0.013 |
| hybrid+clef@50 | model (searched) | with an abstention threshold | 0.95 | 238 | 0 | 0.000 | n/a | 0 | 0 | 238 | 50% | 0.000 (0.00 to 0.00) | 0.626 | 1.000 |
| hybrid+clef-flash@20 | plain | answer always | - | 400 | 0 | 1.000 | 0.120 | 48 | 352 | 0 | 50% | 0.880 (0.85 to 0.91) | n/a | 0.000 |
| hybrid+clef-flash@20 | plain | reserved | - | 400 | 0 | 0.965 | 0.124 | 48 | 338 | 14 | 50% | 0.845 (0.81 to 0.88) | 1.000 | 0.050 |
| hybrid+clef-flash@20 | plain | with an abstention threshold | 0.95 | 400 | 0 | 0.028 | 0.455 | 5 | 6 | 389 | 50% | 0.015 (0.00 to 0.03) | 0.710 | 0.979 |
| hybrid+clef-flash@20 | model | answer always | - | 400 | 0 | 1.000 | 0.115 | 46 | 354 | 0 | 50% | 0.885 (0.85 to 0.92) | n/a | 0.000 |
| hybrid+clef-flash@20 | model | reserved | - | 400 | 0 | 0.973 | 0.118 | 46 | 343 | 11 | 50% | 0.858 (0.82 to 0.89) | 1.000 | 0.039 |
| hybrid+clef-flash@20 | model | with an abstention threshold | 0.95 | 400 | 0 | 0.020 | 0.750 | 6 | 2 | 392 | 50% | 0.005 (0.00 to 0.01) | 0.709 | 0.993 |
| hybrid+clef-flash@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.113 | 27 | 211 | 0 | 50% | 0.887 (0.85 to 0.92) | n/a | 0.000 |
| hybrid+clef-flash@20 | model (searched) | reserved | - | 238 | 0 | 0.987 | 0.115 | 27 | 208 | 3 | 50% | 0.874 (0.83 to 0.92) | 1.000 | 0.019 |
| hybrid+clef-flash@20 | model (searched) | with an abstention threshold | 0.95 | 238 | 0 | 0.013 | 0.667 | 2 | 1 | 235 | 50% | 0.004 (0.00 to 0.01) | 0.681 | 0.994 |
| hybrid+clef-flash@50 | plain | answer always | - | 400 | 0 | 1.000 | 0.140 | 56 | 344 | 0 | 50% | 0.860 (0.83 to 0.89) | n/a | 0.000 |
| hybrid+clef-flash@50 | plain | reserved | - | 400 | 0 | 0.980 | 0.143 | 56 | 336 | 8 | 50% | 0.840 (0.81 to 0.88) | 1.000 | 0.031 |
| hybrid+clef-flash@50 | plain | with an abstention threshold | 0.95 | 400 | 0 | 0.007 | 0.333 | 1 | 2 | 397 | 50% | 0.005 (0.00 to 0.01) | 0.642 | 0.992 |
| hybrid+clef-flash@50 | model | answer always | - | 400 | 0 | 1.000 | 0.145 | 58 | 342 | 0 | 50% | 0.855 (0.82 to 0.89) | n/a | 0.000 |
| hybrid+clef-flash@50 | model | reserved | - | 400 | 0 | 0.988 | 0.147 | 58 | 337 | 5 | 50% | 0.843 (0.81 to 0.88) | 1.000 | 0.020 |
| hybrid+clef-flash@50 | model | with an abstention threshold | 0.95 | 400 | 0 | 0.005 | 1.000 | 2 | 0 | 398 | 50% | 0.000 (0.00 to 0.00) | 0.643 | 1.000 |
| hybrid+clef-flash@50 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.139 | 33 | 205 | 0 | 50% | 0.861 (0.82 to 0.90) | n/a | 0.000 |
| hybrid+clef-flash@50 | model (searched) | reserved | - | 238 | 0 | 0.996 | 0.139 | 33 | 204 | 1 | 50% | 0.857 (0.82 to 0.89) | 1.000 | 0.007 |
| hybrid+clef-flash@50 | model (searched) | with an abstention threshold | 0.95 | 238 | 0 | 0.004 | 1.000 | 1 | 0 | 237 | 50% | 0.000 (0.00 to 0.00) | 0.629 | 1.000 |

### Latency

| arm | source | decision p50 ms | p95 | server p50 ms | p95 | retrieval p50 ms | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid@20 | plain | - | - | - | - | 587 | 618 |
| hybrid@20 | model | - | - | - | - | 607 | 1,802 |
| hybrid@20 | model (searched) | - | - | - | - | 1,205 | 1,809 |
| hybrid+strands@20 | plain | 81 | 128 | - | - | 587 | 618 |
| hybrid+strands@20 | model | 84 | 129 | - | - | 607 | 1,802 |
| hybrid+strands@20 | model (searched) | 87 | 129 | - | - | 1,205 | 1,809 |
| hybrid@50 | plain | - | - | - | - | 587 | 618 |
| hybrid@50 | model | - | - | - | - | 607 | 1,802 |
| hybrid@50 | model (searched) | - | - | - | - | 1,205 | 1,809 |
| hybrid+strands@50 | plain | 234 | 309 | - | - | 587 | 618 |
| hybrid+strands@50 | model | 240 | 299 | - | - | 607 | 1,802 |
| hybrid+strands@50 | model (searched) | 243 | 285 | - | - | 1,205 | 1,809 |
| hybrid+clm-local@20 | plain | 339 | 545 | 338 | 544 | 588 | 617 |
| hybrid+clm-local@20 | model | 279 | 540 | 278 | 539 | 596 | 1,773 |
| hybrid+clm-local@20 | model (searched) | 202 | 547 | 202 | 546 | 1,173 | 1,782 |
| hybrid+clm-local@50 | plain | 369 | 813 | 368 | 812 | 588 | 617 |
| hybrid+clm-local@50 | model | 230 | 746 | 229 | 745 | 596 | 1,773 |
| hybrid+clm-local@50 | model (searched) | 157 | 524 | 156 | 523 | 1,173 | 1,782 |
| hybrid+clef@20 | plain | 517 | 763 | - | - | 587 | 619 |
| hybrid+clef@20 | model | 514 | 751 | - | - | 595 | 1,781 |
| hybrid+clef@20 | model (searched) | 505 | 736 | - | - | 1,201 | 1,786 |
| hybrid+clef@50 | plain | 782 | 1,115 | - | - | 587 | 619 |
| hybrid+clef@50 | model | 751 | 1,067 | - | - | 595 | 1,781 |
| hybrid+clef@50 | model (searched) | 736 | 1,056 | - | - | 1,201 | 1,786 |
| hybrid+clef-flash@20 | plain | 341 | 596 | - | - | 591 | 632 |
| hybrid+clef-flash@20 | model | 340 | 616 | - | - | 602 | 1,787 |
| hybrid+clef-flash@20 | model (searched) | 326 | 658 | - | - | 1,202 | 1,805 |
| hybrid+clef-flash@50 | plain | 589 | 771 | - | - | 591 | 632 |
| hybrid+clef-flash@50 | model | 575 | 814 | - | - | 602 | 1,787 |
| hybrid+clef-flash@50 | model (searched) | 529 | 816 | - | - | 1,202 | 1,805 |

### Cost per 1,000 searches

| arm | source | asks / search | input tokens / search | USD | CLM busy USD | CLM wall USD |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 0.00 | 0 | 0.0000 | - | - |
| hybrid@20 | model | 0.00 | 0 | 0.0000 | - | - |
| hybrid@20 | model (searched) | 0.00 | 0 | 0.0000 | - | - |
| hybrid+strands@20 | plain | 1.00 | 1,149 | local | - | - |
| hybrid+strands@20 | model | 1.00 | 1,129 | local | - | - |
| hybrid+strands@20 | model (searched) | 1.00 | 1,120 | local | - | - |
| hybrid@50 | plain | 0.00 | 0 | 0.0000 | - | - |
| hybrid@50 | model | 0.00 | 0 | 0.0000 | - | - |
| hybrid@50 | model (searched) | 0.00 | 0 | 0.0000 | - | - |
| hybrid+strands@50 | plain | 1.00 | 2,390 | local | - | - |
| hybrid+strands@50 | model | 1.00 | 2,358 | local | - | - |
| hybrid+strands@50 | model (searched) | 1.00 | 2,342 | local | - | - |
| hybrid+clm-local@20 | plain | 1.00 | 391 | local | - | - |
| hybrid+clm-local@20 | model | 1.00 | 288 | local | - | - |
| hybrid+clm-local@20 | model (searched) | 1.00 | 209 | local | - | - |
| hybrid+clm-local@50 | plain | 1.00 | 377 | local | - | - |
| hybrid+clm-local@50 | model | 1.00 | 300 | local | - | - |
| hybrid+clm-local@50 | model (searched) | 1.00 | 217 | local | - | - |
| hybrid+clef@20 | plain | 1.00 | 1,112 | 0.2668 | - | - |
| hybrid+clef@20 | model | 1.00 | 1,106 | 0.2653 | - | - |
| hybrid+clef@20 | model (searched) | 1.00 | 1,111 | 0.2667 | - | - |
| hybrid+clef@50 | plain | 1.00 | 2,589 | 0.6214 | - | - |
| hybrid+clef@50 | model | 1.00 | 2,556 | 0.6135 | - | - |
| hybrid+clef@50 | model (searched) | 1.00 | 2,544 | 0.6106 | - | - |
| hybrid+clef-flash@20 | plain | 1.00 | 1,112 | 0.1000 | - | - |
| hybrid+clef-flash@20 | model | 1.00 | 1,106 | 0.0995 | - | - |
| hybrid+clef-flash@20 | model (searched) | 1.00 | 1,111 | 0.1000 | - | - |
| hybrid+clef-flash@50 | plain | 1.00 | 2,589 | 0.2330 | - | - |
| hybrid+clef-flash@50 | model | 1.00 | 2,556 | 0.2301 | - | - |
| hybrid+clef-flash@50 | model (searched) | 1.00 | 2,544 | 0.2290 | - | - |

Per arm, all sources together: the summed seconds of its decisions, which CLM's wall figure prices, next to the arm's own clock time and the asks the decision cache replayed (hits) or sent (misses) during it.

| arm | searches | errors | decision s | arm clock s | cache hits | cache misses |
|---|---:|---:|---:|---:|---:|---:|
| hybrid@20 | 800 | 0 | - | 271.0 | - | - |
| hybrid+strands@20 | 800 | 0 | 69.7 | 46.1 | 289 | 511 |
| hybrid@50 | 800 | 0 | - | 0.0 | - | - |
| hybrid+strands@50 | 800 | 0 | 190.0 | 132.3 | 252 | 548 |
| hybrid+clm-local@20 | 800 | 0 | 224.0 | 128.1 | 289 | 511 |
| hybrid+clm-local@50 | 800 | 0 | 261.8 | 154.8 | 252 | 548 |
| hybrid+clef@20 | 800 | 0 | 429.0 | 277.2 | 289 | 511 |
| hybrid+clef@50 | 800 | 0 | 629.4 | 432.5 | 252 | 548 |
| hybrid+clef-flash@20 | 800 | 0 | 311.3 | 198.9 | 289 | 511 |
| hybrid+clef-flash@50 | 800 | 0 | 456.0 | 306.6 | 252 | 548 |

## Dev

### Thresholds

| threshold key | decider | K | payload | searches (positives + negatives) | τ | U at τ | U at 0.00 | coverage at τ | wrong-tool rate at τ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `strands-decider-2B-hobson-v19@127.0.0.1:8000\|tool-choice-v1\|db6d5d57dd5971bd\|choice` | strands | 20 | reserved option, FULL, questions [[21]] | 200 (100 + 100) | 0.95 | -0.010 | -0.680 | 0.050 | 0.030 |
| `strands-decider-2B-hobson-v19@127.0.0.1:8000\|tool-choice-v1\|5be950b6c011df2d\|choice` | strands | 50 | reserved option, FULL, questions [[51]] | 200 (100 + 100) | 0.70 | -0.010 | -0.690 | 0.060 | 0.035 |
| `strands-decider-2B-hobson-v19@127.0.0.1:8000\|tool-choice-v1\|55db837d3c1774fe\|choice` | strands | 20 | reserved option, BRIEF, questions [[21]] | 200 (100 + 100) | 0.95 | -0.060 | -0.700 | 0.110 | 0.085 |
| `strands-decider-2B-hobson-v19@127.0.0.1:8000\|tool-choice-v1\|38dc60cd902e7d70\|choice` | strands | 50 | reserved option, BRIEF, questions [[51]] | 200 (100 + 100) | 0.95 | -0.015 | -0.705 | 0.025 | 0.020 |
| `clm-latest@127.0.0.1:8700\|tool-choice-v1\|db6d5d57dd5971bd\|choice` | clm-local | 20 | reserved option, FULL, questions [[21]] | 200 (100 + 100) | 0.90 | 0.020 | -0.800 | 0.020 | 0.000 |
| `clm-latest@127.0.0.1:8700\|tool-choice-v1\|5be950b6c011df2d\|choice` | clm-local | 50 | reserved option, FULL, questions [[51]] | 200 (100 + 100) | 0.80 | 0.020 | -0.850 | 0.020 | 0.000 |
| `clm-latest@127.0.0.1:8700\|tool-choice-v1\|55db837d3c1774fe\|choice` | clm-local | 20 | reserved option, BRIEF, questions [[21]] | 200 (100 + 100) | 0.90 | 0.010 | -0.775 | 0.040 | 0.015 |
| `clm-latest@127.0.0.1:8700\|tool-choice-v1\|38dc60cd902e7d70\|choice` | clm-local | 50 | reserved option, BRIEF, questions [[51]] | 200 (100 + 100) | 0.90 | 0.005 | -0.880 | 0.005 | 0.000 |
| `clef@api.cloudflare.com\|tool-choice-v1\|db6d5d57dd5971bd\|choice` | clef | 20 | reserved option, FULL, questions [[21]] | 200 (100 + 100) | 0.95 | -0.025 | -0.585 | 0.025 | 0.025 |
| `clef@api.cloudflare.com\|tool-choice-v1\|5be950b6c011df2d\|choice` | clef | 50 | reserved option, FULL, questions [[51]] | 200 (100 + 100) | 0.95 | -0.030 | -0.660 | 0.030 | 0.030 |
| `clef@api.cloudflare.com\|tool-choice-v1\|55db837d3c1774fe\|choice` | clef | 20 | reserved option, BRIEF, questions [[21]] | 200 (100 + 100) | 0.95 | -0.045 | -0.625 | 0.045 | 0.045 |
| `clef@api.cloudflare.com\|tool-choice-v1\|38dc60cd902e7d70\|choice` | clef | 50 | reserved option, BRIEF, questions [[51]] | 200 (100 + 100) | 0.95 | -0.030 | -0.610 | 0.030 | 0.030 |
| `clef-flash@api.cloudflare.com\|tool-choice-v1\|db6d5d57dd5971bd\|choice` | clef-flash | 20 | reserved option, FULL, questions [[21]] | 200 (100 + 100) | 0.90 | -0.035 | -0.605 | 0.055 | 0.045 |
| `clef-flash@api.cloudflare.com\|tool-choice-v1\|5be950b6c011df2d\|choice` | clef-flash | 50 | reserved option, FULL, questions [[51]] | 200 (100 + 100) | 0.95 | -0.010 | -0.645 | 0.010 | 0.010 |
| `clef-flash@api.cloudflare.com\|tool-choice-v1\|55db837d3c1774fe\|choice` | clef-flash | 20 | reserved option, BRIEF, questions [[21]] | 200 (100 + 100) | 0.95 | -0.030 | -0.585 | 0.050 | 0.040 |
| `clef-flash@api.cloudflare.com\|tool-choice-v1\|38dc60cd902e7d70\|choice` | clef-flash | 50 | reserved option, BRIEF, questions [[51]] | 200 (100 + 100) | 0.95 | -0.010 | -0.650 | 0.010 | 0.010 |

### P@1 by dev run

P@1 on positives (their count in brackets), one column per dev run that made each search once; the ablations' settings are in the runs table.

| arm | source | `20261004T172920Z` | `20261004T173216Z` | `20261004T163119Z` | `20261004T173616Z` | `20261004T174644Z` | `20261004T180157Z` | `20261004T180955Z` | `20261004T181450Z` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) |
| hybrid@20 | model | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) |
| hybrid@20 | model (searched) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) |
| hybrid+strands@20 | plain | 0.280 (50) | 0.240 (50) | - | - | - | - | - | - |
| hybrid+strands@20 | model | 0.260 (50) | 0.300 (50) | - | - | - | - | - | - |
| hybrid+strands@20 | model (searched) | 0.267 (30) | 0.333 (30) | - | - | - | - | - | - |
| hybrid@50 | plain | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) |
| hybrid@50 | model | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) |
| hybrid@50 | model (searched) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) |
| hybrid+strands@50 | plain | 0.280 (50) | 0.260 (50) | - | - | - | - | - | - |
| hybrid+strands@50 | model | 0.300 (50) | 0.300 (50) | - | - | - | - | - | - |
| hybrid+strands@50 | model (searched) | 0.300 (30) | 0.300 (30) | - | - | - | - | - | - |
| hybrid+clm-local@20 | plain | - | - | 0.100 (50) | 0.120 (50) | - | - | - | - |
| hybrid+clm-local@20 | model | - | - | 0.100 (50) | 0.140 (50) | - | - | - | - |
| hybrid+clm-local@20 | model (searched) | - | - | 0.067 (30) | 0.133 (30) | - | - | - | - |
| hybrid+clm-local@50 | plain | - | - | 0.100 (50) | 0.120 (50) | - | - | - | - |
| hybrid+clm-local@50 | model | - | - | 0.100 (50) | 0.100 (50) | - | - | - | - |
| hybrid+clm-local@50 | model (searched) | - | - | 0.067 (30) | 0.067 (30) | - | - | - | - |
| hybrid+clef@20 | plain | - | - | - | - | 0.280 (50) | 0.300 (50) | - | - |
| hybrid+clef@20 | model | - | - | - | - | 0.320 (50) | 0.340 (50) | - | - |
| hybrid+clef@20 | model (searched) | - | - | - | - | 0.333 (30) | 0.367 (30) | - | - |
| hybrid+clef@50 | plain | - | - | - | - | 0.300 (50) | 0.320 (50) | - | - |
| hybrid+clef@50 | model | - | - | - | - | 0.320 (50) | 0.380 (50) | - | - |
| hybrid+clef@50 | model (searched) | - | - | - | - | 0.300 (30) | 0.400 (30) | - | - |
| hybrid+clef-flash@20 | plain | - | - | - | - | - | - | 0.280 (50) | 0.300 (50) |
| hybrid+clef-flash@20 | model | - | - | - | - | - | - | 0.320 (50) | 0.340 (50) |
| hybrid+clef-flash@20 | model (searched) | - | - | - | - | - | - | 0.333 (30) | 0.367 (30) |
| hybrid+clef-flash@50 | plain | - | - | - | - | - | - | 0.300 (50) | 0.320 (50) |
| hybrid+clef-flash@50 | model | - | - | - | - | - | - | 0.300 (50) | 0.320 (50) |
| hybrid+clef-flash@50 | model (searched) | - | - | - | - | - | - | 0.300 (30) | 0.333 (30) |

### Determinism

Largest change of one probability (a card's or the reserved option's) between repeats of the same search, over every repeated search of the arm.

| run | arm | searches repeated | largest abs Δp | at | within 0.01 |
|---|---|---:|---:|---|---|
| `20261004T173356Z` | hybrid+strands@20 | 20 | 0.0000 | apibank_query_38 (plain, positive) | yes |
| `20261004T173844Z` | hybrid+clm-local@20 | 20 | 0.0000 | apibank_query_38 (plain, positive) | yes |
| `20261004T180701Z` | hybrid+clef@20 | 20 | 0.0000 | apibank_query_38 (plain, positive) | yes |
| `20261004T181826Z` | hybrid+clef-flash@20 | 20 | 0.0000 | apibank_query_38 (plain, positive) | yes |

### Errors

No dev search failed.

## Token heuristic against Jev's count

For every exchange with Jev, the input tokens Jev reported against the heuristic's estimate that the planner budgets with: a least-squares line (reported = intercept + slope x estimate), and the ratios reported / estimate. A ratio under 1 means the heuristic counted more than Jev billed.

| split | Jev exchanges | intercept | slope | median reported / estimated | share under 1 |
|---|---:|---:|---:|---:|---:|
| dev | 0 | n/a | n/a | n/a | n/a |
| heldout | 0 | n/a | n/a | n/a | n/a |

## Notes

- **Outcomes.** A search is answered when the decider names a tool, correct when that tool is a gold one, and wrong otherwise. A negative is the same search with its gold tools removed from the candidates, so every answer to it is wrong.
- **P@1** counts positives only and ignores abstention: is the first card of the decider's ranking a gold tool? `hybrid P@1` asks the same of retrieval order on the same searches, and `ceiling` is the share of them with a gold tool among the K candidates. `Δ vs hybrid` is the decider's P@1 minus retrieval's own P@1 on the same positives. Its interval resamples tasks with both answers drawn together, so it says whether the decision helped even where the two separate intervals overlap.
- **Answer always.** This reinterprets the same ranking obtained with the none option available: its first card is chosen, ignoring abstention. It does not show what the model would answer without that option. The dev ablation tested that setting separately; for logprob it also changed the number of rounds.
- **Rules.** `reserved`: the decider abstains when its reserved "none of these" option is at least as likely as its best card. `with an abstention threshold`: it also abstains when that card's probability is below τ, the threshold dev chose for the search's threshold key (model, prompt version, payload shape, question kind). Probabilities compare only within one question, so a τ never crosses keys.
- **Rates.** coverage: answered / searches. Selective accuracy: correct / answered. Wrong-tool rate: wrong / searches. Abstention precision: abstentions on searches without a gold tool among the candidates / abstentions. Abstention recall: those abstentions / searches without a gold tool among the candidates.
- **Choosing τ.** Per threshold key, on dev: of τ = 0.00, 0.05, ..., 0.95, the one with the highest U = (correct - wrong) / searches, where abstaining counts 0; of equal U, the lower τ wins. It uses the dev runs that made each search once, both query sources, positives and negatives; a search that several of those runs made counts once. Fallback searches of the `model` source (its writer made no search, so the request was used) repeat their `plain` search and count twice in U(τ) and in the searches counts of the thresholds table. The held-out risk-coverage tables are there to read, never to choose.
- **Intervals.** 95% percentile bootstrap, 2,000 resamples with seed 0, resampling tasks: a task's positive search, its negative and their repeats are drawn together.
- **Errors.** A search whose decider raised `DecisionError` counts as an error and stays out of every rate, latency and cost figure.
- **Latency.** Decision: the critical path of the decider's calls as the client timed them, network included; a call the decision cache replays keeps the time the original call took. Server: the same by the server's own clock, where it reports one. Retrieval: the first hybrid retrieval of the search's queries, measured live; a query whose vector the embedding cache already held skips the embeddings call, so it reads faster than a cold one.
- **Cost.** Jev: reported input tokens at the price its declared limits give. Logprob: the reported usage, priced by genai-prices. CLM is paid in GPU time, at the list price in the run's manifest, two ways. Busy prices the server's own seconds per search, as if the GPU never sat idle: a lower bound. Wall prices the summed seconds of the arm's decision calls as the client saw them, as if one container served the searches one after another. It leaves out the cold starts, the idle gaps between arms and the scale-down window, which Modal also bills. Modal's CPU and memory charges come on top of both and are not included. Every arm also embeds its queries for hybrid retrieval, at the same cost for each arm, which the F1 retrieval results report.
- **Negatives** stand in for a catalog without the gold tools: the gold ids are dropped from a deeper retrieval, so the other tools' order can move slightly from what a smaller catalog would give.

## Risk-coverage over the threshold grid

### Dev: `strands-decider-2B-hobson-v19@127.0.0.1:8000|tool-choice-v1|db6d5d57dd5971bd|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.950 | 0.142 | 27 | 163 | 10 | 50% | 0.815 | -0.680 |
| 0.05 | 0.950 | 0.142 | 27 | 163 | 10 | 50% | 0.815 | -0.680 |
| 0.10 | 0.950 | 0.142 | 27 | 163 | 10 | 50% | 0.815 | -0.680 |
| 0.15 | 0.930 | 0.145 | 27 | 159 | 14 | 50% | 0.795 | -0.660 |
| 0.20 | 0.920 | 0.147 | 27 | 157 | 16 | 50% | 0.785 | -0.650 |
| 0.25 | 0.810 | 0.167 | 27 | 135 | 38 | 50% | 0.675 | -0.540 |
| 0.30 | 0.675 | 0.193 | 26 | 109 | 65 | 50% | 0.545 | -0.415 |
| 0.35 | 0.595 | 0.202 | 24 | 95 | 81 | 50% | 0.475 | -0.355 |
| 0.40 | 0.550 | 0.182 | 20 | 90 | 90 | 50% | 0.450 | -0.350 |
| 0.45 | 0.415 | 0.205 | 17 | 66 | 117 | 50% | 0.330 | -0.245 |
| 0.50 | 0.350 | 0.200 | 14 | 56 | 130 | 50% | 0.280 | -0.210 |
| 0.55 | 0.330 | 0.197 | 13 | 53 | 134 | 50% | 0.265 | -0.200 |
| 0.60 | 0.255 | 0.216 | 11 | 40 | 149 | 50% | 0.200 | -0.145 |
| 0.65 | 0.235 | 0.234 | 11 | 36 | 153 | 50% | 0.180 | -0.125 |
| 0.70 | 0.210 | 0.262 | 11 | 31 | 158 | 50% | 0.155 | -0.100 |
| 0.75 | 0.170 | 0.265 | 9 | 25 | 166 | 50% | 0.125 | -0.080 |
| 0.80 | 0.115 | 0.304 | 7 | 16 | 177 | 50% | 0.080 | -0.045 |
| 0.85 | 0.100 | 0.300 | 6 | 14 | 180 | 50% | 0.070 | -0.040 |
| 0.90 | 0.090 | 0.222 | 4 | 14 | 182 | 50% | 0.070 | -0.050 |
| **0.95** | 0.050 | 0.400 | 4 | 6 | 190 | 50% | 0.030 | -0.010 |

### Dev: `strands-decider-2B-hobson-v19@127.0.0.1:8000|tool-choice-v1|5be950b6c011df2d|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.980 | 0.148 | 29 | 167 | 4 | 50% | 0.835 | -0.690 |
| 0.05 | 0.980 | 0.148 | 29 | 167 | 4 | 50% | 0.835 | -0.690 |
| 0.10 | 0.970 | 0.149 | 29 | 165 | 6 | 50% | 0.825 | -0.680 |
| 0.15 | 0.795 | 0.182 | 29 | 130 | 41 | 50% | 0.650 | -0.505 |
| 0.20 | 0.625 | 0.232 | 29 | 96 | 75 | 50% | 0.480 | -0.335 |
| 0.25 | 0.515 | 0.262 | 27 | 76 | 97 | 50% | 0.380 | -0.245 |
| 0.30 | 0.390 | 0.244 | 19 | 59 | 122 | 50% | 0.295 | -0.200 |
| 0.35 | 0.330 | 0.227 | 15 | 51 | 134 | 50% | 0.255 | -0.180 |
| 0.40 | 0.265 | 0.226 | 12 | 41 | 147 | 50% | 0.205 | -0.145 |
| 0.45 | 0.215 | 0.279 | 12 | 31 | 157 | 50% | 0.155 | -0.095 |
| 0.50 | 0.175 | 0.343 | 12 | 23 | 165 | 50% | 0.115 | -0.055 |
| 0.55 | 0.145 | 0.345 | 10 | 19 | 171 | 50% | 0.095 | -0.045 |
| 0.60 | 0.120 | 0.333 | 8 | 16 | 176 | 50% | 0.080 | -0.040 |
| 0.65 | 0.080 | 0.312 | 5 | 11 | 184 | 50% | 0.055 | -0.030 |
| **0.70** | 0.060 | 0.417 | 5 | 7 | 188 | 50% | 0.035 | -0.010 |
| 0.75 | 0.050 | 0.300 | 3 | 7 | 190 | 50% | 0.035 | -0.020 |
| 0.80 | 0.050 | 0.300 | 3 | 7 | 190 | 50% | 0.035 | -0.020 |
| 0.85 | 0.040 | 0.125 | 1 | 7 | 192 | 50% | 0.035 | -0.030 |
| 0.90 | 0.025 | 0.200 | 1 | 4 | 195 | 50% | 0.020 | -0.015 |
| 0.95 | 0.025 | 0.200 | 1 | 4 | 195 | 50% | 0.020 | -0.015 |

### Dev: `strands-decider-2B-hobson-v19@127.0.0.1:8000|tool-choice-v1|55db837d3c1774fe|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.970 | 0.139 | 27 | 167 | 6 | 50% | 0.835 | -0.700 |
| 0.05 | 0.970 | 0.139 | 27 | 167 | 6 | 50% | 0.835 | -0.700 |
| 0.10 | 0.970 | 0.139 | 27 | 167 | 6 | 50% | 0.835 | -0.700 |
| 0.15 | 0.970 | 0.139 | 27 | 167 | 6 | 50% | 0.835 | -0.700 |
| 0.20 | 0.945 | 0.143 | 27 | 162 | 11 | 50% | 0.810 | -0.675 |
| 0.25 | 0.870 | 0.149 | 26 | 148 | 26 | 50% | 0.740 | -0.610 |
| 0.30 | 0.750 | 0.167 | 25 | 125 | 50 | 50% | 0.625 | -0.500 |
| 0.35 | 0.685 | 0.175 | 24 | 113 | 63 | 50% | 0.565 | -0.445 |
| 0.40 | 0.595 | 0.185 | 22 | 97 | 81 | 50% | 0.485 | -0.375 |
| 0.45 | 0.515 | 0.214 | 22 | 81 | 97 | 50% | 0.405 | -0.295 |
| 0.50 | 0.470 | 0.223 | 21 | 73 | 106 | 50% | 0.365 | -0.260 |
| 0.55 | 0.420 | 0.179 | 15 | 69 | 116 | 50% | 0.345 | -0.270 |
| 0.60 | 0.340 | 0.206 | 14 | 54 | 132 | 50% | 0.270 | -0.200 |
| 0.65 | 0.300 | 0.233 | 14 | 46 | 140 | 50% | 0.230 | -0.160 |
| 0.70 | 0.255 | 0.255 | 13 | 38 | 149 | 50% | 0.190 | -0.125 |
| 0.75 | 0.205 | 0.293 | 12 | 29 | 159 | 50% | 0.145 | -0.085 |
| 0.80 | 0.175 | 0.314 | 11 | 24 | 165 | 50% | 0.120 | -0.065 |
| 0.85 | 0.150 | 0.267 | 8 | 22 | 170 | 50% | 0.110 | -0.070 |
| 0.90 | 0.135 | 0.259 | 7 | 20 | 173 | 50% | 0.100 | -0.065 |
| **0.95** | 0.110 | 0.227 | 5 | 17 | 178 | 50% | 0.085 | -0.060 |

### Dev: `strands-decider-2B-hobson-v19@127.0.0.1:8000|tool-choice-v1|38dc60cd902e7d70|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.985 | 0.142 | 28 | 169 | 3 | 50% | 0.845 | -0.705 |
| 0.05 | 0.985 | 0.142 | 28 | 169 | 3 | 50% | 0.845 | -0.705 |
| 0.10 | 0.985 | 0.142 | 28 | 169 | 3 | 50% | 0.845 | -0.705 |
| 0.15 | 0.860 | 0.163 | 28 | 144 | 28 | 50% | 0.720 | -0.580 |
| 0.20 | 0.690 | 0.196 | 27 | 111 | 62 | 50% | 0.555 | -0.420 |
| 0.25 | 0.575 | 0.217 | 25 | 90 | 85 | 50% | 0.450 | -0.325 |
| 0.30 | 0.440 | 0.216 | 19 | 69 | 112 | 50% | 0.345 | -0.250 |
| 0.35 | 0.370 | 0.243 | 18 | 56 | 126 | 50% | 0.280 | -0.190 |
| 0.40 | 0.335 | 0.239 | 16 | 51 | 133 | 50% | 0.255 | -0.175 |
| 0.45 | 0.265 | 0.264 | 14 | 39 | 147 | 50% | 0.195 | -0.125 |
| 0.50 | 0.225 | 0.311 | 14 | 31 | 155 | 50% | 0.155 | -0.085 |
| 0.55 | 0.175 | 0.343 | 12 | 23 | 165 | 50% | 0.115 | -0.055 |
| 0.60 | 0.145 | 0.310 | 9 | 20 | 171 | 50% | 0.100 | -0.055 |
| 0.65 | 0.130 | 0.308 | 8 | 18 | 174 | 50% | 0.090 | -0.050 |
| 0.70 | 0.105 | 0.381 | 8 | 13 | 179 | 50% | 0.065 | -0.025 |
| 0.75 | 0.100 | 0.350 | 7 | 13 | 180 | 50% | 0.065 | -0.030 |
| 0.80 | 0.085 | 0.235 | 4 | 13 | 183 | 50% | 0.065 | -0.045 |
| 0.85 | 0.075 | 0.133 | 2 | 13 | 185 | 50% | 0.065 | -0.055 |
| 0.90 | 0.055 | 0.091 | 1 | 10 | 189 | 50% | 0.050 | -0.045 |
| **0.95** | 0.025 | 0.200 | 1 | 4 | 195 | 50% | 0.020 | -0.015 |

### Dev: `clm-latest@127.0.0.1:8700|tool-choice-v1|db6d5d57dd5971bd|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.900 | 0.056 | 10 | 170 | 20 | 50% | 0.850 | -0.800 |
| 0.05 | 0.900 | 0.056 | 10 | 170 | 20 | 50% | 0.850 | -0.800 |
| 0.10 | 0.900 | 0.056 | 10 | 170 | 20 | 50% | 0.850 | -0.800 |
| 0.15 | 0.900 | 0.056 | 10 | 170 | 20 | 50% | 0.850 | -0.800 |
| 0.20 | 0.900 | 0.056 | 10 | 170 | 20 | 50% | 0.850 | -0.800 |
| 0.25 | 0.795 | 0.063 | 10 | 149 | 41 | 50% | 0.745 | -0.695 |
| 0.30 | 0.705 | 0.064 | 9 | 132 | 59 | 50% | 0.660 | -0.615 |
| 0.35 | 0.625 | 0.072 | 9 | 116 | 75 | 50% | 0.580 | -0.535 |
| 0.40 | 0.520 | 0.077 | 8 | 96 | 96 | 50% | 0.480 | -0.440 |
| 0.45 | 0.425 | 0.094 | 8 | 77 | 115 | 50% | 0.385 | -0.345 |
| 0.50 | 0.295 | 0.136 | 8 | 51 | 141 | 50% | 0.255 | -0.215 |
| 0.55 | 0.225 | 0.178 | 8 | 37 | 155 | 50% | 0.185 | -0.145 |
| 0.60 | 0.190 | 0.184 | 7 | 31 | 162 | 50% | 0.155 | -0.120 |
| 0.65 | 0.145 | 0.241 | 7 | 22 | 171 | 50% | 0.110 | -0.075 |
| 0.70 | 0.115 | 0.304 | 7 | 16 | 177 | 50% | 0.080 | -0.045 |
| 0.75 | 0.090 | 0.278 | 5 | 13 | 182 | 50% | 0.065 | -0.040 |
| 0.80 | 0.070 | 0.357 | 5 | 9 | 186 | 50% | 0.045 | -0.020 |
| 0.85 | 0.055 | 0.455 | 5 | 6 | 189 | 50% | 0.030 | -0.005 |
| **0.90** | 0.020 | 1.000 | 4 | 0 | 196 | 50% | 0.000 | 0.020 |
| 0.95 | 0.020 | 1.000 | 4 | 0 | 196 | 50% | 0.000 | 0.020 |

### Dev: `clm-latest@127.0.0.1:8700|tool-choice-v1|5be950b6c011df2d|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.950 | 0.053 | 10 | 180 | 10 | 50% | 0.900 | -0.850 |
| 0.05 | 0.950 | 0.053 | 10 | 180 | 10 | 50% | 0.900 | -0.850 |
| 0.10 | 0.950 | 0.053 | 10 | 180 | 10 | 50% | 0.900 | -0.850 |
| 0.15 | 0.940 | 0.053 | 10 | 178 | 12 | 50% | 0.890 | -0.840 |
| 0.20 | 0.810 | 0.062 | 10 | 152 | 38 | 50% | 0.760 | -0.710 |
| 0.25 | 0.725 | 0.062 | 9 | 136 | 55 | 50% | 0.680 | -0.635 |
| 0.30 | 0.575 | 0.070 | 8 | 107 | 85 | 50% | 0.535 | -0.495 |
| 0.35 | 0.425 | 0.094 | 8 | 77 | 115 | 50% | 0.385 | -0.345 |
| 0.40 | 0.330 | 0.121 | 8 | 58 | 134 | 50% | 0.290 | -0.250 |
| 0.45 | 0.260 | 0.115 | 6 | 46 | 148 | 50% | 0.230 | -0.200 |
| 0.50 | 0.215 | 0.140 | 6 | 37 | 157 | 50% | 0.185 | -0.155 |
| 0.55 | 0.205 | 0.146 | 6 | 35 | 159 | 50% | 0.175 | -0.145 |
| 0.60 | 0.165 | 0.152 | 5 | 28 | 167 | 50% | 0.140 | -0.115 |
| 0.65 | 0.130 | 0.192 | 5 | 21 | 174 | 50% | 0.105 | -0.080 |
| 0.70 | 0.105 | 0.238 | 5 | 16 | 179 | 50% | 0.080 | -0.055 |
| 0.75 | 0.070 | 0.286 | 4 | 10 | 186 | 50% | 0.050 | -0.030 |
| **0.80** | 0.020 | 1.000 | 4 | 0 | 196 | 50% | 0.000 | 0.020 |
| 0.85 | 0.020 | 1.000 | 4 | 0 | 196 | 50% | 0.000 | 0.020 |
| 0.90 | 0.010 | 1.000 | 2 | 0 | 198 | 50% | 0.000 | 0.010 |
| 0.95 | 0.000 | n/a | 0 | 0 | 200 | 50% | 0.000 | 0.000 |

### Dev: `clm-latest@127.0.0.1:8700|tool-choice-v1|55db837d3c1774fe|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.895 | 0.067 | 12 | 167 | 21 | 50% | 0.835 | -0.775 |
| 0.05 | 0.895 | 0.067 | 12 | 167 | 21 | 50% | 0.835 | -0.775 |
| 0.10 | 0.895 | 0.067 | 12 | 167 | 21 | 50% | 0.835 | -0.775 |
| 0.15 | 0.895 | 0.067 | 12 | 167 | 21 | 50% | 0.835 | -0.775 |
| 0.20 | 0.830 | 0.072 | 12 | 154 | 34 | 50% | 0.770 | -0.710 |
| 0.25 | 0.790 | 0.070 | 11 | 147 | 42 | 50% | 0.735 | -0.680 |
| 0.30 | 0.760 | 0.072 | 11 | 141 | 48 | 50% | 0.705 | -0.650 |
| 0.35 | 0.660 | 0.083 | 11 | 121 | 68 | 50% | 0.605 | -0.550 |
| 0.40 | 0.550 | 0.091 | 10 | 100 | 90 | 50% | 0.500 | -0.450 |
| 0.45 | 0.440 | 0.114 | 10 | 78 | 112 | 50% | 0.390 | -0.340 |
| 0.50 | 0.320 | 0.156 | 10 | 54 | 136 | 50% | 0.270 | -0.220 |
| 0.55 | 0.295 | 0.169 | 10 | 49 | 141 | 50% | 0.245 | -0.195 |
| 0.60 | 0.245 | 0.184 | 9 | 40 | 151 | 50% | 0.200 | -0.155 |
| 0.65 | 0.225 | 0.200 | 9 | 36 | 155 | 50% | 0.180 | -0.135 |
| 0.70 | 0.165 | 0.273 | 9 | 24 | 167 | 50% | 0.120 | -0.075 |
| 0.75 | 0.140 | 0.250 | 7 | 21 | 172 | 50% | 0.105 | -0.070 |
| 0.80 | 0.110 | 0.318 | 7 | 15 | 178 | 50% | 0.075 | -0.040 |
| 0.85 | 0.085 | 0.412 | 7 | 10 | 183 | 50% | 0.050 | -0.015 |
| **0.90** | 0.040 | 0.625 | 5 | 3 | 192 | 50% | 0.015 | 0.010 |
| 0.95 | 0.010 | 0.500 | 1 | 1 | 198 | 50% | 0.005 | 0.000 |

### Dev: `clm-latest@127.0.0.1:8700|tool-choice-v1|38dc60cd902e7d70|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.980 | 0.051 | 10 | 186 | 4 | 50% | 0.930 | -0.880 |
| 0.05 | 0.980 | 0.051 | 10 | 186 | 4 | 50% | 0.930 | -0.880 |
| 0.10 | 0.960 | 0.052 | 10 | 182 | 8 | 50% | 0.910 | -0.860 |
| 0.15 | 0.930 | 0.054 | 10 | 176 | 14 | 50% | 0.880 | -0.830 |
| 0.20 | 0.850 | 0.059 | 10 | 160 | 30 | 50% | 0.800 | -0.750 |
| 0.25 | 0.745 | 0.067 | 10 | 139 | 51 | 50% | 0.695 | -0.645 |
| 0.30 | 0.595 | 0.076 | 9 | 110 | 81 | 50% | 0.550 | -0.505 |
| 0.35 | 0.490 | 0.092 | 9 | 89 | 102 | 50% | 0.445 | -0.400 |
| 0.40 | 0.390 | 0.090 | 7 | 71 | 122 | 50% | 0.355 | -0.320 |
| 0.45 | 0.355 | 0.099 | 7 | 64 | 129 | 50% | 0.320 | -0.285 |
| 0.50 | 0.285 | 0.105 | 6 | 51 | 143 | 50% | 0.255 | -0.225 |
| 0.55 | 0.230 | 0.130 | 6 | 40 | 154 | 50% | 0.200 | -0.170 |
| 0.60 | 0.150 | 0.133 | 4 | 26 | 170 | 50% | 0.130 | -0.110 |
| 0.65 | 0.150 | 0.133 | 4 | 26 | 170 | 50% | 0.130 | -0.110 |
| 0.70 | 0.120 | 0.167 | 4 | 20 | 176 | 50% | 0.100 | -0.080 |
| 0.75 | 0.080 | 0.250 | 4 | 12 | 184 | 50% | 0.060 | -0.040 |
| 0.80 | 0.055 | 0.182 | 2 | 9 | 189 | 50% | 0.045 | -0.035 |
| 0.85 | 0.025 | 0.200 | 1 | 4 | 195 | 50% | 0.020 | -0.015 |
| **0.90** | 0.005 | 1.000 | 1 | 0 | 199 | 50% | 0.000 | 0.005 |
| 0.95 | 0.000 | n/a | 0 | 0 | 200 | 50% | 0.000 | 0.000 |

### Dev: `clef@api.cloudflare.com|tool-choice-v1|db6d5d57dd5971bd|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.885 | 0.169 | 30 | 147 | 23 | 50% | 0.735 | -0.585 |
| 0.05 | 0.885 | 0.169 | 30 | 147 | 23 | 50% | 0.735 | -0.585 |
| 0.10 | 0.885 | 0.169 | 30 | 147 | 23 | 50% | 0.735 | -0.585 |
| 0.15 | 0.855 | 0.175 | 30 | 141 | 29 | 50% | 0.705 | -0.555 |
| 0.20 | 0.845 | 0.178 | 30 | 139 | 31 | 50% | 0.695 | -0.545 |
| 0.25 | 0.775 | 0.194 | 30 | 125 | 45 | 50% | 0.625 | -0.475 |
| 0.30 | 0.690 | 0.203 | 28 | 110 | 62 | 50% | 0.550 | -0.410 |
| 0.35 | 0.650 | 0.215 | 28 | 102 | 70 | 50% | 0.510 | -0.370 |
| 0.40 | 0.615 | 0.203 | 25 | 98 | 77 | 50% | 0.490 | -0.365 |
| 0.45 | 0.530 | 0.217 | 23 | 83 | 94 | 50% | 0.415 | -0.300 |
| 0.50 | 0.485 | 0.216 | 21 | 76 | 103 | 50% | 0.380 | -0.275 |
| 0.55 | 0.420 | 0.238 | 20 | 64 | 116 | 50% | 0.320 | -0.220 |
| 0.60 | 0.330 | 0.227 | 15 | 51 | 134 | 50% | 0.255 | -0.180 |
| 0.65 | 0.305 | 0.230 | 14 | 47 | 139 | 50% | 0.235 | -0.165 |
| 0.70 | 0.215 | 0.256 | 11 | 32 | 157 | 50% | 0.160 | -0.105 |
| 0.75 | 0.170 | 0.235 | 8 | 26 | 166 | 50% | 0.130 | -0.090 |
| 0.80 | 0.125 | 0.240 | 6 | 19 | 175 | 50% | 0.095 | -0.065 |
| 0.85 | 0.110 | 0.136 | 3 | 19 | 178 | 50% | 0.095 | -0.080 |
| 0.90 | 0.075 | 0.000 | 0 | 15 | 185 | 50% | 0.075 | -0.075 |
| **0.95** | 0.025 | 0.000 | 0 | 5 | 195 | 50% | 0.025 | -0.025 |

### Dev: `clef@api.cloudflare.com|tool-choice-v1|5be950b6c011df2d|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.970 | 0.160 | 31 | 163 | 6 | 50% | 0.815 | -0.660 |
| 0.05 | 0.970 | 0.160 | 31 | 163 | 6 | 50% | 0.815 | -0.660 |
| 0.10 | 0.950 | 0.163 | 31 | 159 | 10 | 50% | 0.795 | -0.640 |
| 0.15 | 0.940 | 0.165 | 31 | 157 | 12 | 50% | 0.785 | -0.630 |
| 0.20 | 0.900 | 0.172 | 31 | 149 | 20 | 50% | 0.745 | -0.590 |
| 0.25 | 0.785 | 0.172 | 27 | 130 | 43 | 50% | 0.650 | -0.515 |
| 0.30 | 0.700 | 0.186 | 26 | 114 | 60 | 50% | 0.570 | -0.440 |
| 0.35 | 0.590 | 0.212 | 25 | 93 | 82 | 50% | 0.465 | -0.340 |
| 0.40 | 0.545 | 0.211 | 23 | 86 | 91 | 50% | 0.430 | -0.315 |
| 0.45 | 0.425 | 0.259 | 22 | 63 | 115 | 50% | 0.315 | -0.205 |
| 0.50 | 0.385 | 0.247 | 19 | 58 | 123 | 50% | 0.290 | -0.195 |
| 0.55 | 0.320 | 0.219 | 14 | 50 | 136 | 50% | 0.250 | -0.180 |
| 0.60 | 0.255 | 0.255 | 13 | 38 | 149 | 50% | 0.190 | -0.125 |
| 0.65 | 0.185 | 0.297 | 11 | 26 | 163 | 50% | 0.130 | -0.075 |
| 0.70 | 0.145 | 0.207 | 6 | 23 | 171 | 50% | 0.115 | -0.085 |
| 0.75 | 0.115 | 0.130 | 3 | 20 | 177 | 50% | 0.100 | -0.085 |
| 0.80 | 0.090 | 0.111 | 2 | 16 | 182 | 50% | 0.080 | -0.070 |
| 0.85 | 0.080 | 0.000 | 0 | 16 | 184 | 50% | 0.080 | -0.080 |
| 0.90 | 0.055 | 0.000 | 0 | 11 | 189 | 50% | 0.055 | -0.055 |
| **0.95** | 0.030 | 0.000 | 0 | 6 | 194 | 50% | 0.030 | -0.030 |

### Dev: `clef@api.cloudflare.com|tool-choice-v1|55db837d3c1774fe|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.945 | 0.169 | 32 | 157 | 11 | 50% | 0.785 | -0.625 |
| 0.05 | 0.945 | 0.169 | 32 | 157 | 11 | 50% | 0.785 | -0.625 |
| 0.10 | 0.945 | 0.169 | 32 | 157 | 11 | 50% | 0.785 | -0.625 |
| 0.15 | 0.945 | 0.169 | 32 | 157 | 11 | 50% | 0.785 | -0.625 |
| 0.20 | 0.910 | 0.176 | 32 | 150 | 18 | 50% | 0.750 | -0.590 |
| 0.25 | 0.850 | 0.188 | 32 | 138 | 30 | 50% | 0.690 | -0.530 |
| 0.30 | 0.775 | 0.200 | 31 | 124 | 45 | 50% | 0.620 | -0.465 |
| 0.35 | 0.735 | 0.211 | 31 | 116 | 53 | 50% | 0.580 | -0.425 |
| 0.40 | 0.690 | 0.196 | 27 | 111 | 62 | 50% | 0.555 | -0.420 |
| 0.45 | 0.675 | 0.200 | 27 | 108 | 65 | 50% | 0.540 | -0.405 |
| 0.50 | 0.565 | 0.239 | 27 | 86 | 87 | 50% | 0.430 | -0.295 |
| 0.55 | 0.500 | 0.250 | 25 | 75 | 100 | 50% | 0.375 | -0.250 |
| 0.60 | 0.445 | 0.258 | 23 | 66 | 111 | 50% | 0.330 | -0.215 |
| 0.65 | 0.385 | 0.299 | 23 | 54 | 123 | 50% | 0.270 | -0.155 |
| 0.70 | 0.315 | 0.286 | 18 | 45 | 137 | 50% | 0.225 | -0.135 |
| 0.75 | 0.255 | 0.275 | 14 | 37 | 149 | 50% | 0.185 | -0.115 |
| 0.80 | 0.205 | 0.244 | 10 | 31 | 159 | 50% | 0.155 | -0.105 |
| 0.85 | 0.160 | 0.219 | 7 | 25 | 168 | 50% | 0.125 | -0.090 |
| 0.90 | 0.130 | 0.269 | 7 | 19 | 174 | 50% | 0.095 | -0.060 |
| **0.95** | 0.045 | 0.000 | 0 | 9 | 191 | 50% | 0.045 | -0.045 |

### Dev: `clef@api.cloudflare.com|tool-choice-v1|38dc60cd902e7d70|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.960 | 0.182 | 35 | 157 | 8 | 50% | 0.785 | -0.610 |
| 0.05 | 0.960 | 0.182 | 35 | 157 | 8 | 50% | 0.785 | -0.610 |
| 0.10 | 0.960 | 0.182 | 35 | 157 | 8 | 50% | 0.785 | -0.610 |
| 0.15 | 0.935 | 0.187 | 35 | 152 | 13 | 50% | 0.760 | -0.585 |
| 0.20 | 0.865 | 0.191 | 33 | 140 | 27 | 50% | 0.700 | -0.535 |
| 0.25 | 0.800 | 0.200 | 32 | 128 | 40 | 50% | 0.640 | -0.480 |
| 0.30 | 0.710 | 0.211 | 30 | 112 | 58 | 50% | 0.560 | -0.410 |
| 0.35 | 0.665 | 0.226 | 30 | 103 | 67 | 50% | 0.515 | -0.365 |
| 0.40 | 0.545 | 0.248 | 27 | 82 | 91 | 50% | 0.410 | -0.275 |
| 0.45 | 0.495 | 0.273 | 27 | 72 | 101 | 50% | 0.360 | -0.225 |
| 0.50 | 0.450 | 0.289 | 26 | 64 | 110 | 50% | 0.320 | -0.190 |
| 0.55 | 0.375 | 0.293 | 22 | 53 | 125 | 50% | 0.265 | -0.155 |
| 0.60 | 0.310 | 0.290 | 18 | 44 | 138 | 50% | 0.220 | -0.130 |
| 0.65 | 0.280 | 0.232 | 13 | 43 | 144 | 50% | 0.215 | -0.150 |
| 0.70 | 0.260 | 0.192 | 10 | 42 | 148 | 50% | 0.210 | -0.160 |
| 0.75 | 0.235 | 0.213 | 10 | 37 | 153 | 50% | 0.185 | -0.135 |
| 0.80 | 0.185 | 0.189 | 7 | 30 | 163 | 50% | 0.150 | -0.115 |
| 0.85 | 0.160 | 0.188 | 6 | 26 | 168 | 50% | 0.130 | -0.100 |
| 0.90 | 0.070 | 0.214 | 3 | 11 | 186 | 50% | 0.055 | -0.040 |
| **0.95** | 0.030 | 0.000 | 0 | 6 | 194 | 50% | 0.030 | -0.030 |

### Dev: `clef-flash@api.cloudflare.com|tool-choice-v1|db6d5d57dd5971bd|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.905 | 0.166 | 30 | 151 | 19 | 50% | 0.755 | -0.605 |
| 0.05 | 0.905 | 0.166 | 30 | 151 | 19 | 50% | 0.755 | -0.605 |
| 0.10 | 0.905 | 0.166 | 30 | 151 | 19 | 50% | 0.755 | -0.605 |
| 0.15 | 0.905 | 0.166 | 30 | 151 | 19 | 50% | 0.755 | -0.605 |
| 0.20 | 0.885 | 0.169 | 30 | 147 | 23 | 50% | 0.735 | -0.585 |
| 0.25 | 0.810 | 0.185 | 30 | 132 | 38 | 50% | 0.660 | -0.510 |
| 0.30 | 0.740 | 0.189 | 28 | 120 | 52 | 50% | 0.600 | -0.460 |
| 0.35 | 0.695 | 0.187 | 26 | 113 | 61 | 50% | 0.565 | -0.435 |
| 0.40 | 0.620 | 0.169 | 21 | 103 | 76 | 50% | 0.515 | -0.410 |
| 0.45 | 0.565 | 0.177 | 20 | 93 | 87 | 50% | 0.465 | -0.365 |
| 0.50 | 0.475 | 0.179 | 17 | 78 | 105 | 50% | 0.390 | -0.305 |
| 0.55 | 0.430 | 0.151 | 13 | 73 | 114 | 50% | 0.365 | -0.300 |
| 0.60 | 0.390 | 0.141 | 11 | 67 | 122 | 50% | 0.335 | -0.280 |
| 0.65 | 0.335 | 0.164 | 11 | 56 | 133 | 50% | 0.280 | -0.225 |
| 0.70 | 0.285 | 0.158 | 9 | 48 | 143 | 50% | 0.240 | -0.195 |
| 0.75 | 0.240 | 0.146 | 7 | 41 | 152 | 50% | 0.205 | -0.170 |
| 0.80 | 0.190 | 0.184 | 7 | 31 | 162 | 50% | 0.155 | -0.120 |
| 0.85 | 0.150 | 0.167 | 5 | 25 | 170 | 50% | 0.125 | -0.100 |
| **0.90** | 0.055 | 0.182 | 2 | 9 | 189 | 50% | 0.045 | -0.035 |
| 0.95 | 0.040 | 0.000 | 0 | 8 | 192 | 50% | 0.040 | -0.040 |

### Dev: `clef-flash@api.cloudflare.com|tool-choice-v1|5be950b6c011df2d|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.945 | 0.159 | 30 | 159 | 11 | 50% | 0.795 | -0.645 |
| 0.05 | 0.945 | 0.159 | 30 | 159 | 11 | 50% | 0.795 | -0.645 |
| 0.10 | 0.935 | 0.160 | 30 | 157 | 13 | 50% | 0.785 | -0.635 |
| 0.15 | 0.920 | 0.163 | 30 | 154 | 16 | 50% | 0.770 | -0.620 |
| 0.20 | 0.855 | 0.175 | 30 | 141 | 29 | 50% | 0.705 | -0.555 |
| 0.25 | 0.775 | 0.194 | 30 | 125 | 45 | 50% | 0.625 | -0.475 |
| 0.30 | 0.680 | 0.206 | 28 | 108 | 64 | 50% | 0.540 | -0.400 |
| 0.35 | 0.635 | 0.213 | 27 | 100 | 73 | 50% | 0.500 | -0.365 |
| 0.40 | 0.545 | 0.220 | 24 | 85 | 91 | 50% | 0.425 | -0.305 |
| 0.45 | 0.490 | 0.214 | 21 | 77 | 102 | 50% | 0.385 | -0.280 |
| 0.50 | 0.390 | 0.192 | 15 | 63 | 122 | 50% | 0.315 | -0.240 |
| 0.55 | 0.335 | 0.194 | 13 | 54 | 133 | 50% | 0.270 | -0.205 |
| 0.60 | 0.255 | 0.255 | 13 | 38 | 149 | 50% | 0.190 | -0.125 |
| 0.65 | 0.215 | 0.209 | 9 | 34 | 157 | 50% | 0.170 | -0.125 |
| 0.70 | 0.185 | 0.189 | 7 | 30 | 163 | 50% | 0.150 | -0.115 |
| 0.75 | 0.155 | 0.129 | 4 | 27 | 169 | 50% | 0.135 | -0.115 |
| 0.80 | 0.135 | 0.074 | 2 | 25 | 173 | 50% | 0.125 | -0.115 |
| 0.85 | 0.085 | 0.118 | 2 | 15 | 183 | 50% | 0.075 | -0.065 |
| 0.90 | 0.050 | 0.000 | 0 | 10 | 190 | 50% | 0.050 | -0.050 |
| **0.95** | 0.010 | 0.000 | 0 | 2 | 198 | 50% | 0.010 | -0.010 |

### Dev: `clef-flash@api.cloudflare.com|tool-choice-v1|55db837d3c1774fe|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.905 | 0.177 | 32 | 149 | 19 | 50% | 0.745 | -0.585 |
| 0.05 | 0.905 | 0.177 | 32 | 149 | 19 | 50% | 0.745 | -0.585 |
| 0.10 | 0.905 | 0.177 | 32 | 149 | 19 | 50% | 0.745 | -0.585 |
| 0.15 | 0.885 | 0.181 | 32 | 145 | 23 | 50% | 0.725 | -0.565 |
| 0.20 | 0.840 | 0.173 | 29 | 139 | 32 | 50% | 0.695 | -0.550 |
| 0.25 | 0.790 | 0.184 | 29 | 129 | 42 | 50% | 0.645 | -0.500 |
| 0.30 | 0.740 | 0.189 | 28 | 120 | 52 | 50% | 0.600 | -0.460 |
| 0.35 | 0.720 | 0.194 | 28 | 116 | 56 | 50% | 0.580 | -0.440 |
| 0.40 | 0.680 | 0.191 | 26 | 110 | 64 | 50% | 0.550 | -0.420 |
| 0.45 | 0.640 | 0.195 | 25 | 103 | 72 | 50% | 0.515 | -0.390 |
| 0.50 | 0.605 | 0.207 | 25 | 96 | 79 | 50% | 0.480 | -0.355 |
| 0.55 | 0.550 | 0.191 | 21 | 89 | 90 | 50% | 0.445 | -0.340 |
| 0.60 | 0.505 | 0.188 | 19 | 82 | 99 | 50% | 0.410 | -0.315 |
| 0.65 | 0.465 | 0.204 | 19 | 74 | 107 | 50% | 0.370 | -0.275 |
| 0.70 | 0.400 | 0.212 | 17 | 63 | 120 | 50% | 0.315 | -0.230 |
| 0.75 | 0.355 | 0.211 | 15 | 56 | 129 | 50% | 0.280 | -0.205 |
| 0.80 | 0.255 | 0.216 | 11 | 40 | 149 | 50% | 0.200 | -0.145 |
| 0.85 | 0.165 | 0.242 | 8 | 25 | 167 | 50% | 0.125 | -0.085 |
| 0.90 | 0.115 | 0.217 | 5 | 18 | 177 | 50% | 0.090 | -0.065 |
| **0.95** | 0.050 | 0.200 | 2 | 8 | 190 | 50% | 0.040 | -0.030 |

### Dev: `clef-flash@api.cloudflare.com|tool-choice-v1|38dc60cd902e7d70|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.970 | 0.165 | 32 | 162 | 6 | 50% | 0.810 | -0.650 |
| 0.05 | 0.970 | 0.165 | 32 | 162 | 6 | 50% | 0.810 | -0.650 |
| 0.10 | 0.950 | 0.168 | 32 | 158 | 10 | 50% | 0.790 | -0.630 |
| 0.15 | 0.925 | 0.173 | 32 | 153 | 15 | 50% | 0.765 | -0.605 |
| 0.20 | 0.885 | 0.181 | 32 | 145 | 23 | 50% | 0.725 | -0.565 |
| 0.25 | 0.835 | 0.192 | 32 | 135 | 33 | 50% | 0.675 | -0.515 |
| 0.30 | 0.810 | 0.198 | 32 | 130 | 38 | 50% | 0.650 | -0.490 |
| 0.35 | 0.710 | 0.218 | 31 | 111 | 58 | 50% | 0.555 | -0.400 |
| 0.40 | 0.605 | 0.231 | 28 | 93 | 79 | 50% | 0.465 | -0.325 |
| 0.45 | 0.510 | 0.245 | 25 | 77 | 98 | 50% | 0.385 | -0.260 |
| 0.50 | 0.480 | 0.240 | 23 | 73 | 104 | 50% | 0.365 | -0.250 |
| 0.55 | 0.435 | 0.241 | 21 | 66 | 113 | 50% | 0.330 | -0.225 |
| 0.60 | 0.395 | 0.241 | 19 | 60 | 121 | 50% | 0.300 | -0.205 |
| 0.65 | 0.370 | 0.257 | 19 | 55 | 126 | 50% | 0.275 | -0.180 |
| 0.70 | 0.320 | 0.250 | 16 | 48 | 136 | 50% | 0.240 | -0.160 |
| 0.75 | 0.265 | 0.208 | 11 | 42 | 147 | 50% | 0.210 | -0.155 |
| 0.80 | 0.180 | 0.250 | 9 | 27 | 164 | 50% | 0.135 | -0.090 |
| 0.85 | 0.145 | 0.172 | 5 | 24 | 171 | 50% | 0.120 | -0.095 |
| 0.90 | 0.075 | 0.267 | 4 | 11 | 185 | 50% | 0.055 | -0.035 |
| **0.95** | 0.010 | 0.000 | 0 | 2 | 198 | 50% | 0.010 | -0.010 |

### Held-out: hybrid+strands@20, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.939 | 0.156 | 117 | 634 | 49 | 50% | 0.792 | -0.646 |
| 0.05 | 0.939 | 0.156 | 117 | 634 | 49 | 50% | 0.792 | -0.646 |
| 0.10 | 0.939 | 0.156 | 117 | 634 | 49 | 50% | 0.792 | -0.646 |
| 0.15 | 0.932 | 0.157 | 117 | 629 | 54 | 50% | 0.786 | -0.640 |
| 0.20 | 0.911 | 0.159 | 116 | 613 | 71 | 50% | 0.766 | -0.621 |
| 0.25 | 0.843 | 0.163 | 110 | 564 | 126 | 50% | 0.705 | -0.568 |
| 0.30 | 0.714 | 0.173 | 99 | 472 | 229 | 50% | 0.590 | -0.466 |
| 0.35 | 0.620 | 0.192 | 95 | 401 | 304 | 50% | 0.501 | -0.383 |
| 0.40 | 0.497 | 0.201 | 80 | 318 | 402 | 50% | 0.398 | -0.297 |
| 0.45 | 0.412 | 0.203 | 67 | 263 | 470 | 50% | 0.329 | -0.245 |
| 0.50 | 0.351 | 0.206 | 58 | 223 | 519 | 50% | 0.279 | -0.206 |
| 0.55 | 0.279 | 0.224 | 50 | 173 | 577 | 50% | 0.216 | -0.154 |
| 0.60 | 0.217 | 0.230 | 40 | 134 | 626 | 50% | 0.168 | -0.117 |
| 0.65 | 0.170 | 0.250 | 34 | 102 | 664 | 50% | 0.128 | -0.085 |
| 0.70 | 0.120 | 0.271 | 26 | 70 | 704 | 50% | 0.087 | -0.055 |
| 0.75 | 0.091 | 0.301 | 22 | 51 | 727 | 50% | 0.064 | -0.036 |
| 0.80 | 0.055 | 0.364 | 16 | 28 | 756 | 50% | 0.035 | -0.015 |
| 0.85 | 0.031 | 0.400 | 10 | 15 | 775 | 50% | 0.019 | -0.006 |
| 0.90 | 0.015 | 0.833 | 10 | 2 | 788 | 50% | 0.003 | 0.010 |
| **0.95** | 0.004 | 1.000 | 3 | 0 | 797 | 50% | 0.000 | 0.004 |

### Held-out: hybrid+strands@50, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.973 | 0.141 | 110 | 668 | 22 | 50% | 0.835 | -0.698 |
| 0.05 | 0.973 | 0.141 | 110 | 668 | 22 | 50% | 0.835 | -0.698 |
| 0.10 | 0.953 | 0.144 | 110 | 652 | 38 | 50% | 0.815 | -0.677 |
| 0.15 | 0.840 | 0.159 | 107 | 565 | 128 | 50% | 0.706 | -0.573 |
| 0.20 | 0.694 | 0.173 | 96 | 459 | 245 | 50% | 0.574 | -0.454 |
| 0.25 | 0.514 | 0.185 | 76 | 335 | 389 | 50% | 0.419 | -0.324 |
| 0.30 | 0.394 | 0.206 | 65 | 250 | 485 | 50% | 0.312 | -0.231 |
| 0.35 | 0.297 | 0.235 | 56 | 182 | 562 | 50% | 0.228 | -0.158 |
| 0.40 | 0.211 | 0.272 | 46 | 123 | 631 | 50% | 0.154 | -0.096 |
| 0.45 | 0.160 | 0.320 | 41 | 87 | 672 | 50% | 0.109 | -0.058 |
| 0.50 | 0.115 | 0.326 | 30 | 62 | 708 | 50% | 0.077 | -0.040 |
| 0.55 | 0.095 | 0.368 | 28 | 48 | 724 | 50% | 0.060 | -0.025 |
| 0.60 | 0.065 | 0.442 | 23 | 29 | 748 | 50% | 0.036 | -0.007 |
| 0.65 | 0.037 | 0.433 | 13 | 17 | 770 | 50% | 0.021 | -0.005 |
| **0.70** | 0.030 | 0.417 | 10 | 14 | 776 | 50% | 0.018 | -0.005 |
| 0.75 | 0.018 | 0.286 | 4 | 10 | 786 | 50% | 0.013 | -0.007 |
| 0.80 | 0.013 | 0.200 | 2 | 8 | 790 | 50% | 0.010 | -0.007 |
| 0.85 | 0.003 | 1.000 | 2 | 0 | 798 | 50% | 0.000 | 0.003 |
| 0.90 | 0.000 | n/a | 0 | 0 | 800 | 50% | 0.000 | 0.000 |
| 0.95 | 0.000 | n/a | 0 | 0 | 800 | 50% | 0.000 | 0.000 |

### Held-out: hybrid+clm-local@20, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.932 | 0.062 | 46 | 700 | 54 | 50% | 0.875 | -0.818 |
| 0.05 | 0.932 | 0.062 | 46 | 700 | 54 | 50% | 0.875 | -0.818 |
| 0.10 | 0.932 | 0.062 | 46 | 700 | 54 | 50% | 0.875 | -0.818 |
| 0.15 | 0.916 | 0.063 | 46 | 687 | 67 | 50% | 0.859 | -0.801 |
| 0.20 | 0.864 | 0.065 | 45 | 646 | 109 | 50% | 0.807 | -0.751 |
| 0.25 | 0.781 | 0.067 | 42 | 583 | 175 | 50% | 0.729 | -0.676 |
| 0.30 | 0.672 | 0.078 | 42 | 496 | 262 | 50% | 0.620 | -0.568 |
| 0.35 | 0.593 | 0.080 | 38 | 436 | 326 | 50% | 0.545 | -0.497 |
| 0.40 | 0.500 | 0.087 | 35 | 365 | 400 | 50% | 0.456 | -0.412 |
| 0.45 | 0.412 | 0.094 | 31 | 299 | 470 | 50% | 0.374 | -0.335 |
| 0.50 | 0.344 | 0.091 | 25 | 250 | 525 | 50% | 0.312 | -0.281 |
| 0.55 | 0.290 | 0.103 | 24 | 208 | 568 | 50% | 0.260 | -0.230 |
| 0.60 | 0.245 | 0.097 | 19 | 177 | 604 | 50% | 0.221 | -0.198 |
| 0.65 | 0.229 | 0.082 | 15 | 168 | 617 | 50% | 0.210 | -0.191 |
| 0.70 | 0.166 | 0.075 | 10 | 123 | 667 | 50% | 0.154 | -0.141 |
| 0.75 | 0.115 | 0.109 | 10 | 82 | 708 | 50% | 0.102 | -0.090 |
| 0.80 | 0.084 | 0.119 | 8 | 59 | 733 | 50% | 0.074 | -0.064 |
| 0.85 | 0.050 | 0.100 | 4 | 36 | 760 | 50% | 0.045 | -0.040 |
| **0.90** | 0.022 | 0.222 | 4 | 14 | 782 | 50% | 0.018 | -0.013 |
| 0.95 | 0.013 | 0.200 | 2 | 8 | 790 | 50% | 0.010 | -0.007 |

### Held-out: hybrid+clm-local@50, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.955 | 0.048 | 37 | 727 | 36 | 50% | 0.909 | -0.863 |
| 0.05 | 0.955 | 0.048 | 37 | 727 | 36 | 50% | 0.909 | -0.863 |
| 0.10 | 0.946 | 0.049 | 37 | 720 | 43 | 50% | 0.900 | -0.854 |
| 0.15 | 0.881 | 0.050 | 35 | 670 | 95 | 50% | 0.838 | -0.794 |
| 0.20 | 0.794 | 0.052 | 33 | 602 | 165 | 50% | 0.752 | -0.711 |
| 0.25 | 0.662 | 0.058 | 31 | 499 | 270 | 50% | 0.624 | -0.585 |
| 0.30 | 0.544 | 0.067 | 29 | 406 | 365 | 50% | 0.507 | -0.471 |
| 0.35 | 0.439 | 0.068 | 24 | 327 | 449 | 50% | 0.409 | -0.379 |
| 0.40 | 0.347 | 0.061 | 17 | 261 | 522 | 50% | 0.326 | -0.305 |
| 0.45 | 0.287 | 0.065 | 15 | 215 | 570 | 50% | 0.269 | -0.250 |
| 0.50 | 0.240 | 0.073 | 14 | 178 | 608 | 50% | 0.223 | -0.205 |
| 0.55 | 0.203 | 0.068 | 11 | 151 | 638 | 50% | 0.189 | -0.175 |
| 0.60 | 0.146 | 0.051 | 6 | 111 | 683 | 50% | 0.139 | -0.131 |
| 0.65 | 0.107 | 0.058 | 5 | 81 | 714 | 50% | 0.101 | -0.095 |
| 0.70 | 0.070 | 0.089 | 5 | 51 | 744 | 50% | 0.064 | -0.058 |
| 0.75 | 0.052 | 0.095 | 4 | 38 | 758 | 50% | 0.048 | -0.043 |
| 0.80 | 0.048 | 0.105 | 4 | 34 | 762 | 50% | 0.043 | -0.037 |
| 0.85 | 0.029 | 0.087 | 2 | 21 | 777 | 50% | 0.026 | -0.024 |
| **0.90** | 0.005 | 0.000 | 0 | 4 | 796 | 50% | 0.005 | -0.005 |
| 0.95 | 0.000 | n/a | 0 | 0 | 800 | 50% | 0.000 | 0.000 |

### Held-out: hybrid+clef@20, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.950 | 0.167 | 127 | 633 | 40 | 50% | 0.791 | -0.632 |
| 0.05 | 0.950 | 0.167 | 127 | 633 | 40 | 50% | 0.791 | -0.632 |
| 0.10 | 0.950 | 0.167 | 127 | 633 | 40 | 50% | 0.791 | -0.632 |
| 0.15 | 0.950 | 0.167 | 127 | 633 | 40 | 50% | 0.791 | -0.632 |
| 0.20 | 0.914 | 0.174 | 127 | 604 | 69 | 50% | 0.755 | -0.596 |
| 0.25 | 0.834 | 0.183 | 122 | 545 | 133 | 50% | 0.681 | -0.529 |
| 0.30 | 0.757 | 0.196 | 119 | 487 | 194 | 50% | 0.609 | -0.460 |
| 0.35 | 0.690 | 0.207 | 114 | 438 | 248 | 50% | 0.547 | -0.405 |
| 0.40 | 0.574 | 0.220 | 101 | 358 | 341 | 50% | 0.448 | -0.321 |
| 0.45 | 0.511 | 0.237 | 97 | 312 | 391 | 50% | 0.390 | -0.269 |
| 0.50 | 0.436 | 0.261 | 91 | 258 | 451 | 50% | 0.323 | -0.209 |
| 0.55 | 0.375 | 0.270 | 81 | 219 | 500 | 50% | 0.274 | -0.172 |
| 0.60 | 0.325 | 0.296 | 77 | 183 | 540 | 50% | 0.229 | -0.133 |
| 0.65 | 0.270 | 0.333 | 72 | 144 | 584 | 50% | 0.180 | -0.090 |
| 0.70 | 0.224 | 0.358 | 64 | 115 | 621 | 50% | 0.144 | -0.064 |
| 0.75 | 0.175 | 0.364 | 51 | 89 | 660 | 50% | 0.111 | -0.048 |
| 0.80 | 0.136 | 0.358 | 39 | 70 | 691 | 50% | 0.087 | -0.039 |
| 0.85 | 0.090 | 0.458 | 33 | 39 | 728 | 50% | 0.049 | -0.007 |
| 0.90 | 0.060 | 0.521 | 25 | 23 | 752 | 50% | 0.029 | 0.003 |
| **0.95** | 0.025 | 0.500 | 10 | 10 | 780 | 50% | 0.013 | 0.000 |

### Held-out: hybrid+clef@50, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.975 | 0.196 | 153 | 627 | 20 | 50% | 0.784 | -0.593 |
| 0.05 | 0.975 | 0.196 | 153 | 627 | 20 | 50% | 0.784 | -0.593 |
| 0.10 | 0.970 | 0.197 | 153 | 623 | 24 | 50% | 0.779 | -0.588 |
| 0.15 | 0.927 | 0.206 | 153 | 589 | 58 | 50% | 0.736 | -0.545 |
| 0.20 | 0.860 | 0.215 | 148 | 540 | 112 | 50% | 0.675 | -0.490 |
| 0.25 | 0.785 | 0.228 | 143 | 485 | 172 | 50% | 0.606 | -0.427 |
| 0.30 | 0.681 | 0.251 | 137 | 408 | 255 | 50% | 0.510 | -0.339 |
| 0.35 | 0.588 | 0.281 | 132 | 338 | 330 | 50% | 0.422 | -0.258 |
| 0.40 | 0.511 | 0.293 | 120 | 289 | 391 | 50% | 0.361 | -0.211 |
| 0.45 | 0.416 | 0.330 | 110 | 223 | 467 | 50% | 0.279 | -0.141 |
| 0.50 | 0.351 | 0.345 | 97 | 184 | 519 | 50% | 0.230 | -0.109 |
| 0.55 | 0.305 | 0.369 | 90 | 154 | 556 | 50% | 0.193 | -0.080 |
| 0.60 | 0.273 | 0.381 | 83 | 135 | 582 | 50% | 0.169 | -0.065 |
| 0.65 | 0.228 | 0.385 | 70 | 112 | 618 | 50% | 0.140 | -0.052 |
| 0.70 | 0.172 | 0.406 | 56 | 82 | 662 | 50% | 0.102 | -0.033 |
| 0.75 | 0.135 | 0.398 | 43 | 65 | 692 | 50% | 0.081 | -0.028 |
| 0.80 | 0.089 | 0.437 | 31 | 40 | 729 | 50% | 0.050 | -0.011 |
| 0.85 | 0.068 | 0.481 | 26 | 28 | 746 | 50% | 0.035 | -0.003 |
| 0.90 | 0.045 | 0.500 | 18 | 18 | 764 | 50% | 0.022 | 0.000 |
| **0.95** | 0.005 | 0.750 | 3 | 1 | 796 | 50% | 0.001 | 0.003 |

### Held-out: hybrid+clef-flash@20, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.969 | 0.121 | 94 | 681 | 25 | 50% | 0.851 | -0.734 |
| 0.05 | 0.969 | 0.121 | 94 | 681 | 25 | 50% | 0.851 | -0.734 |
| 0.10 | 0.969 | 0.121 | 94 | 681 | 25 | 50% | 0.851 | -0.734 |
| 0.15 | 0.969 | 0.121 | 94 | 681 | 25 | 50% | 0.851 | -0.734 |
| 0.20 | 0.932 | 0.125 | 93 | 653 | 54 | 50% | 0.816 | -0.700 |
| 0.25 | 0.896 | 0.128 | 92 | 625 | 83 | 50% | 0.781 | -0.666 |
| 0.30 | 0.858 | 0.133 | 91 | 595 | 114 | 50% | 0.744 | -0.630 |
| 0.35 | 0.782 | 0.144 | 90 | 536 | 174 | 50% | 0.670 | -0.557 |
| 0.40 | 0.705 | 0.160 | 90 | 474 | 236 | 50% | 0.593 | -0.480 |
| 0.45 | 0.608 | 0.171 | 83 | 403 | 314 | 50% | 0.504 | -0.400 |
| 0.50 | 0.545 | 0.183 | 80 | 356 | 364 | 50% | 0.445 | -0.345 |
| 0.55 | 0.476 | 0.189 | 72 | 309 | 419 | 50% | 0.386 | -0.296 |
| 0.60 | 0.410 | 0.216 | 71 | 257 | 472 | 50% | 0.321 | -0.233 |
| 0.65 | 0.370 | 0.236 | 70 | 226 | 504 | 50% | 0.282 | -0.195 |
| 0.70 | 0.301 | 0.257 | 62 | 179 | 559 | 50% | 0.224 | -0.146 |
| 0.75 | 0.249 | 0.276 | 55 | 144 | 601 | 50% | 0.180 | -0.111 |
| 0.80 | 0.191 | 0.314 | 48 | 105 | 647 | 50% | 0.131 | -0.071 |
| 0.85 | 0.133 | 0.387 | 41 | 65 | 694 | 50% | 0.081 | -0.030 |
| 0.90 | 0.069 | 0.436 | 24 | 31 | 745 | 50% | 0.039 | -0.009 |
| **0.95** | 0.024 | 0.579 | 11 | 8 | 781 | 50% | 0.010 | 0.004 |

### Held-out: hybrid+clef-flash@50, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.984 | 0.145 | 114 | 673 | 13 | 50% | 0.841 | -0.699 |
| 0.05 | 0.984 | 0.145 | 114 | 673 | 13 | 50% | 0.841 | -0.699 |
| 0.10 | 0.984 | 0.145 | 114 | 673 | 13 | 50% | 0.841 | -0.699 |
| 0.15 | 0.956 | 0.149 | 114 | 651 | 35 | 50% | 0.814 | -0.671 |
| 0.20 | 0.915 | 0.153 | 112 | 620 | 68 | 50% | 0.775 | -0.635 |
| 0.25 | 0.848 | 0.165 | 112 | 566 | 122 | 50% | 0.708 | -0.568 |
| 0.30 | 0.754 | 0.176 | 106 | 497 | 197 | 50% | 0.621 | -0.489 |
| 0.35 | 0.685 | 0.182 | 100 | 448 | 252 | 50% | 0.560 | -0.435 |
| 0.40 | 0.598 | 0.195 | 93 | 385 | 322 | 50% | 0.481 | -0.365 |
| 0.45 | 0.502 | 0.209 | 84 | 318 | 398 | 50% | 0.398 | -0.292 |
| 0.50 | 0.429 | 0.224 | 77 | 266 | 457 | 50% | 0.333 | -0.236 |
| 0.55 | 0.360 | 0.257 | 74 | 214 | 512 | 50% | 0.268 | -0.175 |
| 0.60 | 0.302 | 0.289 | 70 | 172 | 558 | 50% | 0.215 | -0.128 |
| 0.65 | 0.224 | 0.324 | 58 | 121 | 621 | 50% | 0.151 | -0.079 |
| 0.70 | 0.194 | 0.323 | 50 | 105 | 645 | 50% | 0.131 | -0.069 |
| 0.75 | 0.156 | 0.368 | 46 | 79 | 675 | 50% | 0.099 | -0.041 |
| 0.80 | 0.119 | 0.411 | 39 | 56 | 705 | 50% | 0.070 | -0.021 |
| 0.85 | 0.077 | 0.371 | 23 | 39 | 738 | 50% | 0.049 | -0.020 |
| 0.90 | 0.037 | 0.500 | 15 | 15 | 770 | 50% | 0.019 | 0.000 |
| **0.95** | 0.006 | 0.600 | 3 | 2 | 795 | 50% | 0.003 | 0.001 |

