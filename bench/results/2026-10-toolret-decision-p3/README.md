# ToolRet decision stage, held-out run 20261005T132939Z+20261005T145540Z

Held-out runs `20261005T132939Z`, `20261005T145540Z`, one per decider: 200 tasks from `bench/tasks/toolret-heldout-200.json` (sha256 `b632db8a5cb3`), `mteb/ToolRetrieval` at `76d45e5` with 44,453 tools, catalog `sha256:7d56b70f3415`; toolhunch 0.1.0, commits `8a39d28`, `5e6d0a6 (dirty)`, Python 3.14.3. Every abstention threshold applied here was chosen on the dev runs (`20261005T124458Z`, `20261005T124909Z`, `20261005T125333Z`, `20261005T144955Z`, `20261005T130805Z`, `20261005T125701Z`, `20261005T125736Z`, `20261005T131210Z`); no held-out search chose one.

**Laya.** Its authors describe the base checkpoints as a fast base to specialise, not a zero-shot decision engine; we run the English one zero-shot. Its questions are planned with Laya's own tokenizer, and an option whose name would repeat its key past 48 tokens is sent as its key alone.

**Laya wide.** Its authors describe the base checkpoints as a fast base to specialise, not a zero-shot decision engine; we run the English one zero-shot. Its questions are planned with Laya's own tokenizer, and an option whose name would repeat its key past 48 tokens is sent as its key alone. It runs with a 512-token option budget in a 1,024-token window, beyond the 192 and 512 the checkpoint ships with, as its model card advises for many options.

**rizzo-flow.** The authors' fine-tune, 4B at q8_0 on llama.cpp with Metal; quantization and hardware change its probabilities, by its authors' account.

**Local deciders.** Laya and Laya wide and rizzo-flow ran on one machine (Apple M5 Max, 128 GB, macOS 26.6.2): their cost reads “local”, and their latency is that machine's, not comparable like for like with a hosted API's.

## Runs

| split | role | run | tasks | sources | negatives | repeat | K | deciders (max detail) | reserved option | model-source fallbacks | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | main | `20261005T124458Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | laya (BRIEF) | on | 20 of 50 | `8a39d28` |
| dev | main | `20261005T124909Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | laya-wide (BRIEF) | on | 20 of 50 | `8a39d28` |
| dev | main | `20261005T125333Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | laya-wide (BRIEF, floor BRIEF) | on | 20 of 50 | `8a39d28` |
| dev | main | `20261005T144955Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | rizzo-flow (FULL) | on | 20 of 50 | `5e6d0a6 (dirty)` |
| dev | main | `20261005T130805Z` | 50 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain, model | yes | 1 | 20, 50 | rizzo-flow (BRIEF) | on | 20 of 50 | `8a39d28` |
| dev | main | `20261005T125701Z` | 10 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 3 | 20 | laya (BRIEF) | on | - | `8a39d28` |
| dev | main | `20261005T125736Z` | 10 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 3 | 20 | laya-wide (BRIEF) | on | - | `8a39d28` |
| dev | main | `20261005T131210Z` | 10 (`bench/tasks/toolret-pilot-50.json`, sha256 `90b0f6009aa5`) | plain | yes | 3 | 20 | rizzo-flow (FULL) | on | - | `8a39d28` |
| heldout | main | `20261005T132939Z` | 200 (`bench/tasks/toolret-heldout-200.json`, sha256 `b632db8a5cb3`) | plain, model | yes | 1 | 20, 50 | laya-wide (BRIEF, floor BRIEF) | on | 81 of 200 | `8a39d28` |
| heldout | main | `20261005T145540Z` | 200 (`bench/tasks/toolret-heldout-200.json`, sha256 `b632db8a5cb3`) | plain, model | yes | 1 | 20, 50 | rizzo-flow (FULL) | on | 81 of 200 | `5e6d0a6 (dirty)` |

| decider | model | prompt version | declared limits |
|---|---|---|---|
| laya | `english@127.0.0.1:8010` | `tool-choice-v1` | laya 0.3.27 serve.py (100 options, 64 questions) and common.py build_head/render_options (48-token option cap); convaiinnovations/laya @ 55cf4c4 rl_agent_config.json (max_len 512, head_max_len 192) (checked 2026-10-05) |
| laya-wide | `english@127.0.0.1:8010` | `tool-choice-v1` | laya 0.3.27 serve.py (100 options, 64 questions) and common.py build_head/render_options (48-token option cap); convaiinnovations/laya @ 55cf4c4 rl_agent_config.json (max_len 512, head_max_len 192); sent with every request: head_max_len 512, max_len 1024 (checked 2026-10-05) |
| rizzo-flow | `rizzo-flow-4b-q8_0@127.0.0.1:8017` | `tool-choice-v1` | Rizzo-AI-Academy/rizzo-flow @ b9ba007: schema.py MAX_SLOTS = 26 and Text max_length=8000 characters for each option; --ctx 8192 bounds each question with its state (prompts.py compile_request); template margin 192 from a probe on 2026-10-05: a minimal request costs 149 tokens (checked 2026-10-05) |
| rizzo-flow | `rizzo-flow-4b-q8_0@127.0.0.1:8017` | `tool-choice-v1` | Rizzo-AI-Academy/rizzo-flow @ b9ba007: schema.py MAX_SLOTS = 26; --ctx 8192 bounds each question with its state (prompts.py compile_request); template margin 192 from a probe on 2026-10-05: a minimal request costs 149 tokens (checked 2026-10-05) |

Model-written queries of `20261005T124458Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261005T124909Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261005T125333Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261005T144955Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261005T130805Z`: `bench/tasks/toolret-pilot-50.model-queries.json` (sha256 `533eb6c81fbe`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261005T132939Z`: `bench/tasks/toolret-heldout-200.model-queries.json` (sha256 `d8df10441cfb`), written by `gpt-5.4-mini-2026-03-17`.

Model-written queries of `20261005T145540Z`: `bench/tasks/toolret-heldout-200.model-queries.json` (sha256 `d8df10441cfb`), written by `gpt-5.4-mini-2026-03-17`.

## Held-out

`plain` searches with the ToolRet request, `model` with the queries a model wrote for the task, and `model (searched)` keeps the tasks where that model did search: on 81 of 200 tasks it answered without searching, and the request itself was used.

### P@1

P@1 measures a relevant tool first on positive requests, ignoring abstention; it does not measure task completion.

| arm | source | positives | P@1 (95% CI) | hybrid P@1 | Δ vs hybrid (95% CI) | ceiling |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 200 | 0.220 (0.17 to 0.28) | 0.220 | - | 0.590 |
| hybrid@20 | model | 200 | 0.215 (0.16 to 0.27) | 0.215 | - | 0.600 |
| hybrid@20 | model (searched) | 119 | 0.202 (0.13 to 0.28) | 0.202 | - | 0.647 |
| hybrid+laya-wide@20 | plain | 200 | 0.200 (0.15 to 0.26) | 0.220 | -0.020 (-0.09 to 0.04) | 0.590 |
| hybrid+laya-wide@20 | model | 200 | 0.185 (0.13 to 0.24) | 0.215 | -0.030 (-0.10 to 0.04) | 0.600 |
| hybrid+laya-wide@20 | model (searched) | 119 | 0.168 (0.10 to 0.24) | 0.202 | -0.034 (-0.12 to 0.05) | 0.647 |
| hybrid@50 | plain | 200 | 0.220 (0.17 to 0.28) | 0.220 | - | 0.715 |
| hybrid@50 | model | 200 | 0.215 (0.16 to 0.27) | 0.215 | - | 0.720 |
| hybrid@50 | model (searched) | 119 | 0.202 (0.13 to 0.28) | 0.202 | - | 0.748 |
| hybrid+laya-wide@50 | plain | 197 | 0.178 (0.13 to 0.23) | 0.223 | -0.046 (-0.11 to 0.01) | 0.711 |
| hybrid+laya-wide@50 | model | 198 | 0.152 (0.10 to 0.20) | 0.212 | -0.061 (-0.12 to 0.00) | 0.717 |
| hybrid+laya-wide@50 | model (searched) | 117 | 0.128 (0.07 to 0.19) | 0.197 | -0.068 (-0.15 to 0.01) | 0.744 |
| hybrid+rizzo-flow@20 | plain | 200 | 0.305 (0.24 to 0.36) | 0.220 | +0.085 (0.04 to 0.13) | 0.590 |
| hybrid+rizzo-flow@20 | model | 200 | 0.295 (0.23 to 0.36) | 0.215 | +0.080 (0.04 to 0.13) | 0.600 |
| hybrid+rizzo-flow@20 | model (searched) | 119 | 0.277 (0.20 to 0.36) | 0.202 | +0.076 (0.02 to 0.14) | 0.647 |
| hybrid+rizzo-flow@50 | plain | 200 | 0.335 (0.27 to 0.40) | 0.220 | +0.115 (0.06 to 0.17) | 0.715 |
| hybrid+rizzo-flow@50 | model | 200 | 0.335 (0.27 to 0.40) | 0.215 | +0.120 (0.07 to 0.18) | 0.720 |
| hybrid+rizzo-flow@50 | model (searched) | 119 | 0.319 (0.24 to 0.40) | 0.202 | +0.118 (0.05 to 0.19) | 0.748 |

**Key-only options, failed searches and lists two rounds cannot hold.** A card whose name would take its option past the model's per-option window is sent as its key alone, the name itself. A card that no question can show fails its search, which counts as an error. A search whose candidates two rounds cannot hold (groups asked at the same time, then one final question) is not applicable, and so is its cell: that is the two-round policy chosen here, not a verdict on what the model could do with the catalog another way.

- hybrid+laya-wide@20, plain: 1.6% of the card options sent as their key alone.
- hybrid+laya-wide@20, model: 1.7% of the card options sent as their key alone.
- hybrid+laya-wide@50, plain: 2.1% of the card options sent as their key alone; 6 failed searches: card 'craft_Vqa_tool_207': its option key alone takes 49 tokens, over max_option_tokens=48 (2); card 'craft_Tabmwp_tool_43': its option key alone takes 49 tokens, over max_option_tokens=48 (2); card 'toolbench_tool_10219': its option key alone takes 53 tokens, over max_option_tokens=48 (2).
- hybrid+laya-wide@50, model: 1.8% of the card options sent as their key alone; 4 failed searches: card 'craft_Vqa_tool_207': its option key alone takes 49 tokens, over max_option_tokens=48 (2); card 'craft_Tabmwp_tool_43': its option key alone takes 49 tokens, over max_option_tokens=48 (2).

### Abstention

Positive and negative requests are pooled below; the negative share is shown beside every row.

| arm | source | rule | τ | searches | errors | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate (95% CI) | abstention precision | abstention recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid@20 | plain | answer always | - | 400 | 0 | 1.000 | 0.110 | 44 | 356 | 0 | 50% | 0.890 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@20 | model | answer always | - | 400 | 0 | 1.000 | 0.107 | 43 | 357 | 0 | 50% | 0.892 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.101 | 24 | 214 | 0 | 50% | 0.899 (0.86 to 0.93) | n/a | 0.000 |
| hybrid+laya-wide@20 | plain | answer always | - | 400 | 0 | 1.000 | 0.100 | 40 | 360 | 0 | 50% | 0.900 (0.87 to 0.93) | n/a | 0.000 |
| hybrid+laya-wide@20 | plain | reserved | - | 400 | 0 | 0.850 | 0.115 | 39 | 301 | 60 | 50% | 0.752 (0.70 to 0.80) | 0.833 | 0.177 |
| hybrid+laya-wide@20 | plain | with an abstention threshold | 0.90 (4 searches without) | 400 | 0 | 0.015 | 0.167 | 1 | 5 | 394 | 50% | 0.013 (0.00 to 0.03) | 0.703 | 0.982 |
| hybrid+laya-wide@20 | model | answer always | - | 400 | 0 | 1.000 | 0.092 | 37 | 363 | 0 | 50% | 0.907 (0.88 to 0.94) | n/a | 0.000 |
| hybrid+laya-wide@20 | model | reserved | - | 400 | 0 | 0.900 | 0.103 | 37 | 323 | 40 | 50% | 0.807 (0.76 to 0.85) | 0.900 | 0.129 |
| hybrid+laya-wide@20 | model | with an abstention threshold | 0.90 (4 searches without) | 400 | 0 | 0.018 | 0.143 | 1 | 6 | 393 | 50% | 0.015 (0.00 to 0.03) | 0.697 | 0.979 |
| hybrid+laya-wide@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.084 | 20 | 218 | 0 | 50% | 0.916 (0.88 to 0.95) | n/a | 0.000 |
| hybrid+laya-wide@20 | model (searched) | reserved | - | 238 | 0 | 0.966 | 0.087 | 20 | 210 | 8 | 50% | 0.882 (0.84 to 0.92) | 0.750 | 0.037 |
| hybrid+laya-wide@20 | model (searched) | with an abstention threshold | 0.90 (2 searches without) | 238 | 0 | 0.025 | 0.167 | 1 | 5 | 232 | 50% | 0.021 (0.00 to 0.05) | 0.672 | 0.969 |
| hybrid@50 | plain | answer always | - | 400 | 0 | 1.000 | 0.110 | 44 | 356 | 0 | 50% | 0.890 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@50 | model | answer always | - | 400 | 0 | 1.000 | 0.107 | 43 | 357 | 0 | 50% | 0.892 (0.86 to 0.92) | n/a | 0.000 |
| hybrid@50 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.101 | 24 | 214 | 0 | 50% | 0.899 (0.86 to 0.93) | n/a | 0.000 |
| hybrid+laya-wide@50 | plain | answer always | - | 394 | 6 | 1.000 | 0.089 | 35 | 359 | 0 | 50% | 0.911 (0.89 to 0.94) | n/a | 0.000 |
| hybrid+laya-wide@50 | plain | reserved | - | 394 | 6 | 0.891 | 0.100 | 35 | 316 | 43 | 50% | 0.802 (0.76 to 0.85) | 0.791 | 0.134 |
| hybrid+laya-wide@50 | plain | with an abstention threshold | 0.00, 0.90, 0.95 | 394 | 6 | 0.089 | 0.114 | 4 | 31 | 359 | 50% | 0.079 (0.05 to 0.12) | 0.641 | 0.906 |
| hybrid+laya-wide@50 | model | answer always | - | 396 | 4 | 1.000 | 0.076 | 30 | 366 | 0 | 50% | 0.924 (0.90 to 0.95) | n/a | 0.000 |
| hybrid+laya-wide@50 | model | reserved | - | 396 | 4 | 0.939 | 0.081 | 30 | 342 | 24 | 50% | 0.864 (0.83 to 0.90) | 0.833 | 0.079 |
| hybrid+laya-wide@50 | model | with an abstention threshold | 0.00, 0.90, 0.95 | 396 | 4 | 0.066 | 0.154 | 4 | 22 | 370 | 50% | 0.056 (0.03 to 0.09) | 0.649 | 0.945 |
| hybrid+laya-wide@50 | model (searched) | answer always | - | 234 | 4 | 1.000 | 0.064 | 15 | 219 | 0 | 50% | 0.936 (0.91 to 0.97) | n/a | 0.000 |
| hybrid+laya-wide@50 | model (searched) | reserved | - | 234 | 4 | 0.987 | 0.065 | 15 | 216 | 3 | 50% | 0.923 (0.89 to 0.96) | 0.667 | 0.014 |
| hybrid+laya-wide@50 | model (searched) | with an abstention threshold | 0.00, 0.90, 0.95 | 234 | 4 | 0.060 | 0.071 | 1 | 13 | 220 | 50% | 0.056 (0.02 to 0.09) | 0.627 | 0.939 |
| hybrid+rizzo-flow@20 | plain | answer always | - | 400 | 0 | 1.000 | 0.152 | 61 | 339 | 0 | 50% | 0.848 (0.82 to 0.88) | n/a | 0.000 |
| hybrid+rizzo-flow@20 | plain | reserved | - | 400 | 0 | 0.915 | 0.164 | 60 | 306 | 34 | 50% | 0.765 (0.72 to 0.81) | 0.912 | 0.110 |
| hybrid+rizzo-flow@20 | plain | with an abstention threshold | 0.95 | 400 | 0 | 0.130 | 0.423 | 22 | 30 | 348 | 50% | 0.075 (0.04 to 0.11) | 0.733 | 0.904 |
| hybrid+rizzo-flow@20 | model | answer always | - | 400 | 0 | 1.000 | 0.147 | 59 | 341 | 0 | 50% | 0.853 (0.82 to 0.88) | n/a | 0.000 |
| hybrid+rizzo-flow@20 | model | reserved | - | 400 | 0 | 0.917 | 0.161 | 59 | 308 | 33 | 50% | 0.770 (0.72 to 0.81) | 0.909 | 0.107 |
| hybrid+rizzo-flow@20 | model | with an abstention threshold | 0.95 | 400 | 0 | 0.107 | 0.442 | 19 | 24 | 357 | 50% | 0.060 (0.03 to 0.09) | 0.720 | 0.918 |
| hybrid+rizzo-flow@20 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.139 | 33 | 205 | 0 | 50% | 0.861 (0.82 to 0.90) | n/a | 0.000 |
| hybrid+rizzo-flow@20 | model (searched) | reserved | - | 238 | 0 | 0.950 | 0.146 | 33 | 193 | 12 | 50% | 0.811 (0.76 to 0.87) | 0.917 | 0.068 |
| hybrid+rizzo-flow@20 | model (searched) | with an abstention threshold | 0.95 | 238 | 0 | 0.088 | 0.381 | 8 | 13 | 217 | 50% | 0.055 (0.02 to 0.10) | 0.687 | 0.925 |
| hybrid+rizzo-flow@50 | plain | answer always | - | 400 | 0 | 1.000 | 0.168 | 67 | 333 | 0 | 50% | 0.833 (0.80 to 0.86) | n/a | 0.000 |
| hybrid+rizzo-flow@50 | plain | reserved | - | 400 | 0 | 0.950 | 0.174 | 66 | 314 | 20 | 50% | 0.785 (0.74 to 0.82) | 0.950 | 0.074 |
| hybrid+rizzo-flow@50 | plain | with an abstention threshold | 0.95 | 400 | 0 | 0.058 | 0.435 | 10 | 13 | 377 | 50% | 0.033 (0.01 to 0.06) | 0.655 | 0.961 |
| hybrid+rizzo-flow@50 | model | answer always | - | 400 | 0 | 1.000 | 0.168 | 67 | 333 | 0 | 50% | 0.833 (0.80 to 0.86) | n/a | 0.000 |
| hybrid+rizzo-flow@50 | model | reserved | - | 400 | 0 | 0.945 | 0.175 | 66 | 312 | 22 | 50% | 0.780 (0.73 to 0.82) | 0.864 | 0.074 |
| hybrid+rizzo-flow@50 | model | with an abstention threshold | 0.95 | 400 | 0 | 0.035 | 0.500 | 7 | 7 | 386 | 50% | 0.018 (0.01 to 0.03) | 0.648 | 0.977 |
| hybrid+rizzo-flow@50 | model (searched) | answer always | - | 238 | 0 | 1.000 | 0.160 | 38 | 200 | 0 | 50% | 0.840 (0.80 to 0.88) | n/a | 0.000 |
| hybrid+rizzo-flow@50 | model (searched) | reserved | - | 238 | 0 | 0.950 | 0.164 | 37 | 189 | 12 | 50% | 0.794 (0.74 to 0.85) | 0.750 | 0.060 |
| hybrid+rizzo-flow@50 | model (searched) | with an abstention threshold | 0.95 | 238 | 0 | 0.025 | 0.500 | 3 | 3 | 232 | 50% | 0.013 (0.00 to 0.03) | 0.634 | 0.987 |

Held-out threshold keys with no dev τ (no dev search under the key in a run that made each search once): the reserved option alone decides their searches under both rules.

- `english@127.0.0.1:8010|tool-choice-v1|daaae964dd77bcd4|choice`: 6 searches of hybrid+laya-wide@20
- `english@127.0.0.1:8010|tool-choice-v1|82c3890d189d307b|choice`: 2 searches of hybrid+laya-wide@20

### Latency

| arm | source | decision p50 ms | p95 | latency basis | server p50 ms | p95 | retrieval p50 ms | p95 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| hybrid@20 | plain | - | - | - | - | - | 597 | 621 |
| hybrid@20 | model | - | - | - | - | - | 610 | 1,861 |
| hybrid@20 | model (searched) | - | - | - | - | - | 1,212 | 1,904 |
| hybrid+laya-wide@20 | plain | 104 | 118 | sum of calls | 100 | 114 | 597 | 621 |
| hybrid+laya-wide@20 | model | 105 | 122 | sum of calls | 102 | 118 | 610 | 1,861 |
| hybrid+laya-wide@20 | model (searched) | 107 | 123 | sum of calls | 104 | 119 | 1,212 | 1,904 |
| hybrid@50 | plain | - | - | - | - | - | 597 | 621 |
| hybrid@50 | model | - | - | - | - | - | 610 | 1,861 |
| hybrid@50 | model (searched) | - | - | - | - | - | 1,212 | 1,904 |
| hybrid+laya-wide@50 | plain | 241 | 311 | sum of calls | 235 | 304 | 597 | 621 |
| hybrid+laya-wide@50 | model | 245 | 315 | sum of calls | 239 | 308 | 610 | 1,863 |
| hybrid+laya-wide@50 | model (searched) | 251 | 325 | sum of calls | 246 | 318 | 1,212 | 1,912 |
| hybrid+rizzo-flow@20 | plain | 434 | 686 | sum of calls | - | - | 586 | 607 |
| hybrid+rizzo-flow@20 | model | 450 | 695 | sum of calls | - | - | 592 | 1,779 |
| hybrid+rizzo-flow@20 | model (searched) | 459 | 678 | sum of calls | - | - | 1,194 | 1,826 |
| hybrid+rizzo-flow@50 | plain | 1,376 | 2,167 | sum of calls | - | - | 586 | 607 |
| hybrid+rizzo-flow@50 | model | 1,340 | 2,009 | sum of calls | - | - | 592 | 1,779 |
| hybrid+rizzo-flow@50 | model (searched) | 1,325 | 1,951 | sum of calls | - | - | 1,194 | 1,826 |

### Cost per 1,000 searches

| arm | source | asks / search | input tokens / search | USD | CLM busy USD | CLM wall USD |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 0.00 | 0 | 0.0000 | - | - |
| hybrid@20 | model | 0.00 | 0 | 0.0000 | - | - |
| hybrid@20 | model (searched) | 0.00 | 0 | 0.0000 | - | - |
| hybrid+laya-wide@20 | plain | 2.99 | 941 | local | - | - |
| hybrid+laya-wide@20 | model | 2.98 | 974 | local | - | - |
| hybrid+laya-wide@20 | model (searched) | 2.98 | 1,017 | local | - | - |
| hybrid@50 | plain | 0.00 | 0 | 0.0000 | - | - |
| hybrid@50 | model | 0.00 | 0 | 0.0000 | - | - |
| hybrid@50 | model (searched) | 0.00 | 0 | 0.0000 | - | - |
| hybrid+laya-wide@50 | plain | 5.01 | 2,116 | local | - | - |
| hybrid+laya-wide@50 | model | 4.98 | 2,164 | local | - | - |
| hybrid+laya-wide@50 | model (searched) | 5.00 | 2,245 | local | - | - |
| hybrid+rizzo-flow@20 | plain | 1.00 | 1,339 | local | - | - |
| hybrid+rizzo-flow@20 | model | 1.00 | 1,318 | local | - | - |
| hybrid+rizzo-flow@20 | model (searched) | 1.00 | 1,307 | local | - | - |
| hybrid+rizzo-flow@50 | plain | 3.00 | 3,848 | local | - | - |
| hybrid+rizzo-flow@50 | model | 3.00 | 3,777 | local | - | - |
| hybrid+rizzo-flow@50 | model (searched) | 3.00 | 3,709 | local | - | - |

Per arm, all sources together: the summed seconds of its decisions on its latency basis, which CLM's wall figure prices, next to the arm's own clock time and the asks the decision cache replayed (hits) or sent (misses) during it.

| arm | searches | errors | not applicable | decision s | latency basis | arm clock s | cache hits | cache misses |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| hybrid@20 | 800 | 0 | - | - | - | 275.6 | - | - |
| hybrid+laya-wide@20 | 800 | 0 | - | 84.9 | sum of calls | 58.1 | 925 | 1,465 |
| hybrid@50 | 800 | 0 | - | - | - | 0.0 | - | - |
| hybrid+laya-wide@50 | 800 | 10 | - | 196.8 | sum of calls | 147.9 | 1,364 | 2,585 |
| hybrid+rizzo-flow@20 | 800 | 0 | - | 371.9 | sum of calls | 0.2 | 800 | 0 |
| hybrid+rizzo-flow@50 | 800 | 0 | - | 1155.6 | sum of calls | 2.7 | 2,395 | 5 |

## Dev

### Thresholds

| threshold key | decider | K | payload | searches (positives + negatives) | τ | U at τ | U at 0.00 | coverage at τ | wrong-tool rate at τ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `english@127.0.0.1:8010\|tool-choice-v1\|bb2167de7b53d632\|choice` | laya | 20 | reserved option, BRIEF, questions [[7, 7, 6], [7]] | 52 (25 + 27) | 0.95 | -0.058 | -0.654 | 0.096 | 0.077 |
| `english@127.0.0.1:8010\|tool-choice-v1\|16b7fa91fb744a41\|choice` | laya | 20 | reserved option, BRIEF, questions [[10, 10], [5]] | 79 (41 + 38) | 0.95 | 0.000 | -0.582 | 0.000 | 0.000 |
| `english@127.0.0.1:8010\|tool-choice-v1\|7f581460a377081f\|choice` | laya | 20 | reserved option, BRIEF, questions [[7, 7, 6], [4]] | 53 (26 + 27) | 0.90 | 0.000 | -0.604 | 0.000 | 0.000 |
| `english@127.0.0.1:8010\|tool-choice-v1\|3e3dcbd59a4c063e\|choice` | laya | 20 | reserved option, BRIEF, questions [[5, 5, 5, 5], [5]] | 12 (6 + 6) | 0.55 | 0.000 | -1.000 | 0.000 | 0.000 |
| `english@127.0.0.1:8010\|tool-choice-v1\|5be43a15ea7d5226\|choice` | laya | 50 | reserved option, BRIEF, questions [[10, 10, 10, 10, 10], [6]] | 19 (10 + 9) | 0.90 | -0.105 | -0.684 | 0.211 | 0.158 |
| `english@127.0.0.1:8010\|tool-choice-v1\|6458bcca55287d0c\|choice` | laya | 50 | reserved option, BRIEF, questions [[9, 9, 8, 8, 8, 8], [7]] | 9 (5 + 4) | 0.90 | 0.111 | -0.556 | 0.111 | 0.000 |
| `english@127.0.0.1:8010\|tool-choice-v1\|f94f42d07daecc05\|choice` | laya | 50 | reserved option, BRIEF, questions [[13, 13, 12, 12], [5]] | 7 (3 + 4) | 0.35 | 0.000 | -0.429 | 0.571 | 0.286 |
| `english@127.0.0.1:8010\|tool-choice-v1\|5f213fe500bb1c64\|choice` | laya-wide | 20 | reserved option, BRIEF, questions [[21]] | 190 (95 + 95) | 0.95 | -0.084 | -0.432 | 0.158 | 0.121 |
| `english@127.0.0.1:8010\|tool-choice-v1\|00ba74610a785db5\|choice` | laya-wide | 20 | reserved option, BRIEF, questions [[10, 10], [5]] | 6 (3 + 3) | 0.35 | 0.000 | -0.333 | 0.000 | 0.000 |
| `english@127.0.0.1:8010\|tool-choice-v1\|c49fe346388fcb9d\|choice` | laya-wide | 20 | reserved option, BRIEF, questions [[20], [3]] | 4 (2 + 2) | 0.65 | 0.000 | -1.000 | 0.000 | 0.000 |
| `english@127.0.0.1:8010\|tool-choice-v1\|31cd1eadfc771440\|choice` | laya-wide | 50 | reserved option, BRIEF, questions [[17, 17, 16], [7]] | 84 (41 + 43) | 0.95 | -0.024 | -0.750 | 0.071 | 0.048 |
| `english@127.0.0.1:8010\|tool-choice-v1\|22910edc46f29766\|choice` | laya-wide | 50 | reserved option, BRIEF, questions [[25, 25], [5]] | 116 (59 + 57) | 0.75 | 0.000 | -0.690 | 0.052 | 0.026 |
| `english@127.0.0.1:8010\|tool-choice-v1\|6b72b81c3e0298d9\|choice` | laya-wide | 20 | reserved option, BRIEF, questions [[10, 10], [5]] | 200 (100 + 100) | 0.90 | 0.005 | -0.605 | 0.015 | 0.005 |
| `english@127.0.0.1:8010\|tool-choice-v1\|cf4aa66e75aac348\|choice` | laya-wide | 50 | reserved option, BRIEF, questions [[13, 13, 12, 12], [9]] | 185 (93 + 92) | 0.90 | 0.000 | -0.762 | 0.022 | 0.011 |
| `english@127.0.0.1:8010\|tool-choice-v1\|5df48cd1df2aa3d3\|choice` | laya-wide | 50 | reserved option, BRIEF, questions [[10, 10, 10, 10, 10], [11]] | 13 (5 + 8) | 0.95 | 0.000 | -1.000 | 0.000 | 0.000 |
| `english@127.0.0.1:8010\|tool-choice-v1\|71e28f2ebd1297c3\|choice` | laya-wide | 50 | reserved option, BRIEF, questions [[17, 17, 16], [7]] | 2 (2 + 0) | 0.00 | 1.000 | 1.000 | 1.000 | 0.000 |
| `rizzo-flow-4b-q8_0@127.0.0.1:8017\|tool-choice-v1\|2c792f772b1dce1b\|choice` | rizzo-flow | 20 | reserved option, FULL, questions [[21]] | 200 (100 + 100) | 0.95 | -0.085 | -0.625 | 0.195 | 0.140 |
| `rizzo-flow-4b-q8_0@127.0.0.1:8017\|tool-choice-v1\|4766c781e268ae66\|choice` | rizzo-flow | 50 | reserved option, FULL, questions [[25, 25], [5]] | 200 (100 + 100) | 0.95 | -0.035 | -0.690 | 0.125 | 0.080 |
| `rizzo-flow-4b-q8_0@127.0.0.1:8017\|tool-choice-v1\|55db837d3c1774fe\|choice` | rizzo-flow | 20 | reserved option, BRIEF, questions [[21]] | 200 (100 + 100) | 0.90 | -0.075 | -0.610 | 0.215 | 0.145 |
| `rizzo-flow-4b-q8_0@127.0.0.1:8017\|tool-choice-v1\|4bd5f48ebf46b08f\|choice` | rizzo-flow | 50 | reserved option, BRIEF, questions [[25, 25], [5]] | 200 (100 + 100) | 0.95 | -0.045 | -0.690 | 0.145 | 0.095 |

### P@1 by dev run

P@1 on positives (their count in brackets), one column per dev run that made each search once; the ablations' settings are in the runs table.

| arm | source | `20261005T124458Z` | `20261005T124909Z` | `20261005T125333Z` | `20261005T144955Z` | `20261005T130805Z` |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | plain | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) |
| hybrid@20 | model | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) |
| hybrid@20 | model (searched) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) |
| hybrid+laya@20 | plain | not applicable (2) | - | - | - | - |
| hybrid+laya@20 | model | not applicable (2) | - | - | - | - |
| hybrid+laya@20 | model (searched) | 0.200 (30) | - | - | - | - |
| hybrid@50 | plain | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) |
| hybrid@50 | model | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) | 0.200 (50) |
| hybrid@50 | model (searched) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) | 0.200 (30) |
| hybrid+laya@50 | plain | not applicable (85) | - | - | - | - |
| hybrid+laya@50 | model | not applicable (80) | - | - | - | - |
| hybrid+laya@50 | model (searched) | not applicable (44) | - | - | - | - |
| hybrid+laya-wide@20 | plain | - | 0.200 (50) | 0.280 (50) | - | - |
| hybrid+laya-wide@20 | model | - | 0.160 (50) | 0.240 (50) | - | - |
| hybrid+laya-wide@20 | model (searched) | - | 0.100 (30) | 0.233 (30) | - | - |
| hybrid+laya-wide@50 | plain | - | 0.160 (50) | 0.160 (50) | - | - |
| hybrid+laya-wide@50 | model | - | 0.200 (50) | 0.180 (50) | - | - |
| hybrid+laya-wide@50 | model (searched) | - | 0.233 (30) | 0.167 (30) | - | - |
| hybrid+rizzo-flow@20 | plain | - | - | - | 0.260 (50) | 0.260 (50) |
| hybrid+rizzo-flow@20 | model | - | - | - | 0.280 (50) | 0.280 (50) |
| hybrid+rizzo-flow@20 | model (searched) | - | - | - | 0.300 (30) | 0.300 (30) |
| hybrid+rizzo-flow@50 | plain | - | - | - | 0.260 (50) | 0.280 (50) |
| hybrid+rizzo-flow@50 | model | - | - | - | 0.260 (50) | 0.260 (50) |
| hybrid+rizzo-flow@50 | model (searched) | - | - | - | 0.233 (30) | 0.300 (30) |

### Determinism

Largest change of one probability (a card's or the reserved option's) between repeats of the same search, over every repeated search of the arm.

| run | arm | searches repeated | largest abs Δp | at | within 0.01 |
|---|---|---:|---:|---|---|
| `20261005T125701Z` | hybrid+laya@20 | 20 | 0.0000 | apibank_query_38 (plain, positive) | yes |
| `20261005T125736Z` | hybrid+laya-wide@20 | 20 | 0.0000 | apibank_query_38 (plain, positive) | yes |
| `20261005T131210Z` | hybrid+rizzo-flow@20 | 20 | 0.0000 | apibank_query_38 (plain, positive) | yes |

### Errors

| run | arm | errors | of them, a card no question can show | not applicable | searches |
|---|---|---:|---:|---:|---:|
| `20261005T124458Z` | hybrid+laya@20 | 0 | 0 | 4 | 200 |
| `20261005T124458Z` | hybrid+laya@50 | 0 | 0 | 165 | 200 |

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
- **Errors.** A search whose decider raised `DecisionError`, whose candidates hold a card no question can show, or whose final question did not fit after round one, counts as an error and stays out of every rate, latency and cost figure. A search whose candidates two rounds cannot hold is not applicable: it is counted apart, and its cell reports no P@1 and no rule, in every table.
- **Latency.** Decision: the decider's calls as the client timed them, network included; a call the decision cache replays keeps the time the original call took. Its basis is the critical path (within a round only the slowest call counts, as round one's calls run at the same time), or the sum of calls for a model asked one request at a time, such as a local server. Server: the same by the server's own clock, where it reports one. The per-arm decision seconds use the same basis. Retrieval: the first hybrid retrieval of the search's queries, measured live; a query whose vector the embedding cache already held skips the embeddings call, so it reads faster than a cold one.
- **Cost.** Jev: reported input tokens at the price its declared limits give. Logprob: the reported usage, priced by genai-prices. CLM is paid in GPU time, at the list price in the run's manifest, two ways. Busy prices the server's own seconds per search, as if the GPU never sat idle: a lower bound. Wall prices the summed seconds of the arm's decision calls as the client saw them, as if one container served the searches one after another. It leaves out the cold starts, the idle gaps between arms and the scale-down window, which Modal also bills. Modal's CPU and memory charges come on top of both and are not included. Every arm also embeds its queries for hybrid retrieval, at the same cost for each arm, which the F1 retrieval results report.
- **Negatives** stand in for a catalog without the gold tools: the gold ids are dropped from a deeper retrieval, so the other tools' order can move slightly from what a smaller catalog would give.

## Risk-coverage over the threshold grid

### Dev: `english@127.0.0.1:8010|tool-choice-v1|bb2167de7b53d632|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.769 | 0.075 | 3 | 37 | 12 | 52% | 0.712 | -0.654 |
| 0.05 | 0.769 | 0.075 | 3 | 37 | 12 | 52% | 0.712 | -0.654 |
| 0.10 | 0.769 | 0.075 | 3 | 37 | 12 | 52% | 0.712 | -0.654 |
| 0.15 | 0.769 | 0.075 | 3 | 37 | 12 | 52% | 0.712 | -0.654 |
| 0.20 | 0.769 | 0.075 | 3 | 37 | 12 | 52% | 0.712 | -0.654 |
| 0.25 | 0.769 | 0.075 | 3 | 37 | 12 | 52% | 0.712 | -0.654 |
| 0.30 | 0.712 | 0.081 | 3 | 34 | 15 | 52% | 0.654 | -0.596 |
| 0.35 | 0.654 | 0.088 | 3 | 31 | 18 | 52% | 0.596 | -0.538 |
| 0.40 | 0.615 | 0.094 | 3 | 29 | 20 | 52% | 0.558 | -0.500 |
| 0.45 | 0.558 | 0.103 | 3 | 26 | 23 | 52% | 0.500 | -0.442 |
| 0.50 | 0.500 | 0.115 | 3 | 23 | 26 | 52% | 0.442 | -0.385 |
| 0.55 | 0.481 | 0.120 | 3 | 22 | 27 | 52% | 0.423 | -0.365 |
| 0.60 | 0.481 | 0.120 | 3 | 22 | 27 | 52% | 0.423 | -0.365 |
| 0.65 | 0.481 | 0.120 | 3 | 22 | 27 | 52% | 0.423 | -0.365 |
| 0.70 | 0.404 | 0.143 | 3 | 18 | 31 | 52% | 0.346 | -0.288 |
| 0.75 | 0.385 | 0.150 | 3 | 17 | 32 | 52% | 0.327 | -0.269 |
| 0.80 | 0.327 | 0.059 | 1 | 16 | 35 | 52% | 0.308 | -0.288 |
| 0.85 | 0.250 | 0.077 | 1 | 12 | 39 | 52% | 0.231 | -0.212 |
| 0.90 | 0.212 | 0.091 | 1 | 10 | 41 | 52% | 0.192 | -0.173 |
| **0.95** | 0.096 | 0.200 | 1 | 4 | 47 | 52% | 0.077 | -0.058 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|16b7fa91fb744a41|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.911 | 0.181 | 13 | 59 | 7 | 48% | 0.747 | -0.582 |
| 0.05 | 0.911 | 0.181 | 13 | 59 | 7 | 48% | 0.747 | -0.582 |
| 0.10 | 0.911 | 0.181 | 13 | 59 | 7 | 48% | 0.747 | -0.582 |
| 0.15 | 0.911 | 0.181 | 13 | 59 | 7 | 48% | 0.747 | -0.582 |
| 0.20 | 0.911 | 0.181 | 13 | 59 | 7 | 48% | 0.747 | -0.582 |
| 0.25 | 0.886 | 0.186 | 13 | 57 | 9 | 48% | 0.722 | -0.557 |
| 0.30 | 0.772 | 0.213 | 13 | 48 | 18 | 48% | 0.608 | -0.443 |
| 0.35 | 0.532 | 0.262 | 11 | 31 | 37 | 48% | 0.392 | -0.253 |
| 0.40 | 0.468 | 0.270 | 10 | 27 | 42 | 48% | 0.342 | -0.215 |
| 0.45 | 0.316 | 0.280 | 7 | 18 | 54 | 48% | 0.228 | -0.139 |
| 0.50 | 0.228 | 0.222 | 4 | 14 | 61 | 48% | 0.177 | -0.127 |
| 0.55 | 0.203 | 0.250 | 4 | 12 | 63 | 48% | 0.152 | -0.101 |
| 0.60 | 0.165 | 0.231 | 3 | 10 | 66 | 48% | 0.127 | -0.089 |
| 0.65 | 0.165 | 0.231 | 3 | 10 | 66 | 48% | 0.127 | -0.089 |
| 0.70 | 0.101 | 0.250 | 2 | 6 | 71 | 48% | 0.076 | -0.051 |
| 0.75 | 0.076 | 0.333 | 2 | 4 | 73 | 48% | 0.051 | -0.025 |
| 0.80 | 0.051 | 0.000 | 0 | 4 | 75 | 48% | 0.051 | -0.051 |
| 0.85 | 0.025 | 0.000 | 0 | 2 | 77 | 48% | 0.025 | -0.025 |
| 0.90 | 0.025 | 0.000 | 0 | 2 | 77 | 48% | 0.025 | -0.025 |
| **0.95** | 0.000 | n/a | 0 | 0 | 79 | 48% | 0.000 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|7f581460a377081f|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.755 | 0.100 | 4 | 36 | 13 | 51% | 0.679 | -0.604 |
| 0.05 | 0.755 | 0.100 | 4 | 36 | 13 | 51% | 0.679 | -0.604 |
| 0.10 | 0.755 | 0.100 | 4 | 36 | 13 | 51% | 0.679 | -0.604 |
| 0.15 | 0.755 | 0.100 | 4 | 36 | 13 | 51% | 0.679 | -0.604 |
| 0.20 | 0.755 | 0.100 | 4 | 36 | 13 | 51% | 0.679 | -0.604 |
| 0.25 | 0.755 | 0.100 | 4 | 36 | 13 | 51% | 0.679 | -0.604 |
| 0.30 | 0.755 | 0.100 | 4 | 36 | 13 | 51% | 0.679 | -0.604 |
| 0.35 | 0.679 | 0.111 | 4 | 32 | 17 | 51% | 0.604 | -0.528 |
| 0.40 | 0.472 | 0.160 | 4 | 21 | 28 | 51% | 0.396 | -0.321 |
| 0.45 | 0.226 | 0.000 | 0 | 12 | 41 | 51% | 0.226 | -0.226 |
| 0.50 | 0.170 | 0.000 | 0 | 9 | 44 | 51% | 0.170 | -0.170 |
| 0.55 | 0.170 | 0.000 | 0 | 9 | 44 | 51% | 0.170 | -0.170 |
| 0.60 | 0.170 | 0.000 | 0 | 9 | 44 | 51% | 0.170 | -0.170 |
| 0.65 | 0.170 | 0.000 | 0 | 9 | 44 | 51% | 0.170 | -0.170 |
| 0.70 | 0.151 | 0.000 | 0 | 8 | 45 | 51% | 0.151 | -0.151 |
| 0.75 | 0.132 | 0.000 | 0 | 7 | 46 | 51% | 0.132 | -0.132 |
| 0.80 | 0.094 | 0.000 | 0 | 5 | 48 | 51% | 0.094 | -0.094 |
| 0.85 | 0.019 | 0.000 | 0 | 1 | 52 | 51% | 0.019 | -0.019 |
| **0.90** | 0.000 | n/a | 0 | 0 | 53 | 51% | 0.000 | 0.000 |
| 0.95 | 0.000 | n/a | 0 | 0 | 53 | 51% | 0.000 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|3e3dcbd59a4c063e|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.000 | 0 | 12 | 0 | 50% | 1.000 | -1.000 |
| 0.05 | 1.000 | 0.000 | 0 | 12 | 0 | 50% | 1.000 | -1.000 |
| 0.10 | 1.000 | 0.000 | 0 | 12 | 0 | 50% | 1.000 | -1.000 |
| 0.15 | 1.000 | 0.000 | 0 | 12 | 0 | 50% | 1.000 | -1.000 |
| 0.20 | 1.000 | 0.000 | 0 | 12 | 0 | 50% | 1.000 | -1.000 |
| 0.25 | 0.583 | 0.000 | 0 | 7 | 5 | 50% | 0.583 | -0.583 |
| 0.30 | 0.500 | 0.000 | 0 | 6 | 6 | 50% | 0.500 | -0.500 |
| 0.35 | 0.417 | 0.000 | 0 | 5 | 7 | 50% | 0.417 | -0.417 |
| 0.40 | 0.250 | 0.000 | 0 | 3 | 9 | 50% | 0.250 | -0.250 |
| 0.45 | 0.250 | 0.000 | 0 | 3 | 9 | 50% | 0.250 | -0.250 |
| 0.50 | 0.083 | 0.000 | 0 | 1 | 11 | 50% | 0.083 | -0.083 |
| **0.55** | 0.000 | n/a | 0 | 0 | 12 | 50% | 0.000 | 0.000 |
| 0.60 | 0.000 | n/a | 0 | 0 | 12 | 50% | 0.000 | 0.000 |
| 0.65 | 0.000 | n/a | 0 | 0 | 12 | 50% | 0.000 | 0.000 |
| 0.70 | 0.000 | n/a | 0 | 0 | 12 | 50% | 0.000 | 0.000 |
| 0.75 | 0.000 | n/a | 0 | 0 | 12 | 50% | 0.000 | 0.000 |
| 0.80 | 0.000 | n/a | 0 | 0 | 12 | 50% | 0.000 | 0.000 |
| 0.85 | 0.000 | n/a | 0 | 0 | 12 | 50% | 0.000 | 0.000 |
| 0.90 | 0.000 | n/a | 0 | 0 | 12 | 50% | 0.000 | 0.000 |
| 0.95 | 0.000 | n/a | 0 | 0 | 12 | 50% | 0.000 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|5be43a15ea7d5226|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.895 | 0.118 | 2 | 15 | 2 | 47% | 0.789 | -0.684 |
| 0.05 | 0.895 | 0.118 | 2 | 15 | 2 | 47% | 0.789 | -0.684 |
| 0.10 | 0.895 | 0.118 | 2 | 15 | 2 | 47% | 0.789 | -0.684 |
| 0.15 | 0.895 | 0.118 | 2 | 15 | 2 | 47% | 0.789 | -0.684 |
| 0.20 | 0.895 | 0.118 | 2 | 15 | 2 | 47% | 0.789 | -0.684 |
| 0.25 | 0.895 | 0.118 | 2 | 15 | 2 | 47% | 0.789 | -0.684 |
| 0.30 | 0.895 | 0.118 | 2 | 15 | 2 | 47% | 0.789 | -0.684 |
| 0.35 | 0.895 | 0.118 | 2 | 15 | 2 | 47% | 0.789 | -0.684 |
| 0.40 | 0.737 | 0.071 | 1 | 13 | 5 | 47% | 0.684 | -0.632 |
| 0.45 | 0.737 | 0.071 | 1 | 13 | 5 | 47% | 0.684 | -0.632 |
| 0.50 | 0.526 | 0.100 | 1 | 9 | 9 | 47% | 0.474 | -0.421 |
| 0.55 | 0.316 | 0.167 | 1 | 5 | 13 | 47% | 0.263 | -0.211 |
| 0.60 | 0.316 | 0.167 | 1 | 5 | 13 | 47% | 0.263 | -0.211 |
| 0.65 | 0.316 | 0.167 | 1 | 5 | 13 | 47% | 0.263 | -0.211 |
| 0.70 | 0.316 | 0.167 | 1 | 5 | 13 | 47% | 0.263 | -0.211 |
| 0.75 | 0.263 | 0.200 | 1 | 4 | 14 | 47% | 0.211 | -0.158 |
| 0.80 | 0.263 | 0.200 | 1 | 4 | 14 | 47% | 0.211 | -0.158 |
| 0.85 | 0.263 | 0.200 | 1 | 4 | 14 | 47% | 0.211 | -0.158 |
| **0.90** | 0.211 | 0.250 | 1 | 3 | 15 | 47% | 0.158 | -0.105 |
| 0.95 | 0.105 | 0.000 | 0 | 2 | 17 | 47% | 0.105 | -0.105 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|6458bcca55287d0c|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.222 | 2 | 7 | 0 | 44% | 0.778 | -0.556 |
| 0.05 | 1.000 | 0.222 | 2 | 7 | 0 | 44% | 0.778 | -0.556 |
| 0.10 | 1.000 | 0.222 | 2 | 7 | 0 | 44% | 0.778 | -0.556 |
| 0.15 | 1.000 | 0.222 | 2 | 7 | 0 | 44% | 0.778 | -0.556 |
| 0.20 | 1.000 | 0.222 | 2 | 7 | 0 | 44% | 0.778 | -0.556 |
| 0.25 | 1.000 | 0.222 | 2 | 7 | 0 | 44% | 0.778 | -0.556 |
| 0.30 | 0.778 | 0.286 | 2 | 5 | 2 | 44% | 0.556 | -0.333 |
| 0.35 | 0.778 | 0.286 | 2 | 5 | 2 | 44% | 0.556 | -0.333 |
| 0.40 | 0.667 | 0.333 | 2 | 4 | 3 | 44% | 0.444 | -0.222 |
| 0.45 | 0.556 | 0.400 | 2 | 3 | 4 | 44% | 0.333 | -0.111 |
| 0.50 | 0.556 | 0.400 | 2 | 3 | 4 | 44% | 0.333 | -0.111 |
| 0.55 | 0.333 | 0.333 | 1 | 2 | 6 | 44% | 0.222 | -0.111 |
| 0.60 | 0.222 | 0.500 | 1 | 1 | 7 | 44% | 0.111 | 0.000 |
| 0.65 | 0.222 | 0.500 | 1 | 1 | 7 | 44% | 0.111 | 0.000 |
| 0.70 | 0.222 | 0.500 | 1 | 1 | 7 | 44% | 0.111 | 0.000 |
| 0.75 | 0.222 | 0.500 | 1 | 1 | 7 | 44% | 0.111 | 0.000 |
| 0.80 | 0.222 | 0.500 | 1 | 1 | 7 | 44% | 0.111 | 0.000 |
| 0.85 | 0.222 | 0.500 | 1 | 1 | 7 | 44% | 0.111 | 0.000 |
| **0.90** | 0.111 | 1.000 | 1 | 0 | 8 | 44% | 0.000 | 0.111 |
| 0.95 | 0.000 | n/a | 0 | 0 | 9 | 44% | 0.000 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|f94f42d07daecc05|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.286 | 2 | 5 | 0 | 57% | 0.714 | -0.429 |
| 0.05 | 1.000 | 0.286 | 2 | 5 | 0 | 57% | 0.714 | -0.429 |
| 0.10 | 1.000 | 0.286 | 2 | 5 | 0 | 57% | 0.714 | -0.429 |
| 0.15 | 1.000 | 0.286 | 2 | 5 | 0 | 57% | 0.714 | -0.429 |
| 0.20 | 1.000 | 0.286 | 2 | 5 | 0 | 57% | 0.714 | -0.429 |
| 0.25 | 1.000 | 0.286 | 2 | 5 | 0 | 57% | 0.714 | -0.429 |
| 0.30 | 0.714 | 0.400 | 2 | 3 | 2 | 57% | 0.429 | -0.143 |
| **0.35** | 0.571 | 0.500 | 2 | 2 | 3 | 57% | 0.286 | 0.000 |
| 0.40 | 0.571 | 0.500 | 2 | 2 | 3 | 57% | 0.286 | 0.000 |
| 0.45 | 0.571 | 0.500 | 2 | 2 | 3 | 57% | 0.286 | 0.000 |
| 0.50 | 0.286 | 0.000 | 0 | 2 | 5 | 57% | 0.286 | -0.286 |
| 0.55 | 0.286 | 0.000 | 0 | 2 | 5 | 57% | 0.286 | -0.286 |
| 0.60 | 0.286 | 0.000 | 0 | 2 | 5 | 57% | 0.286 | -0.286 |
| 0.65 | 0.143 | 0.000 | 0 | 1 | 6 | 57% | 0.143 | -0.143 |
| 0.70 | 0.000 | n/a | 0 | 0 | 7 | 57% | 0.000 | 0.000 |
| 0.75 | 0.000 | n/a | 0 | 0 | 7 | 57% | 0.000 | 0.000 |
| 0.80 | 0.000 | n/a | 0 | 0 | 7 | 57% | 0.000 | 0.000 |
| 0.85 | 0.000 | n/a | 0 | 0 | 7 | 57% | 0.000 | 0.000 |
| 0.90 | 0.000 | n/a | 0 | 0 | 7 | 57% | 0.000 | 0.000 |
| 0.95 | 0.000 | n/a | 0 | 0 | 7 | 57% | 0.000 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|5f213fe500bb1c64|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.558 | 0.113 | 12 | 94 | 84 | 50% | 0.495 | -0.432 |
| 0.05 | 0.558 | 0.113 | 12 | 94 | 84 | 50% | 0.495 | -0.432 |
| 0.10 | 0.558 | 0.113 | 12 | 94 | 84 | 50% | 0.495 | -0.432 |
| 0.15 | 0.558 | 0.113 | 12 | 94 | 84 | 50% | 0.495 | -0.432 |
| 0.20 | 0.558 | 0.113 | 12 | 94 | 84 | 50% | 0.495 | -0.432 |
| 0.25 | 0.553 | 0.114 | 12 | 93 | 85 | 50% | 0.489 | -0.426 |
| 0.30 | 0.553 | 0.114 | 12 | 93 | 85 | 50% | 0.489 | -0.426 |
| 0.35 | 0.547 | 0.115 | 12 | 92 | 86 | 50% | 0.484 | -0.421 |
| 0.40 | 0.526 | 0.120 | 12 | 88 | 90 | 50% | 0.463 | -0.400 |
| 0.45 | 0.505 | 0.125 | 12 | 84 | 94 | 50% | 0.442 | -0.379 |
| 0.50 | 0.458 | 0.126 | 11 | 76 | 103 | 50% | 0.400 | -0.342 |
| 0.55 | 0.437 | 0.133 | 11 | 72 | 107 | 50% | 0.379 | -0.321 |
| 0.60 | 0.395 | 0.147 | 11 | 64 | 115 | 50% | 0.337 | -0.279 |
| 0.65 | 0.347 | 0.167 | 11 | 55 | 124 | 50% | 0.289 | -0.232 |
| 0.70 | 0.337 | 0.172 | 11 | 53 | 126 | 50% | 0.279 | -0.221 |
| 0.75 | 0.305 | 0.190 | 11 | 47 | 132 | 50% | 0.247 | -0.189 |
| 0.80 | 0.268 | 0.137 | 7 | 44 | 139 | 50% | 0.232 | -0.195 |
| 0.85 | 0.247 | 0.149 | 7 | 40 | 143 | 50% | 0.211 | -0.174 |
| 0.90 | 0.221 | 0.167 | 7 | 35 | 148 | 50% | 0.184 | -0.147 |
| **0.95** | 0.158 | 0.233 | 7 | 23 | 160 | 50% | 0.121 | -0.084 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|00ba74610a785db5|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.333 | 0.000 | 0 | 2 | 4 | 50% | 0.333 | -0.333 |
| 0.05 | 0.333 | 0.000 | 0 | 2 | 4 | 50% | 0.333 | -0.333 |
| 0.10 | 0.333 | 0.000 | 0 | 2 | 4 | 50% | 0.333 | -0.333 |
| 0.15 | 0.333 | 0.000 | 0 | 2 | 4 | 50% | 0.333 | -0.333 |
| 0.20 | 0.333 | 0.000 | 0 | 2 | 4 | 50% | 0.333 | -0.333 |
| 0.25 | 0.333 | 0.000 | 0 | 2 | 4 | 50% | 0.333 | -0.333 |
| 0.30 | 0.333 | 0.000 | 0 | 2 | 4 | 50% | 0.333 | -0.333 |
| **0.35** | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.40 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.45 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.50 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.55 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.60 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.65 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.70 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.75 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.80 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.85 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.90 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |
| 0.95 | 0.000 | n/a | 0 | 0 | 6 | 50% | 0.000 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|c49fe346388fcb9d|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.05 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.10 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.15 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.20 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.25 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.30 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.35 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.40 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.45 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.50 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.55 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| 0.60 | 1.000 | 0.000 | 0 | 4 | 0 | 50% | 1.000 | -1.000 |
| **0.65** | 0.000 | n/a | 0 | 0 | 4 | 50% | 0.000 | 0.000 |
| 0.70 | 0.000 | n/a | 0 | 0 | 4 | 50% | 0.000 | 0.000 |
| 0.75 | 0.000 | n/a | 0 | 0 | 4 | 50% | 0.000 | 0.000 |
| 0.80 | 0.000 | n/a | 0 | 0 | 4 | 50% | 0.000 | 0.000 |
| 0.85 | 0.000 | n/a | 0 | 0 | 4 | 50% | 0.000 | 0.000 |
| 0.90 | 0.000 | n/a | 0 | 0 | 4 | 50% | 0.000 | 0.000 |
| 0.95 | 0.000 | n/a | 0 | 0 | 4 | 50% | 0.000 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|31cd1eadfc771440|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.798 | 0.030 | 2 | 65 | 17 | 51% | 0.774 | -0.750 |
| 0.05 | 0.798 | 0.030 | 2 | 65 | 17 | 51% | 0.774 | -0.750 |
| 0.10 | 0.798 | 0.030 | 2 | 65 | 17 | 51% | 0.774 | -0.750 |
| 0.15 | 0.798 | 0.030 | 2 | 65 | 17 | 51% | 0.774 | -0.750 |
| 0.20 | 0.798 | 0.030 | 2 | 65 | 17 | 51% | 0.774 | -0.750 |
| 0.25 | 0.738 | 0.032 | 2 | 60 | 22 | 51% | 0.714 | -0.690 |
| 0.30 | 0.679 | 0.035 | 2 | 55 | 27 | 51% | 0.655 | -0.631 |
| 0.35 | 0.655 | 0.036 | 2 | 53 | 29 | 51% | 0.631 | -0.607 |
| 0.40 | 0.524 | 0.045 | 2 | 42 | 40 | 51% | 0.500 | -0.476 |
| 0.45 | 0.524 | 0.045 | 2 | 42 | 40 | 51% | 0.500 | -0.476 |
| 0.50 | 0.476 | 0.050 | 2 | 38 | 44 | 51% | 0.452 | -0.429 |
| 0.55 | 0.440 | 0.054 | 2 | 35 | 47 | 51% | 0.417 | -0.393 |
| 0.60 | 0.429 | 0.056 | 2 | 34 | 48 | 51% | 0.405 | -0.381 |
| 0.65 | 0.298 | 0.080 | 2 | 23 | 59 | 51% | 0.274 | -0.250 |
| 0.70 | 0.190 | 0.125 | 2 | 14 | 68 | 51% | 0.167 | -0.143 |
| 0.75 | 0.190 | 0.125 | 2 | 14 | 68 | 51% | 0.167 | -0.143 |
| 0.80 | 0.179 | 0.133 | 2 | 13 | 69 | 51% | 0.155 | -0.131 |
| 0.85 | 0.167 | 0.143 | 2 | 12 | 70 | 51% | 0.143 | -0.119 |
| 0.90 | 0.119 | 0.200 | 2 | 8 | 74 | 51% | 0.095 | -0.071 |
| **0.95** | 0.071 | 0.333 | 2 | 4 | 78 | 51% | 0.048 | -0.024 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|22910edc46f29766|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.931 | 0.130 | 14 | 94 | 8 | 49% | 0.810 | -0.690 |
| 0.05 | 0.931 | 0.130 | 14 | 94 | 8 | 49% | 0.810 | -0.690 |
| 0.10 | 0.931 | 0.130 | 14 | 94 | 8 | 49% | 0.810 | -0.690 |
| 0.15 | 0.931 | 0.130 | 14 | 94 | 8 | 49% | 0.810 | -0.690 |
| 0.20 | 0.931 | 0.130 | 14 | 94 | 8 | 49% | 0.810 | -0.690 |
| 0.25 | 0.922 | 0.131 | 14 | 93 | 9 | 49% | 0.802 | -0.681 |
| 0.30 | 0.716 | 0.157 | 13 | 70 | 33 | 49% | 0.603 | -0.491 |
| 0.35 | 0.629 | 0.178 | 13 | 60 | 43 | 49% | 0.517 | -0.405 |
| 0.40 | 0.526 | 0.180 | 11 | 50 | 55 | 49% | 0.431 | -0.336 |
| 0.45 | 0.440 | 0.196 | 10 | 41 | 65 | 49% | 0.353 | -0.267 |
| 0.50 | 0.345 | 0.250 | 10 | 30 | 76 | 49% | 0.259 | -0.172 |
| 0.55 | 0.216 | 0.280 | 7 | 18 | 91 | 49% | 0.155 | -0.095 |
| 0.60 | 0.155 | 0.278 | 5 | 13 | 98 | 49% | 0.112 | -0.069 |
| 0.65 | 0.155 | 0.278 | 5 | 13 | 98 | 49% | 0.112 | -0.069 |
| 0.70 | 0.103 | 0.333 | 4 | 8 | 104 | 49% | 0.069 | -0.034 |
| **0.75** | 0.052 | 0.500 | 3 | 3 | 110 | 49% | 0.026 | 0.000 |
| 0.80 | 0.026 | 0.000 | 0 | 3 | 113 | 49% | 0.026 | -0.026 |
| 0.85 | 0.017 | 0.000 | 0 | 2 | 114 | 49% | 0.017 | -0.017 |
| 0.90 | 0.000 | n/a | 0 | 0 | 116 | 49% | 0.000 | 0.000 |
| 0.95 | 0.000 | n/a | 0 | 0 | 116 | 49% | 0.000 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|6b72b81c3e0298d9|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.815 | 0.129 | 21 | 142 | 37 | 50% | 0.710 | -0.605 |
| 0.05 | 0.815 | 0.129 | 21 | 142 | 37 | 50% | 0.710 | -0.605 |
| 0.10 | 0.815 | 0.129 | 21 | 142 | 37 | 50% | 0.710 | -0.605 |
| 0.15 | 0.815 | 0.129 | 21 | 142 | 37 | 50% | 0.710 | -0.605 |
| 0.20 | 0.815 | 0.129 | 21 | 142 | 37 | 50% | 0.710 | -0.605 |
| 0.25 | 0.810 | 0.130 | 21 | 141 | 38 | 50% | 0.705 | -0.600 |
| 0.30 | 0.650 | 0.154 | 20 | 110 | 70 | 50% | 0.550 | -0.450 |
| 0.35 | 0.520 | 0.163 | 17 | 87 | 96 | 50% | 0.435 | -0.350 |
| 0.40 | 0.430 | 0.198 | 17 | 69 | 114 | 50% | 0.345 | -0.260 |
| 0.45 | 0.355 | 0.183 | 13 | 58 | 129 | 50% | 0.290 | -0.225 |
| 0.50 | 0.240 | 0.188 | 9 | 39 | 152 | 50% | 0.195 | -0.150 |
| 0.55 | 0.160 | 0.219 | 7 | 25 | 168 | 50% | 0.125 | -0.090 |
| 0.60 | 0.115 | 0.304 | 7 | 16 | 177 | 50% | 0.080 | -0.045 |
| 0.65 | 0.100 | 0.300 | 6 | 14 | 180 | 50% | 0.070 | -0.040 |
| 0.70 | 0.090 | 0.333 | 6 | 12 | 182 | 50% | 0.060 | -0.030 |
| 0.75 | 0.065 | 0.231 | 3 | 10 | 187 | 50% | 0.050 | -0.035 |
| 0.80 | 0.035 | 0.429 | 3 | 4 | 193 | 50% | 0.020 | -0.005 |
| 0.85 | 0.020 | 0.500 | 2 | 2 | 196 | 50% | 0.010 | 0.000 |
| **0.90** | 0.015 | 0.667 | 2 | 1 | 197 | 50% | 0.005 | 0.005 |
| 0.95 | 0.005 | 1.000 | 1 | 0 | 199 | 50% | 0.000 | 0.005 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|cf4aa66e75aac348|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.903 | 0.078 | 13 | 154 | 18 | 50% | 0.832 | -0.762 |
| 0.05 | 0.903 | 0.078 | 13 | 154 | 18 | 50% | 0.832 | -0.762 |
| 0.10 | 0.903 | 0.078 | 13 | 154 | 18 | 50% | 0.832 | -0.762 |
| 0.15 | 0.903 | 0.078 | 13 | 154 | 18 | 50% | 0.832 | -0.762 |
| 0.20 | 0.843 | 0.083 | 13 | 143 | 29 | 50% | 0.773 | -0.703 |
| 0.25 | 0.697 | 0.093 | 12 | 117 | 56 | 50% | 0.632 | -0.568 |
| 0.30 | 0.508 | 0.117 | 11 | 83 | 91 | 50% | 0.449 | -0.389 |
| 0.35 | 0.395 | 0.110 | 8 | 65 | 112 | 50% | 0.351 | -0.308 |
| 0.40 | 0.297 | 0.145 | 8 | 47 | 130 | 50% | 0.254 | -0.211 |
| 0.45 | 0.238 | 0.182 | 8 | 36 | 141 | 50% | 0.195 | -0.151 |
| 0.50 | 0.195 | 0.222 | 8 | 28 | 149 | 50% | 0.151 | -0.108 |
| 0.55 | 0.151 | 0.250 | 7 | 21 | 157 | 50% | 0.114 | -0.076 |
| 0.60 | 0.141 | 0.269 | 7 | 19 | 159 | 50% | 0.103 | -0.065 |
| 0.65 | 0.114 | 0.143 | 3 | 18 | 164 | 50% | 0.097 | -0.081 |
| 0.70 | 0.097 | 0.167 | 3 | 15 | 167 | 50% | 0.081 | -0.065 |
| 0.75 | 0.086 | 0.125 | 2 | 14 | 169 | 50% | 0.076 | -0.065 |
| 0.80 | 0.086 | 0.125 | 2 | 14 | 169 | 50% | 0.076 | -0.065 |
| 0.85 | 0.054 | 0.200 | 2 | 8 | 175 | 50% | 0.043 | -0.032 |
| **0.90** | 0.022 | 0.500 | 2 | 2 | 181 | 50% | 0.011 | 0.000 |
| 0.95 | 0.011 | 0.500 | 1 | 1 | 183 | 50% | 0.005 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|5df48cd1df2aa3d3|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.000 | 0 | 13 | 0 | 62% | 1.000 | -1.000 |
| 0.05 | 1.000 | 0.000 | 0 | 13 | 0 | 62% | 1.000 | -1.000 |
| 0.10 | 1.000 | 0.000 | 0 | 13 | 0 | 62% | 1.000 | -1.000 |
| 0.15 | 1.000 | 0.000 | 0 | 13 | 0 | 62% | 1.000 | -1.000 |
| 0.20 | 1.000 | 0.000 | 0 | 13 | 0 | 62% | 1.000 | -1.000 |
| 0.25 | 1.000 | 0.000 | 0 | 13 | 0 | 62% | 1.000 | -1.000 |
| 0.30 | 0.923 | 0.000 | 0 | 12 | 1 | 62% | 0.923 | -0.923 |
| 0.35 | 0.769 | 0.000 | 0 | 10 | 3 | 62% | 0.769 | -0.769 |
| 0.40 | 0.615 | 0.000 | 0 | 8 | 5 | 62% | 0.615 | -0.615 |
| 0.45 | 0.615 | 0.000 | 0 | 8 | 5 | 62% | 0.615 | -0.615 |
| 0.50 | 0.462 | 0.000 | 0 | 6 | 7 | 62% | 0.462 | -0.462 |
| 0.55 | 0.462 | 0.000 | 0 | 6 | 7 | 62% | 0.462 | -0.462 |
| 0.60 | 0.462 | 0.000 | 0 | 6 | 7 | 62% | 0.462 | -0.462 |
| 0.65 | 0.462 | 0.000 | 0 | 6 | 7 | 62% | 0.462 | -0.462 |
| 0.70 | 0.462 | 0.000 | 0 | 6 | 7 | 62% | 0.462 | -0.462 |
| 0.75 | 0.462 | 0.000 | 0 | 6 | 7 | 62% | 0.462 | -0.462 |
| 0.80 | 0.462 | 0.000 | 0 | 6 | 7 | 62% | 0.462 | -0.462 |
| 0.85 | 0.462 | 0.000 | 0 | 6 | 7 | 62% | 0.462 | -0.462 |
| 0.90 | 0.308 | 0.000 | 0 | 4 | 9 | 62% | 0.308 | -0.308 |
| **0.95** | 0.000 | n/a | 0 | 0 | 13 | 62% | 0.000 | 0.000 |

### Dev: `english@127.0.0.1:8010|tool-choice-v1|71e28f2ebd1297c3|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.05 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.10 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.15 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.20 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.25 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.30 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.35 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.40 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.45 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.50 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.55 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.60 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.65 | 1.000 | 1.000 | 2 | 0 | 0 | 0% | 0.000 | 1.000 |
| 0.70 | 0.000 | n/a | 0 | 0 | 2 | 0% | 0.000 | 0.000 |
| 0.75 | 0.000 | n/a | 0 | 0 | 2 | 0% | 0.000 | 0.000 |
| 0.80 | 0.000 | n/a | 0 | 0 | 2 | 0% | 0.000 | 0.000 |
| 0.85 | 0.000 | n/a | 0 | 0 | 2 | 0% | 0.000 | 0.000 |
| 0.90 | 0.000 | n/a | 0 | 0 | 2 | 0% | 0.000 | 0.000 |
| 0.95 | 0.000 | n/a | 0 | 0 | 2 | 0% | 0.000 | 0.000 |

### Dev: `rizzo-flow-4b-q8_0@127.0.0.1:8017|tool-choice-v1|2c792f772b1dce1b|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.895 | 0.151 | 27 | 152 | 21 | 50% | 0.760 | -0.625 |
| 0.05 | 0.895 | 0.151 | 27 | 152 | 21 | 50% | 0.760 | -0.625 |
| 0.10 | 0.895 | 0.151 | 27 | 152 | 21 | 50% | 0.760 | -0.625 |
| 0.15 | 0.895 | 0.151 | 27 | 152 | 21 | 50% | 0.760 | -0.625 |
| 0.20 | 0.885 | 0.153 | 27 | 150 | 23 | 50% | 0.750 | -0.615 |
| 0.25 | 0.885 | 0.153 | 27 | 150 | 23 | 50% | 0.750 | -0.615 |
| 0.30 | 0.885 | 0.153 | 27 | 150 | 23 | 50% | 0.750 | -0.615 |
| 0.35 | 0.875 | 0.154 | 27 | 148 | 25 | 50% | 0.740 | -0.605 |
| 0.40 | 0.840 | 0.161 | 27 | 141 | 32 | 50% | 0.705 | -0.570 |
| 0.45 | 0.770 | 0.169 | 26 | 128 | 46 | 50% | 0.640 | -0.510 |
| 0.50 | 0.710 | 0.183 | 26 | 116 | 58 | 50% | 0.580 | -0.450 |
| 0.55 | 0.585 | 0.188 | 22 | 95 | 83 | 50% | 0.475 | -0.365 |
| 0.60 | 0.540 | 0.204 | 22 | 86 | 92 | 50% | 0.430 | -0.320 |
| 0.65 | 0.505 | 0.208 | 21 | 80 | 99 | 50% | 0.400 | -0.295 |
| 0.70 | 0.445 | 0.225 | 20 | 69 | 111 | 50% | 0.345 | -0.245 |
| 0.75 | 0.420 | 0.238 | 20 | 64 | 116 | 50% | 0.320 | -0.220 |
| 0.80 | 0.395 | 0.228 | 18 | 61 | 121 | 50% | 0.305 | -0.215 |
| 0.85 | 0.325 | 0.215 | 14 | 51 | 135 | 50% | 0.255 | -0.185 |
| 0.90 | 0.265 | 0.264 | 14 | 39 | 147 | 50% | 0.195 | -0.125 |
| **0.95** | 0.195 | 0.282 | 11 | 28 | 161 | 50% | 0.140 | -0.085 |

### Dev: `rizzo-flow-4b-q8_0@127.0.0.1:8017|tool-choice-v1|4766c781e268ae66|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.950 | 0.137 | 26 | 164 | 10 | 50% | 0.820 | -0.690 |
| 0.05 | 0.950 | 0.137 | 26 | 164 | 10 | 50% | 0.820 | -0.690 |
| 0.10 | 0.950 | 0.137 | 26 | 164 | 10 | 50% | 0.820 | -0.690 |
| 0.15 | 0.950 | 0.137 | 26 | 164 | 10 | 50% | 0.820 | -0.690 |
| 0.20 | 0.950 | 0.137 | 26 | 164 | 10 | 50% | 0.820 | -0.690 |
| 0.25 | 0.950 | 0.137 | 26 | 164 | 10 | 50% | 0.820 | -0.690 |
| 0.30 | 0.925 | 0.141 | 26 | 159 | 15 | 50% | 0.795 | -0.665 |
| 0.35 | 0.920 | 0.141 | 26 | 158 | 16 | 50% | 0.790 | -0.660 |
| 0.40 | 0.905 | 0.144 | 26 | 155 | 19 | 50% | 0.775 | -0.645 |
| 0.45 | 0.875 | 0.149 | 26 | 149 | 25 | 50% | 0.745 | -0.615 |
| 0.50 | 0.745 | 0.154 | 23 | 126 | 51 | 50% | 0.630 | -0.515 |
| 0.55 | 0.695 | 0.165 | 23 | 116 | 61 | 50% | 0.580 | -0.465 |
| 0.60 | 0.645 | 0.178 | 23 | 106 | 71 | 50% | 0.530 | -0.415 |
| 0.65 | 0.530 | 0.189 | 20 | 86 | 94 | 50% | 0.430 | -0.330 |
| 0.70 | 0.490 | 0.204 | 20 | 78 | 102 | 50% | 0.390 | -0.290 |
| 0.75 | 0.410 | 0.220 | 18 | 64 | 118 | 50% | 0.320 | -0.230 |
| 0.80 | 0.355 | 0.239 | 17 | 54 | 129 | 50% | 0.270 | -0.185 |
| 0.85 | 0.270 | 0.315 | 17 | 37 | 146 | 50% | 0.185 | -0.100 |
| 0.90 | 0.220 | 0.295 | 13 | 31 | 156 | 50% | 0.155 | -0.090 |
| **0.95** | 0.125 | 0.360 | 9 | 16 | 175 | 50% | 0.080 | -0.035 |

### Dev: `rizzo-flow-4b-q8_0@127.0.0.1:8017|tool-choice-v1|55db837d3c1774fe|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.880 | 0.153 | 27 | 149 | 24 | 50% | 0.745 | -0.610 |
| 0.05 | 0.880 | 0.153 | 27 | 149 | 24 | 50% | 0.745 | -0.610 |
| 0.10 | 0.880 | 0.153 | 27 | 149 | 24 | 50% | 0.745 | -0.610 |
| 0.15 | 0.880 | 0.153 | 27 | 149 | 24 | 50% | 0.745 | -0.610 |
| 0.20 | 0.870 | 0.155 | 27 | 147 | 26 | 50% | 0.735 | -0.600 |
| 0.25 | 0.870 | 0.155 | 27 | 147 | 26 | 50% | 0.735 | -0.600 |
| 0.30 | 0.845 | 0.154 | 26 | 143 | 31 | 50% | 0.715 | -0.585 |
| 0.35 | 0.820 | 0.159 | 26 | 138 | 36 | 50% | 0.690 | -0.560 |
| 0.40 | 0.765 | 0.170 | 26 | 127 | 47 | 50% | 0.635 | -0.505 |
| 0.45 | 0.725 | 0.166 | 24 | 121 | 55 | 50% | 0.605 | -0.485 |
| 0.50 | 0.675 | 0.178 | 24 | 111 | 65 | 50% | 0.555 | -0.435 |
| 0.55 | 0.620 | 0.177 | 22 | 102 | 76 | 50% | 0.510 | -0.400 |
| 0.60 | 0.515 | 0.214 | 22 | 81 | 97 | 50% | 0.405 | -0.295 |
| 0.65 | 0.480 | 0.229 | 22 | 74 | 104 | 50% | 0.370 | -0.260 |
| 0.70 | 0.445 | 0.236 | 21 | 68 | 111 | 50% | 0.340 | -0.235 |
| 0.75 | 0.405 | 0.235 | 19 | 62 | 119 | 50% | 0.310 | -0.215 |
| 0.80 | 0.350 | 0.257 | 18 | 52 | 130 | 50% | 0.260 | -0.170 |
| 0.85 | 0.280 | 0.286 | 16 | 40 | 144 | 50% | 0.200 | -0.120 |
| **0.90** | 0.215 | 0.326 | 14 | 29 | 157 | 50% | 0.145 | -0.075 |
| 0.95 | 0.190 | 0.289 | 11 | 27 | 162 | 50% | 0.135 | -0.080 |

### Dev: `rizzo-flow-4b-q8_0@127.0.0.1:8017|tool-choice-v1|4bd5f48ebf46b08f|choice`

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.960 | 0.141 | 27 | 165 | 8 | 50% | 0.825 | -0.690 |
| 0.05 | 0.960 | 0.141 | 27 | 165 | 8 | 50% | 0.825 | -0.690 |
| 0.10 | 0.960 | 0.141 | 27 | 165 | 8 | 50% | 0.825 | -0.690 |
| 0.15 | 0.960 | 0.141 | 27 | 165 | 8 | 50% | 0.825 | -0.690 |
| 0.20 | 0.960 | 0.141 | 27 | 165 | 8 | 50% | 0.825 | -0.690 |
| 0.25 | 0.960 | 0.141 | 27 | 165 | 8 | 50% | 0.825 | -0.690 |
| 0.30 | 0.940 | 0.144 | 27 | 161 | 12 | 50% | 0.805 | -0.670 |
| 0.35 | 0.915 | 0.148 | 27 | 156 | 17 | 50% | 0.780 | -0.645 |
| 0.40 | 0.875 | 0.149 | 26 | 149 | 25 | 50% | 0.745 | -0.615 |
| 0.45 | 0.765 | 0.157 | 24 | 129 | 47 | 50% | 0.645 | -0.525 |
| 0.50 | 0.720 | 0.160 | 23 | 121 | 56 | 50% | 0.605 | -0.490 |
| 0.55 | 0.610 | 0.172 | 21 | 101 | 78 | 50% | 0.505 | -0.400 |
| 0.60 | 0.540 | 0.176 | 19 | 89 | 92 | 50% | 0.445 | -0.350 |
| 0.65 | 0.515 | 0.184 | 19 | 84 | 97 | 50% | 0.420 | -0.325 |
| 0.70 | 0.435 | 0.207 | 18 | 69 | 113 | 50% | 0.345 | -0.255 |
| 0.75 | 0.375 | 0.227 | 17 | 58 | 125 | 50% | 0.290 | -0.205 |
| 0.80 | 0.350 | 0.243 | 17 | 53 | 130 | 50% | 0.265 | -0.180 |
| 0.85 | 0.295 | 0.237 | 14 | 45 | 141 | 50% | 0.225 | -0.155 |
| 0.90 | 0.195 | 0.256 | 10 | 29 | 161 | 50% | 0.145 | -0.095 |
| **0.95** | 0.145 | 0.345 | 10 | 19 | 171 | 50% | 0.095 | -0.045 |

### Held-out: hybrid+laya-wide@20, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.875 | 0.109 | 76 | 624 | 100 | 50% | 0.780 | -0.685 |
| 0.05 | 0.875 | 0.109 | 76 | 624 | 100 | 50% | 0.780 | -0.685 |
| 0.10 | 0.875 | 0.109 | 76 | 624 | 100 | 50% | 0.780 | -0.685 |
| 0.15 | 0.875 | 0.109 | 76 | 624 | 100 | 50% | 0.780 | -0.685 |
| 0.20 | 0.875 | 0.109 | 76 | 624 | 100 | 50% | 0.780 | -0.685 |
| 0.25 | 0.848 | 0.112 | 76 | 602 | 122 | 50% | 0.752 | -0.657 |
| 0.30 | 0.700 | 0.120 | 67 | 493 | 240 | 50% | 0.616 | -0.532 |
| 0.35 | 0.569 | 0.130 | 59 | 396 | 345 | 50% | 0.495 | -0.421 |
| 0.40 | 0.453 | 0.144 | 52 | 310 | 438 | 50% | 0.388 | -0.323 |
| 0.45 | 0.330 | 0.174 | 46 | 218 | 536 | 50% | 0.273 | -0.215 |
| 0.50 | 0.244 | 0.215 | 42 | 153 | 605 | 50% | 0.191 | -0.139 |
| 0.55 | 0.194 | 0.232 | 36 | 119 | 645 | 50% | 0.149 | -0.104 |
| 0.60 | 0.138 | 0.282 | 31 | 79 | 690 | 50% | 0.099 | -0.060 |
| 0.65 | 0.096 | 0.247 | 19 | 58 | 723 | 50% | 0.072 | -0.049 |
| 0.70 | 0.064 | 0.255 | 13 | 38 | 749 | 50% | 0.048 | -0.031 |
| 0.75 | 0.043 | 0.265 | 9 | 25 | 766 | 50% | 0.031 | -0.020 |
| 0.80 | 0.028 | 0.227 | 5 | 17 | 778 | 50% | 0.021 | -0.015 |
| 0.85 | 0.019 | 0.267 | 4 | 11 | 785 | 50% | 0.014 | -0.009 |
| 0.90 | 0.011 | 0.222 | 2 | 7 | 791 | 50% | 0.009 | -0.006 |
| 0.95 | 0.004 | 0.333 | 1 | 2 | 797 | 50% | 0.003 | -0.001 |

### Held-out: hybrid+laya-wide@50, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.915 | 0.090 | 65 | 658 | 67 | 50% | 0.833 | -0.751 |
| 0.05 | 0.915 | 0.090 | 65 | 658 | 67 | 50% | 0.833 | -0.751 |
| 0.10 | 0.915 | 0.090 | 65 | 658 | 67 | 50% | 0.833 | -0.751 |
| 0.15 | 0.915 | 0.090 | 65 | 658 | 67 | 50% | 0.833 | -0.751 |
| 0.20 | 0.837 | 0.095 | 63 | 598 | 129 | 50% | 0.757 | -0.677 |
| 0.25 | 0.684 | 0.111 | 60 | 480 | 250 | 50% | 0.608 | -0.532 |
| 0.30 | 0.544 | 0.121 | 52 | 378 | 360 | 50% | 0.478 | -0.413 |
| 0.35 | 0.447 | 0.122 | 43 | 310 | 437 | 50% | 0.392 | -0.338 |
| 0.40 | 0.373 | 0.125 | 37 | 258 | 495 | 50% | 0.327 | -0.280 |
| 0.45 | 0.313 | 0.138 | 34 | 213 | 543 | 50% | 0.270 | -0.227 |
| 0.50 | 0.281 | 0.135 | 30 | 192 | 568 | 50% | 0.243 | -0.205 |
| 0.55 | 0.225 | 0.157 | 28 | 150 | 612 | 50% | 0.190 | -0.154 |
| 0.60 | 0.185 | 0.171 | 25 | 121 | 644 | 50% | 0.153 | -0.122 |
| 0.65 | 0.144 | 0.193 | 22 | 92 | 676 | 50% | 0.116 | -0.089 |
| 0.70 | 0.114 | 0.167 | 15 | 75 | 700 | 50% | 0.095 | -0.076 |
| 0.75 | 0.084 | 0.197 | 13 | 53 | 724 | 50% | 0.067 | -0.051 |
| 0.80 | 0.067 | 0.113 | 6 | 47 | 737 | 50% | 0.059 | -0.052 |
| 0.85 | 0.053 | 0.119 | 5 | 37 | 748 | 50% | 0.047 | -0.041 |
| 0.90 | 0.043 | 0.147 | 5 | 29 | 756 | 50% | 0.037 | -0.030 |
| 0.95 | 0.034 | 0.185 | 5 | 22 | 763 | 50% | 0.028 | -0.022 |

### Held-out: hybrid+rizzo-flow@20, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.916 | 0.162 | 119 | 614 | 67 | 50% | 0.767 | -0.619 |
| 0.05 | 0.916 | 0.162 | 119 | 614 | 67 | 50% | 0.767 | -0.619 |
| 0.10 | 0.916 | 0.162 | 119 | 614 | 67 | 50% | 0.767 | -0.619 |
| 0.15 | 0.916 | 0.162 | 119 | 614 | 67 | 50% | 0.767 | -0.619 |
| 0.20 | 0.916 | 0.162 | 119 | 614 | 67 | 50% | 0.767 | -0.619 |
| 0.25 | 0.910 | 0.163 | 119 | 609 | 72 | 50% | 0.761 | -0.613 |
| 0.30 | 0.892 | 0.167 | 119 | 595 | 86 | 50% | 0.744 | -0.595 |
| 0.35 | 0.858 | 0.171 | 117 | 569 | 114 | 50% | 0.711 | -0.565 |
| 0.40 | 0.818 | 0.177 | 116 | 538 | 146 | 50% | 0.672 | -0.527 |
| 0.45 | 0.756 | 0.185 | 112 | 493 | 195 | 50% | 0.616 | -0.476 |
| 0.50 | 0.666 | 0.197 | 105 | 428 | 267 | 50% | 0.535 | -0.404 |
| 0.55 | 0.581 | 0.213 | 99 | 366 | 335 | 50% | 0.458 | -0.334 |
| 0.60 | 0.537 | 0.221 | 95 | 335 | 370 | 50% | 0.419 | -0.300 |
| 0.65 | 0.484 | 0.217 | 84 | 303 | 413 | 50% | 0.379 | -0.274 |
| 0.70 | 0.412 | 0.239 | 79 | 251 | 470 | 50% | 0.314 | -0.215 |
| 0.75 | 0.378 | 0.238 | 72 | 230 | 498 | 50% | 0.287 | -0.198 |
| 0.80 | 0.307 | 0.280 | 69 | 177 | 554 | 50% | 0.221 | -0.135 |
| 0.85 | 0.246 | 0.299 | 59 | 138 | 603 | 50% | 0.172 | -0.099 |
| 0.90 | 0.189 | 0.318 | 48 | 103 | 649 | 50% | 0.129 | -0.069 |
| **0.95** | 0.119 | 0.432 | 41 | 54 | 705 | 50% | 0.068 | -0.016 |

### Held-out: hybrid+rizzo-flow@50, for reading only

| τ | coverage | selective accuracy | correct | wrong | abstained | negatives | wrong-tool rate | U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.948 | 0.174 | 132 | 626 | 42 | 50% | 0.782 | -0.618 |
| 0.05 | 0.948 | 0.174 | 132 | 626 | 42 | 50% | 0.782 | -0.618 |
| 0.10 | 0.948 | 0.174 | 132 | 626 | 42 | 50% | 0.782 | -0.618 |
| 0.15 | 0.948 | 0.174 | 132 | 626 | 42 | 50% | 0.782 | -0.618 |
| 0.20 | 0.948 | 0.174 | 132 | 626 | 42 | 50% | 0.782 | -0.618 |
| 0.25 | 0.948 | 0.174 | 132 | 626 | 42 | 50% | 0.782 | -0.618 |
| 0.30 | 0.946 | 0.174 | 132 | 625 | 43 | 50% | 0.781 | -0.616 |
| 0.35 | 0.914 | 0.176 | 129 | 602 | 69 | 50% | 0.752 | -0.591 |
| 0.40 | 0.889 | 0.180 | 128 | 583 | 89 | 50% | 0.729 | -0.569 |
| 0.45 | 0.819 | 0.192 | 126 | 529 | 145 | 50% | 0.661 | -0.504 |
| 0.50 | 0.733 | 0.203 | 119 | 467 | 214 | 50% | 0.584 | -0.435 |
| 0.55 | 0.635 | 0.226 | 115 | 393 | 292 | 50% | 0.491 | -0.347 |
| 0.60 | 0.530 | 0.236 | 100 | 324 | 376 | 50% | 0.405 | -0.280 |
| 0.65 | 0.463 | 0.238 | 88 | 282 | 430 | 50% | 0.352 | -0.242 |
| 0.70 | 0.386 | 0.262 | 81 | 228 | 491 | 50% | 0.285 | -0.184 |
| 0.75 | 0.312 | 0.276 | 69 | 181 | 550 | 50% | 0.226 | -0.140 |
| 0.80 | 0.246 | 0.299 | 59 | 138 | 603 | 50% | 0.172 | -0.099 |
| 0.85 | 0.190 | 0.316 | 48 | 104 | 648 | 50% | 0.130 | -0.070 |
| 0.90 | 0.121 | 0.361 | 35 | 62 | 703 | 50% | 0.077 | -0.034 |
| **0.95** | 0.046 | 0.459 | 17 | 20 | 763 | 50% | 0.025 | -0.004 |

