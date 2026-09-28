# ToolRet retrieval, run 20260928T154543Z

50 tasks from `bench/tasks/toolret-pilot-50.json` (sha256 `90b0f6009aa5`), `mteb/ToolRetrieval` at `76d45e5`: 44,453 tools, catalog `sha256:7d56b70f3415`. toolhunch 0.1.0, commit `9a43c92`, Python 3.14.3.

Query modes:

- `plain`: the query text alone, which is what an agent has.
- `instructed`: ToolRet's formats, `{instruction} {query}` for lexical arms and `Instruct: {instruction}\nQuery: {query}` for `dense`; `hybrid` gets the lexical one for both halves. ToolRet's instructions were written by GPT-4o from the target tools (paper, section 3.3), so this mode hints at the answer: `plain` is the realistic setting.

## plain

| arm | R@1 | R@5 | R@10 | R@20 | R@50 | nDCG@10 (95% CI) | P@1 (95% CI) | p50 ms | p95 ms | index s | emb. tokens | USD | USD / 1k queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| keywords | 0.020 | 0.040 | 0.040 | 0.047 | 0.047 | 0.028 (0.00-0.08) | 0.020 (0.00-0.06) | 112.9 | 129.7 | 0.11 | 0 | 0.0000 | 0.0000 |
| bm25 | 0.152 | 0.267 | 0.293 | 0.383 | 0.413 | 0.248 (0.15-0.35) | 0.220 (0.12-0.34) | 4.0 | 10.8 | 0.41 | 0 | 0.0000 | 0.0000 |
| bm25-stem | 0.138 | 0.283 | 0.293 | 0.337 | 0.398 | 0.246 (0.15-0.35) | 0.180 (0.08-0.30) | 4.3 | 11.4 | 0.48 | 0 | 0.0000 | 0.0000 |
| bm25-namedesc | 0.125 | 0.207 | 0.240 | 0.350 | 0.413 | 0.197 (0.11-0.29) | 0.180 (0.08-0.28) | 3.2 | 8.5 | 0.34 | 0 | 0.0000 | 0.0000 |
| bm25-raw | 0.152 | 0.300 | 0.337 | 0.370 | 0.448 | 0.276 (0.18-0.38) | 0.240 (0.12-0.36) | 7.5 | 17.8 | 1.43 | 0 | 0.0000 | 0.0000 |
| bm25s-toolret | 0.132 | 0.210 | 0.243 | 0.290 | 0.420 | 0.202 (0.11-0.30) | 0.200 (0.10-0.32) | 0.3 | 0.4 | 0.98 | 0 | 0.0000 | 0.0000 |
| dense | 0.103 | 0.302 | 0.337 | 0.388 | 0.425 | 0.242 (0.15-0.33) | 0.120 (0.04-0.20) | 878.5 | 1175.6 | 212.91 | 1,449,953 | 0.0290 | 0.0009 |
| hybrid | 0.145 | 0.300 | 0.333 | 0.407 | 0.492 | 0.264 (0.17-0.37) | 0.200 (0.10-0.32) | 640.2 | 667.8 | 5.99 | 4,495 | 0.0001 | 0.0009 |

## instructed

| arm | R@1 | R@5 | R@10 | R@20 | R@50 | nDCG@10 (95% CI) | P@1 (95% CI) | p50 ms | p95 ms | index s | emb. tokens | USD | USD / 1k queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| keywords | 0.020 | 0.040 | 0.060 | 0.060 | 0.080 | 0.037 (0.00-0.09) | 0.020 (0.00-0.06) | 118.1 | 137.5 | 0.11 | 0 | 0.0000 | 0.0000 |
| bm25 | 0.292 | 0.403 | 0.430 | 0.517 | 0.593 | 0.383 (0.26-0.50) | 0.360 (0.22-0.50) | 12.9 | 17.0 | 0.41 | 0 | 0.0000 | 0.0000 |
| bm25-stem | 0.292 | 0.417 | 0.507 | 0.542 | 0.588 | 0.414 (0.30-0.53) | 0.380 (0.24-0.52) | 11.9 | 16.6 | 0.48 | 0 | 0.0000 | 0.0000 |
| bm25-namedesc | 0.225 | 0.350 | 0.390 | 0.463 | 0.560 | 0.328 (0.21-0.44) | 0.280 (0.16-0.40) | 11.3 | 18.9 | 0.34 | 0 | 0.0000 | 0.0000 |
| bm25-raw | 0.252 | 0.393 | 0.450 | 0.507 | 0.602 | 0.374 (0.26-0.48) | 0.340 (0.22-0.46) | 17.3 | 28.4 | 1.43 | 0 | 0.0000 | 0.0000 |
| bm25s-toolret | 0.208 | 0.397 | 0.515 | 0.552 | 0.605 | 0.372 (0.27-0.48) | 0.260 (0.16-0.38) | 0.5 | 0.8 | 0.98 | 0 | 0.0000 | 0.0000 |
| dense | 0.257 | 0.430 | 0.507 | 0.517 | 0.592 | 0.397 (0.28-0.52) | 0.300 (0.18-0.44) | 875.3 | 1097.3 | 212.91 | 1,449,953 | 0.0290 | 0.0019 |
| hybrid | 0.282 | 0.432 | 0.502 | 0.563 | 0.643 | 0.414 (0.31-0.53) | 0.340 (0.22-0.46) | 832.0 | 1017.9 | 5.99 | 4,495 | 0.0001 | 0.0018 |

Cost: the run billed 1,454,448 embedding tokens, $0.0291 with `text-embedding-3-small`. The catalog is embedded once and shared through the embedding cache, so the first arm that needs it carries the index cost.

Notes:

- Relevance is binary (ToolRet qrels). R@k: share of the relevant tools in the top k. nDCG@10: log2 discount. P@1: the first result is relevant. 95% CIs: percentile bootstrap over tasks (1,000 resamples, seed 0); with few tasks they are wide, and differences inside them are not findings.
- Latency is per search, after one warm-up search per arm that builds the index (`index s`). `dense` and `hybrid` latency is mostly the pure-Python cosine over every card vector, plus an embeddings API round trip when the query text is not cached yet (`hybrid`'s plain queries repeat `dense`'s, so they skip it).
- `keywords` is Pydantic AI's keyword search, copied and parity-tested. Inside an agent `ToolSearch(max_results=10)` keeps its first 10 results, so R@20 and R@50 describe the ranking, not what the model would see.
- `bm25s-toolret` replays ToolRet's own BM25 baseline (bm25s defaults, raw tool JSON, English stop words) on this corpus.
- Embedding tokens and USD are what the run billed per arm, for both modes; texts already in the embedding cache cost nothing. USD / 1k queries is what embedding 1,000 such queries costs in deployment (their cl100k tokens, cached or not), without the one-off index.
