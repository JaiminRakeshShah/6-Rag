# Retrieval evals comparison

Golden set: `data/gold/policy_rag_golden.json`. 50 questions. Ranking metrics use the 47 questions that name a supporting section. The 3 abstain questions have no relevant chunk, so they are timed and left out of the ranking means.

A chunk is relevant when its section is listed in `supporting_sections`. If that section sets `chunk_index`, only that chunk counts. The ranked list is `hybrid_search` after reciprocal rank fusion and the cross-encoder. The mean relevant set is 1.09 chunks.

## Baseline, k = 10

Recorded with `python -m src.eval_retrieval`. Each index contributes 10 chunks, and the metrics are computed on the 10 chunks returned after reranking.

| Metric | Value |
|---|---:|
| MRR | 0.7557 |
| Precision | 0.1085 |
| nDCG@10 | 0.8074 |
| Latency | 1.181 s |
| Recall@10 | 1.0000 |
| Precision@10 | 0.1085 |

Scored questions: 47. Timed questions: 50. Precision and precision@10 match because every scored query returned 10 chunks. Recall@10 is 1, so precision@10 is the mean relevant-set size divided by 10.

## Same ranking, fewer chunks kept

The first stage still fetches 10 chunks from dense search and 10 from BM25. The rows keep the first 1, 3, 5, or 10 chunks of that reranked list. Thirty of 47 questions already place the first relevant chunk at rank 1.

| Chunks kept | Recall | MRR | nDCG | Precision | Questions missing evidence |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.596 | 0.638 | 0.638 | 0.638 | 21 / 47 |
| 3 | 0.830 | 0.727 | 0.747 | 0.305 | 8 / 47 |
| 5 | 0.894 | 0.742 | 0.773 | 0.196 | 5 / 47 |
| 10 | 1.000 | 0.756 | 0.807 | 0.109 | 0 / 47 |

First relevant rank on this list: rank 1 on 30 questions, rank 2 on 7, rank 3 on 2, rank 4 on 2, rank 5 on 1, rank 7 on 3, rank 8 on 1, rank 10 on 1.

Ten kept chunks is the only cutoff that still contains every evidence chunk. Five misses 5 questions. Three misses 8.

## Smaller first-stage fetch

Dense search and BM25 each contribute 10, 5, or 3 chunks before fusion and the cross-encoder. Latency is the mean over all 50 questions in this run. Compare these times with each other. The baseline latency above is from the earlier run.

| Fetched per index | Recall of the full list | Recall of the top 5 | Questions never retrieved | Mean latency |
|---:|---:|---:|---:|---:|
| 10 | 1.000 | 0.894 | 0 / 47 | 1.39 s |
| 5 | 1.000 | 0.957 | 0 / 47 | 0.88 s |
| 3 | 0.957 | 0.957 | 2 / 47 | 0.50 s |

Fetching 5 from each index still reaches recall 1.00 on the full reranked list, and its top 5 misses 2 questions. Fetching 3 leaves 2 questions with no relevant chunk in the candidate pool. A longer final list cannot recover those.

## Hybrid ratio against the reranked list

Both lists are built from the same 10 dense neighbors and 10 BM25 hits. The hybrid ratio rank is `dense_weight * cosine similarity + (1 - dense_weight) * normalized BM25`, with no cross-encoder. The current final list is reciprocal rank fusion followed by the cross-encoder. From here the cutoffs to keep are 5 and 10.

| Ranker | Chunks kept | Recall | MRR | nDCG | Precision | Questions missing evidence |
|---|---:|---:|---:|---:|---:|---:|
| RRF + cross-encoder | 5 | 0.894 | 0.742 | 0.773 | 0.196 | 5 / 47 |
| RRF only | 5 | 0.883 | 0.658 | 0.711 | 0.187 | 6 / 47 |
| Ratio 0.7 / 0.3 | 5 | 0.883 | 0.698 | 0.740 | 0.187 | 6 / 47 |
| Ratio 0.5 / 0.5 | 5 | 0.915 | 0.690 | 0.741 | 0.196 | 5 / 47 |
| Ratio 0.9 / 0.1 | 5 | 0.883 | 0.682 | 0.728 | 0.187 | 6 / 47 |
| RRF + cross-encoder | 10 | 1.000 | 0.756 | 0.807 | 0.109 | 0 / 47 |
| RRF only | 10 | 1.000 | 0.672 | 0.751 | 0.109 | 0 / 47 |
| Ratio 0.7 / 0.3 | 10 | 0.936 | 0.706 | 0.758 | 0.100 | 4 / 47 |
| Ratio 0.5 / 0.5 | 10 | 1.000 | 0.698 | 0.770 | 0.109 | 0 / 47 |
| Ratio 0.9 / 0.1 | 10 | 0.904 | 0.686 | 0.735 | 0.096 | 5 / 47 |

The 0.7 / 0.3 ratio does not improve the final list. At 5 chunks it misses 6 questions instead of 5, and MRR falls from 0.742 to 0.698. At 10 chunks it misses 4 questions that the reranked list still contains. An even split keeps recall at 1.00 for 10 chunks, and its rank quality stays below the cross-encoder. The reranked list remains the one to use.

## Weighted RRF plus the cross-encoder

The candidate pool is still 10 dense neighbors and 10 BM25 hits. Reciprocal rank fusion then uses a dense/sparse split, and the cross-encoder reranks that pool. A 0.5 / 0.5 split is the same ordering as equal RRF, so it matches the current list. A 0.7 / 0.3 split changes the order only when two cross-encoder scores tie. That happened on 2 of 47 questions, and the relevant chunk moved down.

| Ranker | Chunks kept | Recall | MRR | nDCG | Precision | Questions missing evidence |
|---|---:|---:|---:|---:|---:|---:|
| Equal RRF + cross-encoder | 5 | 0.894 | 0.742 | 0.773 | 0.196 | 5 / 47 |
| RRF 0.5 / 0.5 + cross-encoder | 5 | 0.894 | 0.742 | 0.773 | 0.196 | 5 / 47 |
| RRF 0.7 / 0.3 + cross-encoder | 5 | 0.894 | 0.731 | 0.765 | 0.196 | 5 / 47 |
| Equal RRF + cross-encoder | 10 | 1.000 | 0.756 | 0.807 | 0.109 | 0 / 47 |
| RRF 0.5 / 0.5 + cross-encoder | 10 | 1.000 | 0.756 | 0.807 | 0.109 | 0 / 47 |
| RRF 0.7 / 0.3 + cross-encoder | 10 | 1.000 | 0.745 | 0.800 | 0.109 | 0 / 47 |

Recall is unchanged. MRR and nDCG are a little lower with the 0.7 / 0.3 split. Equal RRF plus the cross-encoder stays the ranking to use.
