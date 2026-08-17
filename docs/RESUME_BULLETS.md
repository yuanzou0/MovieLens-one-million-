# Resume bullets

## Recommended version

- Built a leakage-controlled offline recommender evaluation pipeline on 1,000,209 MovieLens
  interactions, using chronological train/validation/test splits and reproducible negative-sampled
  ranking across RMSE, MAE, Precision/Recall/NDCG/Hit Rate@10, coverage, and novelty.
- Benchmarked Bayesian popularity against sparse truncated SVD; identified an accuracy-diversity
  trade-off in which SVD expanded catalog coverage from 14.2% to 48.8% (3.4x) and increased novelty
  by 8.0%, then diagnosed performance differences across 1,762 users segmented by activity.
- Modularized data loading, modeling, and evaluation into a one-command experiment with pinned
  dependencies, deterministic seeds, machine-readable results, and unit-tested split/candidate logic.

## Compact one-bullet version

- Developed a reproducible MovieLens 1M recommender benchmark with chronological holdouts and
  leakage-safe Top-K evaluation; compared popularity and sparse SVD across accuracy, coverage,
  novelty, and user-activity segments, finding a 3.4x SVD coverage gain at an accuracy trade-off.

## Claims to avoid

Do not list NMF or SVD++ as compared models, do not reuse the original Precision@10/Recall@10, and
do not state that the hybrid model improves cold start. Those claims are not supported by the
retained experiments. Exact figures above come from `results/metrics.json` and should be updated if
the protocol or seed changes.

