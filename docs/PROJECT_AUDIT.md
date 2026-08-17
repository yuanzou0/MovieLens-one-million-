# Project audit and claim policy

This audit separates reproducible evidence from results that appeared in the original course
notebook or report.

## Corrections applied

- Removed NMF from project claims: it was imported but never trained or evaluated.
- Renamed the notebook section `Matrix Decomposition-SVM` to `Matrix Decomposition-SVD`.
- Removed SVD++ from model-comparison claims because its final training and evaluation cells have
  no retained output.
- Replaced random 80/20 splitting in the primary experiment with a global chronological 70/10/20
  train/validation/test split. Timestamp ties stay in one partition.
- Limited model selection to train and validation data. The test period is evaluated only after
  the model configuration is fixed.
- Replaced ranking over test-observed items with candidate generation from test positives plus 100
  seeded, never-interacted negatives per user.
- Added NDCG@10, Hit Rate@10, catalog coverage, novelty, and mean training-item popularity.
- Added low/medium/high user-activity slices based only on pre-test history.
- Reclassified `Report.pdf` and the original notebook outputs as legacy course artifacts. Their
  random-split numbers are not directly comparable to the new temporal protocol.

## Legacy results that must not be presented as current

| Claim | Evidence in original artifact | Decision |
|---|---:|---|
| NMF comparison | Import only | Remove |
| SVD RMSE / MAE | Notebook: 0.8612 / 0.6756; report: 0.8587 / 0.6774 | Label legacy and inconsistent |
| SVD Precision@10 / Recall@10 | 0.7654 / 0.6438 over test-observed items only | Withdraw |
| SVD++ performance | No final output | Do not claim |
| Hybrid cold-start improvement | No controlled cold-start split | Do not claim |
| Numba speedup | Popularity: 0.14x including JIT; average-rating: 1.89x | Do not generalize |

## Resume-safe interpretation

The strongest result is not that SVD wins every metric. Under the corrected protocol, the
popularity baseline is stronger on sampled ranking accuracy, while truncated SVD recommends a
broader, less-popular catalog. This is a useful product trade-off and a more credible data-science
story than selectively reporting the highest old metric.

Suggested bullet:

> Built a leakage-controlled MovieLens 1M evaluation pipeline using chronological train/validation/test
> splits and sampled-candidate ranking; compared a Bayesian popularity baseline with sparse truncated
> SVD across RMSE/MAE, Precision/Recall/NDCG/Hit Rate@10, coverage, novelty, and user-activity segments.

Add exact numbers only from `results/metrics.json`, generated in the environment used for the resume.

