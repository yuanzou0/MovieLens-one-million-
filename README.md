# MovieLens 1M Recommender: Leakage-Safe Offline Evaluation

An offline recommendation project focused on **credible experimentation**, not model count. The
current pipeline compares a Bayesian popularity baseline with sparse truncated SVD using a
chronological holdout and realistic sampled-candidate ranking.

> **Status:** `run_experiment.py` is the reproducible primary experiment. `Final_Code.ipynb` and
> `Report.pdf` are retained as legacy course artifacts and contain earlier random-split analyses.
> Their metrics are not comparable with the current protocol.

## Why this revision matters

The original course submission used a random 80/20 split and tuned SVD with cross-validation over
data later reused for testing. It also ranked only items already present in the test set, which made
the reported Top-K scores optimistic. The revised experiment fixes both issues:

1. Sort all interactions by timestamp and split them 70% / 10% / 20% into train, validation, and
   test periods. Timestamp ties never cross a boundary.
2. Use only train and validation data for model choice. Evaluate the test period once.
3. For every eligible test user, rank all relevant test movies (rating >= 4) together with 100
   seeded movies the user never interacted with.
4. Report rating accuracy, ranking quality, catalog reach, novelty, popularity bias, and results by
   pre-test user activity.

## Reproduced results

MovieLens 1M, seed 42, global chronological split, 100 sampled negatives per user:

| Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Coverage@10 | Novelty@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bayesian popularity | 0.9718 | 0.7744 | 0.6232 | 0.2146 | 0.6636 | 0.9535 | 0.1422 | 10.0743 |
| Truncated SVD (50 factors) | 1.0792 | 0.8718 | 0.5276 | 0.1883 | 0.5798 | 0.9376 | **0.4878** | **10.8785** |

These values are computed from 200,041 test interactions and 288,460 ranking candidates across
1,762 eligible users. With this protocol, popularity is the stronger accuracy baseline; SVD trades
accuracy for substantially higher coverage and novelty. That trade-off is the defensible result.

Sampled metrics depend on candidate construction and are not directly comparable with full-catalog
ranking or with results using a different negative count. See `results/metrics.json` for the full
protocol and low/medium/high user-activity slices.

## Reproduce

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py --models popularity svd --negatives 100
pytest -q
```

Outputs:

- `results/metrics.json`: complete protocol, metrics, parameters, and activity slices
- `results/model_comparison.csv`: compact model comparison table

Optional Optuna search uses **train/validation only** and retrains the selected SVD configuration on
train + validation before the single test evaluation:

```bash
python run_experiment.py --models svd --n-trials 10 --negatives 100
```

## Repository layout

| Path | Purpose |
|---|---|
| `run_experiment.py` | Reproducible command-line experiment |
| `src/data.py` | Typed MovieLens loader |
| `src/evaluation.py` | Temporal split, candidate construction, and metrics |
| `src/models.py` | Bayesian popularity and sparse truncated SVD models |
| `tests/` | Protocol and metric unit tests |
| `results/` | Machine-readable reproduced outputs |
| `docs/PROJECT_AUDIT.md` | Claim-by-claim audit and resume guidance |
| `Final_Code.ipynb` | Legacy exploratory course notebook |
| `Report.pdf` | Legacy course report |

## What is and is not claimed

- **NMF is not claimed:** it was imported in the old notebook but never trained or evaluated.
- **SVD++ is not claimed:** final output was not retained, so no comparison can be verified.
- **No cold-start improvement is claimed:** the old hybrid used adaptive weights but did not run a
  controlled new-item experiment. The revised output includes user-activity slices, a first step
  toward that analysis.
- **No general Numba speedup is claimed:** the old popularity run was 0.14x when JIT compilation was
  included, while one average-rating run was 1.89x. A defensible benchmark needs warm-up and repeated
  timings with uncertainty.
- **Legacy SVD values are not current:** the old notebook retained RMSE 0.8612 and MAE 0.6756 under a
  different random-split protocol; the report listed 0.8587 and 0.6774. Neither is used above.

## Data and license

MovieLens data is provided by [GroupLens](https://grouplens.org/datasets/movielens/1m/) and remains
subject to its own usage terms. Repository code is released under the [MIT License](LICENSE).

Dataset citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History
and Context*. ACM Transactions on Interactive Intelligent Systems 5(4). DOI:
[10.1145/2827872](https://doi.org/10.1145/2827872).

