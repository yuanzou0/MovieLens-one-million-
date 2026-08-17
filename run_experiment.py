from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_movielens
from src.evaluation import (
    activity_segments,
    build_sampled_candidates,
    ranking_metrics,
    rating_metrics,
    temporal_split,
)
from src.models import PopularityBaseline, TruncatedSVD


SEED = 42
DEFAULT_SVD_PARAMS = {
    "n_factors": 50,
    "random_state": SEED,
}


def tune_svd(train: pd.DataFrame, validation: pd.DataFrame, n_trials: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {"n_factors": trial.suggest_categorical("n_factors", [25, 50, 100]), "random_state": SEED}
        model = TruncatedSVD(**params).fit(train)
        return rating_metrics(model, validation)["rmse"]

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials)
    return {**study.best_params, "random_state": SEED}


def evaluate_model(name, model, test, candidates, training_ratings, catalog_size, segments):
    result = {
        "model": name,
        **rating_metrics(model, test),
        **ranking_metrics(model, candidates, training_ratings, catalog_size, k=10),
    }
    result["by_user_activity"] = {}
    for segment in ["low", "medium", "high"]:
        users = [user for user, label in segments.items() if label == segment]
        subset = candidates[candidates["user_id"].isin(users)]
        result["by_user_activity"][segment] = ranking_metrics(
            model, subset, training_ratings, catalog_size, k=10
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe MovieLens 1M evaluation")
    parser.add_argument("--data-dir", default="ml-1m")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--models", nargs="+", choices=["popularity", "svd"], default=["popularity", "svd"])
    parser.add_argument("--n-trials", type=int, default=0, help="Optuna trials on train/validation only")
    parser.add_argument("--negatives", type=int, default=100)
    args = parser.parse_args()

    np.random.seed(SEED)
    ratings, movies = load_movielens(args.data_dir)
    split = temporal_split(ratings)
    train_validation = pd.concat([split.train, split.validation], ignore_index=True)
    catalog = movies["movie_id"].to_numpy()
    candidates = build_sampled_candidates(
        train_validation, split.test, catalog, negatives_per_user=args.negatives, seed=SEED
    )
    segments = activity_segments(train_validation, candidates["user_id"].unique())

    protocol = {
        "seed": SEED,
        "split": "global chronological 70/10/20; timestamp ties kept together",
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "train_end_timestamp": split.train_end_timestamp,
        "validation_end_timestamp": split.validation_end_timestamp,
        "relevance_threshold": 4.0,
        "negatives_per_user": args.negatives,
        "candidate_policy": "all test positives plus sampled never-interacted items",
        "test_policy": "evaluated once after model and hyperparameters are fixed",
    }
    results = []
    if "popularity" in args.models:
        model = PopularityBaseline().fit(train_validation)
        results.append(evaluate_model("Popularity", model, split.test, candidates, train_validation, len(catalog), segments))
    if "svd" in args.models:
        params = tune_svd(split.train, split.validation, args.n_trials) if args.n_trials else DEFAULT_SVD_PARAMS
        model = TruncatedSVD(**params).fit(train_validation)
        svd_result = evaluate_model("SVD", model, split.test, candidates, train_validation, len(catalog), segments)
        svd_result["parameters"] = params
        results.append(svd_result)

    segment_counts = pd.Series(segments).value_counts().sort_index().to_dict()
    payload = {"protocol": protocol, "evaluation_user_segments": segment_counts, "results": results}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(results).drop(columns=["parameters", "by_user_activity"], errors="ignore").to_csv(
        output_dir / "model_comparison.csv", index=False
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
