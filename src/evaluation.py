from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class ScoringModel(Protocol):
    def predict_many(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_timestamp: int
    validation_end_timestamp: int


def temporal_split(
    ratings: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
) -> TemporalSplit:
    """Globally split interactions by timestamp; ties never cross a boundary."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction leaves no test data")

    ordered = ratings.sort_values(["timestamp", "user_id", "movie_id"], kind="stable")
    timestamps = ordered["timestamp"].to_numpy()
    train_target = max(1, int(round(len(ordered) * train_fraction)))
    validation_target = max(
        train_target + 1, int(round(len(ordered) * (train_fraction + validation_fraction)))
    )
    validation_target = min(validation_target, len(ordered) - 1)
    train_end = int(timestamps[train_target - 1])
    validation_end = int(timestamps[validation_target - 1])

    train = ordered[ordered["timestamp"] <= train_end].copy()
    validation = ordered[
        (ordered["timestamp"] > train_end) & (ordered["timestamp"] <= validation_end)
    ].copy()
    test = ordered[ordered["timestamp"] > validation_end].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("one temporal partition is empty")
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    return TemporalSplit(train, validation, test, train_end, validation_end)


def rating_metrics(model: ScoringModel, test: pd.DataFrame) -> dict[str, float]:
    truth = test["rating"].to_numpy(dtype=float)
    predictions = model.predict_many(
        test["user_id"].to_numpy(), test["movie_id"].to_numpy()
    )
    error = truth - predictions
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def build_sampled_candidates(
    history: pd.DataFrame,
    test: pd.DataFrame,
    catalog: np.ndarray,
    relevance_threshold: float = 4.0,
    negatives_per_user: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Build per-user ranking sets from test positives and never-interacted negatives."""
    rng = np.random.default_rng(seed)
    catalog = np.asarray(sorted(set(map(int, catalog))), dtype=np.int32)
    all_interactions = pd.concat(
        [history[["user_id", "movie_id"]], test[["user_id", "movie_id"]]],
        ignore_index=True,
    ).groupby("user_id")["movie_id"].agg(lambda values: set(map(int, values)))
    positives = test[test["rating"] >= relevance_threshold].groupby("user_id")["movie_id"]

    rows: list[tuple[int, int, int]] = []
    for user_id, values in positives:
        positive_items = sorted(set(map(int, values)))
        interacted = all_interactions.loc[user_id]
        negative_pool = np.array([item for item in catalog if item not in interacted], dtype=np.int32)
        n_negative = min(negatives_per_user, len(negative_pool))
        sampled = rng.choice(negative_pool, size=n_negative, replace=False)
        rows.extend((int(user_id), item, 1) for item in positive_items)
        rows.extend((int(user_id), int(item), 0) for item in sampled)
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "relevant"])


def ranking_metrics(
    model: ScoringModel,
    candidates: pd.DataFrame,
    training_ratings: pd.DataFrame,
    catalog_size: int,
    k: int = 10,
) -> dict[str, float | int]:
    """Compute macro ranking and beyond-accuracy metrics on fixed candidates."""
    scored = candidates.copy()
    scored["score"] = model.predict_many(
        scored["user_id"].to_numpy(), scored["movie_id"].to_numpy()
    )
    scored = scored.sort_values(
        ["user_id", "score", "movie_id"], ascending=[True, False, True], kind="stable"
    )

    item_counts = training_ratings["movie_id"].value_counts()
    total_interactions = len(training_ratings)
    precision, recall, ndcg, hit_rate = [], [], [], []
    recommended_items: set[int] = set()
    novelty_values, popularity_values = [], []

    for _, group in scored.groupby("user_id", sort=False):
        top = group.head(k)
        labels = top["relevant"].to_numpy(dtype=float)
        n_relevant = int(group["relevant"].sum())
        hits = float(labels.sum())
        precision.append(hits / k)
        recall.append(hits / n_relevant)
        discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
        dcg = float(np.sum(labels * discounts))
        ideal_len = min(n_relevant, k)
        idcg = float(np.sum(discounts[:ideal_len]))
        ndcg.append(dcg / idcg if idcg else 0.0)
        hit_rate.append(float(hits > 0))

        items = top["movie_id"].astype(int).tolist()
        recommended_items.update(items)
        counts = np.array([item_counts.get(item, 0) for item in items], dtype=float)
        popularity_values.extend(counts.tolist())
        novelty_values.extend((-np.log2((counts + 1) / (total_interactions + catalog_size))).tolist())

    return {
        f"precision@{k}": float(np.mean(precision)),
        f"recall@{k}": float(np.mean(recall)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"hit_rate@{k}": float(np.mean(hit_rate)),
        f"coverage@{k}": len(recommended_items) / catalog_size,
        f"novelty@{k}": float(np.mean(novelty_values)),
        f"mean_train_popularity@{k}": float(np.mean(popularity_values)),
        "evaluated_users": len(precision),
        "candidates": len(candidates),
    }


def activity_segments(history: pd.DataFrame, users: np.ndarray) -> dict[int, str]:
    """Label evaluation users as low, medium, or high activity by train-history tertiles."""
    counts = history.groupby("user_id").size().reindex(users, fill_value=0)
    q1, q2 = counts.quantile([1 / 3, 2 / 3]).tolist()
    return {
        int(user): "low" if count <= q1 else "medium" if count <= q2 else "high"
        for user, count in counts.items()
    }
