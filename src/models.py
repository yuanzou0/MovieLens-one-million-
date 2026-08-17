from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityBaseline:
    """Train-only Bayesian item means, with popularity as a deterministic tie-breaker."""

    def __init__(self, prior_weight: float = 25.0):
        self.prior_weight = prior_weight

    def fit(self, ratings: pd.DataFrame) -> "PopularityBaseline":
        self.global_mean = float(ratings["rating"].mean())
        stats = ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
        stats["score"] = (
            stats["mean"] * stats["count"] + self.global_mean * self.prior_weight
        ) / (stats["count"] + self.prior_weight)
        self.scores = stats["score"].to_dict()
        return self

    def predict_many(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        del user_ids
        return np.fromiter(
            (self.scores.get(int(item), self.global_mean) for item in item_ids),
            dtype=float,
            count=len(item_ids),
        )


class TruncatedSVD:
    """Mean-centered truncated SVD trained directly on a sparse ratings matrix."""

    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(self, ratings: pd.DataFrame) -> "TruncatedSVD":
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds

        self.global_mean = float(ratings["rating"].mean())
        users = np.sort(ratings["user_id"].unique())
        items = np.sort(ratings["movie_id"].unique())
        self.user_to_index = {int(value): index for index, value in enumerate(users)}
        self.item_to_index = {int(value): index for index, value in enumerate(items)}
        user_index = ratings["user_id"].map(self.user_to_index).to_numpy()
        item_index = ratings["movie_id"].map(self.item_to_index).to_numpy()
        user_means = ratings.groupby("user_id")["rating"].mean().reindex(users).to_numpy()
        self.user_means = user_means
        centered = ratings["rating"].to_numpy(dtype=float) - user_means[user_index]
        matrix = csr_matrix((centered, (user_index, item_index)), shape=(len(users), len(items)))
        k = min(self.n_factors, min(matrix.shape) - 1)
        u, singular_values, vt = svds(matrix, k=k, rng=np.random.default_rng(self.random_state))
        order = np.argsort(singular_values)[::-1]
        self.user_factors = u[:, order] * singular_values[order]
        self.item_factors = vt[order, :].T
        return self

    def predict_many(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        predictions = np.full(len(item_ids), self.global_mean, dtype=float)
        for index, (user, item) in enumerate(zip(user_ids, item_ids)):
            user_index = self.user_to_index.get(int(user))
            item_index = self.item_to_index.get(int(item))
            if user_index is None:
                continue
            predictions[index] = self.user_means[user_index]
            if item_index is not None:
                predictions[index] += self.user_factors[user_index] @ self.item_factors[item_index]
        return np.clip(predictions, 1.0, 5.0)
