import numpy as np
import pandas as pd

from src.evaluation import build_sampled_candidates, ranking_metrics, temporal_split


class FixedModel:
    def predict_many(self, user_ids, item_ids):
        del user_ids
        return np.asarray(item_ids, dtype=float)


def test_temporal_split_is_strictly_ordered():
    ratings = pd.DataFrame(
        {
            "user_id": [1] * 10,
            "movie_id": range(10),
            "rating": [4] * 10,
            "timestamp": range(10),
        }
    )
    split = temporal_split(ratings)
    assert split.train.timestamp.max() < split.validation.timestamp.min()
    assert split.validation.timestamp.max() < split.test.timestamp.min()


def test_sampled_negatives_have_no_known_interactions():
    history = pd.DataFrame({"user_id": [1, 1], "movie_id": [1, 2], "rating": [5, 3]})
    test = pd.DataFrame({"user_id": [1], "movie_id": [3], "rating": [5]})
    candidates = build_sampled_candidates(history, test, np.arange(1, 8), negatives_per_user=3)
    negatives = set(candidates.loc[candidates.relevant == 0, "movie_id"])
    assert negatives.isdisjoint({1, 2, 3})
    assert set(candidates.loc[candidates.relevant == 1, "movie_id"]) == {3}


def test_ranking_metrics_known_top_k():
    candidates = pd.DataFrame(
        {"user_id": [1, 1, 1], "movie_id": [3, 2, 1], "relevant": [1, 0, 0]}
    )
    train = pd.DataFrame({"movie_id": [1, 2, 3], "rating": [3, 4, 5]})
    result = ranking_metrics(FixedModel(), candidates, train, catalog_size=3, k=2)
    assert result["precision@2"] == 0.5
    assert result["recall@2"] == 1.0
    assert result["hit_rate@2"] == 1.0

