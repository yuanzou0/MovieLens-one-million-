from __future__ import annotations

from pathlib import Path

import pandas as pd


RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 1M ratings and movie metadata from ``data_dir``."""
    data_dir = Path(data_dir)
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        names=RATING_COLUMNS,
        engine="python",
        encoding="latin-1",
    )
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        names=MOVIE_COLUMNS,
        engine="python",
        encoding="latin-1",
    )
    ratings = ratings.astype(
        {"user_id": "int32", "movie_id": "int32", "rating": "float32", "timestamp": "int64"}
    )
    movies = movies.astype({"movie_id": "int32"})
    return ratings, movies

