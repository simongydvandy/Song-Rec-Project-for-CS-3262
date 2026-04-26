"""
User-based collaborative filtering KNN — Simon's component (part 2).

Each user is encoded by their demographic vector.
Given a target user, the K nearest demographic neighbours are found,
and their songs are ranked by frequency and returned as recommendations.
This acts as a cold-start fallback when song history is sparse.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_and_clean, build_user_features


class CollabKNN:
    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors
        self._df: pd.DataFrame | None = None
        self._user_ids: list[str] | None = None
        self._user_matrix: np.ndarray | None = None  # shape: (n_users, n_features)

    def fit(self, df: pd.DataFrame, user_features: pd.DataFrame):
        """
        Aggregate per-row user features into one vector per unique user (mean),
        then store for nearest-neighbour lookup.
        """
        self._df = df.copy().reset_index(drop=True)
        user_feat_df = user_features.copy()
        user_feat_df["user_id"] = df["user_id"].values

        # One representative vector per user (mean of their rows)
        agg = user_feat_df.groupby("user_id").mean()
        self._user_ids = list(agg.index)
        self._user_matrix = agg.values
        return self

    def _get_user_vector(self, user_id: str) -> np.ndarray | None:
        if user_id in self._user_ids:
            idx = self._user_ids.index(user_id)
            return self._user_matrix[idx]
        return None

    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_own: bool = True,
    ) -> pd.DataFrame:
        """
        Find K nearest neighbour users by demographic similarity, then
        recommend their songs ranked by submission frequency.
        """
        vec = self._get_user_vector(user_id)
        if vec is None:
            raise ValueError(f"Unknown user_id: {user_id}")

        sims = cosine_similarity(
            vec.reshape(1, -1), self._user_matrix
        ).flatten()

        # Exclude self
        self_idx = self._user_ids.index(user_id)
        sims[self_idx] = -1.0

        neighbor_idxs = np.argsort(sims)[::-1][: self.k_neighbors]
        neighbor_ids = [self._user_ids[i] for i in neighbor_idxs]

        neighbor_songs = self._df[self._df["user_id"].isin(neighbor_ids)]
        if exclude_own:
            own_songs = set(self._df[self._df["user_id"] == user_id]["song"])
            neighbor_songs = neighbor_songs[~neighbor_songs["song"].isin(own_songs)]

        top = (
            neighbor_songs.groupby(["song", "artist", "genre"])
            .size()
            .reset_index(name="neighbor_count")
            .sort_values("neighbor_count", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return top

    def recommend_from_vector(
        self,
        demo_vector: np.ndarray,
        exclude_songs: list[str] | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Recommend for a new (unseen) user given their demographic vector."""
        sims = cosine_similarity(
            demo_vector.reshape(1, -1), self._user_matrix
        ).flatten()

        neighbor_idxs = np.argsort(sims)[::-1][: self.k_neighbors]
        neighbor_ids = [self._user_ids[i] for i in neighbor_idxs]

        neighbor_songs = self._df[self._df["user_id"].isin(neighbor_ids)]
        if exclude_songs:
            neighbor_songs = neighbor_songs[~neighbor_songs["song"].isin(exclude_songs)]

        top = (
            neighbor_songs.groupby(["song", "artist", "genre"])
            .size()
            .reset_index(name="neighbor_count")
            .sort_values("neighbor_count", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return top


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean()
    uf = build_user_features(df)

    model = CollabKNN(k_neighbors=5).fit(df, uf)

    sample_user = df["user_id"].value_counts().index[0]
    print(f"User: {sample_user}")
    print("Their songs:")
    print(df[df["user_id"] == sample_user][["song", "genre"]].to_string(index=False))
    print("\nTop-5 collaborative recommendations:")
    print(model.recommend(sample_user, top_k=5).to_string(index=False))
