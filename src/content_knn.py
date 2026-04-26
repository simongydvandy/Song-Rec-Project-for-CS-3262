"""
Content-based KNN recommendation — Simon's component (part 1).

Each song is represented as a one-hot/ordinal feature vector.
A user's preference profile = mean vector of all songs they submitted.
Recommendations are ranked by cosine similarity to the user profile.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_and_clean, build_song_features


class ContentKNN:
    def __init__(self):
        self._df: pd.DataFrame | None = None
        self._song_feats: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, song_features: pd.DataFrame):
        self._df = df.copy().reset_index(drop=True)
        self._song_feats = song_features.reset_index(drop=True)
        return self

    def _user_profile(self, user_id: str, exclude_idx: list[int] | None = None) -> np.ndarray:
        """Mean song-feature vector for a user, optionally excluding specific row indices."""
        mask = self._df["user_id"] == user_id
        if exclude_idx:
            mask = mask & ~self._df.index.isin(exclude_idx)
        idxs = self._df[mask].index.tolist()
        if not idxs:
            return np.zeros(self._song_feats.shape[1])
        return self._song_feats.iloc[idxs].values.mean(axis=0)

    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_own: bool = True,
    ) -> pd.DataFrame:
        """
        Return top-K songs most similar to the user's profile.
        exclude_own=True filters out songs the user already submitted.
        """
        profile = self._user_profile(user_id)
        sims = cosine_similarity(
            profile.reshape(1, -1), self._song_feats.values
        ).flatten()

        results = self._df.copy()
        results["similarity"] = sims

        if exclude_own:
            results = results[results["user_id"] != user_id]

        top = (
            results.sort_values("similarity", ascending=False)
            .drop_duplicates(subset=["song"])
            .head(top_k)[["song", "artist", "genre", "release_year", "similarity"]]
            .reset_index(drop=True)
        )
        return top

    def recommend_from_profile(
        self,
        profile_vector: np.ndarray,
        exclude_songs: list[str] | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Recommend given an explicit profile vector (useful for cold-start demos)."""
        sims = cosine_similarity(
            profile_vector.reshape(1, -1), self._song_feats.values
        ).flatten()

        results = self._df.copy()
        results["similarity"] = sims

        if exclude_songs:
            results = results[~results["song"].isin(exclude_songs)]

        top = (
            results.sort_values("similarity", ascending=False)
            .drop_duplicates(subset=["song"])
            .head(top_k)[["song", "artist", "genre", "release_year", "similarity"]]
            .reset_index(drop=True)
        )
        return top


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean()
    sf = build_song_features(df)

    model = ContentKNN().fit(df, sf)

    # Pick a user who submitted at least 1 song
    sample_user = df["user_id"].value_counts().index[0]
    print(f"User: {sample_user}")
    print("Their songs:")
    print(df[df["user_id"] == sample_user][["song", "genre"]].to_string(index=False))
    print("\nTop-5 content-based recommendations:")
    print(model.recommend(sample_user, top_k=5).to_string(index=False))
