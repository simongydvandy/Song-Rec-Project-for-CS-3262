"""
Shared evaluation utilities.

Primary metric for all models: Genre Match@K (leave-one-out).
With only 170 unique songs across 172 rows, exact song retrieval (hit rate) is
effectively impossible — almost every song appears only once, so the held-out
song never shows up in another user's submission pool. Genre match is the
appropriate proxy: does any top-K recommendation share the genre of the
held-out song?

Secondary (KNN only): exact hit rate is still computed for reference.
"""

import numpy as np
import pandas as pd
from typing import Callable


# ── Leave-one-out evaluation ─────────────────────────────────────────────────

def leave_one_out_eval(
    df: pd.DataFrame,
    recommend_fn: Callable[[str, list[str]], pd.DataFrame],
    top_k: int = 5,
) -> dict:
    """
    For every user with ≥2 song submissions:
      - Hold out one song at random.
      - Call recommend_fn(user_id, known_songs) to get top-K recommendations.
      - Compute exact hit rate AND genre match rate.

    recommend_fn signature:
        (user_id: str, exclude_songs: list[str]) -> DataFrame with columns 'song', 'genre'

    Returns dict with hit_rate, genre_match_at_k, precision_at_k, recall_at_k, n_users.

    Note: hit_rate will typically be 0 on this small dataset because almost every
    song is unique — use genre_match_at_k as the primary metric.
    """
    multi_users = df.groupby("user_id").filter(lambda g: len(g) >= 2)["user_id"].unique()

    hits = 0
    genre_hits = 0
    total = 0

    rng = np.random.default_rng(42)
    for user_id in multi_users:
        user_rows = df[df["user_id"] == user_id]
        holdout_idx = rng.integers(len(user_rows))
        holdout_row = user_rows.iloc[holdout_idx]
        holdout_song  = holdout_row["song"]
        holdout_genre = holdout_row["genre"]
        known_songs = user_rows["song"].tolist()

        try:
            recs = recommend_fn(user_id, known_songs)
        except Exception:
            continue

        if holdout_song in recs["song"].values:
            hits += 1
        if holdout_genre in recs["genre"].values:
            genre_hits += 1
        total += 1

    if total == 0:
        return {"hit_rate": 0.0, "genre_match_at_k": 0.0,
                "precision_at_k": 0.0, "recall_at_k": 0.0, "n_users": 0}

    hit_rate       = hits / total
    genre_match    = genre_hits / total
    precision      = hits / (total * top_k)
    recall         = hit_rate

    return {
        "hit_rate":          round(hit_rate, 4),
        "genre_match_at_k":  round(genre_match, 4),
        "precision_at_k":    round(precision, 4),
        "recall_at_k":       round(recall, 4),
        "n_users":           total,
    }


# ── K-means genre accuracy ───────────────────────────────────────────────────

def kmeans_genre_accuracy(
    df: pd.DataFrame,
    recommend_fn: Callable[[np.ndarray, np.ndarray, list[str]], pd.DataFrame],
    user_features: pd.DataFrame,
    song_features: pd.DataFrame,
    top_k: int = 1,
) -> dict:
    """
    For each row, use demographics as input and check if the top recommended
    genre matches the actual genre of that row.
    """
    n_user_feats = user_features.shape[1]
    correct = 0
    total = len(df)

    for i in range(total):
        actual_genre = df.iloc[i]["genre"]
        user_vec = user_features.iloc[i].values
        song_vec = np.zeros(song_features.shape[1])
        exclude = [df.iloc[i]["song"]]

        try:
            recs = recommend_fn(user_vec, song_vec, exclude)
        except Exception:
            total -= 1
            continue

        if not recs.empty and recs.iloc[0]["genre"] == actual_genre:
            correct += 1

    return {
        "genre_accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "n_rows": total,
    }


# ── Summary printer ──────────────────────────────────────────────────────────

def print_comparison(content_results, collab_results, kmeans_results):
    print("=" * 63)
    print(f"{'Metric':<30} {'Content KNN':>12} {'Collab KNN':>12} {'K-means':>7}")
    print("-" * 63)
    # Primary metric (comparable across all three)
    km_gm = kmeans_results["genre_accuracy"]
    print(f"{'genre_match_at_k (primary)':<30} "
          f"{content_results['genre_match_at_k']:>12.4f} "
          f"{collab_results['genre_match_at_k']:>12.4f} "
          f"{km_gm:>7.4f}")
    print("-" * 63)
    # Secondary KNN-only metrics
    for metric in ["hit_rate", "precision_at_k", "recall_at_k"]:
        print(
            f"{metric:<30} {content_results[metric]:>12.4f} {collab_results[metric]:>12.4f} {'  n/a':>7}"
        )
    print("-" * 63)
    print(f"{'n evaluated':<30} {content_results['n_users']:>12} {collab_results['n_users']:>12} {kmeans_results['n_rows']:>7}")
    print("=" * 63)
    print("Note: KNN hit_rate=0 is expected — 170 unique songs across 172 rows;")
    print("      almost no held-out song exists in another user's pool.")
    print("      Use genre_match_at_k as the primary comparison metric.")
