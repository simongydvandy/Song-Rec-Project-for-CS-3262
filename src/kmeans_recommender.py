"""
K-means clustering-based recommendation — Xizhi's component.

Workflow:
1. Build joint feature vectors (user demographics + song features).
2. Scale with StandardScaler (K-means is distance-sensitive).
3. Choose K via elbow (inertia) and silhouette analysis.
4. Fit KMeans and assign every row to a cluster.
5. Recommend: given a query user's demographics, find their cluster and return
   the most common songs/genres in that cluster.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_and_clean, build_joint_features, build_user_features


class KMeansRecommender:
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self._df: pd.DataFrame | None = None
        self._joint: pd.DataFrame | None = None
        self._scaled: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    # ── Fit ─────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, joint_features: pd.DataFrame):
        """Fit scaler + KMeans on the joint feature matrix."""
        self._df = df.copy().reset_index(drop=True)
        self._joint = joint_features.reset_index(drop=True)
        self._scaled = self.scaler.fit_transform(self._joint.values)
        self._labels = self.model.fit_predict(self._scaled)
        self._df["cluster"] = self._labels
        return self

    # ── Recommend ───────────────────────────────────────────────────────────

    def recommend(
        self,
        user_demo_vector: np.ndarray,
        song_feature_vector: np.ndarray,
        exclude_songs: list[str] | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Assign the query vector to the nearest cluster centroid and return
        the top-K most frequently submitted songs in that cluster.

        Parameters
        ----------
        user_demo_vector  : 1-D array matching build_user_features column order
        song_feature_vector: 1-D array matching build_song_features column order
                             (can be zeros if only demographics are known)
        exclude_songs     : song names already known to the user
        top_k             : how many songs to return
        """
        joint_vec = np.concatenate([user_demo_vector, song_feature_vector])
        scaled_vec = self.scaler.transform(joint_vec.reshape(1, -1))

        # Distance to each centroid → assign to nearest
        distances = np.linalg.norm(
            self.model.cluster_centers_ - scaled_vec, axis=1
        )
        cluster_id = int(np.argmin(distances))

        cluster_songs = self._df[self._df["cluster"] == cluster_id]
        if exclude_songs:
            cluster_songs = cluster_songs[~cluster_songs["song"].isin(exclude_songs)]

        recommendations = (
            cluster_songs.groupby(["song", "artist", "genre"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return recommendations

    def cluster_genre_profile(self) -> pd.DataFrame:
        """Return genre distribution per cluster (normalised)."""
        grouped = (
            self._df.groupby(["cluster", "genre"])
            .size()
            .reset_index(name="count")
        )
        totals = grouped.groupby("cluster")["count"].transform("sum")
        grouped["proportion"] = grouped["count"] / totals
        return grouped.sort_values(["cluster", "proportion"], ascending=[True, False])


# ── Elbow / Silhouette analysis ──────────────────────────────────────────────

def elbow_analysis(
    scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> tuple[list[float], list[float]]:
    """Return (inertias, silhouette_scores) for a range of K values."""
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(scaled, labels))
    return inertias, sil_scores


def plot_elbow(k_range, inertias, sil_scores):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, marker="o")
    ax1.set_xlabel("Number of clusters K")
    ax1.set_ylabel("Inertia (within-cluster SSE)")
    ax1.set_title("Elbow Method")

    ax2.plot(list(k_range), sil_scores, marker="o", color="orange")
    ax2.set_xlabel("Number of clusters K")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")

    plt.tight_layout()
    return fig


# ── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean()
    joint = build_joint_features(df)
    user_feats = build_user_features(df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(joint.values)

    k_range = range(2, 11)
    inertias, sil_scores = elbow_analysis(scaled, k_range=k_range)
    print("K  | Inertia   | Silhouette")
    for k, ine, sil in zip(k_range, inertias, sil_scores):
        print(f"{k:2d} | {ine:9.2f} | {sil:.4f}")

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"\nBest K by silhouette: {best_k}")

    rec = KMeansRecommender(n_clusters=best_k)
    rec.fit(df, joint)

    # Demo: recommend for the first row's user demographics
    demo_vec = user_feats.iloc[0].values
    song_vec = np.zeros(joint.shape[1] - user_feats.shape[1])
    results = rec.recommend(demo_vec, song_vec, top_k=5)
    print("\nSample recommendations:")
    print(results)
