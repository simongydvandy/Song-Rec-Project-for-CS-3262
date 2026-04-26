# Song Recommendation Project — CS 3262 Applied Machine Learning

**Team:** Simon (Yiding) Gou (`gouy`) · Xizhi Li (`lix71`)

---

## Overview

A music recommendation system built on a 172-entry survey dataset of Vanderbilt students, each sharing demographic information and a song they enjoy. We implement and compare two approaches:

| Approach | Owner | Type |
|----------|-------|------|
| KNN (content-based + collaborative filtering) | Simon | Similarity-based |
| K-means clustering | Xizhi | Unsupervised clustering |

---

## Dataset

`Song Recommendations.csv` — 172 responses, 120 unique users, 170 unique songs.

| Column | Description |
|--------|-------------|
| User ID | Anonymous student identifier |
| Gender | Male / Female |
| Hometown | City / Suburban / Rural |
| User Language | Primary daily language(s) |
| Song / Artist | Submitted song |
| Genre | 14 genre categories (after cleaning) |
| Song Language | Language(s) of the song |
| Release Year | Era bin: Pre-1960 → 2020+ |

---

## Project Structure

```
├── Song Recommendations.csv   # Raw survey data
├── src/
│   ├── preprocess.py          # Cleaning + feature engineering (shared)
│   ├── content_knn.py         # Content-based KNN (Simon)
│   ├── collab_knn.py          # User-based collaborative KNN (Simon)
│   ├── kmeans_recommender.py  # K-means clustering recommender (Xizhi)
│   └── evaluate.py            # Shared evaluation metrics
└── notebooks/
    ├── 01_EDA.ipynb            # Exploratory data analysis
    ├── 02_preprocessing.ipynb  # Feature matrix walkthrough
    ├── 03_kmeans_xizhi.ipynb   # K-means: elbow analysis, cluster profiles
    └── 04_knn_simon.ipynb      # KNN models + full 3-way comparison
```

---

## Approaches

### Simon — KNN Recommendation

**Content-based filtering** (`content_knn.py`)
- Each song is one-hot encoded (genre, language, release era).
- A user's profile = mean vector of their submitted songs.
- Recommendations ranked by cosine similarity to the profile.

**User-based collaborative filtering** (`collab_knn.py`)
- Each user is encoded by demographics (gender, hometown, language).
- K nearest demographic neighbours are found via cosine similarity.
- Songs submitted by neighbours (not yet seen by the user) are recommended.

### Xizhi — K-means Clustering

**Cluster-based recommendation** (`kmeans_recommender.py`)
- Each row is represented as a joint vector: user demographics + song features.
- Features scaled with `StandardScaler`; optimal K chosen by silhouette score.
- A new user is assigned to the nearest cluster centroid, and the most popular songs in that cluster are recommended.

---

## Results

Evaluation metric: **Genre Match@5** (leave-one-out, users with ≥2 submissions).  
Exact song hit rate is 0 for all models by construction — with 170 unique songs across 172 rows, the held-out song almost never appears in another user's pool.

| Model | Genre Match@5 |
|-------|:-------------:|
| Content-based KNN | **0.9545** |
| Collaborative KNN | 0.7727 |
| K-means | 0.1529 |

Content-based KNN performs best because genre/language/era features directly encode preference similarity. K-means genre accuracy is lower because it optimises for cluster cohesion across the full joint feature space, not genre prediction specifically.

---

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Run notebooks in order (01 → 04) from the project root, or import from `src/` directly:

```python
import sys; sys.path.insert(0, 'src')
from preprocess import load_and_clean, build_song_features, build_user_features
from content_knn import ContentKNN
from collab_knn import CollabKNN
from kmeans_recommender import KMeansRecommender
```
