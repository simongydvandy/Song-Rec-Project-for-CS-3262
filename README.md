# Song Recommendation Project — CS 3262 Applied Machine Learning

**Team:** Simon (Yiding) Gou (`gouy`) · Xizhi Li (`lix71`)

---
## 1. Problem Statement
The recommendation task involves developing a music recommendation system using a 172-entry survey dataset containing Vanderbilt students' demographic information and song preferences. The approach compares distinct machine learning methodologies to generate recommendations. A K-Nearest Neighbors (KNN) model is utilized for similarity-based recommendations, combining content-based and collaborative filtering. An unsupervised K-means clustering model was initially implemented but yielded poor recommendations. A supervised Random Forest model was subsequently developed to replace the clustering approach and improve predictive accuracy.
## Overview
| Approach | Owner | Type |
|----------|-------|------|
| EDA | Simon and Xizhi |
| KNN (content-based + collaborative filtering) | Simon | Similarity-based |
| K-means clustering | Xizhi | Unsupervised clustering | (This was done first but has a poor recommendation, so the Random Forest is done later)
|Randrom Forest | Xizhi | Supervised learning |
Note: For EDA and data preprocessings, Simon and Xizhi finished them collaboratively.

## 2. Data Description
This analysis will use the Song Recommendations Dataset, sourced from student-provided survey responses. The song dataset provides information on individual song preferences and the demographic backgrounds of the respondents. It includes 172 entries and contains 10 variables that capture various aspects of each recommendation.

The dataset features

Timestamp (string): The date and time the recommendation was submitted. Your Unique ID (string): A unique, consistent identifier for each respondent. Your Gender (string): The gender identity of the user. Your hometown (string): The type of environment where the user was raised (e.g., City, Suburban, Rural). What language do you primarily use in daily life? (string): The primary language or languages spoken by the respondent. Song name (string): The title of the recommended musical track. Artist (string): The musician or group who performed the song. Genre (string): The musical category or style of the song. Language of the song (string): The language(s) featured in the song's lyrics. Song release year (string): The time period or specific era of the song’s release (e.g., 2020+, 2010–2019).
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
    ├── 03_kmeans_xizhi.ipynb   # K-means: elbow analysis, cluster profiles; 
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
