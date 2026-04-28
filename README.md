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
    ├── 03_kmeans_random forest_xizhi.ipynb   # K-means: elbow analysis, cluster profiles; Random forest: feature importance
    └── 04_knn_simon.ipynb      # KNN models + full 3-way comparison
```


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
## 3. Data Preprocessing
Data Preprocessing

Cleaning Methodology
The dataset was filtered to a final count of 170 valid records. Two samples were dropped because they contained missing or incomplete entries in essential fields such as the song name or primary genre, which are required for similarity calculations.

Standardizing text strings consolidated diverse user inputs into 14 distinct genres and 9 song language categories.

Variable names were systematically cleaned using prefixes such as genre_, slang_, and ulang_ to distinguish between song and user attributes and prevent naming conflicts during encoding.

A final integrity check verified that the feature matrix contained zero missing (NaN) values.

Feature Engineering

Multi-hot and one-hot encoding were used for nominal categorical variables to allow for multiple attributes per entry, such as a song belonging to more than one genre.

Multi-hot encoding represents data by assigning 1.0 to each applicable category and 0.0 to others.  

Ordinal encoding was applied to temporal data to preserve the chronological relationship between release eras.

The release_year_ord column mapped specific intervals to sequential values: 3.0 for "2000–2009", 4.0 for "2010–2019", and 5.0 for "2020+".

User demographics and song features were concatenated into a 34-dimensional joint numerical feature matrix.

Model Justification
Distance-based algorithms like KNN and K-Means require numerical inputs to calculate spatial proximity using metrics like cosine similarity or Euclidean distance.

Multi-hot encoding is necessary for these models to process categorical data without assuming a false mathematical hierarchy between independent groups like "Pop" and "Rock".  

Ordinal encoding ensures that distance-based models recognize the relative temporal proximity between release years.

Tree-based models like Random Forest require numerical features to perform threshold splits during training.

Using ordinal values for years allows Random Forest to make logical chronological splits (e.g., year >= 4.0) to capture modern listening trends.

## 4. EDA: Explaoratory Data Analysis
- Data Issues
The raw dataset contained incomplete records, necessitating the removal of two samples with missing entries in critical fields, such as the song name or genre, to ensure accurate similarity calculations. Additionally, inconsistent text inputs required standardization to consolidate the data into clean categories.

- Patterns & Characteristics
Visualizations of the dataset revealed several distinct trends. The genre distribution demonstrated that Pop is the most prevalent genre, comprising approximately 37% of the dataset, followed by Rock and R&B/Soul. Demographic correlations indicated that Suburban users slightly favor Pop music, while City users exhibit more diverse genre preferences. Furthermore, the dataset exhibits a strong recency bias, with approximately 47% of the submitted songs released in the 2020+ era. There is also a massive user language skew toward English, spoken by roughly 99% of the users, with Chinese and Hindi appearing as the next most common languages. These distinct demographic and temporal patterns justify the inclusion of user attributes alongside song features to effectively cluster and recommend music.

- Data Preprocessing
Cleaning Methodology
During the data cleaning phase, the dataset was filtered to a final count of 170 valid records. Two samples were dropped because they contained missing or incomplete entries in essential fields such as the song name or primary genre, which are required for subsequent similarity calculations. Standardizing the text strings successfully consolidated diverse user inputs into 14 distinct genres and 9 song language categories. To prevent naming conflicts during encoding and clearly distinguish between song and user attributes, variable names were systematically cleaned using prefixes such as genre_, slang_, and ulang_. Finally, an integrity check verified that the resulting feature matrix contained zero missing (NaN) values.

- Feature Engineering
To prepare the data for modeling, multi-hot and one-hot encoding were utilized for nominal categorical variables. This approach allowed for multiple attributes per entry, such as representing a single song that belongs to multiple genres, by assigning a value of 1.0 to each applicable category and 0.0 to others. For temporal data, ordinal encoding was applied to preserve the chronological relationship between release eras. Specifically, the release_year_ord column mapped intervals to sequential values, assigning 3.0 for "2000–2009", 4.0 for "2010–2019", and 5.0 for "2020+". Ultimately, user demographics and song features were concatenated into a comprehensive 34-dimensional joint numerical feature matrix.

- Model Justification
These preprocessing steps were specifically tailored to support our chosen machine learning models. Distance-based algorithms like KNN and K-Means require numerical inputs to calculate spatial proximity using metrics such as cosine similarity or Euclidean distance. Multi-hot encoding is essential for these models to process categorical data without falsely assuming a mathematical hierarchy between independent groups, like "Pop" and "Rock". Furthermore, ordinal encoding ensures that distance-based models correctly interpret the relative temporal proximity between release years. For tree-based models like Random Forest, which require numerical features to perform threshold splits during training, utilizing ordinal values for years allows the algorithm to make logical chronological splits (e.g., year >= 4.0) and effectively capture modern listening trends.

---

Approaches
Simon — KNN Recommendation
Content-based filtering (content_knn.py)

Each song is one-hot encoded (genre, language, release era).

A user's profile = mean vector of their submitted songs.

Recommendations ranked by cosine similarity to the profile.

User-based collaborative filtering (collab_knn.py)

Each user is encoded by demographics (gender, hometown, language).

K nearest demographic neighbours are found via cosine similarity.

Songs submitted by neighbours (not yet seen by the user) are recommended.

Xizhi — K-means Clustering & Random Forest
Cluster-based recommendation (kmeans_recommender.py)

Each row is represented as a joint vector: user demographics + song features.

Features scaled with StandardScaler; optimal K chosen by silhouette score.

A new user is assigned to the nearest cluster centroid, and the most popular songs in that cluster are recommended.

Limitation: K-means clustering is less efficient and effective for this dataset due to the high-dimensional, sparse nature of the multi-hot encoded categorical data (the "curse of dimensionality"). This sparsity dilutes the meaning of distance metrics and forces rigid, unpersonalized cluster assignments.

Supervised classification (random_forest_recommender.py)

Frames recommendation as a classification problem, predicting the likelihood of a user engaging with a particular song based on the joint feature matrix.

Handles the sparse multi-hot columns and ordinal temporal data natively by making logical, discrete threshold splits (e.g., isolating the presence of a specific genre or filtering by release_year_ord >= 4.0).

Recommendations are generated by scoring unseen songs and ranking them based on the tree ensemble's predicted probability of a positive match.

---

## 5. User Preference Analysis
---
Methodology
Cross-tabulation with row-wise normalization was utilized and visualized as Seaborn heatmaps to account for the uneven distribution of user demographics and reveal true proportional preferences.

Hometown vs. Genre Preferences

Suburban Users: Exhibit a strong preference for Pop music, representing 0.41 of their normalized distribution. This is higher than both Rural (0.38) and City (0.33) users.

City Users: Display a broader, more diverse listening profile. While Pop remains the top genre at 0.33, their secondary preferences are evenly split between Hip Hop/Rap (0.13) and Indie/Alternative (0.13), followed closely by R&B/Soul (0.10) and Rock (0.10).

Gender vs. Genre Preferences

Female Users: Show a highly concentrated preference for Pop music at 0.49, with a sharp drop-off to their second most popular genre, Indie/Alternative, at 0.12.

Male Users: Demonstrate a more varied genre distribution. Their preference for Pop is significantly lower at 0.27, followed closely by Rock (0.21) and Hip Hop/Rap (0.15).

Language Correlation

While English dominates the user base (~99%), minority language speakers also form distinct demographic segments. Approximately 35 users speak Chinese, and roughly 10 speak Hindi (similar to the proportion of Spanish speakers).

Summary Insight
These specific correlations directly validate the modeling strategy. The distinct genre distributions across genders, hometowns, and languages justify utilizing user attributes as joint features in the Collaborative Filtering and K-Means models to successfully recommend songs even when prior listening history is sparse.


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
