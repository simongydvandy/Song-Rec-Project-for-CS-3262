"""
Shared preprocessing utilities for the Song Recommendation project.

Produces two feature matrices from the raw survey CSV:
  - song_features : one-hot genre + song-language + release-era (per row)
  - user_features : one-hot gender + hometown + multi-hot user-language (per unique user)
Also returns a cleaned DataFrame with normalised values.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "Song Recommendations.csv"

# ── Column name aliases (raw headers are long) ──────────────────────────────
COL_TIMESTAMP   = "Timestamp"
COL_USER_ID     = "Your Unique ID\nNote, you will need to use this ID for the rest of the semester to keep it  consistent (all lower case)"
COL_GENDER      = "Your Gender"
COL_HOMETOWN    = "Your hometown"
COL_USER_LANG   = "What language do you primarily use in daily life? select all that applied"
COL_SONG        = "Song name"
COL_ARTIST      = "Artist"
COL_GENRE       = "Genre"
COL_SONG_LANG   = "Language of the song (select all that applied)"
COL_YEAR        = "Song release year"

# Friendly short names used internally
SHORT = {
    COL_USER_ID:   "user_id",
    COL_GENDER:    "gender",
    COL_HOMETOWN:  "hometown",
    COL_USER_LANG: "user_lang",
    COL_SONG:      "song",
    COL_ARTIST:    "artist",
    COL_GENRE:     "genre",
    COL_SONG_LANG: "song_lang",
    COL_YEAR:      "release_year",
}

RELEASE_ORDER = ["Pre-1960", "1960–1979", "1980–1999", "2000–2009", "2010–2019", "2020+"]


# ── Cleaning helpers ─────────────────────────────────────────────────────────

def _normalise_genre(g: str) -> str:
    g = g.strip()
    if g in ("Videogame Music", "Video Game"):
        return "Video Game Music"
    if g == "Rock + Opera + Ballad":
        return "Rock"
    return g


def _normalise_song_lang(raw) -> list[str]:
    if not isinstance(raw, str):
        return ["Instrumental"]
    instrumental_aliases = {"none", "no lyrics", "no language", "instrumental"}
    langs = [t.strip() for t in raw.split(";") if t.strip()]
    normalised = []
    for lang in langs:
        if lang.lower() in instrumental_aliases:
            normalised.append("Instrumental")
        else:
            normalised.append(lang.capitalize())
    return normalised or ["Instrumental"]


def _normalise_user_lang(raw) -> list[str]:
    if not isinstance(raw, str):
        return []
    langs = [t.strip() for t in raw.split(";") if t.strip()]
    return [l.capitalize() for l in langs]


# ── Main load/clean function ─────────────────────────────────────────────────

def load_and_clean(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load CSV, rename columns, clean values, drop rows with missing demographics."""
    df = pd.read_csv(path)

    # Rename to short names
    df = df.rename(columns=SHORT)

    # Normalise user_id
    df["user_id"] = df["user_id"].str.lower().str.strip()

    # Drop rows missing gender or hometown (only ~2 rows)
    df = df.dropna(subset=["gender", "hometown"]).reset_index(drop=True)

    # Clean genre
    df["genre"] = df["genre"].apply(_normalise_genre)

    # Expand multi-value columns into list columns
    df["song_lang_list"] = df["song_lang"].apply(_normalise_song_lang)
    df["user_lang_list"] = df["user_lang"].apply(_normalise_user_lang)

    return df


# ── Song feature matrix ──────────────────────────────────────────────────────

def build_song_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode genre, song language(s), and release year for each row.
    Returns a DataFrame with the same index as df.
    """
    # Genre one-hot
    genre_dummies = pd.get_dummies(df["genre"], prefix="genre")

    # Song language multi-hot
    all_song_langs = sorted({lang for langs in df["song_lang_list"] for lang in langs})
    lang_data = {
        f"slang_{lang}": df["song_lang_list"].apply(lambda lst: int(lang in lst))
        for lang in all_song_langs
    }
    song_lang_dummies = pd.DataFrame(lang_data, index=df.index)

    # Release year ordinal (integer 0–5, preserves ordering)
    year_map = {era: i for i, era in enumerate(RELEASE_ORDER)}
    year_ord = df["release_year"].map(year_map).rename("release_year_ord")

    features = pd.concat([genre_dummies, song_lang_dummies, year_ord], axis=1)
    return features.astype(float)


# ── User feature matrix ──────────────────────────────────────────────────────

def build_user_features(df: pd.DataFrame, top_n_langs: int = 6) -> pd.DataFrame:
    """
    Per-row user demographic features: one-hot gender + hometown + multi-hot user language.
    top_n_langs keeps only the N most-common user languages to avoid extreme sparsity.
    Returns a DataFrame with the same index as df.
    """
    # Gender one-hot
    gender_dummies = pd.get_dummies(df["gender"], prefix="gender")

    # Hometown one-hot
    hometown_dummies = pd.get_dummies(df["hometown"], prefix="hometown")

    # User language multi-hot (top N only)
    from collections import Counter
    lang_counts = Counter(lang for langs in df["user_lang_list"] for lang in langs)
    top_langs = [lang for lang, _ in lang_counts.most_common(top_n_langs)]
    lang_data = {
        f"ulang_{lang}": df["user_lang_list"].apply(lambda lst: int(lang in lst))
        for lang in top_langs
    }
    user_lang_dummies = pd.DataFrame(lang_data, index=df.index)

    features = pd.concat([gender_dummies, hometown_dummies, user_lang_dummies], axis=1)
    return features.astype(float)


# ── Combined (joint) feature matrix ─────────────────────────────────────────

def build_joint_features(df: pd.DataFrame, top_n_langs: int = 6) -> pd.DataFrame:
    """Concatenate user + song features into a single matrix (used by K-means)."""
    user_feats = build_user_features(df, top_n_langs=top_n_langs)
    song_feats = build_song_features(df)
    return pd.concat([user_feats, song_feats], axis=1)


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean()
    print(f"Cleaned shape: {df.shape}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Genres: {sorted(df['genre'].unique())}")

    sf = build_song_features(df)
    uf = build_user_features(df)
    jf = build_joint_features(df)
    print(f"Song feature matrix : {sf.shape}")
    print(f"User feature matrix : {uf.shape}")
    print(f"Joint feature matrix: {jf.shape}")
    assert sf.isna().sum().sum() == 0, "NaNs in song features"
    assert uf.isna().sum().sum() == 0, "NaNs in user features"
    print("All checks passed.")
