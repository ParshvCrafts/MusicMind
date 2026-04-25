"""
Preprocesses spotify_songs.csv into songs_processed.csv.
Samples top 150 songs per genre by popularity, derives mood from audio features.
Run from the musicmind/ directory: python data/preprocess.py
"""
from pathlib import Path
import pandas as pd


def derive_mood(energy: float, valence: float, danceability: float, acousticness: float) -> str:
    """Deterministic mood derivation from Spotify audio features."""
    if valence > 0.7 and energy > 0.7:
        return "happy"
    if valence < 0.3 and energy > 0.7:
        return "intense"
    if valence < 0.3 and energy < 0.4:
        return "sad"
    if energy < 0.4 and acousticness > 0.5:
        return "chill"
    if 0.4 <= energy <= 0.7 and valence >= 0.4:
        return "focused"
    if danceability > 0.8:
        return "party"
    if energy < 0.3:
        return "ambient"
    return "relaxed"


def _add_instrumental_bucket(
    df_raw: pd.DataFrame,
    existing_ids: set,
    n: int = 150,
) -> pd.DataFrame:
    """Sample high-instrumentalness tracks not already in the main genre sample.

    Filters instrumentalness > 0.5, excludes already-selected track_ids,
    takes top n by popularity, assigns genre/subgenre = 'instrumental'.
    """
    inst = df_raw[
        (df_raw["instrumentalness"] > 0.5)
        & (~df_raw["track_id"].isin(existing_ids))
    ].copy()

    if inst.empty:
        return inst  # nothing to add

    inst = inst.sort_values("track_popularity", ascending=False).head(n)
    inst["playlist_genre"] = "instrumental"
    inst["playlist_subgenre"] = "instrumental"
    return inst


def preprocess(
    input_path: str = "data/spotify_songs.csv",
    output_path: str = "data/songs_processed.csv",
    samples_per_genre: int = 150,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # Drop duplicates on track_id; some songs appear in multiple playlists
    df = df.drop_duplicates(subset="track_id")

    # Sample top N per genre by popularity
    main_sample = (
        df.sort_values("track_popularity", ascending=False)
        .groupby("playlist_genre")
        .head(samples_per_genre)
        .reset_index(drop=True)
    )

    # 7th bucket: high-instrumentalness tracks not already sampled
    inst_bucket = _add_instrumental_bucket(df, set(main_sample["track_id"]))

    # Combine and reset index
    df = pd.concat([main_sample, inst_bucket], ignore_index=True)

    # Derive mood column
    df["mood"] = df.apply(
        lambda r: derive_mood(
            float(r["energy"]),
            float(r["valence"]),
            float(r["danceability"]),
            float(r["acousticness"]),
        ),
        axis=1,
    )

    # Rename columns to internal standard
    df = df.rename(columns={
        "track_id": "song_id",
        "track_name": "title",
        "track_artist": "artist",
        "playlist_genre": "genre",
        "playlist_subgenre": "subgenre",
        "track_popularity": "popularity",
        "tempo": "bpm",
    })

    # Select only the columns MusicMind uses
    cols = [
        "song_id", "title", "artist", "genre", "subgenre", "mood",
        "energy", "bpm", "valence", "danceability", "acousticness",
        "popularity", "instrumentalness", "speechiness", "liveness",
    ]
    df = df[cols]

    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} songs to {output_path}")
    print(f"\nGenre distribution:\n{df['genre'].value_counts().to_string()}")
    print(f"\nMood distribution:\n{df['mood'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    preprocess()
