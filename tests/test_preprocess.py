import pandas as pd
import pytest
from pathlib import Path

_HERE = Path(__file__).parent.parent


def test_preprocess_has_instrumental_bucket(tmp_path):
    """After preprocess(), songs_processed.csv must contain instrumental rows."""
    from data.preprocess import preprocess
    output = tmp_path / "out.csv"
    preprocess(
        input_path=str(_HERE / "data" / "spotify_songs.csv"),
        output_path=str(output),
    )
    df = pd.read_csv(output)
    inst_rows = df[df["genre"] == "instrumental"]
    assert len(inst_rows) > 0, "No instrumental rows found"
    assert (inst_rows["instrumentalness"] > 0.5).all(), \
        "Instrumental bucket songs must have instrumentalness > 0.5"


def test_instrumental_bucket_no_duplicates(tmp_path):
    """Songs in the instrumental bucket must not duplicate genre-sampled songs."""
    from data.preprocess import preprocess
    output = tmp_path / "out.csv"
    preprocess(
        input_path=str(_HERE / "data" / "spotify_songs.csv"),
        output_path=str(output),
    )
    df = pd.read_csv(output)
    assert df["song_id"].nunique() == len(df), "Duplicate song_ids found"
