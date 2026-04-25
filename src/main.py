"""
Command-line runner for the Music Recommender Simulation.

Loads songs from CSV, runs the recommender for multiple user profiles,
and prints ranked results with explanations.
"""

import sys
import os

# Ensure the src package is importable when running with `python -m src.main`
sys.path.insert(0, os.path.dirname(__file__))

from recommender import load_songs, recommend_songs


# ---------------------------------------------------------------------------
# User profiles
# ---------------------------------------------------------------------------
PROFILES = {
    "Happy Pop Fan": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
    },
    "Chill Lofi Listener": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "likes_acoustic": True,
    },
    "Intense Rock Lover": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.9,
        "likes_acoustic": False,
    },
    "Mellow Acoustic (edge case: sad + high energy)": {
        "genre": "acoustic",
        "mood": "sad",
        "energy": 0.9,
        "likes_acoustic": True,
    },
    "EDM Party Mode": {
        "genre": "edm",
        "mood": "happy",
        "energy": 0.95,
        "likes_acoustic": False,
    },
}


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_recommendations(profile_name: str, prefs: dict, songs: list, k: int = 5) -> None:
    """Run the recommender for one profile and display results."""
    print_header(f"Profile: {profile_name}")
    print(f"  Preferences: genre={prefs.get('genre')}, mood={prefs.get('mood')}, "
          f"energy={prefs.get('energy')}, acoustic={prefs.get('likes_acoustic', False)}")
    print("-" * 60)

    results = recommend_songs(prefs, songs, k=k)

    for rank, (song, score, explanation) in enumerate(results, 1):
        print(f"\n  #{rank}  {song['title']} by {song['artist']}")
        print(f"       Genre: {song['genre']} | Mood: {song['mood']} | Energy: {song['energy']}")
        print(f"       Score: {score:.2f}")
        print(f"       Why:   {explanation}")

    print()


def main() -> None:
    """Entry point — load data, run all profiles, display results."""
    songs = load_songs("data/songs.csv")
    print(f"\nLoaded {len(songs)} songs from catalog.\n")

    for name, prefs in PROFILES.items():
        print_recommendations(name, prefs, songs, k=5)

    # ---------- Sensitivity experiment: double energy, halve genre ----------
    print_header("EXPERIMENT: Weight Shift (2x energy, 0.5x genre)")
    # Temporarily patch weights for the experiment
    import recommender as rec
    original_genre = rec.WEIGHT_GENRE
    original_energy = rec.WEIGHT_ENERGY
    rec.WEIGHT_GENRE = original_genre * 0.5   # halve genre
    rec.WEIGHT_ENERGY = original_energy * 2.0  # double energy

    prefs = PROFILES["Happy Pop Fan"]
    results = recommend_songs(prefs, songs, k=5)
    print(f"  Profile: Happy Pop Fan (genre weight={rec.WEIGHT_GENRE}, energy weight={rec.WEIGHT_ENERGY})")
    print("-" * 60)
    for rank, (song, score, explanation) in enumerate(results, 1):
        print(f"\n  #{rank}  {song['title']} by {song['artist']}")
        print(f"       Score: {score:.2f}")
        print(f"       Why:   {explanation}")

    # Restore original weights
    rec.WEIGHT_GENRE = original_genre
    rec.WEIGHT_ENERGY = original_energy

    # ---------- Sensitivity experiment: remove mood check ----------
    print_header("EXPERIMENT: Feature Removal (mood check disabled)")
    original_mood = rec.WEIGHT_MOOD
    rec.WEIGHT_MOOD = 0.0

    results = recommend_songs(prefs, songs, k=5)
    print(f"  Profile: Happy Pop Fan (mood weight={rec.WEIGHT_MOOD})")
    print("-" * 60)
    for rank, (song, score, explanation) in enumerate(results, 1):
        print(f"\n  #{rank}  {song['title']} by {song['artist']}")
        print(f"       Score: {score:.2f}")
        print(f"       Why:   {explanation}")

    rec.WEIGHT_MOOD = original_mood
    print()


if __name__ == "__main__":
    main()
