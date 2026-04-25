"""Core recommendation engine for the Music Recommender Simulation."""

import csv
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Song:
    """Represents a song and its audio attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ---------------------------------------------------------------------------
# Scoring weights — tweak these to change recommendation behaviour
# ---------------------------------------------------------------------------
WEIGHT_GENRE = 2.0
WEIGHT_MOOD = 1.0
WEIGHT_ENERGY = 1.5
WEIGHT_VALENCE = 0.5
WEIGHT_DANCEABILITY = 0.5
WEIGHT_ACOUSTICNESS = 0.8


# ---------------------------------------------------------------------------
# OOP implementation (used by tests)
# ---------------------------------------------------------------------------
class Recommender:
    """OOP implementation of the recommendation logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> float:
        """Compute a numeric relevance score for a song given a user profile."""
        score = 0.0

        # Categorical matches
        if song.genre.lower() == user.favorite_genre.lower():
            score += WEIGHT_GENRE
        if song.mood.lower() == user.favorite_mood.lower():
            score += WEIGHT_MOOD

        # Numerical proximity: reward songs closer to user targets (0‒1 scale)
        energy_sim = 1.0 - abs(song.energy - user.target_energy)
        score += WEIGHT_ENERGY * energy_sim

        # Valence bonus — happier moods pair with higher valence
        score += WEIGHT_VALENCE * song.valence

        # Danceability bonus
        score += WEIGHT_DANCEABILITY * song.danceability

        # Acousticness preference
        if user.likes_acoustic:
            score += WEIGHT_ACOUSTICNESS * song.acousticness
        else:
            score += WEIGHT_ACOUSTICNESS * (1.0 - song.acousticness)

        return round(score, 4)

    def _explain(self, user: UserProfile, song: Song) -> str:
        """Build a human-readable explanation of why a song was recommended."""
        reasons: List[str] = []

        if song.genre.lower() == user.favorite_genre.lower():
            reasons.append(f"genre match: {song.genre} (+{WEIGHT_GENRE})")
        if song.mood.lower() == user.favorite_mood.lower():
            reasons.append(f"mood match: {song.mood} (+{WEIGHT_MOOD})")

        energy_sim = 1.0 - abs(song.energy - user.target_energy)
        reasons.append(f"energy similarity: {energy_sim:.2f} (+{WEIGHT_ENERGY * energy_sim:.2f})")

        reasons.append(f"valence: {song.valence:.2f} (+{WEIGHT_VALENCE * song.valence:.2f})")
        reasons.append(f"danceability: {song.danceability:.2f} (+{WEIGHT_DANCEABILITY * song.danceability:.2f})")

        if user.likes_acoustic:
            reasons.append(f"acousticness: {song.acousticness:.2f} (+{WEIGHT_ACOUSTICNESS * song.acousticness:.2f})")
        else:
            reasons.append(f"non-acoustic bonus: {1.0 - song.acousticness:.2f} (+{WEIGHT_ACOUSTICNESS * (1.0 - song.acousticness):.2f})")

        return "; ".join(reasons)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked by relevance to the user profile."""
        scored = [(self._score(user, s), s) for s in self.songs]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [song for _, song in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation for why a song was recommended."""
        return self._explain(user, song)


# ---------------------------------------------------------------------------
# Functional implementation (used by main.py)
# ---------------------------------------------------------------------------
def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return a list of dictionaries."""
    songs: List[Dict] = []

    # Resolve path relative to this file's directory when running as module
    if not os.path.isabs(csv_path):
        base = os.path.dirname(os.path.abspath(__file__))
        # Go up one level from src/ to project root
        project_root = os.path.dirname(base)
        csv_path = os.path.join(project_root, csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields to proper types
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)

    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a single song against user preferences; return (score, reasons)."""
    score = 0.0
    reasons: List[str] = []

    # Genre match
    if song.get("genre", "").lower() == user_prefs.get("genre", "").lower():
        score += WEIGHT_GENRE
        reasons.append(f"genre match: {song['genre']} (+{WEIGHT_GENRE})")

    # Mood match
    if song.get("mood", "").lower() == user_prefs.get("mood", "").lower():
        score += WEIGHT_MOOD
        reasons.append(f"mood match: {song['mood']} (+{WEIGHT_MOOD})")

    # Energy similarity
    target_energy = user_prefs.get("energy", 0.5)
    energy_sim = 1.0 - abs(song["energy"] - target_energy)
    energy_pts = WEIGHT_ENERGY * energy_sim
    score += energy_pts
    reasons.append(f"energy similarity {energy_sim:.2f} (+{energy_pts:.2f})")

    # Valence bonus
    val_pts = WEIGHT_VALENCE * song["valence"]
    score += val_pts
    reasons.append(f"valence {song['valence']:.2f} (+{val_pts:.2f})")

    # Danceability bonus
    dance_pts = WEIGHT_DANCEABILITY * song["danceability"]
    score += dance_pts
    reasons.append(f"danceability {song['danceability']:.2f} (+{dance_pts:.2f})")

    # Acousticness — respect user preference if given
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    if likes_acoustic:
        ac_pts = WEIGHT_ACOUSTICNESS * song["acousticness"]
    else:
        ac_pts = WEIGHT_ACOUSTICNESS * (1.0 - song["acousticness"])
    score += ac_pts
    reasons.append(f"acousticness preference (+{ac_pts:.2f})")

    return round(score, 4), reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank all songs by score and return the top k as (song, score, explanation)."""
    scored: List[Tuple[Dict, float, str]] = []
    for song in songs:
        pts, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored.append((song, pts, explanation))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]
