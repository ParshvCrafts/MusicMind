"""Tests for the Scorer Agent — scoring formula and profile-driven weight derivation."""
import pytest
from agents.scorer import ScoringWeights, score_and_rank, _derive_initial_weights
from agents.intent_parser import MoodProfile


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_profile(**kwargs) -> MoodProfile:
    defaults = dict(
        is_music_request=True,
        activity="coding",
        mood_keywords=["focused", "chill"],
        preferred_genres=["edm", "pop"],
        target_energy=0.4,
        target_valence=0.5,
        tempo_preference="medium",
        instrumentalness_preference=0.7,
        popularity_preference=0.5,
        reasoning="test",
        raw_query="coding focus music no lyrics",
    )
    defaults.update(kwargs)
    return MoodProfile(**defaults)


@pytest.fixture
def coding_profile() -> MoodProfile:
    return _make_profile()


@pytest.fixture
def workout_profile() -> MoodProfile:
    return _make_profile(
        activity="workout",
        mood_keywords=["intense", "energetic"],
        preferred_genres=["edm", "rock"],
        target_energy=0.88,
        target_valence=0.6,
        instrumentalness_preference=0.1,
        popularity_preference=0.5,
        raw_query="high energy gym music",
    )


@pytest.fixture
def underground_profile() -> MoodProfile:
    return _make_profile(
        activity="listening",
        preferred_genres=["edm"],
        target_energy=0.5,
        popularity_preference=0.1,  # strongly prefer obscure
        instrumentalness_preference=0.3,
        raw_query="obscure underground electronic",
    )


@pytest.fixture
def instrumental_profile() -> MoodProfile:
    return _make_profile(
        activity="deep focus",
        mood_keywords=["focused"],
        preferred_genres=["edm"],
        target_energy=0.35,
        instrumentalness_preference=0.95,  # purely instrumental
        popularity_preference=0.5,
        raw_query="purely instrumental acoustic music zero vocals",
    )


@pytest.fixture
def sample_songs() -> list[dict]:
    return [
        {
            "song_id": "a1", "title": "Focus Flow", "artist": "Artist A",
            "genre": "edm", "subgenre": "ambient", "mood": "focused",
            "energy": 0.42, "valence": 0.52, "danceability": 0.6,
            "acousticness": 0.1, "bpm": 115.0, "popularity": 70,
            "instrumentalness": 0.75, "speechiness": 0.04, "liveness": 0.1,
        },
        {
            "song_id": "b2", "title": "Pump Up", "artist": "Artist B",
            "genre": "rock", "subgenre": "metal", "mood": "intense",
            "energy": 0.95, "valence": 0.28, "danceability": 0.55,
            "acousticness": 0.03, "bpm": 158.0, "popularity": 80,
            "instrumentalness": 0.05, "speechiness": 0.06, "liveness": 0.15,
        },
        {
            "song_id": "c3", "title": "Underground Beats", "artist": "Artist C",
            "genre": "edm", "subgenre": "techno", "mood": "focused",
            "energy": 0.55, "valence": 0.45, "danceability": 0.7,
            "acousticness": 0.05, "bpm": 130.0, "popularity": 15,
            "instrumentalness": 0.85, "speechiness": 0.03, "liveness": 0.08,
        },
    ]


# ── Basic scoring tests ───────────────────────────────────────────────────────


def test_score_and_rank_sorted_descending(coding_profile, sample_songs):
    results = score_and_rank(sample_songs, coding_profile)
    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i + 1]["score"]


def test_score_includes_breakdown(coding_profile, sample_songs):
    results = score_and_rank(sample_songs, coding_profile)
    assert "score_breakdown" in results[0]
    breakdown = results[0]["score_breakdown"]
    for key in ("energy", "genre", "mood", "acousticness", "instrumentalness", "speechiness_penalty"):
        assert key in breakdown


def test_empty_candidates_returns_empty(coding_profile):
    assert score_and_rank([], coding_profile) == []


def test_score_and_rank_preserves_song_fields(coding_profile, sample_songs):
    results = score_and_rank(sample_songs, coding_profile)
    for field in ("title", "artist", "genre", "score"):
        assert field in results[0]


def test_score_is_float_rounded_to_4dp(coding_profile, sample_songs):
    results = score_and_rank(sample_songs, coding_profile)
    score = results[0]["score"]
    assert isinstance(score, float)
    assert score == round(score, 4)


# ── Profile-driven weight derivation ─────────────────────────────────────────


class TestDeriveInitialWeights:
    def test_coding_profile_boosts_instrumentalness(self, coding_profile):
        w = _derive_initial_weights(coding_profile)
        # instrumentalness_preference=0.7 > 0.6 → weight should be > 1.0
        assert w.instrumentalness_weight > 1.0

    def test_coding_profile_penalizes_speechiness(self, coding_profile):
        w = _derive_initial_weights(coding_profile)
        # coding activity → speechiness_penalty should be boosted
        assert w.speechiness_penalty > 0.1

    def test_workout_profile_boosts_energy_weight(self, workout_profile):
        w = _derive_initial_weights(workout_profile)
        assert w.energy_weight >= 2.0

    def test_workout_profile_boosts_danceability(self, workout_profile):
        w = _derive_initial_weights(workout_profile)
        assert w.danceability_weight >= 1.0

    def test_underground_profile_penalizes_popularity(self, underground_profile):
        w = _derive_initial_weights(underground_profile)
        # popularity_preference=0.1 → popularity_weight should be negative
        assert w.popularity_weight < 0.0

    def test_instrumental_profile_maximizes_inst_weight(self, instrumental_profile):
        w = _derive_initial_weights(instrumental_profile)
        # instrumentalness_preference=0.95 → very high instrumentalness_weight
        assert w.instrumentalness_weight >= 2.0
        # And very high speechiness_penalty
        assert w.speechiness_penalty >= 1.5


# ── Feature weight correctness ────────────────────────────────────────────────


class TestFeatureWeightEffects:
    def test_instrumental_profile_ranks_instrumental_song_higher(
        self, instrumental_profile, sample_songs
    ):
        """Song c3 has instrumentalness=0.85 and should rank highest for a purely instrumental query."""
        results = score_and_rank(sample_songs, instrumental_profile)
        top = results[0]
        # The most instrumental song should rank first
        assert float(top.get("instrumentalness", 0)) >= 0.5, (
            f"Expected top song to have high instrumentalness, got: {top.get('instrumentalness')}"
        )

    def test_underground_profile_penalizes_popular_songs(
        self, underground_profile, sample_songs
    ):
        """song b2 (popularity=80) should be ranked BELOW song c3 (popularity=15) for underground query."""
        results = score_and_rank(sample_songs, underground_profile)
        c3 = next(r for r in results if r["song_id"] == "c3")
        b2 = next(r for r in results if r["song_id"] == "b2")
        # Underground profile penalizes popularity — c3 (popularity 15) should score higher than b2 (80)
        assert c3["score"] > b2["score"], (
            f"Underground track (pop=15, score={c3['score']}) should beat "
            f"popular track (pop=80, score={b2['score']})"
        )

    def test_workout_profile_favors_high_energy_song(self, workout_profile, sample_songs):
        """'Pump Up' (energy=0.95) should rank first for workout queries."""
        results = score_and_rank(sample_songs, workout_profile)
        assert results[0]["title"] == "Pump Up"

    def test_popularity_weight_can_be_negative(self, underground_profile):
        w = _derive_initial_weights(underground_profile)
        assert w.popularity_weight < 0
        # Verify ScoringWeights accepts negative popularity_weight
        assert ScoringWeights(popularity_weight=-0.5)

    def test_speechiness_breakdown_is_negative(self, coding_profile, sample_songs):
        """The speechiness_penalty component in breakdown should be ≤ 0."""
        results = score_and_rank(sample_songs, coding_profile)
        for r in results:
            assert r["score_breakdown"]["speechiness_penalty"] <= 0


import json
import tempfile


def test_liked_artist_gets_bonus(tmp_path, monkeypatch):
    """A liked artist's song scores higher due to memory bonus."""
    from agents.scorer import score_and_rank
    from agents.intent_parser import MoodProfile

    memory = {"liked_artists": {"The Artist": 3}, "disliked_artists": {}}
    mem_file = tmp_path / "user_memory.json"
    mem_file.write_text(json.dumps(memory))

    # Monkeypatch the memory path inside scorer
    import agents.scorer as scorer_mod
    monkeypatch.setattr(scorer_mod, "_MEMORY_PATH", mem_file)

    profile = MoodProfile(
        is_music_request=True, activity="coding", mood_keywords=[],
        preferred_genres=[], target_energy=0.4, target_valence=0.5,
        tempo_preference="medium", instrumentalness_preference=0.3,
        popularity_preference=0.5, reasoning="", raw_query="test",
    )
    candidates = [
        {"song_id": "1", "title": "Liked Song", "artist": "The Artist",
         "genre": "pop", "mood": "focused", "energy": 0.4, "valence": 0.5,
         "danceability": 0.5, "acousticness": 0.3, "popularity": 50,
         "instrumentalness": 0.1, "speechiness": 0.05, "liveness": 0.1, "bpm": 100},
        {"song_id": "2", "title": "Other Song", "artist": "Other Artist",
         "genre": "pop", "mood": "focused", "energy": 0.4, "valence": 0.5,
         "danceability": 0.5, "acousticness": 0.3, "popularity": 50,
         "instrumentalness": 0.1, "speechiness": 0.05, "liveness": 0.1, "bpm": 100},
    ]
    results = score_and_rank(candidates, profile)
    liked_score = next(r["score"] for r in results if r["artist"] == "The Artist")
    other_score = next(r["score"] for r in results if r["artist"] == "Other Artist")
    assert liked_score > other_score, "Liked artist should score higher"
    assert results[0]["score_breakdown"]["memory_bonus"] == pytest.approx(0.4)


def test_disliked_artist_gets_penalty(tmp_path, monkeypatch):
    from agents.scorer import score_and_rank
    from agents.intent_parser import MoodProfile
    import agents.scorer as scorer_mod

    memory = {"liked_artists": {}, "disliked_artists": {"Bad Artist": 2}}
    mem_file = tmp_path / "user_memory.json"
    mem_file.write_text(json.dumps(memory))
    monkeypatch.setattr(scorer_mod, "_MEMORY_PATH", mem_file)

    profile = MoodProfile(
        is_music_request=True, activity="coding", mood_keywords=[],
        preferred_genres=[], target_energy=0.4, target_valence=0.5,
        tempo_preference="medium", instrumentalness_preference=0.3,
        popularity_preference=0.5, reasoning="", raw_query="test",
    )
    candidates = [
        {"song_id": "1", "title": "Disliked Song", "artist": "Bad Artist",
         "genre": "pop", "mood": "focused", "energy": 0.4, "valence": 0.5,
         "danceability": 0.5, "acousticness": 0.3, "popularity": 50,
         "instrumentalness": 0.1, "speechiness": 0.05, "liveness": 0.1, "bpm": 100},
        {"song_id": "2", "title": "Neutral Song", "artist": "Neutral Artist",
         "genre": "pop", "mood": "focused", "energy": 0.4, "valence": 0.5,
         "danceability": 0.5, "acousticness": 0.3, "popularity": 50,
         "instrumentalness": 0.1, "speechiness": 0.05, "liveness": 0.1, "bpm": 100},
    ]
    results = score_and_rank(candidates, profile)
    disliked_score = next(r["score"] for r in results if r["artist"] == "Bad Artist")
    neutral_score = next(r["score"] for r in results if r["artist"] == "Neutral Artist")
    assert disliked_score < neutral_score


def test_missing_memory_file_is_graceful(tmp_path, monkeypatch):
    from agents.scorer import score_and_rank, _load_user_memory
    import agents.scorer as scorer_mod
    monkeypatch.setattr(scorer_mod, "_MEMORY_PATH", tmp_path / "nonexistent.json")
    memory = _load_user_memory()
    assert memory == {"liked_artists": {}, "disliked_artists": {}}


def test_weight_adjustment_falls_back_on_llm_failure(monkeypatch):
    """When the LLM returns invalid JSON (Groq 400), fall back to base weights."""
    from unittest.mock import MagicMock, patch
    from agents.scorer import _adjust_weights_from_feedback, ScoringWeights
    from agents.intent_parser import MoodProfile

    profile = MoodProfile(
        is_music_request=True, activity="coding", mood_keywords=[],
        preferred_genres=[], target_energy=0.4, target_valence=0.5,
        tempo_preference="medium", instrumentalness_preference=0.3,
        popularity_preference=0.5, reasoning="", raw_query="test",
    )
    base = ScoringWeights()

    with patch("agents.scorer.ChatGroq") as mock_groq:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("Groq 400")
        mock_groq.return_value = mock_llm

        # Should not raise — should fall back to base_weights
        result = _adjust_weights_from_feedback("Increase instrumentalness_weight", profile, base)

    assert result.genre_weight == base.genre_weight
    assert result.energy_weight == base.energy_weight
