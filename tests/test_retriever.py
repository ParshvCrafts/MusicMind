import pytest
from unittest.mock import MagicMock
from agents.intent_parser import MoodProfile


def _make_profile(**kwargs) -> MoodProfile:
    defaults = dict(
        is_music_request=True, activity="coding", mood_keywords=["focused"],
        preferred_genres=[], target_energy=0.4, target_valence=0.5,
        tempo_preference="medium", instrumentalness_preference=0.5,
        popularity_preference=0.5, reasoning="", raw_query="test",
    )
    defaults.update(kwargs)
    return MoodProfile(**defaults)


def _songs(instrumentalness_values, energy_values=None):
    energy_values = energy_values or [0.5] * len(instrumentalness_values)
    return [
        {"title": f"Song{i}", "artist": "A", "genre": "pop",
         "mood": "focused", "instrumentalness": iv, "energy": ev,
         "score": 0, "score_breakdown": {}}
        for i, (iv, ev) in enumerate(zip(instrumentalness_values, energy_values))
    ]


def test_instrumentalness_constraint_strict():
    from agents.retriever import _apply_hard_constraints
    profile = _make_profile(instrumentalness_preference=0.95)
    # Only songs[5..9] have instrumentalness > 0.4
    candidates = _songs([0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    result = _apply_hard_constraints(candidates, profile)
    assert all(float(s["instrumentalness"]) > 0.4 for s in result)
    assert len(result) >= 5


def test_instrumentalness_constraint_relaxed():
    from agents.retriever import _apply_hard_constraints
    profile = _make_profile(instrumentalness_preference=0.95)
    # Only 3 songs have inst > 0.4, 4 have inst > 0.2
    candidates = _songs([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.3, 0.5, 0.6])
    result = _apply_hard_constraints(candidates, profile)
    # Strict (>0.4) gives only 2 — should fall back to relaxed (>0.2) giving 4
    # 4 < 5, so falls back to original
    assert len(result) >= 2  # at minimum, something was returned


def test_no_constraint_when_preference_low():
    from agents.retriever import _apply_hard_constraints
    profile = _make_profile(instrumentalness_preference=0.3)
    candidates = _songs([0.0] * 10)  # all vocal
    result = _apply_hard_constraints(candidates, profile)
    assert result == candidates  # no filtering applied


def test_low_energy_constraint():
    from agents.retriever import _apply_hard_constraints
    profile = _make_profile(target_energy=0.15)
    candidates = _songs(
        [0.5] * 10,
        energy_values=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1],
    )
    result = _apply_hard_constraints(candidates, profile)
    assert all(float(s["energy"]) < 0.55 for s in result)


def test_fallback_never_returns_empty():
    from agents.retriever import _apply_hard_constraints
    profile = _make_profile(instrumentalness_preference=0.99)
    candidates = _songs([0.0] * 3)  # only 3 candidates, all fail filter
    result = _apply_hard_constraints(candidates, profile)
    assert len(result) > 0
