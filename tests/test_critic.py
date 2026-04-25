import pytest
from unittest.mock import MagicMock, patch
from agents.intent_parser import MoodProfile


def _make_profile() -> MoodProfile:
    return MoodProfile(
        is_music_request=True, activity="coding", mood_keywords=["focused"],
        preferred_genres=[], target_energy=0.4, target_valence=0.5,
        tempo_preference="medium", instrumentalness_preference=0.8,
        popularity_preference=0.5, reasoning="", raw_query="focus music",
    )


def _make_songs():
    return [
        {"title": f"Song{i}", "artist": "Artist", "genre": "edm", "mood": "focused",
         "energy": 0.4, "valence": 0.5, "acousticness": 0.3, "instrumentalness": 0.6,
         "speechiness": 0.05, "score": 3.0, "score_breakdown": {}}
        for i in range(5)
    ]


def test_evaluate_recommendations_ab_returns_both_models():
    from agents.critic import evaluate_recommendations_ab, CriticVerdict

    mock_verdict = CriticVerdict(
        passed=True, overall_score=7.5, energy_alignment=8.0,
        diversity_score=6.0, explanation_quality=7.0, feedback="", issues=[],
    )

    with patch("agents.critic.ChatGroq") as mock_groq:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_verdict
        mock_groq.return_value = mock_llm

        result = evaluate_recommendations_ab(_make_songs(), _make_profile(), ["exp"] * 5)

    assert "fast" in result
    assert "quality" in result
    assert result["fast"]["passed"] is True
    assert result["quality"]["overall_score"] == pytest.approx(7.5)


def test_evaluate_recommendations_ab_graceful_on_failure():
    from agents.critic import evaluate_recommendations_ab

    with patch("agents.critic.ChatGroq") as mock_groq:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("API error")
        mock_groq.return_value = mock_llm

        result = evaluate_recommendations_ab(_make_songs(), _make_profile(), ["exp"] * 5)

    # Both models failed — should return graceful fallback dicts, not raise
    assert "fast" in result
    assert result["fast"].get("error") is True
    assert result["fast"]["passed"] is True  # graceful pass on error
