"""Tests for the Intent Parser agent (Node 1 in the LangGraph pipeline)."""
import pytest
from unittest.mock import patch, MagicMock

from agents.intent_parser import MoodProfile, parse_intent


def _make_profile(**kwargs) -> MoodProfile:
    """Helper to create a MoodProfile with sensible defaults."""
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
        reasoning="test profile",
        raw_query="",
    )
    defaults.update(kwargs)
    return MoodProfile(**defaults)


class TestMoodProfile:
    def test_default_values(self):
        p = MoodProfile()
        assert p.target_energy == 0.5
        assert p.target_valence == 0.5
        assert p.mood_keywords == []
        assert p.preferred_genres == []
        assert p.is_music_request is True
        assert p.popularity_preference == 0.5
        assert p.instrumentalness_preference == 0.3

    def test_energy_clamped_above(self):
        with pytest.raises(Exception):
            MoodProfile(target_energy=1.5)

    def test_energy_clamped_below(self):
        with pytest.raises(Exception):
            MoodProfile(target_energy=-0.1)

    def test_valence_clamped_above(self):
        with pytest.raises(Exception):
            MoodProfile(target_valence=1.1)

    def test_instrumentalness_preference_valid(self):
        p = MoodProfile(instrumentalness_preference=0.8)
        assert p.instrumentalness_preference == 0.8

    def test_popularity_preference_valid_range(self):
        p = MoodProfile(popularity_preference=0.1)
        assert p.popularity_preference == 0.1

    def test_popularity_preference_clamped(self):
        with pytest.raises(Exception):
            MoodProfile(popularity_preference=1.5)

    def test_is_music_request_defaults_true(self):
        p = MoodProfile()
        assert p.is_music_request is True

    def test_is_music_request_can_be_false(self):
        p = MoodProfile(is_music_request=False)
        assert p.is_music_request is False


class TestParseIntent:
    @patch("agents.intent_parser.ChatGroq")
    def test_coding_query_low_energy(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = _make_profile(target_energy=0.4, activity="coding")

        result = parse_intent("I'm coding late at night, need focus music")

        assert result.target_energy < 0.6
        assert result.raw_query == "I'm coding late at night, need focus music"

    @patch("agents.intent_parser.ChatGroq")
    def test_workout_query_high_energy(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = _make_profile(target_energy=0.85, activity="workout")

        result = parse_intent("Pump up gym music, heavy beats")

        assert result.target_energy > 0.7

    @patch("agents.intent_parser.ChatGroq")
    def test_raw_query_is_preserved(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = _make_profile()

        result = parse_intent("surprise me with something interesting")

        assert result.raw_query == "surprise me with something interesting"

    @patch("agents.intent_parser.ChatGroq")
    def test_non_music_query_sets_is_music_request_false(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = _make_profile(is_music_request=False)

        result = parse_intent("What is the capital of France?")

        assert result.is_music_request is False

    @patch("agents.intent_parser.ChatGroq")
    def test_underground_query_sets_low_popularity_preference(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = _make_profile(
            popularity_preference=0.1,
            preferred_genres=["edm"],
        )

        result = parse_intent("Give me the most obscure underground electronic tracks")

        assert result.popularity_preference < 0.3

    @patch("agents.intent_parser.ChatGroq")
    def test_instrumental_query_high_inst_preference(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = _make_profile(instrumentalness_preference=0.95)

        result = parse_intent("Purely instrumental music with absolutely zero vocals")

        assert result.instrumentalness_preference > 0.8

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            parse_intent("test query")
