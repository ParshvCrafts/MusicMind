"""
Integration tests for the LangGraph Orchestrator (agents/orchestrator.py).

All external dependencies (LLM calls, FAISS retrieval) are mocked so these
tests run fast and offline — no GROQ_API_KEY or data files required.
"""
import pytest
from unittest.mock import MagicMock, patch

from agents.orchestrator import build_graph, AgentState
from agents.intent_parser import MoodProfile
from rag.knowledge_base import MusicKnowledgeBase


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_profile(is_music_request: bool = True, **kwargs) -> MoodProfile:
    defaults = dict(
        is_music_request=is_music_request,
        activity="coding",
        mood_keywords=["focused"],
        preferred_genres=["edm"],
        target_energy=0.4,
        target_valence=0.5,
        tempo_preference="medium",
        instrumentalness_preference=0.7,
        popularity_preference=0.5,
        reasoning="test",
        raw_query="test query",
    )
    defaults.update(kwargs)
    return MoodProfile(**defaults)


def _make_song(n: int, **overrides) -> dict:
    base = {
        "song_id": str(n),
        "title": f"Song {n}",
        "artist": f"Artist {n}",
        "genre": "edm",
        "subgenre": "ambient",
        "mood": "focused",
        "energy": 0.4,
        "valence": 0.5,
        "danceability": 0.6,
        "acousticness": 0.1,
        "bpm": 110.0,
        "popularity": 60,
        "instrumentalness": 0.7,
        "speechiness": 0.04,
        "liveness": 0.1,
        "score": round(3.5 + n * 0.01, 4),
        "score_breakdown": {},
    }
    base.update(overrides)
    return base


def _make_kb() -> MusicKnowledgeBase:
    kb = MagicMock(spec=MusicKnowledgeBase)
    kb.retrieve_songs.return_value = [_make_song(i) for i in range(20)]
    kb.retrieve_knowledge.return_value = ["chunk about music theory"]
    return kb


_INITIAL: AgentState = {
    "query": "test query",
    "mood_profile": None,
    "retrieved_knowledge": [],
    "candidate_songs": [],
    "scored_songs": [],
    "explanations": [],
    "critic_feedback": "",
    "final_recommendations": [],
    "retry_count": 0,
    "agent_trace": [],
    "rejection_reason": "",
}


def _run(
    critic_passes_on_call: int = 1,
    profile_override: MoodProfile = None,
) -> AgentState:
    """Build and invoke graph with mocked agents."""
    profile = profile_override or _make_profile()
    kb = _make_kb()
    songs = [_make_song(i) for i in range(5)]

    call_counter = {"n": 0}

    def _fake_evaluate(songs_list, prof, exps):
        call_counter["n"] += 1
        return ("", True) if call_counter["n"] >= critic_passes_on_call else ("songs are off", False)

    with (
        patch("agents.orchestrator.parse_intent", return_value=profile),
        patch("agents.orchestrator.retrieve_context",
              return_value=(["knowledge chunk"], [_make_song(i) for i in range(20)])),
        patch("agents.orchestrator.score_and_rank", return_value=songs),
        patch("agents.orchestrator.generate_explanations",
              return_value=[f"Explanation {i}" for i in range(5)]),
        patch("agents.orchestrator.evaluate_recommendations", side_effect=_fake_evaluate),
    ):
        graph = build_graph(kb)
        return graph.invoke(dict(_INITIAL))


# ── Graph structure ───────────────────────────────────────────────────────────


class TestGraphStructure:
    def test_build_graph_returns_compiled_graph(self):
        kb = _make_kb()
        with (
            patch("agents.orchestrator.parse_intent"),
            patch("agents.orchestrator.retrieve_context"),
            patch("agents.orchestrator.score_and_rank"),
            patch("agents.orchestrator.generate_explanations"),
            patch("agents.orchestrator.evaluate_recommendations"),
        ):
            graph = build_graph(kb)
        assert callable(getattr(graph, "invoke", None))

    def test_all_5_nodes_present(self):
        kb = _make_kb()
        with (
            patch("agents.orchestrator.parse_intent"),
            patch("agents.orchestrator.retrieve_context"),
            patch("agents.orchestrator.score_and_rank"),
            patch("agents.orchestrator.generate_explanations"),
            patch("agents.orchestrator.evaluate_recommendations"),
        ):
            graph = build_graph(kb)
        node_names = set(graph.get_graph().nodes.keys())
        for expected in ("parse_intent", "retrieve", "score", "explain", "critic"):
            assert expected in node_names, f"Missing node: {expected}"


# ── Happy-path invocation ─────────────────────────────────────────────────────


class TestGraphInvocation:
    def test_successful_run_returns_final_recommendations(self):
        result = _run(critic_passes_on_call=1)
        assert len(result["final_recommendations"]) == 5

    def test_final_recommendations_have_explanation_field(self):
        result = _run(critic_passes_on_call=1)
        for rec in result["final_recommendations"]:
            assert "explanation" in rec

    def test_agent_trace_populated(self):
        result = _run(critic_passes_on_call=1)
        assert len(result.get("agent_trace", [])) >= 4

    def test_retry_count_zero_on_first_pass(self):
        result = _run(critic_passes_on_call=1)
        assert result["retry_count"] == 0

    def test_retry_increments_when_critic_fails_once(self):
        result = _run(critic_passes_on_call=2)
        assert result["retry_count"] == 1

    def test_hard_cap_at_two_retries(self):
        kb = _make_kb()
        songs = [_make_song(i) for i in range(5)]
        with (
            patch("agents.orchestrator.parse_intent", return_value=_make_profile()),
            patch("agents.orchestrator.retrieve_context",
                  return_value=(["chunk"], [_make_song(i) for i in range(20)])),
            patch("agents.orchestrator.score_and_rank", return_value=songs),
            patch("agents.orchestrator.generate_explanations",
                  return_value=[f"Exp {i}" for i in range(5)]),
            patch("agents.orchestrator.evaluate_recommendations",
                  return_value=("always fails", False)),
        ):
            graph = build_graph(kb)
            result = graph.invoke(dict(_INITIAL))
        assert result["retry_count"] == 2
        assert len(result["final_recommendations"]) > 0

    def test_mood_profile_in_state(self):
        result = _run(critic_passes_on_call=1)
        assert result["mood_profile"] is not None
        assert result["mood_profile"].activity == "coding"

    def test_rejection_reason_empty_on_valid_query(self):
        result = _run(critic_passes_on_call=1)
        assert result.get("rejection_reason", "") == ""


# ── Domain rejection ──────────────────────────────────────────────────────────


class TestDomainRejection:
    def test_non_music_query_sets_rejection_reason(self):
        """When is_music_request=False, the graph ends early with a rejection_reason."""
        non_music_profile = _make_profile(is_music_request=False)
        kb = _make_kb()

        with (
            patch("agents.orchestrator.parse_intent", return_value=non_music_profile),
            # retrieve / score / explain / critic should NEVER be called
            patch("agents.orchestrator.retrieve_context") as mock_retrieve,
            patch("agents.orchestrator.score_and_rank") as mock_score,
            patch("agents.orchestrator.generate_explanations") as mock_explain,
            patch("agents.orchestrator.evaluate_recommendations") as mock_critic,
        ):
            graph = build_graph(kb)
            result = graph.invoke(dict(_INITIAL))

        assert result.get("rejection_reason"), "rejection_reason should be non-empty"
        assert result["final_recommendations"] == []
        # These nodes must never run after domain rejection
        mock_retrieve.assert_not_called()
        mock_score.assert_not_called()
        mock_explain.assert_not_called()
        mock_critic.assert_not_called()

    def test_rejection_reason_in_agent_trace(self):
        non_music_profile = _make_profile(is_music_request=False)
        kb = _make_kb()

        with patch("agents.orchestrator.parse_intent", return_value=non_music_profile):
            graph = build_graph(kb)
            result = graph.invoke(dict(_INITIAL))

        assert any("reject" in step.lower() for step in result.get("agent_trace", []))


# ── run_query wrapper ─────────────────────────────────────────────────────────


class TestRunQuery:
    def test_run_query_returns_agent_state(self):
        from agents.orchestrator import run_query

        kb = _make_kb()
        songs = [_make_song(i) for i in range(5)]
        with (
            patch("agents.orchestrator.parse_intent", return_value=_make_profile()),
            patch("agents.orchestrator.retrieve_context",
                  return_value=(["chunk"], [_make_song(i) for i in range(20)])),
            patch("agents.orchestrator.score_and_rank", return_value=songs),
            patch("agents.orchestrator.generate_explanations",
                  return_value=[f"Exp {i}" for i in range(5)]),
            patch("agents.orchestrator.evaluate_recommendations", return_value=("", True)),
        ):
            result = run_query("test query", kb)

        assert isinstance(result, dict)
        assert len(result["final_recommendations"]) > 0
        assert result.get("rejection_reason", "") == ""


# ── A/B Mode tests ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_kb():
    return _make_kb()


def test_run_query_accepts_ab_mode(mock_kb):
    """run_query must accept ab_mode kwarg without error."""
    from agents.orchestrator import run_query

    mock_profile = MagicMock()
    mock_profile.is_music_request = False
    mock_profile.raw_query = "test"

    with patch("agents.orchestrator.parse_intent", return_value=mock_profile):
        state = run_query("test query", mock_kb, ab_mode=False)
    assert "ab_mode" in state


def test_ab_mode_true_sets_verdict_field(mock_kb):
    """When ab_mode=True and recommendations exist, ab_critic_verdict is populated."""
    from agents.orchestrator import run_query

    profile = MoodProfile(
        is_music_request=True, activity="coding", mood_keywords=["focused"],
        preferred_genres=[], target_energy=0.4, target_valence=0.5,
        tempo_preference="medium", instrumentalness_preference=0.5,
        popularity_preference=0.5, reasoning="", raw_query="test",
    )
    mock_song = {
        "song_id": "1", "title": "T", "artist": "A", "genre": "pop", "mood": "focused",
        "energy": 0.4, "valence": 0.5, "danceability": 0.5, "acousticness": 0.3,
        "popularity": 50, "instrumentalness": 0.1, "speechiness": 0.05,
        "liveness": 0.1, "bpm": 100, "score": 3.0, "score_breakdown": {},
    }
    ab_verdict = {"fast": {"passed": True, "overall_score": 7.0, "issues": []},
                  "quality": {"passed": True, "overall_score": 8.0, "issues": []}}

    with patch("agents.orchestrator.parse_intent", return_value=profile), \
         patch("agents.orchestrator.retrieve_context", return_value=([], [mock_song] * 5)), \
         patch("agents.orchestrator.score_and_rank", return_value=[mock_song] * 10), \
         patch("agents.orchestrator.generate_explanations", return_value=["exp"] * 5), \
         patch("agents.orchestrator.evaluate_recommendations", return_value=("", True)), \
         patch("agents.orchestrator.evaluate_recommendations_ab", return_value=ab_verdict):

        state = run_query("test", mock_kb, ab_mode=True)

    assert state.get("ab_critic_verdict") is not None
    assert "fast" in state["ab_critic_verdict"]
