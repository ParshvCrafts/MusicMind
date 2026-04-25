"""
Orchestrator — the LangGraph StateGraph that wires all 5 agent nodes.

Graph shape:
  parse_intent → [domain check] → retrieve → score → explain → critic
                      ↓                                          ↓
                  END (rejection)              final_recommendations set → END
                                                      ↓
                                              retry_count < 2 → score (retry loop)

Entry point: build_graph(kb) → compiled graph
Convenience:  run_query(query, kb) → final AgentState dict
"""
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.intent_parser import MoodProfile, parse_intent
from agents.retriever import retrieve_context
from agents.scorer import score_and_rank
from agents.explainer import generate_explanations
from agents.critic import evaluate_recommendations, evaluate_recommendations_ab
from rag.knowledge_base import MusicKnowledgeBase
from reliability.logger import get_logger

_logger = get_logger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    query: str
    mood_profile: Optional[MoodProfile]
    retrieved_knowledge: list[str]
    candidate_songs: list[dict]
    scored_songs: list[dict]
    explanations: list[str]
    critic_feedback: str
    final_recommendations: list[dict]
    retry_count: int
    agent_trace: list[str]
    rejection_reason: str   # non-empty → pipeline was rejected before retrieval
    ab_mode: bool                          # True to run A/B critic comparison
    ab_critic_verdict: Optional[dict]      # Populated when ab_mode=True on final pass


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_graph(kb: MusicKnowledgeBase):
    """Build and compile the LangGraph StateGraph.

    Conditional edges:
      After parse_intent  → "retrieve" (music request) | "end" (domain rejected)
      After critic        → "end" (passed) | "score" (retry, up to 2 retries)
    """
    graph = StateGraph(AgentState)

    # ── Node definitions ─────────────────────────────────────────────────────

    def node_parse_intent(state: AgentState) -> dict:
        _logger.info("node_parse_intent", extra={"query": state["query"][:80]})
        profile = parse_intent(state["query"])

        if not profile.is_music_request:
            rejection = (
                "This doesn't appear to be a music request. "
                "MusicMind recommends songs based on mood, activity, or genre. "
                "Try: 'I need focus music for coding' or 'upbeat songs for a party'."
            )
            _logger.info("node_parse_intent: domain rejected")
            return {
                "mood_profile": profile,
                "rejection_reason": rejection,
                "agent_trace": state["agent_trace"] + [
                    f"Query rejected: not a music request ({state['query'][:60]})"
                ],
            }

        msg = (
            f"Intent parsed: activity={profile.activity}, "
            f"energy={profile.target_energy:.2f}, valence={profile.target_valence:.2f}, "
            f"genres={profile.preferred_genres}, inst_pref={profile.instrumentalness_preference:.2f}, "
            f"pop_pref={profile.popularity_preference:.2f}"
        )
        return {
            "mood_profile": profile,
            "agent_trace": state["agent_trace"] + [msg],
        }

    def node_retrieve(state: AgentState) -> dict:
        _logger.info("node_retrieve")
        knowledge, candidates = retrieve_context(state["mood_profile"], kb)
        msg = (
            f"Retrieved {len(candidates)} candidates, "
            f"{len(knowledge)} knowledge chunks"
        )
        return {
            "retrieved_knowledge": knowledge,
            "candidate_songs": candidates,
            "agent_trace": state["agent_trace"] + [msg],
        }

    def node_score(state: AgentState) -> dict:
        _logger.info("node_score", extra={"retry": state["retry_count"]})
        scored = score_and_rank(
            state["candidate_songs"],
            state["mood_profile"],
            state.get("critic_feedback", ""),
        )
        top = scored[0]["title"] if scored else "none"
        retry_label = f" (retry {state['retry_count']})" if state["retry_count"] > 0 else ""
        msg = f"Scored {len(scored)} songs{retry_label} — top: {top}"
        return {
            "scored_songs": scored[:10],
            "agent_trace": state["agent_trace"] + [msg],
        }

    def node_explain(state: AgentState) -> dict:
        _logger.info("node_explain")
        explanations = generate_explanations(
            state["scored_songs"][:5],
            state["mood_profile"],
            state["retrieved_knowledge"],
        )
        return {
            "explanations": explanations,
            "agent_trace": state["agent_trace"] + ["Explanations generated"],
        }

    def node_critic(state: AgentState) -> dict:
        _logger.info("node_critic", extra={"retry": state["retry_count"]})
        feedback, passed = evaluate_recommendations(
            state["scored_songs"][:5],
            state["mood_profile"],
            state["explanations"],
        )

        will_end = passed or state["retry_count"] >= 2

        # A/B critic comparison (only on final pass)
        ab_verdict = None
        if will_end and state.get("ab_mode", False):
            try:
                ab_verdict = evaluate_recommendations_ab(
                    state["scored_songs"][:5],
                    state["mood_profile"],
                    state["explanations"],
                )
            except Exception as exc:
                _logger.warning("A/B critic failed: %s", exc)

        if will_end:
            final = [
                {**song, "explanation": state["explanations"][i]}
                for i, song in enumerate(state["scored_songs"][:5])
            ]
            retries_used = state["retry_count"]
            msg = (
                f"Critic passed ✓ (retries: {retries_used})"
                if passed
                else f"Hard cap reached after {retries_used} retries — returning best attempt"
            )
            result: dict = {
                "final_recommendations": final,
                "agent_trace": state["agent_trace"] + [msg],
            }
            if ab_verdict is not None:
                result["ab_critic_verdict"] = ab_verdict
            return result

        short = feedback[:120] + "..." if len(feedback) > 120 else feedback
        msg = f"Critic retry {state['retry_count'] + 1}/2 — {short}"
        return {
            "critic_feedback": feedback,
            "retry_count": state["retry_count"] + 1,
            "agent_trace": state["agent_trace"] + [msg],
        }

    # ── Conditional routing ──────────────────────────────────────────────────

    def _after_parse(state: AgentState) -> str:
        return "end" if state.get("rejection_reason") else "retrieve"

    def _after_critic(state: AgentState) -> str:
        return "end" if state.get("final_recommendations") else "score"

    # ── Wire the graph ───────────────────────────────────────────────────────

    graph.add_node("parse_intent", node_parse_intent)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("score", node_score)
    graph.add_node("explain", node_explain)
    graph.add_node("critic", node_critic)

    graph.set_entry_point("parse_intent")
    graph.add_conditional_edges(
        "parse_intent",
        _after_parse,
        {"retrieve": "retrieve", "end": END},
    )
    graph.add_edge("retrieve", "score")
    graph.add_edge("score", "explain")
    graph.add_edge("explain", "critic")
    graph.add_conditional_edges(
        "critic",
        _after_critic,
        {"end": END, "score": "score"},
    )

    return graph.compile()


# ── Convenience wrapper ───────────────────────────────────────────────────────


def run_query(query: str, kb: MusicKnowledgeBase, ab_mode: bool = False) -> AgentState:
    """Convenience function for CLI usage. Returns the final AgentState."""
    graph = build_graph(kb)
    initial: AgentState = {
        "query": query,
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
        "ab_mode": ab_mode,
        "ab_critic_verdict": None,
    }
    return graph.invoke(initial)
