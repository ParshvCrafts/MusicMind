"""
Retriever Agent — Node 2 of the MusicMind LangGraph pipeline.
Builds a semantic query from the MoodProfile and retrieves:
  - Top 20 song candidates from the FAISS songs index
  - Top 3 knowledge chunks from the FAISS knowledge index
"""
from agents.intent_parser import MoodProfile
from rag.knowledge_base import MusicKnowledgeBase


def _apply_hard_constraints(
    candidates: list[dict],
    profile: "MoodProfile",
) -> list[dict]:
    """Post-retrieval hard filter based on MoodProfile constraints.

    Applies instrumentalness and energy bounds. Each constraint tries a strict
    threshold first; if fewer than 5 candidates survive, relaxes to a softer
    threshold; if still fewer than 5, keeps the original list (never empty).
    """
    MIN_CANDIDATES = 5
    filtered = list(candidates)

    # ── Instrumentalness constraint ───────────────────────────────────────
    if profile.instrumentalness_preference > 0.7:
        strict = [s for s in filtered if float(s.get("instrumentalness", 0)) > 0.4]
        if len(strict) >= MIN_CANDIDATES:
            filtered = strict
        else:
            relaxed = [s for s in filtered if float(s.get("instrumentalness", 0)) > 0.2]
            if len(relaxed) >= MIN_CANDIDATES:
                filtered = relaxed
            # else: keep current filtered (don't shrink below 5)

    # ── Energy constraint (applied on top of instrumentalness result) ────
    if profile.target_energy < 0.25:
        strict = [s for s in filtered if float(s.get("energy", 1.0)) < 0.45]
        if len(strict) >= MIN_CANDIDATES:
            filtered = strict
        else:
            relaxed = [s for s in filtered if float(s.get("energy", 1.0)) < 0.55]
            if len(relaxed) >= MIN_CANDIDATES:
                filtered = relaxed
    elif profile.target_energy > 0.85:
        strict = [s for s in filtered if float(s.get("energy", 0.0)) > 0.7]
        if len(strict) >= MIN_CANDIDATES:
            filtered = strict
        else:
            relaxed = [s for s in filtered if float(s.get("energy", 0.0)) > 0.6]
            if len(relaxed) >= MIN_CANDIDATES:
                filtered = relaxed

    return filtered if filtered else candidates  # safety: never return empty


def retrieve_context(
    profile: MoodProfile,
    kb: MusicKnowledgeBase,
    n_songs: int = 20,
    n_knowledge: int = 3,
) -> tuple[list[str], list[dict]]:
    """Return (knowledge_chunks, candidate_songs) from FAISS indexes."""
    # Build a rich semantic query from the structured profile
    query_parts: list[str] = []

    if profile.activity:
        query_parts.append(f"music for {profile.activity}")
    if profile.mood_keywords:
        query_parts.append(" ".join(profile.mood_keywords))
    if profile.preferred_genres:
        query_parts.append(" ".join(profile.preferred_genres))

    # Append numerical targets as natural language hints
    query_parts.append(f"energy level {profile.target_energy:.1f}")
    query_parts.append(f"valence {profile.target_valence:.1f}")
    query_parts.append(f"tempo {profile.tempo_preference}")

    # Fall back to raw query if nothing was extracted
    query = " ".join(query_parts) if query_parts else profile.raw_query

    knowledge = kb.retrieve_knowledge(query, n=n_knowledge)
    candidates = kb.retrieve_songs(query, n=n_songs)
    candidates = _apply_hard_constraints(candidates, profile)  # post-retrieval hard filter

    return knowledge, candidates
