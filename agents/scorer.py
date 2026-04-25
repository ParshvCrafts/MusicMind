"""
Scorer Agent — Node 3 of the MusicMind LangGraph pipeline.
Applies a weighted scoring formula to rank candidate songs.

Two-stage weight resolution:
  1. _derive_initial_weights(profile) — deterministic, profile-driven defaults.
     Uses MoodProfile fields (activity, instrumentalness_preference,
     popularity_preference, target_energy) to set weights BEFORE any LLM call.
     This means the first-pass scorer is already well-calibrated.
  2. _adjust_weights_from_feedback(feedback, profile, base_weights) — LLM-driven.
     Only called when critic_feedback is non-empty (retry scenario).
     The LLM receives the current weights as context and adjusts them.

Design principle: never make an LLM call on the first pass (no feedback yet).
"""
import logging
import os
import json
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from agents.intent_parser import MoodProfile

_logger = logging.getLogger(__name__)


_MEMORY_PATH = Path(__file__).parent.parent / "data" / "user_memory.json"


def _load_user_memory() -> dict:
    """Load user memory JSON. Returns empty memory on any error."""
    try:
        if _MEMORY_PATH.exists():
            return json.loads(_MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"liked_artists": {}, "disliked_artists": {}}


# ── Schema ────────────────────────────────────────────────────────────────────


class ScoringWeights(BaseModel):
    genre_weight: float = Field(2.0, ge=0.0, le=4.0)
    mood_weight: float = Field(1.0, ge=0.0, le=3.0)
    energy_weight: float = Field(1.5, ge=0.1, le=3.0)
    valence_weight: float = Field(0.5, ge=0.0, le=2.0)
    danceability_weight: float = Field(0.5, ge=0.0, le=2.0)
    acousticness_weight: float = Field(0.8, ge=0.0, le=3.0)
    popularity_weight: float = Field(
        0.3,
        ge=-1.0,  # negative = penalise popular songs (underground queries)
        le=1.0,
    )
    instrumentalness_weight: float = Field(0.2, ge=0.0, le=3.0)
    speechiness_penalty: float = Field(0.1, ge=0.0, le=2.0)


# ── Stage 1: deterministic profile-driven weights ─────────────────────────────


def _derive_initial_weights(profile: MoodProfile) -> ScoringWeights:
    """Compute smart initial weights directly from the MoodProfile.

    No LLM call — pure logic based on activity, instrumentalness_preference,
    popularity_preference, and target_energy.  This ensures the very first
    scoring pass is well-calibrated even before any critic feedback.
    """
    w = ScoringWeights()

    # ── Activity-based adjustments ────────────────────────────────────────
    # NOTE: Check more-specific patterns before generic ones to avoid
    # substring conflicts (e.g. "work" is a substring of "workout").
    activity = (profile.activity or "").lower()

    if any(a in activity for a in ("workout", "gym", "exercise", "lifting", "running", "hiit")):
        w.energy_weight = 2.5
        w.danceability_weight = 1.2
        w.mood_weight = 1.5
        w.speechiness_penalty = 0.0   # vocal hype tracks are fine for gym

    elif any(a in activity for a in ("cod", "focus", "study", "read", "work")):
        w.energy_weight = 1.0          # mid-energy preferred; don't over-amplify
        w.instrumentalness_weight = 1.5
        w.speechiness_penalty = 0.6
        w.danceability_weight = 0.2    # danceability unimportant for focus

    elif any(a in activity for a in ("sleep", "wind down", "bed", "rest", "nap")):
        w.energy_weight = 2.0         # still important to get low energy right
        w.acousticness_weight = 1.5
        w.instrumentalness_weight = 1.0
        w.danceability_weight = 0.1

    elif any(a in activity for a in ("party", "danc", "celebrat", "social")):
        w.energy_weight = 2.0
        w.danceability_weight = 1.5
        w.speechiness_penalty = 0.0

    elif any(a in activity for a in ("meditat", "yoga", "calm", "relax")):
        w.energy_weight = 2.0
        w.acousticness_weight = 1.5
        w.instrumentalness_weight = 1.2
        w.speechiness_penalty = 0.8

    elif any(a in activity for a in ("dinner", "dinner party", "elegant", "sophisticat")):
        w.energy_weight = 1.2
        w.valence_weight = 1.0
        w.acousticness_weight = 1.2
        w.danceability_weight = 0.2

    # ── Instrumentalness preference ───────────────────────────────────────
    # Profile sets 0.0 = vocals fine, 1.0 = purely instrumental required
    inst_pref = profile.instrumentalness_preference
    if inst_pref > 0.6:
        # Scale instrumentalness_weight from 1.5 to 3.0
        w.instrumentalness_weight = max(
            w.instrumentalness_weight,
            round(1.5 + (inst_pref - 0.6) * 3.75, 2),  # 0.6→1.5, 1.0→3.0
        )
        # Scale speechiness_penalty from 0.5 to 2.0
        w.speechiness_penalty = max(
            w.speechiness_penalty,
            round(0.5 + (inst_pref - 0.6) * 3.75, 2),
        )

    # ── Popularity preference ─────────────────────────────────────────────
    pop_pref = profile.popularity_preference
    if pop_pref < 0.35:
        # Underground: penalise popular tracks
        w.popularity_weight = round(-0.5 + pop_pref * (0.5 / 0.35), 2)
    elif pop_pref > 0.65:
        # Mainstream: boost popular tracks
        w.popularity_weight = round(0.3 + (pop_pref - 0.65) * (0.7 / 0.35), 2)
    # else: keep default 0.3 (slight popularity boost)

    # ── Mood keyword hints ────────────────────────────────────────────────
    mood_lower = {m.lower() for m in profile.mood_keywords}
    if mood_lower & {"acoustic", "unplugged", "organic", "raw"}:
        w.acousticness_weight = max(w.acousticness_weight, 1.5)
    if mood_lower & {"instrumental", "no vocals", "no lyrics"}:
        w.instrumentalness_weight = max(w.instrumentalness_weight, 2.0)
        w.speechiness_penalty = max(w.speechiness_penalty, 1.0)
    if mood_lower & {"underground", "obscure", "indie", "deep cut", "hidden"}:
        w.popularity_weight = min(w.popularity_weight, -0.3)
    if mood_lower & {"danceable", "groove", "dance"}:
        w.danceability_weight = max(w.danceability_weight, 1.2)

    return w


# ── Stage 2: LLM-driven weight adjustment from critic feedback ───────────────

_WEIGHT_ADJUSTMENT_PROMPT = """\
You are adjusting music recommendation scoring weights based on critic feedback.

Current weights (derived from user profile):
  genre={genre_weight}, mood={mood_weight}, energy={energy_weight},
  valence={valence_weight}, danceability={danceability_weight},
  acousticness={acousticness_weight}, popularity={popularity_weight},
  instrumentalness={instrumentalness_weight}, speechiness_penalty={speechiness_penalty}

User activity: {activity}
Popularity preference: {pop_pref:.1f} (0=underground, 0.5=neutral, 1=mainstream)
Instrumentalness preference: {inst_pref:.1f} (0=vocals OK, 1=purely instrumental)

Critic feedback to address:
{feedback}

Adjust weights to directly address the feedback. Rules:
- To fix "too much vocal / needs instrumental": increase instrumentalness_weight (max 3.0), speechiness_penalty (max 2.0)
- To fix "too energetic": keep energy_weight high but add valence context
- To fix "not energetic enough": increase energy_weight
- To fix "low diversity": reduce genre_weight to ≤ 1.0
- To fix "too popular / need obscure": reduce popularity_weight (min -1.0)
- To fix "too obscure / need hits": increase popularity_weight (max 1.0)
- To fix "wrong acousticness": increase or decrease acousticness_weight
- All weights must stay within their allowed ranges (popularity: -1.0 to 1.0, all others ≥ 0)
Return only the adjusted ScoringWeights with no explanation."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(_logger, logging.WARNING),
    reraise=True,
)
def _invoke_weight_adjustment(structured_llm, prompt: str) -> ScoringWeights:
    """Retry-wrapped LLM call for weight adjustment. Separated for tenacity compatibility."""
    return structured_llm.invoke(prompt)


def _adjust_weights_from_feedback(
    feedback: str,
    profile: MoodProfile,
    base_weights: ScoringWeights,
) -> ScoringWeights:
    """Use LLM to translate critic feedback into adjusted ScoringWeights.

    Returns base_weights unchanged when feedback is empty (no LLM call).
    Falls back to base_weights gracefully on any LLM failure (e.g. Groq 400
    when the model returns arithmetic expressions instead of literal values).
    """
    if not feedback:
        return base_weights

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Copy .env.example to .env and add your Groq API key."
        )

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
    structured = llm.with_structured_output(ScoringWeights)

    prompt = _WEIGHT_ADJUSTMENT_PROMPT.format(
        genre_weight=base_weights.genre_weight,
        mood_weight=base_weights.mood_weight,
        energy_weight=base_weights.energy_weight,
        valence_weight=base_weights.valence_weight,
        danceability_weight=base_weights.danceability_weight,
        acousticness_weight=base_weights.acousticness_weight,
        popularity_weight=base_weights.popularity_weight,
        instrumentalness_weight=base_weights.instrumentalness_weight,
        speechiness_penalty=base_weights.speechiness_penalty,
        activity=profile.activity or "general",
        pop_pref=profile.popularity_preference,
        inst_pref=profile.instrumentalness_preference,
        feedback=feedback,
    )

    try:
        return _invoke_weight_adjustment(structured, prompt)
    except Exception as exc:
        # Groq sometimes returns arithmetic expressions (e.g. "0.2 * 2") in
        # structured output, producing a 400. Gracefully fall back to the
        # profile-derived base weights so the pipeline never crashes on retry.
        _logger.warning(
            "Weight adjustment LLM failed after retries: %s — using base weights.", exc
        )
        return base_weights


# ── Scoring formula ───────────────────────────────────────────────────────────


def score_and_rank(
    candidates: list[dict],
    profile: MoodProfile,
    critic_feedback: str = "",
) -> list[dict]:
    """Score and rank candidate songs against the MoodProfile.

    Returns list sorted descending by score (rounded to 4 dp).
    Each dict gets 'score' (float) and 'score_breakdown' (dict) added.

    Weight resolution:
      - No feedback (first pass): _derive_initial_weights(profile) — no LLM call.
      - With feedback (retry): _adjust_weights_from_feedback(...) — LLM adjusts.
    """
    if not candidates:
        return []

    base_weights = _derive_initial_weights(profile)
    weights = _adjust_weights_from_feedback(critic_feedback, profile, base_weights)

    memory = _load_user_memory()
    liked_artists = {a.lower() for a in memory.get("liked_artists", {})}
    disliked_artists = {a.lower() for a in memory.get("disliked_artists", {})}

    preferred_genres = {g.lower() for g in profile.preferred_genres}
    preferred_moods = {m.lower() for m in profile.mood_keywords}

    scored: list[dict] = []
    for song in candidates:
        score = 0.0
        breakdown: dict[str, float] = {}

        # Genre match
        g = weights.genre_weight if str(song.get("genre", "")).lower() in preferred_genres else 0.0
        score += g
        breakdown["genre"] = round(g, 4)

        # Mood match
        m = weights.mood_weight if str(song.get("mood", "")).lower() in preferred_moods else 0.0
        score += m
        breakdown["mood"] = round(m, 4)

        # Energy proximity
        energy_diff = abs(float(song.get("energy", 0.5)) - profile.target_energy)
        e = weights.energy_weight * (1.0 - energy_diff)
        score += e
        breakdown["energy"] = round(e, 4)

        # Valence proximity
        valence_diff = abs(float(song.get("valence", 0.5)) - profile.target_valence)
        v = weights.valence_weight * (1.0 - valence_diff)
        score += v
        breakdown["valence"] = round(v, 4)

        # Danceability (absolute, not proximity — high danceability generally desirable)
        d = weights.danceability_weight * float(song.get("danceability", 0.5))
        score += d
        breakdown["danceability"] = round(d, 4)

        # Acousticness proximity to instrumentalness_preference
        # For high inst_pref queries, acousticness IS a feature, not a nuisance
        ac = weights.acousticness_weight * float(song.get("acousticness", 0.3))
        score += ac
        breakdown["acousticness"] = round(ac, 4)

        # Popularity (positive or negative depending on popularity_weight sign)
        pop = weights.popularity_weight * (float(song.get("popularity", 50)) / 100.0)
        score += pop
        breakdown["popularity"] = round(pop, 4)

        # Instrumentalness bonus — rewards vocal-free tracks
        inst = weights.instrumentalness_weight * float(song.get("instrumentalness", 0.0))
        score += inst
        breakdown["instrumentalness"] = round(inst, 4)

        # Speechiness penalty — penalises heavy vocal / rap tracks
        speech_pen = weights.speechiness_penalty * float(song.get("speechiness", 0.0))
        score -= speech_pen
        breakdown["speechiness_penalty"] = round(-speech_pen, 4)

        # Session memory bonus/penalty
        artist_lower = str(song.get("artist", "")).lower()
        mem_bonus = 0.0
        if artist_lower in liked_artists:
            mem_bonus = 0.4
        elif artist_lower in disliked_artists:
            mem_bonus = -0.4
        score += mem_bonus
        breakdown["memory_bonus"] = round(mem_bonus, 4)

        scored.append({**song, "score": round(score, 4), "score_breakdown": breakdown})

    return sorted(scored, key=lambda x: x["score"], reverse=True)
