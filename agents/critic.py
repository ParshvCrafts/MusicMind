"""
Critic Agent — Node 5 of the MusicMind LangGraph pipeline.
Self-evaluates the top-5 recommendations using a structured LLM verdict.
Returns (feedback_string, passed_bool). If passed=False and retry_count < 2,
the Orchestrator loops back to the Scorer with the feedback.

Implementation notes:
  - Uses llama-3.3-70b-versatile (same as explainer) because the 8B model
    occasionally produces malformed JSON for complex structured schemas,
    causing Groq 400 errors.
  - tenacity retries 3 times with exponential backoff before falling back
    to a graceful pass (so the pipeline never crashes on transient API errors).
"""
import logging
import os

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.intent_parser import MoodProfile

_logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────


class CriticVerdict(BaseModel):
    passed: bool = Field(description="True if overall_score >= 6.5")
    overall_score: float = Field(description="Overall quality score 0–10")
    energy_alignment: float = Field(description="0–10: do song energy levels match requested energy?")
    diversity_score: float = Field(
        description="0–10: penalize if all recommendations are from the same genre"
    )
    explanation_quality: float = Field(
        description="0–10: are explanations specific and grounded, not generic?"
    )
    feedback: str = Field(
        description="Actionable feedback for the scorer if passed=False. Empty string if passed=True."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific problems found",
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_CRITIC_PROMPT = """\
You are a critical evaluator of a music recommendation AI system.

User's original request: "{query}"
Inferred profile: activity={activity}, energy_target={energy:.2f}, valence_target={valence:.2f}

Top 5 recommendations:
{recs_text}

Explanations given:
{explanations_text}

Evaluate these recommendations on 4 dimensions (score each 0–10):

1. ENERGY ALIGNMENT: Do the songs' actual energy values match the user's energy target of {energy:.2f}?
   Songs more than 0.3 away from target should be penalized.

2. DIVERSITY: Are there at least 2 different genres represented? If all 5 are the same genre, score below 4.

3. EXPLANATION QUALITY: Are the explanations specific and grounded in music features?
   Generic phrases like "this matches your vibe" score below 3.

4. OVERALL COHERENCE: Would a music expert agree these songs fit the user's request?

Pass threshold: overall_score >= 6.5. Be a rigorous critic.
If failed, write specific actionable feedback for the scorer, for example:
  "Increase instrumentalness_weight — songs have too many vocals for focus context."
  "Reduce energy_weight — songs are too energetic for a sleep/relaxation context."
  "Lower genre_weight to 0.8 to allow more diversity across genres."
  "Reduce popularity_weight — user wants obscure tracks, not mainstream hits."

If acousticness or instrumentalness are central to the request, evaluate those explicitly.
"""


# ── Prompt builder ────────────────────────────────────────────────────────────


def _build_critic_prompt(
    songs: list[dict], profile: "MoodProfile", explanations: list[str]
) -> str:
    """Build the critic evaluation prompt string."""
    recs_text = "\n".join(
        f"{i + 1}. {s.get('title', '?')} by {s.get('artist', '?')} "
        f"| genre={s.get('genre', '?')} | energy={float(s.get('energy', 0.5)):.2f} "
        f"| valence={float(s.get('valence', 0.5)):.2f} "
        f"| acousticness={float(s.get('acousticness', 0.3)):.2f} "
        f"| instrumentalness={float(s.get('instrumentalness', 0.0)):.2f} "
        f"| speechiness={float(s.get('speechiness', 0.0)):.2f} "
        f"| score={s.get('score', '?')}"
        for i, s in enumerate(songs)
    )
    explanations_text = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(explanations))
    return _CRITIC_PROMPT.format(
        query=profile.raw_query,
        activity=profile.activity or "general",
        energy=profile.target_energy,
        valence=profile.target_valence,
        recs_text=recs_text,
        explanations_text=explanations_text,
    )


# ── Retry-wrapped LLM invocation ─────────────────────────────────────────────


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(_logger, logging.WARNING),
    reraise=True,
)
def _invoke_critic(structured_llm, prompt: str) -> CriticVerdict:
    """Invoke the structured critic LLM with retry on transient failures."""
    return structured_llm.invoke(prompt)


# ── Public API ────────────────────────────────────────────────────────────────


def evaluate_recommendations(
    songs: list[dict],
    profile: MoodProfile,
    explanations: list[str],
) -> tuple[str, bool]:
    """Evaluate the top-5 recommendations.

    Returns (feedback, passed) where feedback is actionable for the scorer.
    If passed=True, feedback is an empty string.

    Uses llama-3.3-70b-versatile for reliable structured output.
    Falls back to (feedback="", passed=True) if all retries are exhausted,
    so the pipeline never crashes on transient API errors.

    Raises ValueError if GROQ_API_KEY is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Copy .env.example to .env and add your Groq API key."
        )

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    structured = llm.with_structured_output(CriticVerdict)

    full_prompt = _build_critic_prompt(songs, profile, explanations)

    try:
        verdict: CriticVerdict = _invoke_critic(structured, full_prompt)
        return verdict.feedback if not verdict.passed else "", verdict.passed
    except Exception as exc:
        _logger.warning(
            "Critic evaluation failed after all retries: %s — defaulting to pass.", exc
        )
        # Graceful fallback: rather than crashing the pipeline, pass through.
        # The hard-cap at retry_count >= 2 provides a safety net.
        return "", True


def evaluate_recommendations_ab(
    songs: list[dict],
    profile: "MoodProfile",
    explanations: list[str],
) -> dict:
    """Run two critics sequentially and return both verdicts.

    Returns dict with keys "fast" and "quality", each containing:
      model, overall_score, energy_alignment, diversity_score,
      explanation_quality, passed, feedback, issues
    Falls back to error dict (passed=True) if a model call fails.

    Raises ValueError if GROQ_API_KEY is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    full_prompt = _build_critic_prompt(songs, profile, explanations)
    results: dict = {}

    for key, model_id in [("fast", "llama-3.1-8b-instant"), ("quality", "llama-3.3-70b-versatile")]:
        llm = ChatGroq(model=model_id, api_key=api_key)
        structured = llm.with_structured_output(CriticVerdict)
        try:
            verdict: CriticVerdict = _invoke_critic(structured, full_prompt)
            results[key] = {
                "model": model_id,
                "overall_score": verdict.overall_score,
                "energy_alignment": verdict.energy_alignment,
                "diversity_score": verdict.diversity_score,
                "explanation_quality": verdict.explanation_quality,
                "passed": verdict.passed,
                "feedback": verdict.feedback,
                "issues": verdict.issues,
            }
        except Exception as exc:
            _logger.warning("A/B critic %s failed: %s", key, exc)
            results[key] = {
                "model": model_id,
                "overall_score": 0.0,
                "energy_alignment": 0.0,
                "diversity_score": 0.0,
                "explanation_quality": 0.0,
                "passed": True,  # graceful fallback
                "feedback": "",
                "issues": [str(exc)],
                "error": True,
            }

    return results
