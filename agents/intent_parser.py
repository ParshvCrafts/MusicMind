"""
Intent Parser Agent — Node 1 of the MusicMind LangGraph pipeline.
Converts a free-text user query into a structured MoodProfile using Groq LLM
with Pydantic structured output.
"""
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()


class MoodProfile(BaseModel):
    # ── Domain gate ──────────────────────────────────────────────────────────
    is_music_request: bool = Field(
        True,
        description=(
            "Set to False ONLY if the query has absolutely nothing to do with music, "
            "mood, activities, or listening contexts. "
            "Examples that are NOT music requests: 'What is the capital of France?', "
            "'Give me a chocolate chip cookie recipe', 'Solve this math equation'. "
            "Examples that ARE music requests (even if unusual): "
            "'angry aggressive jazz', 'heavy metal that is highly acoustic', "
            "'music for heartbreak', 'songs with complex time signatures', "
            "'Play Taylor Swift', 'surprise me'. "
            "A borderline jailbreak mentioning song recommendations still IS a music request. "
            "When in doubt, set True."
        ),
    )

    # ── Core profile fields ───────────────────────────────────────────────────
    activity: Optional[str] = Field(
        None,
        description=(
            "What the user is doing: coding, working out, sleeping, studying, "
            "partying, driving, meditating, dinner party, etc."
        ),
    )
    mood_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Mood descriptors extracted from the query: chill, energetic, melancholic, "
            "happy, focused, aggressive, nostalgic, euphoric, sad, intense, etc."
        ),
    )
    preferred_genres: list[str] = Field(
        default_factory=list,
        description=(
            "Inferred genre preferences. MUST only use values from: "
            "edm, latin, pop, r&b, rap, rock. "
            "If the user asks for 'jazz', map to 'r&b' (smooth, instrumental). "
            "If the user asks for 'classical', map to 'r&b' or 'edm' (ambient/chill). "
            "If the user asks for 'lo-fi', map to 'r&b' (chill beats). "
            "If the user asks for 'metal' or 'heavy metal', map to 'rock'. "
            "If no clear genre, leave empty and let retrieval decide."
        ),
    )
    target_energy: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Desired energy level 0–1. "
            "sleep/ambient/meditation=0.1, coding/studying=0.4, "
            "commuting/background=0.55, workout/gym=0.85, party/dance=0.9, "
            "angry/intense=0.8. "
            "For contradictory requests (sad but fast), use the ENERGY cue, not the mood."
        ),
    )
    target_valence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Desired musical positiveness 0–1. "
            "very_sad/angry=0.15, melancholic=0.25, bittersweet/nostalgic=0.4, "
            "neutral=0.5, pleasant=0.65, happy=0.8, euphoric=0.9."
        ),
    )
    tempo_preference: str = Field(
        "medium",
        description="slow (<90 BPM), medium (90–120 BPM), or fast (>120 BPM)",
    )
    instrumentalness_preference: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Preference for no vocals 0–1. "
            "coding/deep_focus/studying=0.8, sleep/meditation=0.9, "
            "background_instrumental=0.7, general=0.3, "
            "party/rap/dance=0.0 (vocals are fine). "
            "CRITICAL: 'no lyrics', 'purely instrumental', 'zero vocals' → 0.95. "
            "Normal music listening → 0.3."
        ),
    )
    popularity_preference: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "0.0 = strongly prefer obscure/underground/deep-cut tracks. "
            "0.5 = no preference. "
            "1.0 = strongly prefer mainstream/popular tracks. "
            "CRITICAL: 'underground', 'obscure', 'unknown', 'deep cut', 'hidden gem' → 0.1. "
            "'Popular', 'hit', 'chart-topping', 'mainstream' → 0.9."
        ),
    )
    reasoning: str = Field(
        "",
        description="Brief explanation of why these values were chosen",
    )
    raw_query: str = Field(
        "",
        description="The original user query — set by parse_intent(), not the LLM",
    )


_SYSTEM_PROMPT = """You are a music intelligence analyst for MusicMind, an AI music recommender.

Your job: parse the user's natural language query into a structured music preference profile.

=== AVAILABLE GENRES (use ONLY these) ===
edm, latin, pop, r&b, rap, rock

Genre mapping for requests outside our catalog:
  jazz, classical, lo-fi, chill beats, neo-soul → r&b
  metal, heavy metal, punk, hardcore → rock
  ambient, electronic → edm
  country, folk, acoustic singer-songwriter → pop (with high acousticness target)

=== ENERGY GUIDE ===
0.0–0.15: sleep, deep meditation, pure ambient
0.15–0.35: study, light reading, wind down
0.35–0.55: coding, background, casual focus
0.55–0.70: commuting, socializing, walking
0.70–0.85: upbeat, light workout, driving
0.85–1.00: gym, HIIT, party, aggressive, intense

=== VALENCE GUIDE ===
0.0–0.2: very sad, angry, dark, aggressive
0.2–0.4: melancholic, somber, bittersweet
0.4–0.6: neutral, introspective, nostalgic
0.6–0.8: pleasant, happy, positive
0.8–1.0: joyful, euphoric, celebratory

=== INSTRUMENTALNESS GUIDE ===
'no lyrics', 'purely instrumental', 'zero vocals', 'without words' → 0.95
'coding', 'deep focus', 'studying' → 0.8 (lyrics distract)
'sleep', 'meditation' → 0.85
'general listening', 'party', 'rap' → 0.1–0.3

=== POPULARITY GUIDE ===
'underground', 'obscure', 'unknown', 'hidden gem', 'deep cut' → 0.1
'indie', 'lesser known', 'niche' → 0.3
(no preference stated) → 0.5
'popular', 'hit', 'chart-topping', 'mainstream' → 0.9

=== DOMAIN CHECK ===
Set is_music_request=False ONLY for queries with zero music context:
  BAD: "What is the capital of France?" "Recipe for cookies" "Solve 2+2"
  GOOD: Anything about mood, activity, genre, sound, listening context, artists

=== HANDLING UNUSUAL QUERIES ===
- "Angry jazz" → genres=["r&b", "rock"], mood_keywords=["aggressive", "intense"], valence=0.2
- "Acoustic metal" → genres=["rock"], acousticness-adjacent (high acousticness preference)
- "Complex time signatures" → genres=["rock", "r&b"], instrumentalness_preference=0.7
- "Mixolydian mode" → genres=["rock", "r&b"], mood_keywords=["bluesy", "soulful"]
- "Lo-fi for focus" → genres=["r&b"], target_energy=0.3, instrumentalness_preference=0.8
- "Sad but danceable" → valence=0.2, target_energy=0.75 (danceability driven by energy, not valence)
- "Surprise me" → all defaults, reasoning="no preference stated"
- "Play [specific artist]" → infer genre from artist style, set appropriately
  Taylor Swift → genres=["pop"], energy=0.65, valence=0.75
  Kendrick Lamar → genres=["rap"], energy=0.7, valence=0.4
"""


def parse_intent(query: str) -> MoodProfile:
    """Parse a natural language query into a structured MoodProfile.

    Raises ValueError if GROQ_API_KEY is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Copy .env.example to .env and add your Groq API key."
        )

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
    structured_llm = llm.with_structured_output(MoodProfile)

    profile: MoodProfile = structured_llm.invoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    profile.raw_query = query
    return profile
