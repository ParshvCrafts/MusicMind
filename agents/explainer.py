"""
Explainer Agent — Node 4 of the MusicMind LangGraph pipeline.
Generates RAG-grounded explanations for all top-5 songs in a SINGLE batch LLM call.
Uses llama-3.3-70b-versatile for highest-quality, user-facing output.
"""
import os
import re

from langchain_groq import ChatGroq

from agents.intent_parser import MoodProfile

# ── Batch prompt ──────────────────────────────────────────────────────────────

_BATCH_EXPLAINER_PROMPT = """\
You are a music curator writing listener-friendly explanations for a music recommendation system.
For each of the songs listed below, write exactly 2 sentences explaining why it was selected
as the best available match for this listener's context.

User's request: "{query}"
Inferred context: activity={activity}, energy_target={energy:.2f}, valence_target={valence:.2f},
instrumentalness_preference={inst_pref:.2f} (0=vocals fine, 1=purely instrumental)

Music Knowledge (draw from this to ground your explanations):
{knowledge_chunks}

Songs to explain:
{songs_block}

Rules for every explanation:
- Frame the explanation around what the song OFFERS for this context.
- Avoid any negative framing: never say a song "doesn't fit", is "less than ideal", or use dismissive language.
- Mention the song title and at least one concrete audio feature value (energy, BPM, acousticness, instrumentalness, etc.).
- No generic phrases: "matches your vibe", "aligns with your needs", "fits your criteria", "based on your preferences".
- If instrumentalness_preference > 0.7: explicitly note the song's instrumentalness and speechiness values.
- Write exactly 2 sentences per song — no more, no less.

Format your response as exactly {n} numbered items:
1. [Two sentences for Song 1]
2. [Two sentences for Song 2]
3. [Two sentences for Song 3]
4. [Two sentences for Song 4]
5. [Two sentences for Song 5]
"""

_ARTIST_NOTE_DEFAULT = ""
_ARTIST_NOTE_TEMPLATE = (
    " Note: user asked for {requested_artist}'s style — explain how this captures "
    "a similar energy or genre feel without claiming it is {requested_artist}'s music."
)


def _build_artist_note(raw_query: str, song_artist: str) -> str:
    query_lower = raw_query.lower()
    for keyword in ("play ", "songs by ", "tracks by ", "music by ", "like "):
        if keyword in query_lower:
            idx = query_lower.find(keyword) + len(keyword)
            requested = raw_query[idx:].split()[0].rstrip(",.!?")
            if requested.lower() not in song_artist.lower():
                return _ARTIST_NOTE_TEMPLATE.format(requested_artist=requested)
    return _ARTIST_NOTE_DEFAULT


def _build_songs_block(top_songs: list[dict], profile: MoodProfile) -> str:
    """Format all songs into a numbered block for the batch prompt."""
    lines = []
    for i, song in enumerate(top_songs, 1):
        artist_note = _build_artist_note(profile.raw_query, song.get("artist", ""))
        lines.append(
            f"Song {i}: \"{song.get('title', 'Unknown')}\" by {song.get('artist', 'Unknown')}\n"
            f"  Genre: {song.get('genre', '')} ({song.get('subgenre', '')}), "
            f"Mood: {song.get('mood', '')}\n"
            f"  Energy: {float(song.get('energy', 0.5)):.2f}, "
            f"Valence: {float(song.get('valence', 0.5)):.2f}, "
            f"BPM: {float(song.get('bpm', 120.0)):.0f}\n"
            f"  Acousticness: {float(song.get('acousticness', 0.3)):.2f}, "
            f"Instrumentalness: {float(song.get('instrumentalness', 0.0)):.2f}, "
            f"Speechiness: {float(song.get('speechiness', 0.0)):.3f}\n"
            f"  Score breakdown: {song.get('score_breakdown', {})}"
            + (f"\n  {artist_note}" if artist_note else "")
        )
    return "\n\n".join(lines)


def _parse_batch_explanations(text: str, n: int = 5) -> list[str]:
    """Parse a numbered list from a batch LLM response. Pads to n if needed."""
    _FALLBACK = "This track was selected as the best available match for your request based on its audio characteristics."

    if not text.strip():
        return [_FALLBACK] * n

    # Split on lines starting with "1.", "2.", etc.
    parts = re.split(r"(?m)^\s*\d+\.\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= n:
        return parts[:n]

    # Fallback: split on double newlines
    fallback_parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(fallback_parts) >= n:
        return fallback_parts[:n]

    # Pad to n with fallback string
    while len(parts) < n:
        parts.append(parts[0] if parts else _FALLBACK)
    return parts[:n]


def generate_explanations(
    top_songs: list[dict],
    profile: MoodProfile,
    knowledge_chunks: list[str],
) -> list[str]:
    """Generate explanations for all songs in a single batch LLM call.

    Returns list of explanation strings in the same order as top_songs.
    Raises ValueError if GROQ_API_KEY is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Copy .env.example to .env and add your Groq API key."
        )

    if not top_songs:
        return []

    n = len(top_songs)
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    knowledge_text = (
        "\n\n".join(knowledge_chunks) if knowledge_chunks else "No additional context available."
    )

    prompt = _BATCH_EXPLAINER_PROMPT.format(
        query=profile.raw_query,
        activity=profile.activity or "general listening",
        energy=profile.target_energy,
        valence=profile.target_valence,
        inst_pref=profile.instrumentalness_preference,
        knowledge_chunks=knowledge_text,
        songs_block=_build_songs_block(top_songs, profile),
        n=n,
    )

    response = llm.invoke(prompt)
    return _parse_batch_explanations(response.content.strip(), n=n)
