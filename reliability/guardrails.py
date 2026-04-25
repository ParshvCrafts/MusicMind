"""
Input and output guardrails for MusicMind.
UserQuery validates and sanitizes raw user input.
RecommendationOutput validates the final recommendations before rendering.
"""
from pydantic import BaseModel, Field, field_validator

_FORBIDDEN = ["ignore previous", "system:", "you are now", "jailbreak", "forget instructions"]


class UserQuery(BaseModel):
    text: str = Field(..., min_length=3, max_length=500)

    @field_validator("text")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        lower_v = v.lower()
        for phrase in _FORBIDDEN:
            if phrase in lower_v:
                raise ValueError(
                    f"Query contains disallowed content ('{phrase}'). "
                    "Please rephrase your music request."
                )
        return v.strip()


class RecommendationOutput(BaseModel):
    songs: list[dict]

    @field_validator("songs")
    @classmethod
    def validate_non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("Recommendation list cannot be empty")
        return v

    @field_validator("songs")
    @classmethod
    def validate_song_fields(cls, songs: list) -> list:
        required = ["title", "artist", "genre", "score", "explanation"]
        for song in songs:
            missing = [f for f in required if f not in song]
            if missing:
                raise ValueError(f"Song missing required fields: {missing}")
        return songs
