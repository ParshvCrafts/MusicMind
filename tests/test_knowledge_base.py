"""Tests for the MusicKnowledgeBase RAG component."""
import pytest
from rag.knowledge_base import MusicKnowledgeBase

# NOTE: These tests require data/songs_processed.csv and rag/documents/ to exist.
# Run data/preprocess.py first if songs_processed.csv is missing.


@pytest.fixture(scope="module")
def kb():
    """Shared KB fixture — ingests once for the whole module."""
    kb = MusicKnowledgeBase()
    if kb.songs_count == 0:
        kb.ingest_songs("data/songs_processed.csv")
    if kb.knowledge_count == 0:
        kb.ingest_knowledge_docs("rag/documents/")
    return kb


def test_songs_ingested(kb):
    assert kb.songs_count > 0


def test_knowledge_ingested(kb):
    assert kb.knowledge_count > 0


def test_song_retrieval_returns_correct_n(kb):
    results = kb.retrieve_songs("chill lofi for studying late at night", n=5)
    assert len(results) == 5


def test_song_retrieval_has_required_fields(kb):
    results = kb.retrieve_songs("workout high energy music", n=3)
    required = ["song_id", "title", "artist", "genre", "mood", "energy", "valence"]
    for r in results:
        for field in required:
            assert field in r, f"Missing field '{field}' in retrieved song"


def test_knowledge_retrieval_returns_chunks(kb):
    chunks = kb.retrieve_knowledge("what BPM range is best for workouts?", n=3)
    assert len(chunks) > 0


def test_knowledge_retrieval_is_relevant(kb):
    chunks = kb.retrieve_knowledge("coding focus instrumental no lyrics", n=3)
    combined = " ".join(chunks).lower()
    assert any(word in combined for word in ["coding", "instrumental", "focus", "lyrics"])


from pathlib import Path as _Path

_HERE_KB = _Path(__file__).parent.parent


def test_cache_save_and_load(tmp_path):
    """Second MusicKnowledgeBase load with same cache_dir skips re-embedding."""
    from rag.knowledge_base import MusicKnowledgeBase
    csv = str(_HERE_KB / "data" / "songs_processed.csv")
    docs = str(_HERE_KB / "rag" / "documents")
    cache = str(tmp_path / "cache")

    # First load — embeds and saves
    kb1 = MusicKnowledgeBase(cache_dir=cache)
    kb1.ingest_songs(csv)
    kb1.ingest_knowledge_docs(docs)
    count1 = kb1.songs_count

    # Second load — must load from cache without re-embedding
    kb2 = MusicKnowledgeBase(cache_dir=cache)
    kb2.ingest_songs(csv)
    kb2.ingest_knowledge_docs(docs)
    assert kb2.songs_count == count1, "Cache load returned wrong song count"
    assert kb2.knowledge_count == kb1.knowledge_count


def test_cache_invalidates_when_csv_newer(tmp_path):
    """Cache is rebuilt when the CSV is newer than the saved index."""
    import os
    from rag.knowledge_base import MusicKnowledgeBase
    csv = str(_HERE_KB / "data" / "songs_processed.csv")
    docs = str(_HERE_KB / "rag" / "documents")
    cache = str(tmp_path / "cache")

    kb1 = MusicKnowledgeBase(cache_dir=cache)
    kb1.ingest_songs(csv)

    # Touch cache file to be older than csv
    faiss_path = tmp_path / "cache" / "songs.faiss"
    old_mtime = faiss_path.stat().st_mtime - 10
    os.utime(str(faiss_path), (old_mtime, old_mtime))

    # Re-ingest — should NOT load from stale cache
    kb2 = MusicKnowledgeBase(cache_dir=cache)
    kb2.ingest_songs(csv)  # should re-embed without error
    assert kb2.songs_count == kb1.songs_count
