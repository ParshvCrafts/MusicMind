"""
FAISS-backed knowledge base for MusicMind.
Manages two indexes: 'songs' (900 Spotify tracks) and 'knowledge' (5 domain docs).

Implementation note: Originally designed for ChromaDB, but switched to FAISS
(faiss-cpu==1.13.2) for the in-process vector store because chromadb's underlying
hnswlib extension consistently crashes on Windows under pytest due to a heap
incompatibility with numpy's scipy-openblas64 BLAS (loaded by the fugue pytest
plugin before any test code runs). FAISS avoids this entirely.

The public API is identical to a ChromaDB-backed implementation:
  - ingest_songs(csv_path)         → embed + index 900 songs
  - ingest_knowledge_docs(path)    → chunk + embed .txt files
  - retrieve_songs(query, n)       → top-n semantically similar songs (list[dict])
  - retrieve_knowledge(query, n)   → top-n relevant text chunks (list[str])
  - songs_count / knowledge_count  → int properties

For production swap: replace FaissCollection with a PersistentClient-backed
collection and the same retrieval logic.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"  # Pinned — do not change mid-project
_EMBED_DIM = 384  # Dimension of all-MiniLM-L6-v2 output


@dataclass
class _FaissCollection:
    """Minimal FAISS wrapper that mimics a ChromaDB collection."""

    _index: Optional[faiss.Index] = field(default=None, init=False)
    _documents: list[str] = field(default_factory=list, init=False)
    _metadatas: list[dict] = field(default_factory=list, init=False)
    _ids: list[str] = field(default_factory=list, init=False)

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """Add or update records. Duplicate IDs are skipped for simplicity."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(_EMBED_DIM)  # inner product (cosine on normed vecs)
        existing = set(self._ids)
        for i, (doc_id, doc, emb) in enumerate(zip(ids, documents, embeddings)):
            if doc_id in existing:
                continue  # skip duplicates
            meta = metadatas[i] if metadatas else {}
            vec = np.array(emb, dtype=np.float32)
            faiss.normalize_L2(vec.reshape(1, -1))  # normalize for cosine similarity
            self._index.add(vec.reshape(1, -1))
            self._ids.append(doc_id)
            self._documents.append(doc)
            self._metadatas.append(meta)

    def query(self, query_embeddings: list[list[float]], n_results: int) -> dict:
        """Return top-n results for each query embedding."""
        if self._index is None or self._index.ntotal == 0:
            return {"metadatas": [[]], "documents": [[]]}

        q = np.array(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(q)
        n = min(n_results, self._index.ntotal)
        _, indices = self._index.search(q, n)

        all_metas = []
        all_docs = []
        for row in indices:
            all_metas.append([self._metadatas[i] for i in row if i >= 0])
            all_docs.append([self._documents[i] for i in row if i >= 0])

        return {"metadatas": all_metas, "documents": all_docs}

    def count(self) -> int:
        return len(self._ids)


class MusicKnowledgeBase:
    def __init__(self, cache_dir: str = "") -> None:
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self._songs = _FaissCollection()
        self._knowledge = _FaissCollection()
        self._cache_dir = Path(cache_dir) if cache_dir else None

    def _save_collection_cache(self, collection: "_FaissCollection", name: str) -> None:
        """Write FAISS index + metadata to cache_dir/{name}.faiss and {name}_meta.pkl."""
        if self._cache_dir is None or collection._index is None:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(collection._index, str(self._cache_dir / f"{name}.faiss"))
        with open(self._cache_dir / f"{name}_meta.pkl", "wb") as fh:
            pickle.dump(
                {"ids": collection._ids, "documents": collection._documents,
                 "metadatas": collection._metadatas},
                fh,
            )

    def _load_collection_cache(
        self, collection: "_FaissCollection", name: str, source_path: str
    ) -> bool:
        """Load cache if it exists and is newer than source_path. Returns True on success."""
        if self._cache_dir is None:
            return False
        faiss_path = self._cache_dir / f"{name}.faiss"
        meta_path = self._cache_dir / f"{name}_meta.pkl"
        if not faiss_path.exists() or not meta_path.exists():
            return False
        # Invalidate if source file is newer than cached index
        try:
            src_mtime = Path(source_path).stat().st_mtime
            cache_mtime = faiss_path.stat().st_mtime
            if src_mtime > cache_mtime:
                return False
        except OSError:
            return False
        collection._index = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as fh:
            data = pickle.load(fh)
        collection._ids = data["ids"]
        collection._documents = data["documents"]
        collection._metadatas = data["metadatas"]
        return True

    def ingest_songs(self, csv_path: str) -> None:
        """Embed and index all songs from songs_processed.csv."""
        if self._load_collection_cache(self._songs, "songs", csv_path):
            print(f"Loaded {self.songs_count} songs from cache.")
            return

        df = pd.read_csv(csv_path)
        texts = [
            (
                f"{row['title']} by {row['artist']}. "
                f"Genre: {row['genre']} ({row.get('subgenre', '')})."
                f" Mood: {row['mood']}. Energy: {float(row['energy']):.2f}."
                f" BPM: {float(row['bpm']):.0f}. Valence: {float(row['valence']):.2f}."
                f" Danceability: {float(row['danceability']):.2f}."
                f" Acousticness: {float(row['acousticness']):.2f}."
                f" Instrumentalness: {float(row.get('instrumentalness', 0.0)):.2f}."
                f" Popularity: {int(row['popularity'])}."
            )
            for _, row in df.iterrows()
        ]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        self._songs.upsert(
            ids=[str(row["song_id"]) for _, row in df.iterrows()],
            documents=texts,
            embeddings=embeddings,
            metadatas=df.to_dict(orient="records"),
        )
        self._save_collection_cache(self._songs, "songs")
        print(f"Ingested {len(df)} songs into FAISS index.")

    def ingest_knowledge_docs(self, docs_path: str) -> None:
        """Chunk and embed all .txt files in docs_path into the knowledge index."""
        doc_dir = Path(docs_path)
        sentinel = str(doc_dir)
        if self._load_collection_cache(self._knowledge, "knowledge", sentinel):
            print(f"Loaded {self.knowledge_count} knowledge chunks from cache.")
            return
        for txt_file in sorted(doc_dir.glob("*.txt")):
            content = txt_file.read_text(encoding="utf-8")
            chunks = self._chunk_text(content, chunk_size=200, overlap=50)
            embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()
            self._knowledge.upsert(
                ids=[f"{txt_file.stem}_{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=embeddings,
            )
        self._save_collection_cache(self._knowledge, "knowledge")
        print(f"Ingested knowledge documents from {docs_path}.")

    def retrieve_songs(self, query: str, n: int = 20) -> list[dict]:
        """Return top-n songs most semantically similar to query."""
        q_emb = self.embedder.encode([query], show_progress_bar=False).tolist()
        results = self._songs.query(query_embeddings=q_emb, n_results=n)
        return results["metadatas"][0]

    def retrieve_knowledge(self, query: str, n: int = 3) -> list[str]:
        """Return top-n knowledge chunks most relevant to query."""
        q_emb = self.embedder.encode([query], show_progress_bar=False).tolist()
        results = self._knowledge.query(query_embeddings=q_emb, n_results=n)
        return results["documents"][0]

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i : i + chunk_size]))
            i += chunk_size - overlap
        return chunks

    @property
    def songs_count(self) -> int:
        return self._songs.count()

    @property
    def knowledge_count(self) -> int:
        return self._knowledge.count()
