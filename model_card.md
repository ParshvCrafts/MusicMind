# Model Card: MusicMind

> Extended from VibeFinder 1.0 (CodePath AI 110 Module 3)

---

## 1. Model / System Name

**MusicMind v2.0**, Agentic Music Recommendation System

---

## 2. Intended Use

MusicMind is an end-to-end AI system that accepts a natural language music request
("I need focus music for late-night coding") and returns ranked song recommendations
with RAG-grounded explanations. It is designed to demonstrate applied AI engineering
concepts, agentic workflows, retrieval-augmented generation, and reliability
evaluation, in the context of CodePath AI 110.

**In-scope:** personal music discovery, mood-based playlist seeding, study/focus/workout
queries against the 1,050-track Spotify-sourced catalog.

**Out-of-scope:** production music streaming, copyright enforcement, real-time Spotify
integration, or recommendation at scale.  The system is not intended to replace a
licensed music platform.

---

## 3. How the System Works

MusicMind uses a **5-node LangGraph agentic pipeline**:

| Node | Name | Model / Method | Output |
|------|------|----------------|--------|
| 1 | Intent Parser | Groq llama-3.1-8b-instant | `MoodProfile` (energy, valence, genres, activity) |
| 2 | FAISS Retriever | Sentence-transformer embeddings | 20 candidate songs + 3 knowledge chunks |
| 3 | Scorer | Weighted formula (deterministic first pass; LLM-adjusted on retry) | Top-10 ranked songs with `score_breakdown` |
| 4 | Explainer | Groq llama-3.3-70b-versatile + RAG | Plain-English explanation per song |
| 5 | Critic | Groq llama-3.3-70b-versatile | Pass/fail verdict + actionable feedback |

**Conditional retry loop:** if the Critic scores below 6.5/10, it writes structured
feedback and the Scorer re-runs with LLM-adjusted weights.  Hard cap at 2 retries
prevents infinite loops.

**Scoring formula (Node 3):**
```
score = genre_match × genre_w
      + mood_match  × mood_w
      + (1 − |energy − target|) × energy_w
      + (1 − |valence − target|) × valence_w
      + danceability × dance_w
      + acousticness × acoustic_w
      + (popularity / 100) × pop_w
      + instrumentalness × inst_w
      − speechiness × speech_pen
      + memory_bonus          # ±0.4 for liked/disliked artists
```

All weights are derived deterministically from the `MoodProfile` on the first pass.
Only on retry does an LLM adjust them based on Critic feedback.

---

## 4. Training Data / Dataset

- **Source:** Spotify Tracks dataset (32,000 rows, Hugging Face).
- **Curated sample:** 1,050 songs, top 150 by popularity per genre (edm, latin, pop,
  r&b, rap, rock) plus 150 high-instrumentalness tracks.
- **Features used:** energy, valence, danceability, acousticness, instrumentalness,
  speechiness, tempo (BPM), popularity, genre, derived mood label.
- **Mood labels:** derived algorithmically from audio features, not human-labeled.
- **No personally identifiable information** is contained in the dataset.

---

## 5. Strengths

- **Transparency:** every recommendation ships with a `score_breakdown` showing exactly
  how many points each audio feature contributed.  Users and developers can audit the
  reasoning without opening a black box.
- **Self-correction:** the Critic agent catches quality failures and drives targeted
  weight adjustments, improving recommendations without human intervention.
- **Guardrail coverage:** prompt-injection phrases are blocked before reaching any LLM;
  output is validated by Pydantic before rendering.  The pipeline never silently
  returns malformed data.
- **Speed on clean queries:** the first scoring pass is purely deterministic, no LLM
  call, so latency is dominated by the Explainer and Critic (~5–10 s total).
- **Reproducible:** 98 unit tests, all mocked, run without any API key in ~97 s.

---

## 6. Limitations and Bias

### Dataset biases

1. **Popularity-skewed sampling:** the catalog was built by selecting the top 150 songs
   per genre by Spotify popularity.  Obscure or underground artists are systematically
   absent, even when the user explicitly requests them.

2. **6-genre ceiling:** all music is bucketed into `edm, latin, pop, r&b, rap, rock,
   instrumental`.  Classical, jazz, metal, country, folk, K-pop, and world music users
   receive approximate mappings that may feel semantically wrong.

3. **Western/English-language bias:** every song in the dataset has an English-language
   title and fits Western genre conventions.  Non-Western music traditions are not
   represented.

4. **Mood labels are algorithmic proxies:** labels like "energetic" or "melancholic"
   were derived from audio features, not human judgment.  Cultural associations between
   audio features and mood vary significantly across listeners.

### Algorithmic biases

5. **Genre dominance:** the genre match term contributes up to +2.0 points (the highest
   single term), creating a filter bubble.  Users who under-specify genre receive more
   diverse but less precise results; users who over-specify get genre-locked lists.

6. **Popularity default bonus:** the default `popularity_weight = +0.3` means mainstream
   hits have a built-in scoring advantage, reinforcing a rich-get-richer effect unless
   the user explicitly requests underground music.

7. **Instrumental catalog gap:** the instrumental bucket was built from EDM tracks with
   low speechiness, not calm lo-fi or classical.  Focus/coding queries requesting
   "instrumental" receive high-energy results that the Critic correctly flags, but the
   2-retry cap limits how much can be corrected.

### Potential misuse

The system processes free-text queries, which means a user could ask for music to
accompany harmful activities.  The input guardrail blocks prompt-injection phrases but
does not classify query intent beyond the music domain.  Out-of-scope queries (e.g.,
trivia, instructions) are caught by the Intent Parser's domain check and rejected with
an explanatory message before any song is retrieved.

---

## 7. Evaluation

### Unit tests

98 tests across 10 test files, all passing, all mocked (no API calls required).  Tests
cover: guardrails, intent parsing, FAISS retrieval, scoring formula, explainer batching,
critic verdict schema, orchestrator state transitions, and the weight-adjustment
fallback path.

```
pytest tests/ -v
# 98 passed, 3 warnings (deprecation from FAISS SWIG bindings) in ~97 s
```

### Eval harness

6 end-to-end test cases with expected energy / valence ranges and minimum quality
thresholds (Critic score ≥ 6.5):

| Case | Query | Expected | Status |
|------|-------|----------|--------|
| 1 | late-night coding focus | energy 0.3–0.55, instrumentalness ≥ 0.3 | PASS |
| 2 | high-energy gym workout | energy ≥ 0.7, genre=rock/edm | PASS |
| 3 | melancholic rainy day | valence ≤ 0.4, energy ≤ 0.5 | PASS |
| 4 | dinner party background | energy 0.5–0.8, valence ≥ 0.5 | PASS |
| 5 | highway driving indie | energy 0.55–0.80 | FAIL* |
| 6 | purely instrumental focus | instrumentalness ≥ 0.7 | FAIL* |

*Cases 5–6 fail due to dataset coverage gaps (no indie/highway genre, EDM-skewed
instrumental bucket), not pipeline logic errors.  The Critic correctly identifies the
mismatch but cannot improve beyond 2 retries.

**Pass rate: 4/6 (66.7%)**

### What surprised me during reliability testing

- The **weight-adjustment LLM** (Node 3 retry) occasionally returned arithmetic
  expressions like `"0.2 * 2"` instead of literal float values in structured output.
  This caused a Groq 400 error that silently failed until the retry decorator and
  graceful fallback were added.

- Increasing `instrumentalness_weight` to 4.0 had **no effect** when the FAISS pool
  contained no truly instrumental candidates, a root-cause vs. symptom confusion that
  took two debugging sessions to fully diagnose.

- The **Critic is overly lenient** on genre diversity when all returned songs share the
  same mood.  It passes "4 rock songs + 1 pop song" as diverse, which a human reviewer
  would flag.

---

## 8. AI Collaboration Reflection

### How AI (Claude) was used

1. **Architecture design:** the 5-node LangGraph structure and the two-stage scoring
   design (deterministic first pass / LLM-adjusted retry only) were co-designed through
   iterative back-and-forth.  AI challenged early designs that called the LLM on every
   scoring pass, which would have tripled latency for clean queries.

2. **Prompt engineering:** the 370-line Intent Parser system prompt, with genre mapping
   rules, activity-to-energy tables, and edge-case examples like "angry jazz", was
   co-written with AI assistance.

3. **Debugging the FAISS migration:** the original design used ChromaDB, which triggered
   a DLL heap conflict under Windows + pytest.  AI diagnosed the root cause and provided
   a drop-in FAISS replacement matching the ChromaDB public API.

4. **Test case generation:** the 6 eval harness cases and their expected ranges were
   generated with AI assistance, including the "purely instrumental focus" case that
   exposed the biggest gap in the dataset.

### One helpful suggestion

The **two-stage scoring design** was AI's suggestion: run deterministic weights on the
first pass (no LLM call), and only invoke the LLM to adjust weights when the Critic
provides feedback.  This kept first-query latency low (~5 s) while preserving full
adaptability on retries.  Without this, every query would have paid an LLM call even
when the first result was already perfect.

### One flawed suggestion

When diagnosing the "instrumental songs not retrieved" bug, AI initially suggested
increasing `instrumentalness_weight` from 2.0 to 4.0 in the Scorer.  This was **wrong**
— the Scorer can only rank songs that FAISS already returned; no amount of weight
adjustment surfaces songs absent from the candidate pool.  The actual fix required two
separate changes: (1) adding high-instrumentalness tracks to the dataset in
`data/preprocess.py`, and (2) adding post-retrieval constraint filtering in the
Retriever.  AI was diagnosing the symptom (low scores) rather than the root cause
(missing candidates).

---

## 9. Future Work

1. **Larger, genre-diverse dataset:** integrate Spotify API or MusicBrainz to cover
   jazz, classical, metal, folk, and K-pop with 500+ tracks per genre.
2. **Cross-session learning:** persist aggregate feedback patterns to build a simple
   collaborative filter on top of the content-based scorer.
3. **Smarter Critic diversity metric:** require ≥ 3 distinct genres in the top-5 and
   a minimum 0.15 spread in energy values before passing.
4. **Latency optimization:** cache Intent Parser outputs for semantically identical
   queries; batch Explainer calls for all songs in one API request.
5. **Streaming UI:** stream Critic and Explainer tokens in real time so the UI feels
   responsive while the 70B model generates.
