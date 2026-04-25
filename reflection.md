## MusicMind v2: AI Collaboration Reflection

### How AI was used during development

AI (Claude) was used in four distinct ways:

1. **Architecture design**: The overall LangGraph node structure (Intent Parser → Retriever → Scorer → Explainer → Critic) was brainstormed collaboratively. AI suggested the two-stage scoring design (deterministic weights first, LLM-adjusted only on retry), which I initially pushed back on as over-engineered, but it turned out to be the right call for keeping the first-pass fast and debuggable.

2. **Prompt engineering**: The 370-line intent parser system prompt (with its genre mapping rules, edge case examples, and activity-to-energy tables) was co-written with AI. It would have taken 3–4x longer to write solo and likely would have missed edge cases like "angry jazz" or "surprise me."

3. **Debugging the FAISS migration**: The original design used ChromaDB with PersistentClient. On Windows, it triggered a heap incompatibility crash under pytest (hnswlib DLL conflict with numpy's scipy-openblas64). AI helped diagnose the root cause and suggested switching to FAISS entirely, providing a drop-in implementation that matched the ChromaDB public API.

4. **Eval test case generation**: The 6 evaluation test cases and their expected energy/valence ranges were generated with AI assistance, which surfaced the "purely instrumental focus music" case that exposed the biggest quality bug in the system.

### One helpful AI suggestion

The **two-stage scoring design** was directly suggested by AI: run deterministic profile-driven weights on the first pass (no LLM call), and only invoke the LLM to adjust weights when the Critic provides feedback on retry. This meant the system was both fast (first query: no extra LLM call) and adaptive (retry: LLM adjusts weights to address critic feedback). Without this guidance I would have called the LLM on every scoring pass, tripling latency for no benefit on clean queries.

### One flawed AI suggestion

AI's first suggestion for the "instrumental songs not being retrieved" bug was to increase `instrumentalness_weight` in the scorer from 2.0 to 4.0. This was **wrong**, the scorer cannot surface songs that aren't in the candidate pool returned by FAISS. The real fix required two separate changes: (1) adding high-instrumentalness tracks to the dataset, and (2) adding post-retrieval constraint filtering. AI initially diagnosed the symptom (low scores) rather than the root cause (no instrumental candidates in the FAISS pool).

### System limitations

1. **6-genre dataset constraint**: The dataset maps all music to `edm, latin, pop, r&b, rap, rock, instrumental`. Classical, jazz, country, folk, and metal users get approximate mappings that may feel wrong.

2. **900-song ceiling**: Even with the instrumental bucket, the catalog is small. A user querying for a very specific combination (e.g., "heavy metal instrumental prog") may find no good matches and get the best available rather than the best possible.

3. **Critic retry cap at 2**: The self-correction loop can only retry twice. For very niche or contradictory queries, two retries are not enough to escape a local optimum in the scoring space.

4. **No cross-session learning**: Session memory persists liked/disliked artists, but there is no long-term preference model. The system does not learn from aggregate feedback patterns across many users or sessions.

5. **Latency**: Even with FAISS caching, each query requires 3–5 LLM API calls (intent parser, explainer, critic, optionally scorer-feedback). Total latency is 5–15 seconds depending on Groq load.
