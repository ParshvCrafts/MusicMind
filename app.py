"""
MusicMind v2 Streamlit UI
Run: streamlit run app.py  (from the musicmind/ directory)
"""
from dotenv import load_dotenv
load_dotenv()

import json
import os
import sys
import urllib.parse
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_DATA_PATH = _HERE / "data" / "songs_processed.csv"
_DOCS_PATH = _HERE / "rag" / "documents"
_MEMORY_PATH = _HERE / "data" / "user_memory.json"

from agents.orchestrator import build_graph, AgentState
from rag.knowledge_base import MusicKnowledgeBase
from reliability.guardrails import UserQuery

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MusicMind",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API key check ─────────────────────────────────────────────────────────────
_api_key = os.environ.get("GROQ_API_KEY", "")
if not _api_key:
    st.error(
        "**GROQ_API_KEY is not set.**\n\n"
        "Create a `.env` file in `musicmind/` with:\n```\nGROQ_API_KEY=gsk_...\n```\n"
        "Get a free key at https://console.groq.com"
    )
    st.stop()

# ── System loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MusicMind — first run ~30 s, subsequent runs use cache...")
def _load_system():
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f"Song data not found at {_DATA_PATH}. Run `python data/preprocess.py` first."
        )
    kb = MusicKnowledgeBase(cache_dir=str(_HERE / "data" / "cache"))
    kb.ingest_songs(str(_DATA_PATH))
    kb.ingest_knowledge_docs(str(_DOCS_PATH))
    graph = build_graph(kb)
    return graph, kb

# ── Session state defaults ────────────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state["query_history"] = []
if "eval_results" not in st.session_state:
    st.session_state["eval_results"] = None
if "eval_timestamp" not in st.session_state:
    st.session_state["eval_timestamp"] = None
if "prefill_query" not in st.session_state:
    st.session_state["prefill_query"] = ""

# ── Helper: save feedback to user_memory.json ────────────────────────────────
def _save_feedback(artist: str, positive: bool) -> None:
    try:
        memory: dict = {"liked_artists": {}, "disliked_artists": {}}
        if _MEMORY_PATH.exists():
            memory = json.loads(_MEMORY_PATH.read_text(encoding="utf-8"))
        bucket = "liked_artists" if positive else "disliked_artists"
        memory[bucket][artist] = memory[bucket].get(artist, 0) + 1
        _MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")
    except Exception:
        pass  # never crash the UI on memory write failure

# ── Chart helpers ─────────────────────────────────────────────────────────────
def _build_radar_chart(recs: list[dict], profile) -> go.Figure:
    """Radar chart: target audio DNA vs recommendations average."""
    axes = ["Energy", "Valence", "Danceability", "Acousticness", "Instrumentalness", "Calm"]

    target_vals = [
        profile.target_energy,
        profile.target_valence,
        0.5,  # danceability target unknown — neutral
        profile.instrumentalness_preference * 0.8,  # rough proxy for acousticness desire
        profile.instrumentalness_preference,
        1.0 - 0.1,  # calm = 1 - default speechiness
    ]

    def avg(key):
        vals = [float(r.get(key, 0.5)) for r in recs]
        return sum(vals) / len(vals) if vals else 0.5

    rec_vals = [
        avg("energy"), avg("valence"), avg("danceability"),
        avg("acousticness"), avg("instrumentalness"),
        1.0 - avg("speechiness"),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=target_vals + [target_vals[0]], theta=axes + [axes[0]],
        fill="toself", name="Your Target",
        line=dict(color="#4CAF50", width=2),
        fillcolor="rgba(76,175,80,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=rec_vals + [rec_vals[0]], theta=axes + [axes[0]],
        fill="toself", name="Recommendations Avg",
        line=dict(color="#2196F3", width=2),
        fillcolor="rgba(33,150,243,0.15)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        title=dict(text="Audio DNA — Target vs Recommendations", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_energy_arc(recs: list[dict]) -> go.Figure:
    """Dual-axis chart: energy bars + BPM line over 5 songs."""
    titles = [f"#{i+1} {r.get('title','?')[:18]}" for i, r in enumerate(recs)]
    energies = [float(r.get("energy", 0.5)) for r in recs]
    bpms = [float(r.get("bpm", 120.0)) for r in recs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=titles, y=energies, name="Energy",
        marker_color=[f"rgba(33,150,243,{0.4 + e * 0.6})" for e in energies],
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=titles, y=bpms, name="BPM", mode="lines+markers",
        line=dict(color="#FF9800", width=2), marker=dict(size=8),
        yaxis="y2",
    ))
    fig.update_layout(
        title=dict(text="Playlist Energy Arc", x=0.5),
        yaxis=dict(title="Energy", range=[0, 1.05], side="left"),
        yaxis2=dict(title="BPM", overlaying="y", side="right", showgrid=False),
        height=300, margin=dict(l=50, r=60, t=40, b=60),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
    )
    return fig


def _build_score_breakdown_chart(song: dict) -> go.Figure:
    """Horizontal bar chart of score_breakdown components."""
    breakdown: dict = song.get("score_breakdown", {})
    if not breakdown:
        return None
    labels = list(breakdown.keys())
    values = [float(v) for v in breakdown.values()]
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=max(200, 30 * len(labels) + 60),
        margin=dict(l=120, r=60, t=20, b=20),
        xaxis=dict(title="Score Contribution"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_eval_chart(results: list[dict]) -> go.Figure:
    labels = [r["query"][:30] + "…" for r in results]
    colors = ["#4CAF50" if r.get("status") == "PASS" else "#F44336" for r in results]
    values = [1] * len(results)
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[r.get("status", "?") for r in results], textposition="inside",
    ))
    fig.update_layout(
        height=220, showlegend=False,
        yaxis=dict(visible=False), xaxis=dict(tickangle=-20),
        margin=dict(l=10, r=10, t=10, b=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_arch_diagram() -> go.Figure:
    """Static flowchart of the MusicMind pipeline."""
    fig = go.Figure()
    nodes = [
        (5, 10.5, "🟢 User Query", "#27ae60"),
        (5, 9.0,  "① Intent Parser", "#2980b9"),
        (5, 7.5,  "② FAISS Retriever", "#2980b9"),
        (5, 6.0,  "③ Scorer", "#2980b9"),
        (5, 4.5,  "④ Explainer (RAG)", "#2980b9"),
        (5, 3.0,  "⑤ Critic", "#e67e22"),
        (5, 1.5,  "🎵 Recommendations", "#27ae60"),
    ]
    for x, y, label, color in nodes:
        fig.add_shape(type="rect", x0=x-2.3, y0=y-0.42, x1=x+2.3, y1=y+0.42,
                      fillcolor=color, line_width=0, layer="below")
        fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>", showarrow=False,
                           font=dict(color="white", size=10))

    arrow_pairs = [(10.08, 9.42), (8.58, 7.92), (7.08, 6.42), (5.58, 4.92), (4.08, 3.42), (2.58, 1.92)]
    for ay, y in arrow_pairs:
        fig.add_annotation(x=5, y=y, ax=5, ay=ay, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#555")

    # Retry loop
    for shape in [
        dict(type="line", x0=7.3, y0=3.0, x1=7.9, y1=3.0),
        dict(type="line", x0=7.9, y0=3.0, x1=7.9, y1=6.0),
        dict(type="line", x0=7.3, y0=6.0, x1=7.9, y1=6.0),
    ]:
        fig.add_shape(**shape, line=dict(color="#e74c3c", width=2))
    fig.add_annotation(x=7.3, y=6.0, ax=7.3, ay=6.4, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowcolor="#e74c3c")
    fig.add_annotation(x=8.2, y=4.5, text="retry<br>≤2×", showarrow=False,
                       font=dict(size=9, color="#e74c3c"))

    fig.update_layout(
        height=460, showlegend=False, margin=dict(l=5, r=5, t=10, b=5),
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[0.8, 11.2]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ab_mode = st.checkbox("🔬 Enable A/B Critic Mode", value=False,
                          help="Run both 8B and 70B critics and compare their verdicts. Adds ~3 sec.")

    st.divider()

    # Query history
    if st.session_state["query_history"]:
        st.subheader("🕘 Recent Queries")
        for item in reversed(st.session_state["query_history"][-10:]):
            label = f"{item['query'][:28]}… → {item['top_rec'][:16]}"
            if st.button(label, key=f"hist_{item['ts']}", use_container_width=True):
                st.session_state["prefill_query"] = item["query"]
                st.rerun()

    st.divider()

    # Eval suite
    if st.button("🧪 Run Evaluation Suite", use_container_width=True):
        try:
            graph, kb = _load_system()
        except FileNotFoundError as exc:
            st.error(str(exc))
        else:
            from reliability.eval_suite import run_eval
            with st.spinner("Running eval cases (~2 min)..."):
                report = run_eval(graph, kb)
            st.session_state["eval_results"] = report["results"]
            st.session_state["eval_timestamp"] = datetime.now().strftime("%H:%M:%S")
            st.success(f"Pass rate: **{report['pass_rate']}**")

    if st.session_state["eval_results"]:
        n_pass = sum(1 for r in st.session_state["eval_results"] if r.get("status") == "PASS")
        n_total = len(st.session_state["eval_results"])
        st.metric(
            "Eval Pass Rate",
            f"{n_pass}/{n_total}",
            delta=f"{n_pass/n_total*100:.0f}%" if n_total > 0 else "0%",
        )
        st.plotly_chart(_build_eval_chart(st.session_state["eval_results"]),
                        use_container_width=True, key="eval_chart")
        st.caption(f"Last run: {st.session_state['eval_timestamp']}")

    st.divider()

    with st.expander("🗺 System Architecture"):
        st.plotly_chart(_build_arch_diagram(), use_container_width=True, key="arch_diagram")

    st.divider()
    st.caption(f"Songs indexed: {_DATA_PATH.name}")
    st.caption(f"Knowledge docs: {len(list(_DOCS_PATH.glob('*.txt')))} files")
    st.caption("MusicMind v2.0 | VibeFinder Extended")
    st.caption("LangGraph · FAISS · Groq")

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("🎵 MusicMind")
st.caption("Agentic AI Music Intelligence — LangGraph · FAISS · Groq")

col_main, col_trace = st.columns([2, 1])

with col_main:
    prefill = st.session_state.get("prefill_query", "")
    st.session_state["prefill_query"] = ""  # reset after reading
    query = st.text_input(
        "What are you in the mood for?",
        value=prefill,
        placeholder="I'm coding late at night and need deep focus without lyrics...",
        key="query_input",
    )
    run_btn = st.button("🔍 Find My Music", type="primary", use_container_width=True)

with col_trace:
    st.subheader("🤖 Agent Trace")
    trace_placeholder = st.empty()

# ── Query handler ─────────────────────────────────────────────────────────────
if run_btn and query.strip():
    try:
        validated = UserQuery(text=query)
    except ValidationError as exc:
        st.error(f"⚠️ Query rejected: {exc.errors()[0]['msg']}")
        st.stop()

    try:
        graph, kb = _load_system()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    initial: AgentState = {
        "query": validated.text,
        "mood_profile": None,
        "retrieved_knowledge": [],
        "candidate_songs": [],
        "scored_songs": [],
        "explanations": [],
        "critic_feedback": "",
        "final_recommendations": [],
        "retry_count": 0,
        "agent_trace": [],
        "rejection_reason": "",
        "ab_mode": ab_mode,
        "ab_critic_verdict": None,
    }

    final_state: dict = {}
    with st.spinner("MusicMind is thinking..."):
        try:
            for state in graph.stream(initial, stream_mode="values"):
                trace_steps = state.get("agent_trace", [])
                if trace_steps:
                    with col_trace:
                        trace_placeholder.markdown(
                            "\n\n".join(f"`{step}`" for step in trace_steps)
                        )
                final_state = state
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            import traceback
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
            st.stop()

    rejection = final_state.get("rejection_reason", "")
    recs: list[dict] = final_state.get("final_recommendations", [])
    profile = final_state.get("mood_profile")
    ab_verdict = final_state.get("ab_critic_verdict")

    with col_main:
        if rejection:
            st.info(f"🎵 **Out of scope:** {rejection}", icon="ℹ️")
        elif not recs:
            st.warning("No recommendations returned. Try adding more detail (activity, genre, mood).")
        else:
            # Save to query history
            top_rec = recs[0].get("title", "?") if recs else "?"
            st.session_state["query_history"].append({
                "query": validated.text, "top_rec": top_rec,
                "ts": datetime.now().isoformat(),
            })

            st.subheader("🎵 Your Recommendations")
            for i, song in enumerate(recs):
                with st.container():
                    c_rank, c_info, c_score = st.columns([0.5, 3, 1])
                    with c_rank:
                        st.metric("", f"#{i + 1}")
                    with c_info:
                        st.markdown(f"**{song.get('title', '?')}** — *{song.get('artist', '?')}*")
                        energy_pct = f"{float(song.get('energy', 0.5)):.0%}"
                        bpm = float(song.get("bpm", 120.0))
                        st.caption(
                            f"🎸 {song.get('genre', '?')} · 😌 {song.get('mood', '?')} · "
                            f"⚡ Energy: {energy_pct} · 🎵 {bpm:.0f} BPM"
                        )
                        st.info(song.get("explanation", "No explanation available."), icon="💬")

                        # YouTube link + score breakdown
                        yt_url = (
                            "https://www.youtube.com/results?search_query="
                            + urllib.parse.quote(f"{song.get('title','')} {song.get('artist','')}")
                        )
                        st.link_button("▶ Listen on YouTube", yt_url)

                        breakdown_fig = _build_score_breakdown_chart(song)
                        if breakdown_fig:
                            with st.expander("Why this song →"):
                                st.plotly_chart(breakdown_fig, use_container_width=True,
                                                key=f"breakdown_{i}")

                    with c_score:
                        st.metric("Score", f"{float(song.get('score', 0)):.2f}")
                    st.divider()

            # Feedback buttons with memory
            st.subheader("How were these picks?")
            top_artist = recs[0].get("artist", "") if recs else ""
            fb1, fb2 = st.columns(2)
            with fb1:
                if st.button("👍 Great picks!", use_container_width=True):
                    if top_artist:
                        _save_feedback(top_artist, positive=True)
                    st.success(f"Thanks! Noted: {top_artist} boosted for future queries.")
            with fb2:
                if st.button("👎 Missed the vibe", use_container_width=True):
                    if top_artist:
                        _save_feedback(top_artist, positive=False)
                    st.warning("Noted! Try adding more detail to your query.")

            # Export CSV
            import io
            csv_cols = ["title", "artist", "genre", "mood", "energy", "bpm",
                        "valence", "danceability", "acousticness", "instrumentalness", "score", "explanation"]
            import pandas as _pd
            csv_df = _pd.DataFrame([
                {c: r.get(c, "") for c in csv_cols} for r in recs
            ])
            csv_buf = io.StringIO()
            csv_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇ Export Recommendations CSV",
                data=csv_buf.getvalue(),
                file_name=f"musicmind_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Mood radar chart
            if profile:
                st.plotly_chart(_build_radar_chart(recs, profile),
                                use_container_width=True, key="radar")

            # Playlist energy arc
            st.plotly_chart(_build_energy_arc(recs), use_container_width=True, key="arc")

            # A/B critic panel
            if ab_verdict:
                with st.expander("🔬 A/B Critic Analysis"):
                    fast = ab_verdict.get("fast", {})
                    quality = ab_verdict.get("quality", {})
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.metric("8B Fast Score", f"{fast.get('overall_score', 0):.1f}/10",
                                  delta="PASS ✓" if fast.get("passed") else "FAIL ✗")
                        if fast.get("issues"):
                            st.caption("Issues: " + "; ".join(fast["issues"][:2]))
                    with cc2:
                        st.metric("70B Quality Score", f"{quality.get('overall_score', 0):.1f}/10",
                                  delta="PASS ✓" if quality.get("passed") else "FAIL ✗")
                        if quality.get("issues"):
                            st.caption("Issues: " + "; ".join(quality["issues"][:2]))

                    # Agreement banner
                    if fast.get("passed") == quality.get("passed"):
                        st.success(
                            f"Models agree: {'PASS ✓' if fast.get('passed') else 'FAIL ✗'}  "
                            f"(8B: {fast.get('overall_score',0):.1f}, "
                            f"70B: {quality.get('overall_score',0):.1f})"
                        )
                    else:
                        st.warning(
                            f"Models disagree — 8B: {'PASS' if fast.get('passed') else 'FAIL'} "
                            f"({fast.get('overall_score',0):.1f}), "
                            f"70B: {'PASS' if quality.get('passed') else 'FAIL'} "
                            f"({quality.get('overall_score',0):.1f})"
                        )
