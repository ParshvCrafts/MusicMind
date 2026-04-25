"""
Evaluation harness for MusicMind.
Runs 5 representative test cases through the full LangGraph pipeline and
produces eval_report.json + a terminal summary table.
"""
import json
from datetime import datetime
from pathlib import Path

# Write the report next to this file's package root (musicmind/) so the path
# is stable regardless of the cwd when eval is invoked.
_REPORT_PATH = Path(__file__).parent.parent / "eval_report.json"

_TEST_CASES = [
    {
        "query": "I'm coding at 2am, need deep focus music without any distracting lyrics",
        "description": "late-night coding focus",
        "expected_energy_range": (0.2, 0.6),
        "expected_valence_range": (0.3, 0.8),
    },
    {
        "query": "Pump up music for leg day at the gym, I want heavy beats",
        "description": "high-energy gym workout",
        "expected_energy_range": (0.7, 1.0),
    },
    {
        "query": "Sad rainy day, feeling nostalgic and a bit melancholic",
        "description": "melancholic rainy day",
        "expected_energy_range": (0.1, 0.5),
        "expected_valence_range": (0.0, 0.5),
    },
    {
        "query": "Background music for a dinner party with friends, upbeat but not too loud",
        "description": "dinner party background",
        "expected_energy_range": (0.4, 0.75),
    },
    {
        "query": "I want to feel like I'm driving on an empty highway at sunset, indie vibes",
        "description": "highway driving indie",
        "expected_energy_range": (0.4, 0.8),
    },
    {
        "query": "I need purely instrumental acoustic music with absolutely zero vocals for deep focus",
        "description": "purely instrumental focus",
        "expected_energy_range": (0.1, 0.55),
        "expected_valence_range": (0.2, 0.8),
        "expected_instrumentalness_min": 0.15,
    },
]

_INITIAL_STATE_TEMPLATE = {
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
    "ab_mode": False,
    "ab_critic_verdict": None,
}


def run_eval(graph, kb) -> dict:
    """Run all test cases through the graph and produce a report.

    Args:
        graph: compiled LangGraph graph from build_graph()
        kb: MusicKnowledgeBase (unused directly — graph has it captured in closure)

    Returns:
        Report dict also written to eval_report.json.
    """
    results = []

    for case in _TEST_CASES:
        initial = {**_INITIAL_STATE_TEMPLATE, "query": case["query"]}

        try:
            final = graph.invoke(initial)
            recs = final.get("final_recommendations", [])

            energy_pass = True
            valence_pass = True
            instrumentalness_pass = True

            if recs and "expected_energy_range" in case:
                avg_energy = sum(float(r.get("energy", 0.5)) for r in recs) / len(recs)
                lo, hi = case["expected_energy_range"]
                energy_pass = lo <= avg_energy <= hi

            if recs and "expected_valence_range" in case:
                avg_valence = sum(float(r.get("valence", 0.5)) for r in recs) / len(recs)
                lo, hi = case["expected_valence_range"]
                valence_pass = lo <= avg_valence <= hi

            if "expected_instrumentalness_min" in case and recs:
                avg_inst = sum(float(r.get("instrumentalness", 0)) for r in recs) / len(recs)
                instrumentalness_pass = avg_inst >= case["expected_instrumentalness_min"]

            all_pass = energy_pass and valence_pass and instrumentalness_pass
            status = "PASS" if (recs and all_pass) else "FAIL"
            results.append({
                "query": case["query"],
                "description": case["description"],
                "num_recs": len(recs),
                "energy_pass": energy_pass,
                "valence_pass": valence_pass,
                "instrumentalness_pass": instrumentalness_pass,
                "top_rec": recs[0]["title"] if recs else "NONE",
                "retries": final.get("retry_count", 0),
                "trace_steps": len(final.get("agent_trace", [])),
                "status": status,
            })

        except Exception as exc:
            results.append({
                "query": case["query"],
                "description": case["description"],
                "status": "ERROR",
                "error": str(exc),
            })

    passed = sum(1 for r in results if r.get("status") == "PASS")
    report = {
        "timestamp": datetime.now().isoformat(),
        "total": len(_TEST_CASES),
        "passed": passed,
        "pass_rate": f"{passed / len(_TEST_CASES) * 100:.1f}%",
        "results": results,
    }

    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 58}")
    print(f"MUSICMIND EVAL: {passed}/{len(_TEST_CASES)} passed ({report['pass_rate']})")
    print(f"{'=' * 58}")
    for r in results:
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        desc = r.get("description", "?")
        top = r.get("top_rec", "?")
        retries = r.get("retries", "?")
        print(f"  [{icon}] {desc:<30} -> {top} (retries={retries})")
    print()

    return report
