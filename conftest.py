"""
Root conftest.py for MusicMind.

Responsibilities:
  1. Add the musicmind/ package root to sys.path so that `agents`, `rag`,
     `reliability`, and `src` are importable regardless of the directory from
     which pytest is invoked.
  2. Stub out GROQ_API_KEY so unit tests that mock the LLM don't need a real
     key to pass (tests that actually call the LLM are skipped in CI via the
     `live_llm` marker).
  3. Silence noisy third-party warnings that pollute test output.
"""
import os
import sys
from pathlib import Path

# ── 1. sys.path — make package root importable ───────────────────────────────
_HERE = Path(__file__).parent  # musicmind/
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── 2. Suppress third-party noise ────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # HuggingFace warning
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # generic telemetry flag

# ── 3. tqdm — disable background monitor thread (safe for pytest) ─────────────
try:
    import tqdm
    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

# ── 4. Stub GROQ_API_KEY so parse_intent / scorer LLM guards don't raise ─────
#    Unit tests that mock ChatGroq never actually call Groq, but the key check
#    in parse_intent() runs before the mock intercepts.  A fake key keeps the
#    guard happy without making real network calls.
os.environ.setdefault("GROQ_API_KEY", "gsk_test_stub_key_for_unit_tests")
