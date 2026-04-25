import pytest


def test_parse_batch_happy_path():
    from agents.explainer import _parse_batch_explanations
    text = (
        "1. Song A is great for focus. Its low energy of 0.3 keeps distractions away.\n"
        "2. Song B has high instrumentalness. It was designed for concentration.\n"
        "3. Song C features ambient textures. BPM of 72 promotes flow state.\n"
        "4. Song D has minimal percussion. Its acousticness of 0.9 soothes the mind.\n"
        "5. Song E was recorded live. Its calm energy suits late-night work sessions."
    )
    result = _parse_batch_explanations(text, n=5)
    assert len(result) == 5
    assert "Song A" in result[0]
    assert "Song E" in result[4]


def test_parse_batch_fallback_padding():
    from agents.explainer import _parse_batch_explanations
    # Only 2 parseable items — should pad to 5
    text = "1. First explanation here.\n2. Second explanation here."
    result = _parse_batch_explanations(text, n=5)
    assert len(result) == 5


def test_parse_batch_empty_fallback():
    from agents.explainer import _parse_batch_explanations
    result = _parse_batch_explanations("", n=5)
    assert len(result) == 5
    assert all(isinstance(r, str) and len(r) > 0 for r in result)


def test_no_negative_framing_in_prompt():
    """The batch prompt must not contain 'does not fit' or 'not suitable'."""
    from agents.explainer import _BATCH_EXPLAINER_PROMPT
    prompt_lower = _BATCH_EXPLAINER_PROMPT.lower()
    assert "does not fit" not in prompt_lower
    assert "not suitable" not in prompt_lower
    assert "not ideal" not in prompt_lower
