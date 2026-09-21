"""Thinking-model support: the trace is split from the answer correctly, and sampled runs never share a cache key."""

from vlm_parking.thinking import settings_key, split_thinking

ANSWER = '{"reason": "x", "governing_panel": null, "verdict": "legal"}'


def test_split_when_the_template_opened_think_in_the_prompt():
    # the usual case: only the closing tag is generated
    thinking, answer = split_thinking(f"arrives 07:33, leaves 08:03... wait, that's after 08:00.\n</think>\n\n{ANSWER}")
    assert thinking == "arrives 07:33, leaves 08:03... wait, that's after 08:00."
    assert answer == ANSWER


def test_split_when_the_model_writes_both_tags():
    thinking, answer = split_thinking(f"<think>\nsome reasoning\n</think>{ANSWER}")
    assert thinking == "some reasoning" and answer == ANSWER


def test_no_closing_tag_means_no_answer():
    # the token budget ran out mid-thought
    thinking, answer = split_thinking("still thinking about whether 08:03 is after")
    assert answer is None and thinking.startswith("still thinking")


def test_splits_at_the_first_closing_tag():
    # vLLM enforces the answer grammar from the first </think>, so that's where the answer starts
    _, answer = split_thinking(f"a</think>{ANSWER}</think>")
    assert answer == f"{ANSWER}</think>"


BASE = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0.0, max_tokens=16384, seed=0, max_side=1024,
            schema_hash="h", reasoning_parser="qwen3")


def test_every_sampling_setting_changes_the_cache_key():
    for field, other in [("seed", 1), ("top_p", 0.8), ("top_k", 40), ("presence_penalty", 1.5), ("temperature", 0.7),
                         ("max_tokens", 8192), ("reasoning_parser", "")]:
        assert settings_key(**BASE) != settings_key(**(BASE | {field: other})), field


def test_cache_key_is_stable():
    assert settings_key(**BASE) == settings_key(**dict(reversed(BASE.items())))
