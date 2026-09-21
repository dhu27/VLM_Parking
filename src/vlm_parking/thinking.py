"""Support for thinking models (scripts/run_inference_thinking.py): splitting the trace from the answer, and the
settings that key a sampled run in the response cache.

A module of its own so that nothing the original runs used (prompts.py, run_inference.py) has to change.
"""

from __future__ import annotations

import json

END_THINK = "</think>"

# The model card's recommended settings for vision-language use (Qwen/Qwen3-VL-8B-Thinking, "Generation
# Hyperparameters > VL"). The card gives a separate text-only set (presence_penalty 1.5), but condition B is
# text-only and condition A isn't, and the study's rule is that only the sign's presentation differs between
# conditions, so one set is used for both.
MODEL_CARD_VL = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}


def split_thinking(text: str) -> tuple[str, str | None]:
    """Split generated text into (thinking, answer).

    The split is at the *first* </think>, because that is where vLLM's reasoning parser ends the thinking and
    starts enforcing the answer grammar. The chat template may open <think> inside the prompt, so only the
    closing tag is guaranteed to appear. No closing tag means the token budget ran out mid-thought: there is
    no answer, and the caller should record a parse failure.
    """
    end = text.find(END_THINK)
    if end == -1:
        return text.strip(), None
    thinking = text[:end].strip()
    if thinking.startswith("<think>"):
        thinking = thinking[len("<think>"):].strip()
    return thinking, text[end + len(END_THINK):].strip()


def settings_key(*, temperature: float, top_p: float, top_k: int, presence_penalty: float, max_tokens: int,
                 seed: int, max_side: int, schema_hash: str, reasoning_parser: str) -> str:
    """Everything about a run that can change an answer, serialized for the cache key.

    Every sampling setting is in here. If top_p or top_k were left out, two runs differing only in those would
    share cache keys, and the second would silently reuse the first's answers.
    """
    return json.dumps({"t": temperature, "top_p": top_p, "top_k": top_k, "presence_penalty": presence_penalty,
                       "max_tokens": max_tokens, "seed": seed, "max_side": max_side, "schema": schema_hash,
                       "reasoning_parser": reasoning_parser}, sort_keys=True)
