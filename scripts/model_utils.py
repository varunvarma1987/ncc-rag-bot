"""
Shared helper for building OpenAI chat completion kwargs across the
pipeline's three LLM call sites (guardrails.py, multi_query_search.py,
generate_answer.py).

Why this exists: switching the default model to gpt-5-mini broke every one
of those calls, because gpt-5-class models reject `temperature=0` outright
("Unsupported value: 'temperature' does not support 0 with this model.
Only the default (1) value is supported") -- a real, documented API
constraint on that model family, not a bug on our end. `seed` alone still
works fine without an explicit temperature, so the fix is to build the
kwargs conditionally per model rather than hardcoding `temperature=0`
everywhere and having it silently break for a whole family of models.
"""

# Prefix-based, not an exhaustive model list -- new model families keep
# appearing, and this is the simplest way to route around a family-wide API
# constraint without hardcoding every individual model name.
MODELS_WITHOUT_TEMPERATURE_CONTROL = ("gpt-5",)


def build_chat_kwargs(model: str, seed: int = 42, temperature: float = 0) -> dict:
    """Return the kwargs to spread into client.chat.completions.create().

    Always includes `seed` for best-effort reproducibility (see
    answer_cache.py for why "best effort" isn't good enough on its own).
    Only includes `temperature` for models that actually accept overriding
    it away from their default.
    """
    kwargs = {"seed": seed}
    if not model.startswith(MODELS_WITHOUT_TEMPERATURE_CONTROL):
        kwargs["temperature"] = temperature
    return kwargs
