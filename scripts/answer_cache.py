"""
A small file-backed cache for AnswerGenerator, keyed by (model, question).

Why this exists (see PROJECT.md, lever #12 -- optimization/caching, and the
investigation that led here): retrieval for this bot depends on an LLM call
to generate query rewrites (multi_query_search.py), and that call turned
out to still vary slightly between runs even at temperature=0 with a fixed
seed -- confirmed by testing the exact same question three times and
getting different rewritten phrasings on one of the three runs. OpenAI's
own docs describe seed-based determinism as "best effort", not guaranteed,
due to backend nondeterminism in how these models are served. That's not
something we can fix by calling the API differently -- but we CAN make sure
a user who asks the exact same question twice always gets the exact same
answer, by remembering what we answered the first time and skipping the
LLM calls entirely on a repeat.

This is a plain JSON file, not a real cache library -- appropriate for a
learning project at this scale (a handful of cached Q&A pairs, no
concurrent writers, no expiry policy needed since the underlying document
doesn't change between runs of this project).
"""

import json
from pathlib import Path

CACHE_PATH = Path(r"c:\Varun\NCC bot\cache\answer_cache.json")


def _normalize(question: str) -> str:
    # Case/whitespace shouldn't matter for cache hits -- "What is X?" and
    # "what is x?" are the same question as far as caching is concerned.
    return " ".join(question.strip().lower().split())


def _make_key(model: str, question: str) -> str:
    # Keyed by model too: swapping ANSWER_MODEL should get its own answers,
    # not silently reuse a different model's cached response.
    return f"{model}::{_normalize(question)}"


class AnswerCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def get(self, model: str, question: str) -> dict | None:
        return self._data.get(_make_key(model, question))

    def set(self, model: str, question: str, result: dict) -> None:
        self._data[_make_key(model, question)] = result
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
