"""
Multi-turn conversation memory for the bot.

The problem this solves: every stage before this (retrieval, hybrid search,
multi-query) only ever sees ONE question in isolation. A natural follow-up
like "what about for walls instead?" carries no useful information on its
own -- BM25 and vector search have no idea what "that" or "instead" refers
to, so retrieval for a bare follow-up like that would be close to random.

The fix used here is a standard RAG pattern called QUESTION CONDENSING: an
LLM is shown the recent conversation history plus the new follow-up, and
asked to rewrite it into a fully self-contained "standalone question" that
means the same thing without needing the earlier turns for context (e.g.
"what about for walls instead?" -> "What R-value thermal break is required
for walls in a Class 2 building?", assuming the previous turn was about
roof thermal breaks). That standalone question is what actually flows into
retrieval -- not the raw follow-up, and not a simple string concatenation
of history + question, which would just confuse the query-rewriting and
retrieval stages with irrelevant leftover phrasing.

Why a SEPARATE LLM call for this rather than just stuffing history into the
main generation prompt: retrieval happens BEFORE generation, and retrieval
needs a clean, self-contained question to search with. If we only fixed
this at the generation step, the WRONG chunks would already have been
retrieved by then -- condensing has to happen upstream of retrieval to do
any good.

Memory itself is just an in-memory list of (question, answer) pairs held
on the AnswerGenerator instance -- no persistence across process restarts,
which is fine for a single chat session. Capped to the last few turns so
the condensing prompt doesn't grow unbounded over a long conversation.
"""

import os
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import build_chat_kwargs

CONDENSE_MODEL = os.environ.get("CONDENSE_MODEL", "gpt-5-mini")
MAX_HISTORY_TURNS = 3

CONDENSE_PROMPT = """Given the conversation history below and a new follow-up question, rewrite the follow-up into a fully self-contained standalone question that means the same thing WITHOUT needing the history to understand it. Resolve any pronouns or implicit references ("that", "it", "instead", "what about...") using the history.

If the follow-up is already self-contained and doesn't depend on the history at all, just return it unchanged.

Conversation history:
{history}

Follow-up question: {question}

Output ONLY the standalone question, nothing else."""


def format_history(history: list[tuple[str, str]]) -> str:
    turns = history[-MAX_HISTORY_TURNS:]
    return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in turns)


def condense_question(question: str, history: list[tuple[str, str]]) -> str:
    if not history:
        return question

    client = OpenAI()
    prompt = CONDENSE_PROMPT.format(history=format_history(history), question=question)
    response = client.chat.completions.create(
        model=CONDENSE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        **build_chat_kwargs(CONDENSE_MODEL),
    )
    return response.choices[0].message.content.strip()


class ConversationMemory:
    """Plain in-memory turn history for one chat session."""

    def __init__(self):
        self.turns: list[tuple[str, str]] = []

    def add(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        self.turns = self.turns[-MAX_HISTORY_TURNS:]

    def clear(self) -> None:
        self.turns = []
