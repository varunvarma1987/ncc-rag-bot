"""
Interactive CLI for the NCC RAG bot -- a thin wrapper around AnswerGenerator
so you can ask questions directly from the terminal instead of scripting
each call.

Usage:
    python scripts/chat_cli.py
    ANSWER_MODEL=gpt-4o python scripts/chat_cli.py   # swap the answering model

Type 'exit' or 'quit' to stop, or 'new'/'reset' to clear conversation memory
and start a fresh multi-turn thread without restarting the process.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_answer import AnswerGenerator


def main() -> None:
    print("Loading NCC bot (embeddings, BM25 index)...")
    generator = AnswerGenerator()
    print(f"Ready. Using model: {generator.model} (override with ANSWER_MODEL env var)")
    print("Type your question, 'new'/'reset' to clear conversation memory, or 'exit'/'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if question.lower() in ("new", "reset"):
            generator.memory.clear()
            print("(conversation memory cleared)\n")
            continue

        result = generator.answer(question)

        # Show the resolved standalone question when it actually differs
        # from what was typed -- makes it visible when conversation memory
        # changed how the question was interpreted (e.g. a follow-up like
        # "what about for walls instead?" resolving pronouns/references
        # using the prior turn).
        if result.get("standalone_question") and result["standalone_question"] != question:
            print(f"(interpreted as: {result['standalone_question']})")

        print("\nBot:", result["answer"])
        if not result.get("blocked") and result["sources"]:
            source_strs = [
                f"{s['clause_code']} (p.{s['page_start']})" if s["clause_code"] else f"p.{s['page_start']}"
                for s in result["sources"]
            ]
            print("Sources:", ", ".join(source_strs))
        print()


if __name__ == "__main__":
    main()
