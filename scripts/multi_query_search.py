"""
Stage 4b of the RAG pipeline: multi-query retrieval, layered on top of the
hybrid search built in hybrid_search.py.

The idea: a single query is just one way of phrasing an information need,
and retrieval (dense OR lexical) only finds what's close to THAT specific
phrasing. If a user asks "fire escape stairs" but the NCC always says
"egress" and "stairway" and never "escape", dense search can still miss it
if the wording gap is big enough, and BM25 will miss it outright (no shared
terms). Multi-query fixes this by asking an LLM to generate a few
alternative phrasings of the SAME question, running our existing hybrid
search for each one, and merging all the result sets -- so a chunk only
needs to match ONE of several phrasings to surface, not the user's exact
wording.

Model choice for generating the variants: gpt-4o-mini, matching the pattern
already used in reference/advanced_rag (1).py's MultiQueryRetriever setup.
This is a good fit for query rewriting specifically -- it's a small,
cheap, fast task (rephrase a sentence a few ways), not the harder job of
actually answering the user's question from retrieved context (that's the
generation stage, lever still to come, where a stronger model choice will
matter more).

Fusion approach: each of the (original query + N variants) produces its own
ranked hybrid-search result list. We treat each list as one more ranked
list to fuse via Reciprocal Rank Fusion (RRF) -- same technique already
used inside hybrid_search.py to combine BM25 and vector rankings, just
applied one level up, across QUERIES instead of across retrieval methods.
The exact clause-code shortcut (if the ORIGINAL query names a real clause)
still takes priority over everything -- rephrasing a query that already
contains an exact code adds noise, not signal, for that part of the query.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_search import HybridSearcher, RRF_K
from model_utils import build_chat_kwargs

load_dotenv()

QUERY_REWRITE_MODEL = os.environ.get("QUERY_REWRITE_MODEL", "gpt-5-mini")
NUM_VARIANTS = 3

QUERY_REWRITE_PROMPT = """You are helping search a building code document (the Australian NCC).
Given a user's question, generate {n} alternative phrasings of the SAME question that might use different terminology a building code would use (e.g. "egress" instead of "escape", "sole-occupancy unit" instead of "apartment").

Return ONLY the {n} alternative phrasings, one per line, no numbering, no extra commentary.

Question: {query}"""


class MultiQuerySearcher:
    def __init__(self):
        self.hybrid_searcher = HybridSearcher()
        self.openai_client = OpenAI()

    def _generate_variants(self, query: str, n: int = NUM_VARIANTS) -> list[str]:
        response = self.openai_client.chat.completions.create(
            model=QUERY_REWRITE_MODEL,
            messages=[{"role": "user", "content": QUERY_REWRITE_PROMPT.format(n=n, query=query)}],
            # Deterministic on purpose: the same question should retrieve the
            # same chunks every time. A previous run at temperature=0.5 was
            # observed to generate different query rewrites (and therefore
            # different retrieved chunks) across runs for the identical
            # question -- a real reliability problem for a bot people need
            # to trust to find the same answer twice. temperature=0 alone
            # wasn't quite enough -- OpenAI's own docs note temperature=0
            # is only "mostly deterministic", not guaranteed, due to
            # internal batching/hardware nondeterminism. Adding a fixed
            # `seed` makes the API return "best effort" reproducible output
            # for identical requests (per OpenAI's documented behavior).
            # build_chat_kwargs handles model-specific quirks -- gpt-5-class
            # models reject an explicit temperature override entirely (only
            # their default of 1 is accepted), so temperature is only
            # included for models that actually support overriding it.
            **build_chat_kwargs(QUERY_REWRITE_MODEL),
        )
        lines = [line.strip() for line in response.choices[0].message.content.split("\n")]
        return [line for line in lines if line]

    def search(self, query: str, k: int = 5, per_query_pool: int = 10) -> dict:
        """Returns {"variants": [...], "results": [...]} so callers/demo code
        can see what queries were actually searched, not just the final list.
        """
        variants = self._generate_variants(query)
        all_queries = [query] + variants

        # The exact clause-code shortcut from the ORIGINAL query only --
        # rephrasing a query that names a real code doesn't help find it,
        # since we already have a guaranteed-correct path for that case.
        exact_matches = self.hybrid_searcher._exact_clause_matches(query)
        exact_ids = {m["id"] for m in exact_matches}

        # Run hybrid search for every phrasing, collect each as its own
        # ranked list of chunk ids for cross-query RRF fusion.
        id_to_result: dict[str, dict] = {}
        ranked_id_lists: list[list[str]] = []
        for q in all_queries:
            results = self.hybrid_searcher.search(q, k=per_query_pool)
            ranked_id_lists.append([r["id"] for r in results])
            for r in results:
                id_to_result.setdefault(r["id"], r)

        rrf_scores: dict[str, float] = {}
        for ranked_ids in ranked_id_lists:
            for rank, doc_id in enumerate(ranked_ids):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        fused_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

        final_results = list(exact_matches)
        for doc_id in fused_ids:
            if doc_id in exact_ids:
                continue
            if len(final_results) >= k:
                break
            result = dict(id_to_result[doc_id])
            result["match_type"] = "multi_query"
            result["rrf_score"] = round(rrf_scores[doc_id], 5)
            final_results.append(result)

        return {"variants": variants, "results": final_results[:k] if not exact_matches else final_results}


def main() -> None:
    searcher = MultiQuerySearcher()

    test_queries = [
        "fire escape stairs for apartments",
        "how much insulation does a roof need",
    ]

    for query in test_queries:
        print("=" * 70)
        print("QUERY:", query)
        outcome = searcher.search(query, k=4)
        print("Generated variants:")
        for v in outcome["variants"]:
            print("  -", v)
        print("Results:")
        for r in outcome["results"]:
            print(f"  [{r['match_type']}] {r['metadata'].get('clause_code') or '(no code)'} "
                  f"| {r['metadata'].get('section_title')}")
            print("   ", r["text"][:120].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
