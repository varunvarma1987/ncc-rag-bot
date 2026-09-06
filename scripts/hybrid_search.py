"""
Stage 4 of the RAG pipeline: retrieval, specifically hybrid search.

Decision (see PROJECT.md, lever #9): dense vector search ALONE isn't enough
here. The NCC is full of short, precise tokens -- clause codes (J3D5,
NSW I4D33), defined terms (Class 2 building), numeric limits (R0.2) -- and
embeddings are built to capture MEANING, not exact surface form. Two
different clauses about similar topics (e.g. J3D5 "roof thermal breaks" vs
J3D6 "wall thermal breaks") can sit close together in embedding space
precisely because they're semantically similar, even when a user wants one
specific one by name. A pure lexical/keyword method doesn't have that
problem: it directly rewards exact term overlap.

So this implements THREE layers, combined:

  1. Exact clause-code shortcut. We already store `clause_code` as exact
     metadata on every chunk (see embed_and_store.py). If the query itself
     contains something shaped like a real clause code, there's no reason
     to rank/guess at all -- we do a direct metadata filter and return
     those chunks with top priority, guaranteed correct.

  2. BM25 lexical search. A classic keyword-scoring algorithm (term
     frequency, weighted by how rare/common each term is across the whole
     corpus) run over the same chunk text as the vector store. Good at
     exact phrase/term matches (e.g. "Class 2 building") that a query might
     phrase very similarly to the source text.

  3. Dense vector search (from embed_and_store.py / Chroma). Good at
     matching MEANING even when the query's wording differs from the
     source text (e.g. "fire escape stairs" retrieving a clause that says
     "egress" and "stairway" but never the word "escape").

Layers 2 and 3 are combined with Reciprocal Rank Fusion (RRF): instead of
trying to make BM25 scores and cosine-similarity scores comparable (they're
on completely different scales), RRF only looks at each result's RANK in
each list and combines those. A chunk ranked #1 by BM25 and #3 by vector
search scores higher overall than one ranked #1 by vector search alone but
absent from BM25's results -- appearing near the top of BOTH lists is
rewarded more than dominating just one.
"""

import json
import re
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
from openai import OpenAI
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent))
from embed_and_store import load_all_chunks, make_chunk_id, EMBEDDING_MODEL, VECTOR_STORE_DIR, COLLECTION_NAME

load_dotenv()

# Same clause-code shape used during extraction (extract_text.py), but
# applied to free-form user queries here -- looser on surrounding context
# since a user might type "what does J3D5 say" or just "J3D5".
CLAUSE_CODE_IN_QUERY_RE = re.compile(r"\b(?:[A-Z]{2,4} )?[A-Z]{1,4}\d+[A-Z]\d+\b")

# RRF's smoothing constant. Higher values flatten the influence of rank
# differences near the top; 60 is the commonly cited default from the
# original RRF paper and works fine without tuning for a corpus this size.
RRF_K = 60

# Deterministic acronym expansion, applied to every query BEFORE search.
# Found via a real failure case: a question using "DDA" couldn't retrieve
# the one chunk in the whole corpus that discusses the Disability
# Discrimination Act, because that chunk spells the name out in full and
# shares zero tokens with the bare acronym "DDA" -- BM25 had nothing to
# match, and dense vector search didn't rank the (short, low-similarity)
# chunk in its top 20 either. The only reason it was ever found at all was
# when multi-query's LLM-generated rewrite happened to expand the acronym
# itself -- a coin flip, not something to depend on. Expanding known
# acronyms up front makes this deterministic instead of relying on LLM luck.
# Not meant to be exhaustive -- just the acronyms actually hit in testing.
ACRONYM_EXPANSIONS = {
    "DDA": "Disability Discrimination Act",
    "NCC": "National Construction Code",
    "BCA": "Building Code of Australia",
}
ACRONYM_RE = re.compile(r"\b(" + "|".join(ACRONYM_EXPANSIONS) + r")\b")


def expand_acronyms(query: str) -> str:
    """Append the expansion alongside the acronym rather than replacing it,
    so a query that also uses the acronym as a real word/clause-code-like
    token elsewhere still matches on that surface form too.
    """
    matches = set(ACRONYM_RE.findall(query))
    if not matches:
        return query
    expansions = " ".join(ACRONYM_EXPANSIONS[m] for m in matches)
    return f"{query} ({expansions})"


# Same idea as acronym expansion, but for the opposite direction: a query
# written as separate words needs to also match the corpus's single-word
# spelling. Found via a real failure case: "minimum thickness of plaster
# board" (two words) retrieved the wrong clauses, because the corpus text
# spells it "plasterboard" (one word) and BM25 tokenizes on whitespace --
# "plaster"+"board" as two tokens shares no token identity with the single
# token "plasterboard". Even after multi-query generated ONE rewritten
# variant using the correct single-word spelling, the other rewritten
# variants (using "gypsum board", "wall lining") still pulled RRF fusion
# weight toward a different, topically-adjacent-but-wrong set of clauses --
# so relying on an LLM rewrite to happen to use the right spelling wasn't
# reliable here either. Fixed the same way as acronyms: deterministic
# normalization before search, not a hope that phrasing lines up.
COMPOUND_TERM_FIXES = {
    "plaster board": "plasterboard",
    "plaster-board": "plasterboard",
}
COMPOUND_TERM_RE = re.compile("|".join(re.escape(k) for k in COMPOUND_TERM_FIXES), re.IGNORECASE)


def normalize_compound_terms(query: str) -> str:
    matches = {m.group(0).lower() for m in COMPOUND_TERM_RE.finditer(query)}
    if not matches:
        return query
    merged_forms = {COMPOUND_TERM_FIXES[m] for m in matches}
    return f"{query} ({' '.join(merged_forms)})"


def expand_query(query: str) -> str:
    """Apply all deterministic query-side fixes before BM25/vector search."""
    return normalize_compound_terms(expand_acronyms(query))


# Found via a real failure case: a chunk describing "150 mm thick concrete
# panel with ... plasterboard" scored essentially zero for the query
# "minimum THICKNESS of plaster board" -- "thick" and "thickness" are
# obviously the same underlying concept to a person, but plain tokenization
# treats them as two unrelated tokens with zero overlap. Same problem for
# "required"/"requirement", "install"/"installation", etc. -- extremely
# common in a regulatory document that uses many word forms of the same
# root concept. A stemmer reduces words to a shared root ("thick", "requir")
# so these variants collide into the same BM25 token instead of missing
# each other. This is a systemic fix, not a one-off patch like the acronym/
# compound-term dictionaries above -- it should help many future queries
# with word-form mismatches, not just this specific one.
_stemmer = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization for BM25, with stemming.

    A clause code like "J3D5" tokenizes+stems to "j3d5" either way (the
    Porter stemmer doesn't touch tokens that already look like codes/
    numbers), so exact-code matching still works through this path too (as
    a secondary signal alongside the metadata shortcut above).
    """
    return [_stemmer.stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]


class HybridSearcher:
    def __init__(self):
        self.openai_client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

        # BM25 needs the full corpus in memory, tokenized, at startup -- fine
        # at ~1800 chunks. We index the same `text` field stored in Chroma
        # as `documents`, not the contextualized embedding_text, so both
        # retrieval paths are scored against the same visible content.
        self.chunks = load_all_chunks()
        self.chunk_ids = [make_chunk_id(c, i) for i, c in enumerate(self.chunks)]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def _exact_clause_matches(self, query: str) -> list[dict]:
        codes = set(CLAUSE_CODE_IN_QUERY_RE.findall(query))
        if not codes:
            return []

        results = []
        for code in codes:
            got = self.collection.get(where={"clause_code": code})
            for doc_id, doc_text, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                results.append({"id": doc_id, "text": doc_text, "metadata": meta, "match_type": "exact_code"})
        return results

    def _bm25_ranked_ids(self, query: str, top_n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [self.chunk_ids[i] for i in ranked_indices]

    def _vector_ranked_ids(self, query: str, top_n: int) -> list[str]:
        q_embedding = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=query
        ).data[0].embedding
        results = self.collection.query(query_embeddings=[q_embedding], n_results=top_n)
        return results["ids"][0]

    def search(self, query: str, k: int = 5, fusion_pool: int = 20) -> list[dict]:
        """Return up to k results: exact clause-code matches first (if any),
        then RRF-fused BM25 + vector results filling the remainder.
        """
        exact_matches = self._exact_clause_matches(query)
        exact_ids = {m["id"] for m in exact_matches}

        # Deterministic query fixes (acronym expansion, compound-term
        # normalization) apply to BM25/vector search only -- the
        # exact-clause-code shortcut above works off the raw query text and
        # doesn't need either fix.
        expanded_query = expand_query(query)
        bm25_ids = self._bm25_ranked_ids(expanded_query, fusion_pool)
        vector_ids = self._vector_ranked_ids(expanded_query, fusion_pool)

        # RRF: sum 1/(RRF_K + rank) across both ranked lists per chunk id.
        rrf_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, doc_id in enumerate(vector_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        fused_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

        id_to_chunk = {cid: chunk for cid, chunk in zip(self.chunk_ids, self.chunks)}
        results = list(exact_matches)
        for doc_id in fused_ids:
            if doc_id in exact_ids:
                continue  # already returned via the exact-match shortcut
            if len(results) >= k:
                break
            chunk = id_to_chunk[doc_id]
            results.append({
                "id": doc_id,
                "text": chunk["text"],
                "metadata": {
                    "doc": chunk["doc"], "clause_code": chunk["clause_code"] or "",
                    "section_title": chunk["section_title"] or "",
                    "page_start": chunk["page_start"], "page_end": chunk["page_end"],
                },
                "match_type": "hybrid",
                "rrf_score": round(rrf_scores[doc_id], 5),
            })

        return results[:k] if not exact_matches else results


def main() -> None:
    searcher = HybridSearcher()

    test_queries = [
        "What does J3D5 require?",
        "roof and ceiling insulation R-value requirements",
        "Class 2 building fire safety",
    ]

    for query in test_queries:
        print("=" * 70)
        print("QUERY:", query)
        for r in searcher.search(query, k=3):
            print(f"  [{r['match_type']}] {r['metadata'].get('clause_code') or '(no code)'} "
                  f"| {r['metadata'].get('section_title')}")
            print("   ", r["text"][:120].replace("\n", " "))


if __name__ == "__main__":
    main()
