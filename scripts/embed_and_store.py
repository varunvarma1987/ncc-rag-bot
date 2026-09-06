"""
Stage 3 of the RAG pipeline: contextualized chunks -> embeddings -> vector DB.

Decisions this script implements (see PROJECT.md):
  - Embedding model (lever #6): OpenAI text-embedding-3-small. Chosen because
    embedding this whole corpus (~1750 chunks, ~500K tokens) costs well
    under $1 with any mainstream model, so cost wasn't the deciding factor --
    quality and setup simplicity were. `-small` was picked over `-large`
    because this is a single-domain corpus (one regulatory document), where
    the quality gap between the two models matters less than in broad
    open-domain retrieval, and the smaller 1536-dim vector is cheaper to
    store/search.
  - Embedding dimensions (lever #7): native 1536, no truncation yet --
    truncating is a cheap experiment we can run later on the SAME embeddings
    if needed, so there's no reason to decide it prematurely.
  - Vector database (lever #8): Chroma, running embedded (no server process)
    with on-disk persistence. Picked for zero infrastructure overhead at
    this corpus size, while keeping metadata filtering available.

What "embedding" means here concretely: we send each chunk's
`embedding_text` (the contextualized version built in add_context.py --
prefix + original clause text) to the OpenAI API, which returns a fixed-length
list of 1536 floats (a vector) that captures the text's meaning in a form we
can compare mathematically (cosine/dot-product similarity) against a query's
own vector at retrieval time. We store that vector in Chroma, but we store
the ORIGINAL clean `text` (not embedding_text) as the retrievable document
content -- the contextual prefix was only useful to steer the embedding
toward the right meaning, we don't want it cluttering what gets shown back
to a user or fed to the LLM later.

Batching: OpenAI's embeddings endpoint accepts many inputs in one request.
We batch (default 100 chunks/request) rather than one call per chunk, purely
for speed -- ~1794 chunks would mean ~1794 round-trips otherwise.
"""

import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CONTEXTUALIZED_DIR = Path(r"c:\Varun\NCC bot\contextualized_chunks")
VECTOR_STORE_DIR = Path(r"c:\Varun\NCC bot\vector_store")
COLLECTION_NAME = "ncc_2022"

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def load_all_chunks() -> list[dict]:
    chunks = []
    for jsonl_name in ["ncc2022-volume-one.jsonl", "ncc2022-volume-onensw.jsonl"]:
        with open(CONTEXTUALIZED_DIR / jsonl_name, encoding="utf-8") as f:
            chunks.extend(json.loads(line) for line in f)
    return chunks


def make_chunk_id(chunk: dict, index: int) -> str:
    # Doc + page + running index keeps ids unique even for duplicate clause
    # codes (e.g. a clause re-opened in a state-specific appendix -- see
    # PROJECT.md lever #2 notes on A1G4 appearing twice).
    doc_stem = chunk["doc"].replace(".pdf", "")
    return f"{doc_stem}_p{chunk['page_start']}_{index}"


def make_metadata(chunk: dict) -> dict:
    # Chroma metadata values must be str/int/float/bool -- no None -- so
    # front_matter chunks (which have clause_code=None) get an empty string
    # instead of null.
    return {
        "doc": chunk["doc"],
        "chunk_type": chunk["chunk_type"],
        "clause_code": chunk["clause_code"] or "",
        "section_title": chunk["section_title"] or "",
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
    }


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set -- check .env")

    client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    # Fresh collection each run -- this is stage 3 of a pipeline we're still
    # tuning (chunking/context decisions upstream may still change), so we
    # want re-running this script to reflect the latest chunks, not append
    # duplicates on top of a stale collection.
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL, "embedding_dimensions": 1536},
    )

    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks. Embedding in batches of {BATCH_SIZE}...")

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[c["embedding_text"] for c in batch],
        )
        embeddings = [item.embedding for item in response.data]

        collection.add(
            ids=[make_chunk_id(c, batch_start + i) for i, c in enumerate(batch)],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[make_metadata(c) for c in batch],
        )

        done = batch_start + len(batch)
        print(f"  {done}/{len(chunks)} chunks embedded and stored")

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} vectors.")
    print(f"Persisted to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
