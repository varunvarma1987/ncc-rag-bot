"""
Learning exercise for lever #10 (HNSW index tuning) -- see PROJECT.md.

HNSW (Hierarchical Navigable Small World) is the graph structure Chroma
builds under the hood to make vector search fast: instead of comparing a
query to every single stored vector (exact/brute-force search), it builds
a multi-layer graph where each vector is connected to a handful of its
nearest neighbors, and search "walks" the graph toward the query point.
That's what makes vector databases scale to millions of vectors -- but it's
an APPROXIMATION, not exact search, and its three main knobs trade off
speed, memory, and recall (how often the true nearest neighbors actually
get found):

  - max_neighbors (often called "M" in HNSW literature): how many graph
    connections each vector keeps to its neighbors. Higher = a richer,
    more connected graph = better recall, but more memory and slower to
    build. Chroma's default is 16.
  - ef_construction: how hard the algorithm searches for good neighbors
    WHILE BUILDING the graph. Higher = a better-quality graph, but slower
    to build (build-time only cost, doesn't affect query speed). Default
    100.
  - ef_search: how hard it searches AT QUERY TIME. Higher = checks more
    candidates before returning results = better recall, slower per
    query. This is the one you'd typically tune live to trade latency for
    accuracy. Default 100 (Chroma's default is actually already generous
    here).

IMPORTANT LEARNING POINT this script demonstrates empirically: these knobs
only matter once approximate search actually starts diverging from exact
search -- which happens at large scale (hundreds of thousands+ vectors).
At our ~1800 vectors, the HNSW graph is so small and densely connected
relative to the data that default settings are already returning
essentially exact results. So we build a SECOND collection with much more
aggressive settings (bigger graph, harder search effort) and compare its
results against the default collection to see whether it changes anything.
Expectation: little to no difference -- proving the "doesn't matter yet"
point rather than just asserting it.

We reuse the embeddings already computed and stored by embed_and_store.py
(via collection.get(include=[...])) rather than re-calling the OpenAI API --
tuning HNSW is a pure vector-index concern, it has nothing to do with what
embedding model produced the vectors.
"""

import sys
from pathlib import Path

import chromadb
from chromadb.api.collection_configuration import CreateCollectionConfiguration, CreateHNSWConfiguration
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from embed_and_store import VECTOR_STORE_DIR, COLLECTION_NAME, EMBEDDING_MODEL

load_dotenv()

TUNED_COLLECTION_NAME = "ncc_2022_tuned_hnsw"

# Deliberately more aggressive than Chroma's defaults (M=16, ef_construction=100,
# ef_search=100), to give the comparison the best possible chance of showing
# a difference if one exists at this corpus size.
TUNED_HNSW = CreateHNSWConfiguration(
    space="cosine",       # equivalent to l2 for OpenAI's unit-normalized embeddings -- see module docstring
    max_neighbors=32,      # 2x the default graph connectivity
    ef_construction=200,   # 2x the default build-time search effort
    ef_search=200,         # 2x the default query-time search effort
)


def build_tuned_collection(chroma_client: chromadb.ClientAPI) -> chromadb.api.models.Collection.Collection:
    default_collection = chroma_client.get_collection(COLLECTION_NAME)
    all_data = default_collection.get(include=["embeddings", "documents", "metadatas"])

    try:
        chroma_client.delete_collection(TUNED_COLLECTION_NAME)
    except Exception:
        pass

    tuned_collection = chroma_client.create_collection(
        name=TUNED_COLLECTION_NAME,
        configuration=CreateCollectionConfiguration(hnsw=TUNED_HNSW),
        metadata={"embedding_model": EMBEDDING_MODEL, "note": "same vectors as ncc_2022, HNSW params tuned"},
    )
    tuned_collection.add(
        ids=all_data["ids"],
        embeddings=all_data["embeddings"],
        documents=all_data["documents"],
        metadatas=all_data["metadatas"],
    )
    return tuned_collection


def compare(query: str, openai_client: OpenAI, default_col, tuned_col, k: int = 5) -> None:
    q_embedding = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=query).data[0].embedding

    default_results = default_col.query(query_embeddings=[q_embedding], n_results=k)
    tuned_results = tuned_col.query(query_embeddings=[q_embedding], n_results=k)

    default_ids = default_results["ids"][0]
    tuned_ids = tuned_results["ids"][0]

    tuned_label = (f"M={TUNED_HNSW['max_neighbors']}, "
                   f"ef_construction={TUNED_HNSW['ef_construction']}, "
                   f"ef_search={TUNED_HNSW['ef_search']}")

    print(f"QUERY: {query}")
    print(f"  default (M=16, ef_construction=100, ef_search=100): {default_ids}")
    print(f"  tuned   ({tuned_label}): {tuned_ids}")
    print(f"  identical top-{k} ids and order: {default_ids == tuned_ids}")
    print()


def main() -> None:
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    openai_client = OpenAI()

    print("Building comparison collection with tuned HNSW settings "
          "(reusing existing embeddings, no new API calls)...")
    tuned_collection = build_tuned_collection(chroma_client)
    default_collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"Done. {tuned_collection.count()} vectors in tuned collection.\n")

    test_queries = [
        "What does J3D5 require?",
        "roof and ceiling insulation R-value requirements",
        "Class 2 building fire safety",
        "accessible adult change facilities requirements",
    ]
    for query in test_queries:
        compare(query, openai_client, default_collection, tuned_collection)


if __name__ == "__main__":
    main()
