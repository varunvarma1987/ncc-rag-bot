"""
Stage 2.5 of the RAG pipeline: chunks -> contextualized chunks.

Decision (see PROJECT.md, lever on contextual/late chunking): UPGRADED from
template-only to template + LLM-generated context. Originally this stage
only built a deterministic metadata prefix (doc/section/clause/page) --
free, instant, and fully accurate, since chunking already extracted that
metadata directly from the document. That's still here and still the
backbone of embedding_text, because it's ground truth, not a guess.

What's new: an LLM-generated one-sentence description is now ALSO added
per chunk, describing what the chunk covers in plain language -- including
any related concepts, defined terms, or externally-referenced standards/
legislation it touches on. Why add this on top of an already-accurate
template: the template tells an embedding WHERE a chunk sits in the
document, but says nothing about WHAT it means in language a person might
actually search with. A real failure case exposed this gap: a chunk about
"a Standard made under the Disability Discrimination Act" was essentially
unreachable by a query using the acronym "DDA", because nothing in the
template or the raw clause text bridges that gap -- we ended up having to
hand-patch that specific case with deterministic acronym expansion in
hybrid_search.py. An LLM-generated description, written with the chunk's
actual content in hand, can proactively surface these connections (e.g.
naturally writing out "Disability Discrimination Act (DDA)" or similar)
for cases we haven't manually discovered yet, rather than requiring us to
hardcode a fix after every individual failure.

What this is NOT: Anthropic's original Contextual Retrieval passes the
model the FULL surrounding document (or a large chunk of it) so it can
describe a chunk's precise place in a much larger narrative. We don't do
that here -- Volume One alone is ~1.76M characters, and conditioning on
that for every one of ~1750 chunks would be slow and needlessly expensive
for what we need. Instead each chunk is described using only its own text
plus the metadata we already have (section/clause) -- narrower context,
but sufficient for the actual gap we're closing (surfacing related
terminology and concepts), at a fraction of the cost and latency.

Cost/time: ~1750 short LLM calls (small input, ~1-sentence output). Run
concurrently (a thread pool, since these are independent, I/O-bound calls)
to keep wall-clock time reasonable rather than looping one call at a time.

Output: each chunk's "embedding_text" = template prefix + LLM-generated
description + original text. The original "text" field is untouched, for
citation/display.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import build_chat_kwargs

load_dotenv()

CHUNKS_DIR = Path(r"c:\Varun\NCC bot\chunks")
OUT_DIR = Path(r"c:\Varun\NCC bot\contextualized_chunks")

CONTEXT_MODEL = os.environ.get("CONTEXT_MODEL", "gpt-5-mini")
MAX_WORKERS = 12  # concurrent API calls -- independent per-chunk work, no ordering dependency

# Friendly display names for the prefix, since the raw PDF filenames aren't
# what we'd want an embedding (or a user-facing citation) to show.
DOC_DISPLAY_NAMES = {
    "ncc2022-volume-one.pdf": "NCC 2022 Volume One",
    "ncc2022-volume-onensw.pdf": "NCC 2022 Volume One - NSW Variations",
}

CONTEXT_GENERATION_PROMPT = """You are preparing a chunk of the NCC 2022 (Australia's National Construction Code) for a search index.

Section: {section_title}
Clause code: {clause_code}

Chunk text:
{text}

Write ONE short sentence (max ~30 words) in plain language describing what this chunk covers -- phrased the way a person might naturally search for it. If the chunk relates to a defined term, an external Act, or a referenced Standard, name it explicitly -- and if that term has a common acronym (e.g. "Disability Discrimination Act (DDA)", "National Construction Code (NCC)"), always write the acronym in parentheses right after the full name, even if the chunk text itself doesn't use the acronym. Do not repeat the clause code. Output ONLY the sentence, nothing else."""


def build_prefix(chunk: dict) -> str:
    """Build the situating prefix from metadata already captured during
    chunking -- no LLM call, no guessing. Still the reliable backbone.
    """
    doc_display = DOC_DISPLAY_NAMES.get(chunk["doc"], chunk["doc"])
    parts = [f"Source: {doc_display}"]

    if chunk.get("section_title"):
        parts.append(f"Section: {chunk['section_title']}")

    if chunk.get("clause_code"):
        parts.append(f"Clause: {chunk['clause_code']}")

    if chunk["page_start"] == chunk["page_end"]:
        parts.append(f"Page: {chunk['page_start']}")
    else:
        parts.append(f"Pages: {chunk['page_start']}-{chunk['page_end']}")

    return "[" + " | ".join(parts) + "]"


def generate_description(client: OpenAI, chunk: dict) -> str:
    prompt = CONTEXT_GENERATION_PROMPT.format(
        section_title=chunk.get("section_title") or "(none)",
        clause_code=chunk.get("clause_code") or "(none)",
        text=chunk["text"],
    )
    response = client.chat.completions.create(
        model=CONTEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        **build_chat_kwargs(CONTEXT_MODEL),
    )
    return response.choices[0].message.content.strip()


def add_context(client: OpenAI, chunk: dict) -> dict:
    prefix = build_prefix(chunk)
    description = generate_description(client, chunk)
    chunk["llm_description"] = description
    chunk["embedding_text"] = f"{prefix}\n{description}\n{chunk['text']}"
    return chunk


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    for jsonl_name in ["ncc2022-volume-one.jsonl", "ncc2022-volume-onensw.jsonl"]:
        in_path = CHUNKS_DIR / jsonl_name
        out_path = OUT_DIR / jsonl_name

        chunks = [json.loads(line) for line in open(in_path, encoding="utf-8")]
        results = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            future_to_index = {
                pool.submit(add_context, client, chunk): i
                for i, chunk in enumerate(chunks)
            }
            done = 0
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                results[i] = future.result()
                done += 1
                if done % 100 == 0 or done == len(chunks):
                    print(f"  {jsonl_name}: {done}/{len(chunks)} chunks contextualized")

        with open(out_path, "w", encoding="utf-8") as f_out:
            for chunk in results:
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"{jsonl_name}: {len(results)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
