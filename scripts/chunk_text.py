"""
Stage 2 of the RAG pipeline: text -> chunks.

Decision (see PROJECT.md, lever #2): structure-aware splitting, not generic
recursive/fixed/semantic chunking. Why: the NCC's own authors already mark
exactly where each rule starts -- a standalone "clause code" line like
"J3D5" or "NSW I4D33" -- immediately followed by the clause's title. That's
a free, accurate boundary signal; there's no reason to guess at boundaries
(semantic chunking) or use blind size-based cuts (fixed/recursive) when the
document tells us where the real units are.

IMPORTANT: a first version of this script found clause headers by regex on
plain text alone ("does this line look like a clause code?"). That produced
huge numbers of false positives -- clause codes also show up in per-section
"quick contents" listings (code + title, no real body) and inside reference
tables (as wrapped cross-reference cell values). Plain text can't tell those
apart from a real heading. So header detection was moved into
scripts/extract_text.py, which checks the PDF's font metadata instead (real
headers are rendered in a distinctive style -- see that file's docstring)
and writes a verified "clause_headers" list into each page's JSON record.
This script trusts that list and just needs to locate WHERE each verified
header sits in the page's plain text.

Strategy per document:
  1. Reassemble all pages into one long string, but remember which page
     range each character position came from (so each chunk can carry
     page_start/page_end for citations).
  2. For each page's verified clause_headers (from stage 1), locate that
     exact header line within the page's text and record its absolute
     character offset. Each located header starts a new chunk; the chunk
     runs until the next header (or end of document).
  3. Anything BEFORE the first clause code (preface, table of contents,
     glossary front matter) has no clause structure to exploit -- it gets
     grouped separately and split with a plain recursive/size-based cut,
     since guessing boundaries there is fine (it's low retrieval-value
     navigation content anyway).
  4. Any individual clause chunk that's unusually large (multi-page clauses,
     e.g. ones containing big reference tables) gets recursively sub-split
     on paragraph/line boundaries so no single chunk blows past MAX_CHARS.

Chunk overlap: 0 between clause chunks -- each clause is already a complete,
self-contained unit, so overlap would just duplicate unrelated neighboring
rules. Overlap IS applied within the recursive fallback split of an
oversized clause, so a sub-split clause doesn't lose context at its own
internal seams.
"""

import json
import re
from bisect import bisect_right
from pathlib import Path

EXTRACTED_DIR = Path(r"c:\Varun\NCC bot\extracted")
OUT_DIR = Path(r"c:\Varun\NCC bot\chunks")

# Fallback size limit for the recursive sub-split of an oversized clause.
# Chosen as a starting point roughly matching one "screen" of dense
# regulatory text -- revisit once we test retrieval quality (see PROJECT.md
# lever #3, still open).
MAX_CHARS = 3000
OVERLAP_CHARS = 200


def recursive_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Plain size-based recursive split, used only as a fallback.

    Tries paragraph breaks first, then line breaks, then just hard-cuts by
    character count -- this is the generic technique, deliberately NOT used
    as the primary strategy since the clause structure is more reliable.
    """
    if len(text) <= max_chars:
        return [text]

    for separator in ["\n\n", "\n", ". "]:
        parts = text.split(separator)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                candidate = current + separator + part if current else part
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # Start the next chunk with the tail of the previous one
                    # (the overlap) so context isn't lost at the seam.
                    current = (current[-overlap:] + separator + part) if current else part
            if current:
                chunks.append(current)
            return chunks

    # No separators worked (single giant unbroken string) -- hard cut.
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars - overlap)]


def find_header_line(page_text: str, code: str, search_from: int) -> int | None:
    """Find the absolute (within page_text) start offset of `code` sitting
    alone on its own line, searching forward from `search_from`.

    Whitespace-tolerant: extract_text.py's plain-text cleanup can leave
    different spacing around a header (e.g. a converted thin-space) than the
    exact string captured from font-metadata spans, so we rebuild the code's
    parts (e.g. "NSW" / "I4D33") joined by "any amount of whitespace" rather
    than requiring an exact character match.
    """
    parts = code.split(" ")
    pattern = re.compile(
        r"(?m)^[ \t]*" + r"\s+".join(re.escape(p) for p in parts) + r"[ \t]*$"
    )
    m = pattern.search(page_text, search_from)
    return m.start() if m else None


def build_page_index_with_headers(pages: list[dict]) -> tuple[str, list[int], list[int], list[tuple[int, str]]]:
    """Join page texts into one string, recording:
      - offsets: char offset each page starts at (for page_for_offset lookup)
      - page_numbers: parallel list of page numbers
      - header_positions: (absolute_offset, clause_code) for every
        font-verified header, located within the page text it came from.
    """
    full_text_parts = []
    offsets = []
    page_numbers = []
    header_positions: list[tuple[int, str]] = []
    pos = 0

    for page in pages:
        offsets.append(pos)
        page_numbers.append(page["page_number"])
        page_text = page["text"]

        search_from = 0
        for code in page["clause_headers"]:
            local_offset = find_header_line(page_text, code, search_from)
            if local_offset is None:
                # Shouldn't normally happen -- the header came from this same
                # page's text. If cleanup ever mangles a line badly enough
                # that it can't be found, skip it rather than mis-locate it.
                print(f"  WARNING: could not locate header {code!r} on page {page['page_number']}")
                continue
            header_positions.append((pos + local_offset, code))
            search_from = local_offset + len(code)

        full_text_parts.append(page_text)
        pos += len(page_text) + 1  # +1 for the "\n" joiner added below

    full_text = "\n".join(full_text_parts)
    return full_text, offsets, page_numbers, header_positions


def page_for_offset(offset: int, offsets: list[int], page_numbers: list[int]) -> int:
    idx = bisect_right(offsets, offset) - 1
    idx = max(0, min(idx, len(page_numbers) - 1))
    return page_numbers[idx]


def chunk_document(doc_name: str, pages: list[dict]) -> list[dict]:
    full_text, offsets, page_numbers, header_positions = build_page_index_with_headers(pages)
    chunks: list[dict] = []

    # Front matter: everything before the first verified clause header.
    if header_positions:
        front_matter = full_text[: header_positions[0][0]].strip()
    else:
        front_matter = full_text.strip()

    # Look up each page's section_title (e.g. "Energy efficiency") by page
    # number, so a chunk can carry the section it belongs to as metadata --
    # this is what the contextual-chunking prefix (see add_context.py) uses
    # to situate a chunk without needing an LLM call to describe it.
    section_title_by_page = {p["page_number"]: p.get("section_title") for p in pages}

    if front_matter:
        for piece in recursive_split(front_matter, MAX_CHARS, OVERLAP_CHARS):
            page_start = page_for_offset(0, offsets, page_numbers)
            chunks.append({
                "doc": doc_name,
                "clause_code": None,
                "chunk_type": "front_matter",
                "section_title": section_title_by_page.get(page_start),
                "page_start": page_start,
                "page_end": page_for_offset(len(front_matter) - 1, offsets, page_numbers),
                "text": piece,
            })

    # One chunk per clause, split at each verified header to the next.
    for i, (start, clause_code) in enumerate(header_positions):
        end = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(full_text)
        clause_text = full_text[start:end].strip()

        page_start = page_for_offset(start, offsets, page_numbers)
        page_end = page_for_offset(end - 1, offsets, page_numbers)

        pieces = recursive_split(clause_text, MAX_CHARS, OVERLAP_CHARS)
        for piece in pieces:
            chunks.append({
                "doc": doc_name,
                "clause_code": clause_code,
                "chunk_type": "clause",
                "section_title": section_title_by_page.get(page_start),
                "page_start": page_start,
                "page_end": page_end,
                "text": piece,
            })

    return chunks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for jsonl_name in ["ncc2022-volume-one.jsonl", "ncc2022-volume-onensw.jsonl"]:
        in_path = EXTRACTED_DIR / jsonl_name
        pages = [json.loads(line) for line in open(in_path, encoding="utf-8")]
        doc_name = pages[0]["doc"]

        chunks = chunk_document(doc_name, pages)

        out_path = OUT_DIR / jsonl_name
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        n_clauses = sum(1 for c in chunks if c["chunk_type"] == "clause")
        n_front = sum(1 for c in chunks if c["chunk_type"] == "front_matter")
        print(f"{doc_name}: {len(chunks)} chunks ({n_clauses} clause, {n_front} front_matter) -> {out_path}")


if __name__ == "__main__":
    main()
