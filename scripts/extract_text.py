"""
Stage 1 of the RAG pipeline: PDF -> text.

Decision (see PROJECT.md, lever #1): plain text extraction + regex cleanup,
NOT layout-aware/OCR extraction. Why that's enough here: we inspected both
PDFs and found real embedded text (not scanned images), a single-column
layout, and only a handful of genuine tables across ~960 combined pages.
A layout-analysis library would add a heavy dependency for very little gain.

We DO keep tables flagged (via PyMuPDF's find_tables()) rather than silently
flattening them, so we know exactly which pages to revisit if a table-heavy
answer ever comes back wrong.

Output: one JSONL file per source PDF, one JSON record per page:
    {"doc": "...", "page_number": 101, "text": "...", "has_table": false,
     "clause_headers": ["J3D5", "J3D6"], "section_title": "Energy efficiency"}

Why page-level JSONL and not one giant text blob:
  - We need page_number preserved as metadata so later, when a chunk is
    retrieved, we can cite "NCC 2022 Volume One, p.101" back to the user.
  - JSONL (one JSON object per line) is easy to stream/chunk later without
    loading an 884-page document into memory as a single string.

Why "clause_headers" is captured here (and not guessed later from plain
text): a first attempt at chunking used plain-text regex to spot lines that
LOOK like a clause code (e.g. "J3D5" alone on a line). That produced
hundreds of false positives, because clause codes also appear: (a) in
per-section "quick contents" listings that pair every code with its title
but have no actual rule body, and (b) inside reference tables, as a
cross-referenced value that happens to land alone on a wrapped table row.
Plain text has no way to tell these apart -- but the PDF's own styling does:
inspecting font metadata (page.get_text("dict")) showed real clause headings
are ALWAYS rendered as an isolated line in Inter-SemiBold, size 12, bold,
with nothing else sharing that line -- while contents-listing codes are
Inter-Regular size 10, and table-cell codes are ArialMT size 10. So this
extraction stage now checks visual styling, not just text shape, to decide
what's really a clause boundary.
"""

import json
import re
from pathlib import Path

import pymupdf as fitz

# A clause code's text shape, e.g. "J3D5", "NSW I4D33", "S14C2": 1-4 upper
# letters + digits + 1 upper letter + digits, optional state-name prefix.
# This alone is NOT enough to identify a real header (see docstring above) --
# it's combined with the font check below.
CLAUSE_CODE_SHAPE_RE = re.compile(r"^(?:[A-Z]{2,4} )?[A-Z]{1,4}\d+[A-Z]\d+$")

# The exact styling real clause headings use in this PDF (found by inspecting
# font metadata on known real vs. fake header lines). Matched loosely
# ("SemiBold" in font name, size within a small tolerance) so minor per-PDF
# font-subset naming differences don't silently break detection.
HEADER_FONT_NAME_HINT = "SemiBold"
HEADER_FONT_SIZE_MIN = 11.5

# Every page carries a small margin marker near the bottom -- e.g. "J3D4"
# followed immediately by "(1 May 2023)" -- showing the current amendment
# date. It's styled IDENTICALLY to a real clause heading (same font/size/
# bold), so the font check alone can't tell them apart. But it's always
# immediately followed by a bare "(<day> <month> <year>)" line, which a real
# heading never is (a real heading is always followed by its title text).
# That gives us a reliable way to exclude it.
REVISION_DATE_LINE_RE = re.compile(r"^\(\d{1,2} \w+ \d{4}\)$")

DOCS_DIR = Path(r"c:\Varun\NCC bot\documents")
OUT_DIR = Path(r"c:\Varun\NCC bot\extracted")

# The constant middle header line every page repeats. We match it loosely
# (both PDFs share the same document title text) so it can be stripped
# regardless of which Part/Section title sits above it.
DOC_TITLE_LINE = "NCC 2022 Volume One - Building Code of Australia"

# Matches the standalone "Page 123" header line.
PAGE_NUM_LINE_RE = re.compile(r"^Page \d+\s*$")

# PDF text extraction leaves stray unicode spacing characters around clause
# codes (thin space \u200a, non-breaking space \xa0). Normalize these to
# regular spaces so downstream chunking/embedding sees clean text.
WHITESPACE_CHARS_RE = re.compile(r"[\u200a\xa0]")


def clean_page_text(raw_text: str) -> tuple[str, str | None]:
    """Strip the repeated 3-line page header and normalize odd whitespace.

    Every page's raw text starts with:
        <Part/Section title>       (varies per page)
        NCC 2022 Volume One - Building Code of Australia   (constant)
        Page <n>                   (constant pattern)
    followed by a blank-ish line, then the actual body content. We drop all
    three header lines since they're pure navigation noise -- keeping them
    would mean every single chunk in the vector store contains the same
    boilerplate sentence, wasting embedding capacity on non-content tokens.

    The section title line is returned separately (not discarded) -- it's
    the NCC Section this page belongs to (e.g. "Energy efficiency", "Fire
    resistance"), which is exactly the kind of "where does this chunk sit in
    the document" metadata a contextual-chunking prefix needs later. No
    reason to throw it away here and try to re-derive it downstream.
    """
    text = WHITESPACE_CHARS_RE.sub(" ", raw_text)
    lines = text.split("\n")

    # The header is always the first two-to-three lines. Walk forward only
    # while we still recognize header-shaped lines, so we don't accidentally
    # eat real body content if a page ever deviates from the pattern.
    i = 0
    section_title = None
    if i < len(lines) and lines[i].strip():
        section_title = lines[i].strip()
        i += 1  # Part/Section title line (e.g. "Fire resistance")
    if i < len(lines) and lines[i].strip() == DOC_TITLE_LINE:
        i += 1
    if i < len(lines) and PAGE_NUM_LINE_RE.match(lines[i].strip()):
        i += 1

    body = "\n".join(lines[i:])
    return body.strip(), section_title


def find_clause_headers(page: fitz.Page) -> list[str]:
    """Return clause codes on this page that are genuine headings, verified
    by their visual styling (not just text shape -- see module docstring).
    """
    # First pass: flatten the page into an ordered list of (text, is_candidate)
    # so we can look one line ahead when deciding whether a candidate is real.
    page_dict = page.get_text("dict")
    ordered_lines: list[tuple[str, bool]] = []
    for block in page_dict["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue

            # Real headers are an ISOLATED line: every span on it shares the
            # heading style. A line mixing heading font with body font (e.g.
            # a title line that wraps) is not a bare clause-code line.
            is_heading_style = all(
                HEADER_FONT_NAME_HINT in s["font"] and s["size"] >= HEADER_FONT_SIZE_MIN
                for s in spans
            )
            line_text = WHITESPACE_CHARS_RE.sub("", "".join(s["text"] for s in spans)).strip()
            is_candidate = is_heading_style and bool(CLAUSE_CODE_SHAPE_RE.match(line_text))
            ordered_lines.append((line_text, is_candidate))

    headers = []
    for i, (line_text, is_candidate) in enumerate(ordered_lines):
        if not is_candidate:
            continue
        # Exclude the bottom-margin revision-date marker (see constant above)
        # -- it shares the exact heading style but is always followed by a
        # bare "(<date>)" line, which a real clause heading never is.
        next_text = ordered_lines[i + 1][0] if i + 1 < len(ordered_lines) else ""
        if REVISION_DATE_LINE_RE.match(next_text):
            continue
        headers.append(line_text)

    return headers


def extract_pdf(pdf_path: Path, out_path: Path) -> None:
    doc = fitz.open(pdf_path)
    print(f"{pdf_path.name}: {doc.page_count} pages")

    with open(out_path, "w", encoding="utf-8") as f:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            raw_text = page.get_text()
            cleaned, section_title = clean_page_text(raw_text)

            # Cheap table detection so we can flag (not yet specially handle)
            # pages worth revisiting -- see PROJECT.md lever #5.
            try:
                has_table = len(page.find_tables().tables) > 0
            except Exception:
                has_table = False

            record = {
                "doc": pdf_path.name,
                "page_number": page_index + 1,  # 1-indexed to match printed "Page N"
                "text": cleaned,
                "has_table": has_table,
                "clause_headers": find_clause_headers(page),
                "section_title": section_title,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    doc.close()
    print(f"  -> wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_name in ["ncc2022-volume-one.pdf", "ncc2022-volume-onensw.pdf"]:
        pdf_path = DOCS_DIR / pdf_name
        out_path = OUT_DIR / (pdf_path.stem + ".jsonl")
        extract_pdf(pdf_path, out_path)


if __name__ == "__main__":
    main()
