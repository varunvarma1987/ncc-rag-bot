# NCC RAG Bot — Project Plan & Decision Log

## Goal

Build a Retrieval-Augmented Generation (RAG) chatbot over the National Construction
Code (NCC) 2022 documents, primarily as a **learning project** for understanding how
RAG systems work end-to-end. Priorities, in order:

1. Understand each lever in the RAG pipeline (not just get a working demo).
2. Make deliberate, documented decisions at each step — with reasoning, not defaults.
3. Ship a working bot that can answer questions against the NCC documents.

Code should be heavily commented — every non-obvious choice gets a comment explaining
*why*, not just *what*.

## Source documents

Located in [documents/](documents/):

| File | Size | Notes |
|---|---|---|
| `ncc2022-volume-one.pdf` | ~14.4 MB, ~900 pages | Full NCC 2022 Volume One |
| `ncc2022-volume-onensw.pdf` | ~1 MB | NSW variation/appendix to Volume One |

Reference/example code already collected in [reference/](reference/) (contextual
retrieval, late chunking, GraphRAG intro, advanced multi-query/hybrid patterns) —
these are worked examples to learn from and adapt, not to copy wholesale.

## High-level pipeline stages

1. **PDF → text extraction** (with structure preserved where possible: headings,
   clause numbers, tables — the NCC is heavily cross-referenced and clause-numbered,
   which matters a lot for retrieval quality).
2. **Chunking** the extracted text.
3. **Embedding** the chunks.
4. **Storing** embeddings in a vector database (+ metadata for hybrid/filtered search).
5. **Retrieval** at query time (vector, keyword, hybrid, graph, agentic).
6. **Generation** — feeding retrieved context + query to an LLM.
7. **Serving/hosting** the bot.

Each stage below has open levers. Status legend: 🔴 undecided · 🟡 leaning · 🟢 decided.

---

## Steps taken so far

A plain-English narrative of what we've done, how, and why, in order. See
the Decision log below for the compact per-lever summary; see the Progress
log at the bottom for dated entries.

### Step 1: convert the PDFs to text

Before picking an extraction method, we opened both PDFs and inspected a
handful of pages spread across each document (not just the start, since
front matter looks nothing like body content). That told us: the text is
real embedded text, not a scanned image (no OCR needed), the layout is a
plain single column (no multi-column reflow to worry about), and every page
repeats the same 3-line header block: a section title that changes per page
(e.g. "Fire resistance"), then the constant document title ("NCC 2022
Volume One - Building Code of Australia"), then "Page N". There's no footer.

Given that, plain text extraction was enough — no need for a heavier
layout-analysis library. We used PyMuPDF's `page.get_text()` per page, then
ran a cleanup pass that:
- strips those 3 repeated header lines (by pattern-matching them, not just
  blindly dropping the first N lines, so a page that ever deviates from the
  pattern doesn't lose real content), and
- normalizes stray unicode whitespace characters (a "thin space" and a
  non-breaking space) that PyMuPDF leaves around clause codes, replacing
  them with plain spaces so they don't pollute the text.

We also ran PyMuPDF's table detector on every page as a diagnostic and
found ~28% of pages contain at least one genuine table (spot-checked to
confirm these weren't false alarms). We didn't build special table handling
yet — tables get flattened into plain row-by-row text like everything
else — but we flagged every such page with `has_table: true` in the output
so we can find and revisit them later without re-running extraction.

Output is one JSON record per page (`{doc, page_number, text, has_table}`),
written as JSONL (one JSON object per line) rather than one giant text
blob — this keeps page numbers attached as metadata (needed later so a
retrieved chunk can be cited back as "p.101"), and lets later stages stream
through the document instead of loading an 884-page string into memory at
once.

### Step 2: chunk the text

**The idea.** Rather than cutting the text every N characters (fixed
chunking) or guessing at topic boundaries via embedding similarity
(semantic chunking), we noticed the NCC already tells us exactly where each
rule begins: every clause has its own short code — like `J3D5` or
`NSW I4D33` — printed as a heading, immediately followed by that clause's
title and body. That's a free, document-authored boundary signal, so the
plan was: split the text into one chunk per clause, using those codes as
the cut points.

**First attempt, and why it failed.** We wrote a regex to spot "a line that
looks like a clause code" (a short pattern of letters then digits) directly
on the plain extracted text, and used every match as a chunk boundary. This
badly over-split the document — out of ~2,700 chunks produced, roughly
2,150 were near-empty junk. Digging into why: clause codes don't only
appear as real headings. They also show up (a) in per-section "quick
contents" pages, which list every clause code paired with its title back
to back with no actual rule body (navigation only), and (b) inside
reference tables, where a clause code is a cross-referenced cell value that
happens to land alone on its own line purely because of how the table
wrapped across the page width. Plain text has no way to tell "this line is
a real heading" apart from "this line happens to contain a clause code" —
they're the same string.

**The fix.** We went back to the PDF itself and inspected font metadata
(`page.get_text("dict")`, which exposes font name, size, and bold/italic
flags per span of text) for a known real heading versus a known fake one.
The difference turned out to be clear and consistent: a genuine clause
heading is rendered as an isolated line in `Inter-SemiBold`, size 12, bold —
and nothing else shares that exact style on that line. The contents-listing
codes are `Inter-Regular`, size 10. The table-cell codes are `ArialMT`,
size 10. So detection was moved from "does this text look like a code?" to
"is this text styled like a heading, AND does it look like a code?" — using
the PDF's own visual formatting as the ground truth, not a guess from text
shape alone.

That surfaced one more wrinkle: a small marker at the bottom of nearly
every page — showing the "current" clause code plus a revision date, e.g.
`J3D4` then `(1 May 2023)` — turned out to be styled in the exact same
heading font. It's page furniture (a print-navigation aid), not a real
clause start. The giveaway was that it's always immediately followed by a
bare date-in-parentheses line, which a real heading never is (a real
heading is always followed by its title text) — so that pattern was used to
filter it out.

With verified headers in hand, chunking became straightforward: reassemble
each document's pages into one long string while remembering which
character range came from which page (so every chunk keeps page-number
metadata), then cut a new chunk at each verified header and run it until
the next one. Anything before the very first header (preface, glossary,
table of contents) has no clause structure to exploit, so it's grouped
separately and split with a plain generic recursive splitter instead — fine
for low-value navigation content. Any individual clause that turned out to
be unusually long (e.g. one containing a large embedded table) got a
recursive fallback split too, capped at 3000 characters with a 200-character
overlap so it doesn't lose context at its own internal seams — ordinary
clause chunks get no overlap, since each is already a complete, self-
contained unit and overlap would just duplicate unrelated neighboring rules.

**Result:** ~1,750 clean, complete, single-clause chunks across both
documents (1571 + 182 clause chunks, plus a small number of front-matter
chunks), each carrying page numbers for citation. Spot-checked several by
hand to confirm each chunk is the full clause, nothing more and nothing
less.

### Step 3: add context to each chunk

A clause chunk read in isolation is often ambiguous — "must have a thermal
break..." means nothing without knowing it's clause J3D5, under Energy
Efficiency. Anthropic's Contextual Retrieval approach fixes this with an
LLM call per chunk to write a situating summary. We didn't need that: stage
2's chunking already captured exactly the metadata an LLM would have had to
re-derive anyway (document, NCC Section, clause code, page). So instead we
templated the prefix directly from that metadata — free, instant,
deterministic, and more accurate than an LLM guess since it's ground truth,
not inference.

One piece of metadata had to be added first: the page's Section title (e.g.
"Energy efficiency") was being read during header-stripping in Step 1 but
then thrown away, since at the time it was just boilerplate to strip. Went
back and captured it instead of discarding it, threaded it through
chunking, and used it as the "Section:" line in the prefix.

Each chunk now gets a new `embedding_text` field — the prefix plus the
original clause text — while the original `text` field is left untouched
for clean display/citation later. Example:

```
[Source: NCC 2022 Volume One | Section: Energy efficiency | Clause: J3D5 | Page: 443]
J3D5
 Roof thermal breaks of a sole-occupancy unit of a Class 2 building...
```

Files: `scripts/extract_text.py` (now also captures `section_title`),
`scripts/chunk_text.py` (propagates it per chunk), `scripts/add_context.py`
(new — builds the prefix, writes `contextualized_chunks/*.jsonl`).

### Step 4: embed and store

Picked OpenAI `text-embedding-3-small` — with a corpus this small (~1794
chunks, ~500K tokens), embedding cost is negligible for any mainstream
model, so the decision came down to quality vs. setup simplicity rather
than price. Started with `-large`, then switched to `-small` since this is
a single-domain regulatory document rather than broad open-domain
retrieval, where the quality gap matters less, and the smaller 1536-dim
vector is cheaper to store and search. Set up an OpenAI API key in `.env`
(gitignored) and verified connectivity with live test calls before
committing to the full run.

For the vector database, picked Chroma running embedded (no server
process) — zero infrastructure for a corpus this size, while still
supporting metadata filtering, and it stays easy to swap out later if we
outgrow it.

`scripts/embed_and_store.py` batches all chunks to the embeddings API (100
at a time), then stores each resulting vector in Chroma together with its
metadata (clause code, section, page numbers) — but stores the *original*
clean chunk text as the retrievable content, not the contextualized prefix
version, since that prefix was only there to steer the embedding.

Ran it against all 1794 chunks, then sanity-checked with a real query
("What R-value thermal break is required for a metal roof in a Class 2
building?") — it correctly retrieved clause J3D5 as the top match, with a
clear distance margin ahead of related-but-wrong clauses. First real
end-to-end proof the pipeline works.

### Step 5: hybrid search

Pure vector search has a specific weakness for this corpus: the NCC is
full of short, precise tokens — clause codes, defined terms, numeric
limits — and embeddings are built to capture meaning, not exact surface
form. Two clauses about similar topics can sit close together in embedding
space (they ARE semantically similar), which is exactly wrong when a user
names one specifically.

Built three layers of retrieval instead of one:

1. **Exact clause-code shortcut** — since every chunk already carries its
   real `clause_code` as metadata, a query containing something shaped like
   a code (e.g. "J3D5") skips ranking entirely and goes straight to a
   metadata filter. No ambiguity possible.
2. **BM25** lexical search over the same chunk text as the vector store —
   catches exact term/phrase matches a dense embedding might blur.
3. **Dense vector search** (the existing Chroma setup) — catches meaning
   matches even when wording differs from the source text.

Layers 2 and 3 are merged with Reciprocal Rank Fusion, which combines two
ranked lists by their positions rather than trying to make BM25 scores and
cosine distances comparable (they're on incompatible scales).

Tested with three queries covering each path: an explicit-code query
correctly triggered the exact shortcut; a defined-term query and a broad
conceptual query both came back through the fused path with on-topic,
correctly-ranked clauses.

### Step 6: HNSW tuning (empirically ruled out, for now)

Before touching this, wanted to know whether tuning would even do anything
observable — HNSW's parameters trade search speed for recall, but that
trade-off only shows up once approximate search starts diverging from
exact search, which needs real scale (hundreds of thousands+ vectors) to
kick in. At ~1800 vectors, the graph is small enough that default settings
should already be near-exact.

Tested this rather than assuming it: reused the already-computed embeddings
to build a second Chroma collection with 4x every default HNSW parameter
(`max_neighbors` 16→64, `ef_construction`/`ef_search` 100→400), then ran 4
test queries against both the default and the aggressively-tuned
collection. Result: identical top-5 results, same order, every time.
Confirms the "doesn't matter yet" intuition empirically instead of taking
it on faith — keeping Chroma's defaults, revisit only if the corpus grows
by orders of magnitude.

Also noticed along the way: Chroma's default distance metric is `l2`
(Euclidean), not `cosine`. Turns out this doesn't matter for us either —
OpenAI embeddings are pre-normalized to unit length, and for unit-length
vectors, `l2` distance and cosine similarity produce mathematically
identical rankings.

### Step 7: multi-query retrieval

A single query is only one way of phrasing a question, and both retrieval
methods we have (BM25, dense vectors) only find what's close to THAT
specific phrasing. Added a layer on top of hybrid search: an LLM
(gpt-4o-mini, cheap and fast — matching the pattern already used in the
reference scripts) generates a few alternative phrasings using NCC-style
terminology, hybrid search runs for each one, and all the result sets get
merged with the same Reciprocal Rank Fusion technique already used inside
hybrid search — just applied across queries this time instead of across
retrieval methods.

Verified this actually helps rather than assuming it: ran the literal
phrase "fire escape stairs for apartments" (deliberately avoiding the
NCC's own vocabulary — it never says "escape" or "apartments") through
plain hybrid search first, then through multi-query. Plain hybrid returned
one questionable match (a Special Use Buildings clause with no real
connection to apartment stairs); multi-query replaced it with two
directly-relevant fire-isolated-stairway clauses instead. Both runs shared
one odd result caused by a BM25 keyword coincidence unrelated to either
method's quality — useful to notice, since it's a limit of BM25 itself
rather than something either approach fixes.

### Step 8: generation — the bot can actually answer now

Everything before this stage only ever produced a ranked list of raw
clause chunks. This step adds the missing piece: an LLM reads the
retrieved chunks plus the user's question and writes an actual answer,
grounded strictly in that retrieved text (not its own background knowledge
about building codes, which could be outdated or apply to the wrong
jurisdiction) — with the system prompt requiring it to cite the specific
clause code and page it drew from, and to say so explicitly if the
retrieved context doesn't answer the question.

Used gpt-4o-mini as the default model, but made it swappable via an
`ANSWER_MODEL` environment variable rather than hardcoding it — worth
being able to test a stronger model on genuinely ambiguous questions later
without touching code.

Hit and fixed a real bug while testing: citations were showing up as
"page.None" for anything that came through the hybrid-search fusion path,
because that path built its own trimmed-down metadata dict that happened to
drop page numbers (the exact-clause-code shortcut path didn't have this bug,
since it passes through full Chroma metadata unchanged). Fixed by adding
the missing fields back in.

After the fix, tested two real questions end-to-end — thermal break
R-values for a metal roof, and whether revolving doors are allowed as
required exits — both came back correctly answered with accurate,
verifiable citations (clause code + page number matching the real source).
This is the first point where the whole pipeline (extract → chunk →
contextualize → embed → hybrid+multi-query retrieve → generate) produces
an actual usable answer to a real question.

### Step 9: guardrails — scope restriction and prompt injection resistance

Before letting anyone actually chat with the bot, added defenses against
two things: someone using it as a general-purpose assistant (off-topic
questions), and someone trying to manipulate it into ignoring its
instructions (prompt injection).

Rather than just adding "don't do this" to the existing generation prompt,
built a separate, narrow-purpose classifier call that runs first: it labels
the incoming question — is this really about the NCC? does it look like an
attempt to override instructions? — and is explicitly told to treat the
whole input as text to classify, never as commands to act on. If either
check fails, a fixed refusal goes back immediately, and the main model that
actually writes prose the user reads never even sees the question. The
reasoning: a single call that both reads untrusted input AND has the
authority to act on it is the easiest thing to manipulate — splitting
detection into its own call means a successful injection there can only
produce a wrong label, not a compromised final answer. The main generation
prompt was hardened too, as a second layer, with retrieved context
explicitly marked as inert data rather than instructions.

Tested against 5 questions covering the real case, a plain off-topic
question, two direct injection attempts, and one that tried to smuggle a
real NCC question alongside an injection in the same message. All four
adversarial/off-topic cases were correctly blocked before generation ran;
the legitimate question was answered normally.

### Step 10: fixing a real bug report (false refusals)

A real test question — "what is the standard size of staircase width to
DDA?" — came back with a blunt "I cannot answer this question" even though
retrieval had found relevant clauses. Chasing this down surfaced three
separate, stacked problems, each worth understanding on its own:

1. **The generation prompt conflated two different situations.** It used
   the same refusal phrase for "this is off-topic/an injection attempt"
   (correct) and "the context doesn't fully answer" (wrong) — so a
   perfectly good partial answer (the NCC defers accessibility dimensions
   to an external standard rather than restating figures itself) was being
   thrown away as a flat refusal instead of being explained honestly.

2. **Retrieval itself was inconsistent between runs.** The same question
   sometimes retrieved different chunks. Traced to the multi-query
   rewriting LLM call — first found it was running at `temperature=0.5`
   (fixed to 0), then discovered even `temperature=0` isn't fully
   reproducible on its own (confirmed by literally running the identical
   question 3 times and getting 2 different sets of rewritten phrasings) —
   this matches OpenAI's own documented caveat that seed-based determinism
   is "best effort," not guaranteed. Added a fixed seed everywhere as a
   partial mitigation, and a persistent answer cache as the real fix: a
   repeated question now always returns its first answer instead of
   re-rolling retrieval every time.

3. **The actual retrieval gap, found while debugging #2**: the one chunk in
   the whole corpus that discusses the Disability Discrimination Act shares
   zero tokens with the acronym "DDA" that was actually typed — BM25 had
   nothing to match, and the chunk wasn't semantically close enough for
   dense search to rank it either. It only ever surfaced when an LLM
   rewrite happened to spell the acronym out — luck, not a real fix. Added
   deterministic acronym expansion (DDA → "Disability Discrimination Act",
   plus NCC and BCA) that runs before both BM25 and vector search, so this
   doesn't depend on an LLM guessing right.

The key lesson from stacking these together: caching (fix #2) only makes
an answer *consistent* — it takes fixing the real retrieval gap (fix #3)
to make it *correct*. Caching alone would have just permanently locked in
whichever answer happened to come first, good or bad.

### Step 11 (not started)

Hosting. See the Decision log for the open levers still to work through.

---

## Decision log

### 1. PDF → text conversion
- Status: 🟢 decided
- Inspected both PDFs first (see `scripts/extract_text.py` docstring): real
  embedded text (not scanned/OCR), single-column layout, consistent 3-line
  page header (`<Part title>` / doc title / `Page N`), no footer.
- Decision: plain text extraction via PyMuPDF (`page.get_text()`), with regex
  cleanup to strip the repeated header lines and normalize stray unicode
  whitespace (` `, `\xa0`). Rejected layout-aware/OCR extraction
  (pymupdf_layout, unstructured.io) as unnecessary overhead given clean
  embedded text.
- Tables: ~28% of Volume One pages (250/884) and NSW doc (12/77) contain at
  least one genuine table (spot-checked, not false positives — e.g. reference
  tables, adoption-history tables). Decision for now: flatten tables into
  prose text (same as everything else) but flag each page with
  `has_table: true` in the output, so we can revisit table-specific handling
  later (see lever #5) without re-running extraction from scratch.
- Output: `extracted/ncc2022-volume-one.jsonl` and
  `extracted/ncc2022-volume-onensw.jsonl` — one JSON record per page:
  `{doc, page_number, text, has_table}`. Page-level granularity chosen so
  page numbers survive as metadata for citations at answer time, and so
  chunking (next stage) can work off natural page boundaries + text rather
  than one giant in-memory string.
- Stats: 884 + 77 pages, ~1.76M + ~153K chars, only 3 near-empty pages.

### 2. Chunking strategy
- Status: 🟢 decided
- Decision: **structure-aware splitting** — one chunk per NCC clause (e.g.
  `J3D5`, `NSW I4D33`), using the document's own clause-code headings as
  boundaries, with a recursive (paragraph/line) fallback split only for
  clauses too large to be one chunk. Rejected: fixed-size (would slice
  clauses mid-rule), semantic chunking (would guess at boundaries the
  document already states explicitly — see discussion below). Late chunking
  (a different axis — context-preservation via embedding order, not boundary
  placement) is still undecided, see lever #13. Contextual chunking (also
  context-preservation) IS decided — see below.
- Why not semantic chunking: semantic chunking earns its keep when a
  document has no reliable explicit structure (essays, transcripts) and you
  need to *infer* where topics change. The NCC already marks every rule's
  start explicitly (a clause code + title), so guessing via embedding
  similarity would be strictly worse than just reading the structure that's
  already there.
- Implementation pitfall hit and fixed: an initial version detected clause
  headers via plain-text regex ("does this line look like a clause code?").
  That produced ~2000+ false-positive splits, because clause codes also
  appear in per-section "quick contents" listings (code+title, no real body)
  and inside reference tables (as wrapped cross-reference values). Plain
  text can't distinguish these from a real heading. Fix: moved header
  detection into `scripts/extract_text.py` using PDF font metadata — real
  clause headings are rendered in a distinct style (Inter-SemiBold, size 12,
  bold, alone on their line) vs. contents-listing text (Inter-Regular, size
  10) and table cells (ArialMT, size 10). Also had to filter out a
  bottom-of-page margin marker (current clause + revision date stamp) that
  shared the exact same heading style but is always followed by a bare
  "(<date>)" line.
- Result: 1571 clause chunks + 36 front-matter chunks (Volume One), 182 +
  5 (NSW doc). Spot-checked: chunks are clean, complete, single-clause units
  with page numbers attached for citation (e.g. `scripts/chunk_text.py`
  output for J3D5 is the full clause text, nothing more, nothing less).
- Files: `scripts/extract_text.py` (now also emits `clause_headers` per
  page), `scripts/chunk_text.py` (locates verified headers in page text,
  splits into chunks), output in `chunks/*.jsonl`.

### 2b. Contextual chunking
- Status: 🟢 decided
- Decision: **template-based**, not LLM-generated. Anthropic's original
  Contextual Retrieval approach asks an LLM to write a situating summary
  per chunk — but we already captured the exact metadata an LLM would have
  had to re-derive (document, NCC Section, clause code, page) during
  chunking. Templating the prefix from that metadata is free, instant,
  deterministic, and strictly more accurate than an LLM guess, since it's
  ground truth rather than inference. Would only reach for the LLM-based
  version if we didn't have reliable structured metadata to draw from.
- Implementation note: had to go back and capture `section_title` in
  `scripts/extract_text.py` — it was being read during header-stripping in
  lever #1 but discarded as pure boilerplate at the time. Threaded through
  `chunk_text.py` per chunk, then used in the new `scripts/add_context.py`,
  which builds `embedding_text = "[Source: ... | Section: ... | Clause: ...
  | Page: ...]\n" + text` per chunk. Original `text` field kept untouched
  for citation/display; only `embedding_text` is meant to go to the
  embedding model. Output: `contextualized_chunks/*.jsonl`.

### 3. Max chunk size
- Status: 🟡 leaning (interim value set, not yet validated against retrieval quality)
- Since chunk boundaries are now clause-driven (see lever #2), "max chunk
  size" mainly matters as a fallback cap for the rare oversized clause (one
  spanning multiple pages, e.g. containing a big table). Set to 3000 chars
  as a starting point (~1 dense page of text); actual clause chunks are
  naturally much smaller (median 662 chars, p90 2953). Revisit once we test
  retrieval quality end-to-end — this is a placeholder, not a validated choice.
- Decision: 3000 chars (fallback only), pending real evaluation.

### 4. Chunk overlap
- Status: 🟡 leaning
- Decision: 0 overlap between clause chunks (each is a complete, self-
  contained rule — overlap would just duplicate unrelated neighboring
  clauses). 200-char overlap applied only within the recursive fallback
  split of an oversized clause, so an internally-split clause doesn't lose
  context at its own sub-split seams.

### 5. Content type handling
- Status: 🔴 undecided
- Consideration: NCC has prose, numbered clauses, tables, and diagrams/figures.
  Tables may need special handling (e.g. row-wise serialization) rather than being
  flattened into prose chunks.
- Decision:

### 6. Embedding model
- Status: 🟢 decided
- Decision: **OpenAI `text-embedding-3-small`**, called via API.
- Reasoning: with ~1750 chunks (~500K tokens total), embedding cost is
  negligible for every mainstream option (well under $1 for the whole
  corpus even with the priciest model) — so "cheap" wasn't actually a
  differentiating factor here. Initially picked `text-embedding-3-large`
  for top-tier quality, then switched to `-small` (1536 dims vs. large's
  3072): the corpus is a single-domain regulatory document (not broad
  open-domain retrieval), where the quality gap between small and large
  models tends to matter less, and the smaller vector is cheaper to store
  and faster to search downstream. Matches the `langchain_openai` provider
  already used in the reference scripts (`reference/*.py`). Considered
  Voyage AI `voyage-3-large` (Anthropic's recommended embedding partner)
  but rejected to avoid a second provider/API key at this corpus size.
- Setup: `OPENAI_API_KEY` stored in `.env` (gitignored), loaded via
  `python-dotenv`. Verified connectivity with a live test call — confirmed
  1536-dimension output.

### 7. Embedding dimensions
- Status: 🟡 leaning
- `text-embedding-3-small`'s native output is 1536 dimensions (confirmed via
  live test call). It's a Matryoshka-trained model, so the API also accepts
  a `dimensions` parameter to truncate to fewer dims (trading some accuracy
  for smaller/faster vectors) without re-embedding from scratch.
- Decision so far: use the native 1536 dims for the first full embedding
  pass. Truncation is still open — revisit once we can measure retrieval
  quality (see optimization lever #12), since truncating is a cheap
  experiment to run later on the same embeddings.

### 8. Vector database
- Status: 🟢 decided
- Decision: **Chroma**, running embedded (file-based, no server process).
- Reasoning: for a learning project at this scale (~1750 chunks), Chroma
  gives zero infrastructure overhead — no server to install/run/manage, just
  a Python library and a local persistence folder — while still supporting
  metadata filtering out of the box. Rejected pgvector (would mean standing
  up Postgres just to hold vectors) and Qdrant (a purpose-built server with
  more production-grade knobs than this corpus size needs right now,
  though its explicit HNSW/hybrid-search controls remain worth exploring
  later if we outgrow Chroma). Chroma exposes a similar-enough interface
  that swapping to a heavier DB later, if the learning goals call for it
  (e.g. explicit HNSW tuning), stays realistic.
- Setup: installed `chromadb`. Persistent storage at `vector_store/`,
  collection name `ncc_2022`.
- Implemented in `scripts/embed_and_store.py`: batches chunks (100 at a
  time) to OpenAI's embeddings endpoint, stores the resulting vectors in
  Chroma alongside metadata (doc, clause_code, section_title, chunk_type,
  page_start/end). Stores the ORIGINAL clean `text` as the retrievable
  document content, not the contextualized `embedding_text` — the prefix
  was only meant to steer the embedding, not to appear in what gets shown
  to a user or handed to the LLM later. Re-running the script rebuilds the
  collection from scratch (fine at this stage while upstream chunking
  decisions may still change).
- Result: all 1794 chunks embedded and stored successfully. Ran a live
  end-to-end retrieval test — query "What R-value thermal break is required
  for a metal roof in a Class 2 building?" correctly returned J3D5 as the
  top match with a clear distance margin over unrelated clauses (J3D6, J4D4).

### 9. Hybrid search (vector + keyword)
- Status: 🟢 decided
- Decision: **three-layer retrieval**, not plain dense-only or a plain
  BM25+vector ensemble:
  1. **Exact clause-code shortcut** — if the query contains something
     shaped like a real clause code (reusing the shape pattern from
     extraction), skip ranking entirely and do a direct Chroma metadata
     filter on `clause_code`. Guaranteed correct, since we already have
     ground-truth codes stored as metadata on every chunk.
  2. **BM25** lexical search (via `rank_bm25`) over the same chunk text
     stored in Chroma — good at exact term/phrase overlap (defined terms
     like "Class 2 building", numeric values like "R0.2") that dense
     embeddings can blur.
  3. **Dense vector search** (existing Chroma setup) — good at matching
     meaning when the query's wording differs from the source text.
  Layers 2 and 3 are combined via **Reciprocal Rank Fusion (RRF)**: instead
  of normalizing incompatible score scales (BM25 scores vs. cosine
  distances), RRF only uses each result's RANK in each list, rewarding
  chunks that place well in both over ones that dominate only one.
- Why not vector-only: two clauses that are topically similar (e.g. J3D5
  "roof thermal breaks" vs J3D6 "wall thermal breaks") sit close together
  in embedding space precisely because they're semantically related, which
  actively hurts when a user names one specifically. Lexical scoring
  doesn't have that failure mode.
- Tested live with 3 queries: an explicit-code query ("What does J3D5
  require?") correctly hit the exact-code shortcut; a defined-term query
  ("roof and ceiling insulation R-value requirements") and a broad
  conceptual query ("Class 2 building fire safety") both returned on-topic,
  correctly-ranked clauses via the fused path.
- File: `scripts/hybrid_search.py`.

### 9b. Multi-query retrieval
- Status: 🟢 decided
- Decision: add multi-query as a layer on top of the existing hybrid
  search — an LLM (gpt-4o-mini, matching the model already used for this
  purpose in `reference/advanced_rag (1).py`'s `MultiQueryRetriever`)
  generates a few alternative phrasings of the user's question using NCC-
  style terminology, hybrid search runs for each phrasing, and all result
  sets are merged via Reciprocal Rank Fusion — the same RRF technique
  already used inside hybrid search to combine BM25 and vector rankings,
  just applied one level up, across queries instead of across retrieval
  methods.
- Why: a single query is only one way of phrasing an information need.
  "fire escape stairs for apartments" shares almost no vocabulary with how
  the NCC actually writes about this ("egress", "sole-occupancy unit",
  "fire-isolated stairway") — BM25 gets no term overlap at all, and dense
  search can still miss the mark if the wording gap is wide enough.
- Tested against plain hybrid search on the literal phrasing
  "fire escape stairs for apartments": plain hybrid returned a
  questionable match (`I2D7`, Special use buildings — not really about
  apartment stairs); multi-query replaced it with `D3D5` and `D2D4`
  (fire-isolated/rising-stair-separation clauses), a more relevant set.
  Both runs also surfaced the same one odd match (`SA C4D18`, grain storage)
  — a BM25 term-overlap quirk on "stairs" shared by both paths, not
  something multi-query introduced or fixed.
- Trade-off accepted: one extra LLM call per query (small/cheap model,
  ~3 short phrasings) adds latency and a little cost versus plain hybrid
  search, in exchange for better recall on queries phrased far from the
  document's own vocabulary. The exact clause-code shortcut still applies
  first, from the ORIGINAL query only — rephrasing an already-exact code
  lookup adds noise, not signal.
- File: `scripts/multi_query_search.py`.

### 10. HNSW index tuning
- Status: 🟢 decided (keep Chroma defaults — empirically verified, not assumed)
- Background: HNSW is the approximate-nearest-neighbor graph Chroma builds
  so it doesn't have to brute-force-compare a query against every stored
  vector. Its knobs — `max_neighbors`/M (graph connectivity), `ef_construction`
  (build-time search effort), `ef_search` (query-time search effort) — trade
  speed/memory for recall, but that trade-off only becomes visible once
  approximate search starts diverging from exact search, which happens at
  large scale (hundreds of thousands+ vectors).
- Also found: Chroma's default `space` for the distance metric is `l2`
  (Euclidean), not `cosine`. For OpenAI embeddings specifically this
  doesn't matter — they're pre-normalized to unit length, and for
  unit-length vectors `l2` and `cosine` produce mathematically identical
  rankings (`‖a−b‖² = 2 − 2·cos(a,b)` when ‖a‖=‖b‖=1). Would matter for a
  non-normalized embedding model.
- Test: built a second Chroma collection reusing the SAME already-computed
  embeddings (no new API calls needed — HNSW tuning is a pure index
  concern, unrelated to the embedding model) with 4x every default
  parameter (`max_neighbors` 16→64, `ef_construction`/`ef_search` 100→400).
  Ran 4 test queries against both the default and tuned collections.
- Result: **identical top-5 results, same order, on every single query.**
  Empirically confirms — rather than just asserting — that HNSW tuning has
  no measurable effect at ~1800 vectors. Keeping Chroma's defaults; revisit
  only if the corpus grows by orders of magnitude.
- File: `scripts/tune_hnsw.py` (comparison collection deleted after the test
  — this was a learning exercise, not a production artifact).

### 11. Scaling approach
- Status: 🔴 undecided
- Options: vertical (bigger single machine/instance) vs horizontal (sharding/
  replicas). For a single 900-page doc corpus, likely overkill to horizontally
  scale — worth discussing as a learning exercise regardless.
- Decision:

### 12. Optimization techniques
- Status: 🔴 undecided
- Levers: dimensionality reduction (PCA/Matryoshka truncation), quantization
  (scalar/binary), batching embedding calls, caching (query cache, embedding cache).
- Decision:

### 13. Late chunking
- Status: 🔴 undecided (also listed under #2 — may be evaluated as an alternative
  to traditional chunking rather than a separate stage)
- Decision:

### 14. Agentic RAG
- Status: 🔴 undecided
- Consideration: could let the bot decide to re-query, decompose multi-part
  questions (e.g. "what's required for Class 2 buildings in NSW vs nationally"),
  or call tools (e.g. compare Volume One vs NSW variation documents).
- Decision:

### 15. Graph RAG
- Status: 🔴 undecided
- Consideration: NCC's cross-referencing between clauses (heavy "refer to Clause X")
  could be a natural fit for a knowledge-graph layer on top of vector retrieval.
- Decision:

### 15b. Generation model (LLM answering)
- Status: 🟢 decided (default), explicitly swappable
- Decision: **gpt-4o-mini** as the default answering model, but made
  swappable without touching code — override via an `ANSWER_MODEL`
  environment variable, or by passing `model=` directly to
  `AnswerGenerator`. Deliberately not hardcoded to one model: a stronger
  model would likely handle genuinely ambiguous regulatory questions more
  carefully, which is worth being able to test rather than lock in
  permanently.
- Design: system prompt instructs the model to answer ONLY from the
  retrieved context chunks (not its own background knowledge about
  building codes, which could be outdated or for the wrong jurisdiction),
  to say explicitly when the context doesn't answer the question, and to
  cite the specific clause code(s) and page number(s) used.
- Implementation bug hit and fixed: citations initially showed
  `page_start: None` for every non-exact-match result. Cause: the hybrid
  search fusion path (`hybrid_search.py`) built a stripped-down metadata
  dict from scratch that only carried `doc`/`clause_code`/`section_title`,
  dropping page numbers — while the exact-clause-code shortcut path passed
  through full Chroma metadata, which does include them. Fixed by adding
  `page_start`/`page_end` to the metadata dict on the fused-results path too.
- Tested live: "What R-value thermal break is required for a metal roof in
  a Class 2 building?" → correctly answered from J3D5 with citation
  "(J3D5, p.443)". "Are revolving doors allowed as required exits?" →
  correctly answered "No" from D3D24 with citation "(D3D24, p.207)". Both
  fully grounded in retrieved text, both citations verified correct.
- File: `scripts/generate_answer.py`.

### 15c. Guardrails (scope restriction + prompt injection resistance)
- Status: 🟢 decided
- Decision: **defense-in-depth**, not a single-prompt defense. Two layers:
  1. A **separate classifier call** (`guardrails.py`, `check_query_safety`)
     runs BEFORE any retrieval or generation. Its only job is to label the
     incoming question on two dimensions — `on_topic` (is this genuinely
     about the NCC 2022?) and `injection_attempt` (does this try to make an
     AI assistant ignore/override instructions, reveal its system prompt,
     or change its role?) — and it's explicitly instructed to treat the
     ENTIRE input as untrusted text to classify, never as commands to obey.
     If either check fails, a fixed refusal ("I cannot answer this
     question.") is returned immediately — the main generation model never
     sees the question at all.
  2. The main generation prompt (`generate_answer.py`) is ALSO hardened as
     a second layer, in case anything reaches it despite the gate: the
     retrieved context is wrapped in `<context>` tags and explicitly
     labeled as data to read for facts, never instructions to follow, and
     the model is told to ignore anything in the user's message that tries
     to override its instructions.
- Why not just one hardened prompt: a single LLM call that both reads
  untrusted text AND has authority to act on instructions is the easiest
  thing to manipulate — if an injection partially succeeds there, it can
  influence the actual answer the user sees. Splitting classification into
  its own narrow-purpose call, gating the real generation call behind its
  result, means the classifier's only possible failure mode is a wrong
  label, not a manipulated final answer.
- Classifier fails CLOSED: if its own JSON output is ever malformed, the
  code treats that as `on_topic: false, injection_attempt: true` rather
  than silently letting the request through.
- Not implemented: a numeric retrieval-relevance gate (e.g. refuse if
  nothing sufficiently similar was found in the vector store). Considered
  and rejected for now — RRF fusion always returns its top-k candidates
  regardless of whether they're actually relevant, so an RRF score alone
  can't reliably signal "nothing relevant exists" without a calibrated
  absolute similarity threshold. The classifier gate plus the generation
  prompt's existing "say so if the context doesn't answer" instruction
  cover this case in practice.
- Tested against 5 questions: 1 legitimate NCC question (answered
  correctly), 1 plain off-topic question, 2 direct injection attempts, and
  1 mixed attempt that smuggled a real NCC question alongside an injection
  attempt in the same message. All 4 adversarial/off-topic cases were
  correctly blocked before reaching the generation model; the legitimate
  question passed through and was answered normally.
- UX fix added on top: a plain greeting ("hi", "hello", "good morning")
  was initially being flagged as off-topic by the guardrail and refused —
  bad first impression. Added a deterministic regex short-circuit
  (`is_greeting()`) that catches messages that are ONLY a greeting and
  returns a canned "Hi I am NCC bot, how can I help!" response before the
  guardrail classifier even runs (cheaper and faster than an LLM call for
  something this simple). Anchored to match the WHOLE message, so a real
  question with a polite opener ("hi, what does J3D5 say about roofs?")
  still goes through the normal pipeline rather than getting short-circuited.
- File: `scripts/guardrails.py` (new), `scripts/generate_answer.py` (wired
  in as the first step of `AnswerGenerator.answer()`).

### 15d. Bug: false refusals on partial-answer questions
- Status: 🟢 fixed
- Symptom (real user report): "what is the standard size of staircase
  width to DDA?" returned "I cannot answer this question" despite
  retrieval finding relevant chunks (D2D9, D2D11, D3D10, and on some runs
  TAS D1P10).
- Root cause #1 (prompt logic): the generation system prompt used the same
  exact refusal phrase for two different situations — "off-topic/injection
  attempt" (correct use) AND "context doesn't fully answer" (wrong use).
  The NCC actually defers accessibility dimensions to an external
  referenced standard (AS 1428.1) rather than restating specific figures
  in its own text, so "no number found" was getting treated as "can't
  answer at all" instead of "here's what IS established, here's what's
  missing." Fixed by explicitly instructing the model to give a partial
  answer (what the context DOES establish, what it doesn't cover) instead
  of refusing outright, and reserving the exact refusal phrase for the
  guardrail-relevant cases only.
- Root cause #2 (retrieval instability): investigating this surfaced that
  the SAME question could retrieve DIFFERENT chunks across repeated runs.
  Traced to `multi_query_search.py`'s query-rewriting LLM call running at
  `temperature=0.5` (fixed to `0`), and even after that, `temperature=0`
  alone still wasn't fully reproducible — confirmed empirically (3 runs of
  the identical question produced 2 different sets of rewritten phrasings).
  This matches OpenAI's own documented caveat: seed-based determinism is
  "best effort", not guaranteed. Added a fixed `seed=42` to every OpenAI
  call in the pipeline (query rewriting, guardrail classifier, generation)
  as a partial mitigation, plus a persistent answer cache (`answer_cache.py`,
  keyed by model+question) as the real fix for consistency — a repeated
  question now always returns its first answer rather than re-rolling the
  dice on retrieval every time.
- Root cause #3 (the actual retrieval gap, found while debugging #2): the
  ONLY chunk in the entire 1794-chunk corpus that mentions the "Disability
  Discrimination Act" by name is `TAS D1P10`. It shares zero tokens with
  the acronym "DDA" the user actually typed, so BM25 had nothing to match;
  dense vector search didn't rank that short chunk in its top 20 either.
  The clause was only ever found when multi-query's LLM-generated rewrite
  happened to spell the acronym out in full — a coin flip, not something
  to depend on. Fixed properly (not just papered over with caching) by
  adding deterministic acronym expansion (`expand_acronyms()` in
  `hybrid_search.py`) that runs BEFORE both BM25 and vector search:
  known regulatory acronyms (DDA, NCC, BCA) get their full name appended to
  the query text. After this fix, `TAS D1P10` reliably ranks #1 for this
  query on every run, cache or no cache.
- Lesson: caching (root cause #2's fix) makes an answer CONSISTENT, but
  only fixing the actual retrieval gap (root cause #3) makes it CORRECT.
  Both were needed — caching alone would have permanently locked in
  whichever answer happened to come first, good or bad.
- Files touched: `scripts/generate_answer.py` (prompt fix, cache wiring),
  `scripts/multi_query_search.py` (temperature/seed), `scripts/guardrails.py`
  (seed), `scripts/hybrid_search.py` (acronym expansion), `scripts/answer_cache.py`
  (new).

### 15e. Bug: false refusal on "plaster board" thickness question
- Status: 🟢 fixed
- Symptom (real user report): "minimum thickness of plaster board to be
  used in buildings?" returned "I cannot answer this question" even though
  the corpus genuinely contains the answer (S28C7: "13 mm fire-protective
  grade plasterboard").
- Root cause #1 (compound-word tokenization mismatch): the user wrote
  "plaster board" (two words); the corpus spells it "plasterboard" (one
  word). BM25 tokenizes on whitespace, so these share zero token identity —
  same class of problem as the earlier DDA acronym bug. Fixed the same way:
  added deterministic compound-term normalization (`normalize_compound_terms()`
  in `hybrid_search.py`) that appends the merged spelling to the query
  before search, generalized alongside acronym expansion into one
  `expand_query()` entry point.
- Root cause #2 (found AFTER fix #1 still didn't fully solve it — a deeper,
  more systemic issue): even after adding "plasterboard" to the query text,
  BM25 still didn't rank the real answer chunk (S28C7) in its top 20. Dug
  into why: the chunk's actual wording uses "thick" ("150 mm **thick**
  concrete panel...") while the query says "**thickness**" — plain
  tokenization treats these as two unrelated tokens with zero overlap, the
  same failure mode as "board"/"boards" or "required"/"requirement" would
  hit anywhere else in this heavily-inflected regulatory document. This
  isn't specific to one query or one term pair — it's a property of BM25
  with no stemming, and would recur constantly across arbitrary questions.
- Fixed generally, not with another one-off word-pair patch: added a Porter
  stemmer (`nltk.stem.PorterStemmer`, no external data/download needed — the
  classic algorithm is self-contained) to `tokenize()`, so "thick" and
  "thickness" both reduce to the same root token ("thick") and collide
  correctly in BM25. Verified this doesn't disturb clause-code matching
  (`j3d5` stems to itself unchanged). This is a systemic fix that should
  help many future word-form mismatches, not just this specific case.
- Verified no regressions: re-ran the full existing test set (thermal break
  question, revolving doors, exact clause-code lookup, off-topic refusal,
  injection attempt, greeting) after the stemming change — all still
  correct.
- Compounding lesson with 15d: two of the last two real bugs found by
  actually testing the bot were both lexical-matching gaps (acronyms,
  compound words, word-form mismatches) that an LLM-based query rewrite
  sometimes papered over by luck but couldn't be relied on to fix — pointing
  at deterministic query/tokenization normalization as the durable fix,
  not better prompting or more retries.
- Files touched: `scripts/hybrid_search.py` (`normalize_compound_terms()`,
  `expand_query()`, stemming added to `tokenize()`), `nltk` added as a
  dependency.

### 15f. LLM model upgrade (gpt-4o-mini → gpt-5-mini)
- Status: 🟢 decided
- Decision: upgraded the default model for ALL three LLM call sites in the
  pipeline (guardrail classifier, multi-query rewriting, main generation)
  from `gpt-4o-mini` to `gpt-5-mini` — a genuine capability step up while
  staying in the cheap/fast "mini" tier rather than jumping to a full-size
  model. Still fully swappable per-call-site via environment variables
  (`GUARDRAIL_MODEL`, `QUERY_REWRITE_MODEL`, `ANSWER_MODEL`), unchanged from
  before.
- Real API incompatibility hit and fixed: gpt-5-class models reject an
  explicit `temperature` override entirely ("Unsupported value: 'temperature'
  does not support 0 with this model. Only the default (1) value is
  supported") — a hard API constraint on this model family, discovered by
  testing before committing to the switch, not assumed. Fixed generally:
  added `scripts/model_utils.py` (`build_chat_kwargs()`), which every LLM
  call in the pipeline now routes through, so `temperature` is only
  included in the request for models that actually accept overriding it.
  `seed` is always included regardless (still works fine without an
  explicit temperature).
- Verified no regressions: re-ran the full existing test set (thermal
  break, plasterboard, off-topic refusal, injection attempt, greeting)
  after the switch — all correct, and the plasterboard answer came back
  noticeably more thorough (listed multiple example thicknesses with
  citations rather than a single figure).
- Files touched: `scripts/model_utils.py` (new), `scripts/guardrails.py`,
  `scripts/multi_query_search.py`, `scripts/generate_answer.py`.

### 15g. LLM-generated contextual chunking (upgrade from template-only)
- Status: 🟢 decided and complete (full corpus re-contextualized, re-embedded, tested)
- Decision: upgrade lever #2b from template-ONLY contextual chunking to
  **template + LLM-generated description**. The template prefix
  (doc/section/clause/page) is kept as-is — it's free, instant, and 100%
  accurate, since it's built from metadata chunking already extracted.
  Added on top: a one-sentence, LLM-written description of what each chunk
  covers in plain language, explicitly naming any related defined terms,
  external Acts, or referenced Standards — including their common acronym
  in parentheses when one exists.
- Why revisit this (see 15d): the original template-only decision noted
  "would only reach for the LLM-based version if we didn't have reliable
  structured metadata to draw from" — true for WHERE a chunk sits in the
  document, but the DDA bug (15d) showed the gap isn't location, it's
  VOCABULARY. A chunk about "a Standard made under the Disability
  Discrimination Act" is unreachable by a query using "DDA" unless
  something bridges that gap. We'd patched that one specific case with
  hardcoded deterministic acronym expansion in hybrid_search.py — useful,
  but doesn't scale to gaps we haven't hit yet. An LLM reading the actual
  chunk content can proactively surface these connections per-chunk,
  rather than requiring a hand-written fix after every individual failure.
- Explicitly NOT Anthropic's original approach: their Contextual Retrieval
  conditions the LLM on the FULL surrounding document so it can describe a
  chunk's place in a much larger narrative. Rejected here as overkill for
  the actual gap being closed — Volume One alone is ~1.76M characters, and
  conditioning every one of ~1750 per-chunk calls on that would be far
  slower and more expensive than what's needed. Each chunk is described
  using only its own text + the section/clause metadata already on hand.
- First-pass prompt didn't fully solve the motivating case: asked the LLM
  to "name relevant external Acts/Standards including any common acronym"
  and got a full, correct description of TAS D1P10's Disability
  Discrimination Act content -- but without the "(DDA)" shorthand actually
  written out. Tightened the prompt to explicitly require the acronym in
  parentheses after the full name; re-tested and confirmed "(DDA)" now
  appears.
- Cost/time: ~1794 short LLM calls (small input, ~1-sentence output),
  run concurrently via a thread pool (12 workers). Estimated ~809K input
  tokens + ~72K output tokens total (computed from actual chunk sizes) --
  nowhere near the cost of Anthropic's full-document-context version.
- Full run completed successfully (1607 + 187 chunks). Spot-checked output
  quality on the known problem chunks -- e.g. TAS D1P10's description now
  reads "Building or part of a building must be accessible per Standards
  made under the Disability Discrimination Act 1992 (Cth) (DDA)..." and
  S28C7's reads "...Rw and Rw + Ctr) in the National Construction Code
  (NCC) 2022" -- both correctly surfacing the acronym bridge we needed.
- Re-ran `scripts/embed_and_store.py` to re-embed the full corpus with the
  updated `embedding_text`, then re-ran the full regression test suite
  (thermal break, plasterboard, DDA staircase, revolving doors, off-topic,
  injection, greeting) against the refreshed vector store. All correct --
  and the plasterboard answer actually got MORE precise than before the
  upgrade: it now cites an exact "13 mm minimum" figure from S5C10 directly,
  rather than only listing example thicknesses. Concrete evidence the
  contextual embedding upgrade improved retrieval quality, not just
  maintained the deterministic-fix baseline from 15d/15e.
- File: `scripts/add_context.py` (rewritten), `scripts/embed_and_store.py`
  (re-run, no code changes needed there).

### 15h. Multi-turn conversation memory
- Status: 🟢 decided and implemented
- Problem: the bot was fully stateless between questions — confirmed by
  checking the code (`AnswerGenerator.answer()` took only a single
  question, no history). A natural follow-up like "what about for walls
  instead?" carries no useful information in isolation — retrieval has no
  idea what "instead" refers to, so a bare follow-up would retrieve close
  to random chunks.
- Decision: **question condensing**, a standard RAG pattern — before
  retrieval runs, a separate LLM call (`conversation.py`, reusing the
  gpt-5-mini default) is shown the recent conversation history plus the new
  follow-up, and rewrites it into a fully self-contained "standalone
  question" (pronouns/references resolved). That standalone question is
  what actually flows into the guardrail check, retrieval, caching, and
  generation — not the raw follow-up, and not a naive concatenation of
  history + question.
- Why condensing has to happen BEFORE retrieval, not just at the final
  generation step: by the time generation runs, retrieval has already
  happened. If the fix were only "give the generation model the history,"
  the WRONG chunks would already be locked in — condensing has to fix the
  question upstream of retrieval to do any good.
- Design choices: memory is a plain in-memory list (`ConversationMemory`,
  no persistence across process restarts — fine for a single chat session),
  capped to the last 3 turns so the condensing prompt doesn't grow
  unbounded. Greetings are checked BEFORE condensing (cheapest possible
  check first) and are never added to memory, since they're not part of
  the real Q&A thread and would just add noise to later condensing calls.
  Blocked/refused answers are also excluded from memory for the same
  reason. The answer cache (lever from 15d) is now keyed on the STANDALONE
  question rather than the raw one, since that's what actually determines
  retrieval — the same raw follow-up text can resolve to different
  standalone questions in different conversations.
- Tested live: asked "What R-value thermal break is required for a metal
  roof in a Class 2 building?" (answered correctly, J3D5), then followed up
  with "What about for walls instead?" — correctly condensed into a full
  standalone question about metal WALLS in a Class 2 building, correctly
  retrieved J3D6 (the wall-specific analog of J3D5), and correctly cited
  R0.2 for walls.
- Added a `new`/`reset` command to `chat_cli.py` to clear conversation
  memory mid-session without restarting the process, and the CLI now shows
  the resolved standalone question when it differs from what was typed
  (transparency into how a follow-up was interpreted).
- Files: `scripts/conversation.py` (new), `scripts/generate_answer.py`
  (wired in), `scripts/chat_cli.py` (reset command, standalone-question display).

### 16. Hosting
- Status: 🔴 undecided
- Options: local-only (learning/dev), simple single-server deploy (FastAPI +
  Chroma/pgvector), managed vector DB + serverless API, full containerized deploy.
- Decision:

---

## Open questions for next session

- Confirm which embedding/LLM provider(s) we have API access to already.
- Decide whether to build this as a Python script pipeline, a notebook, or an app
  (e.g. FastAPI + simple frontend) from the start.
- Decide evaluation approach: how will we know if a chunking/retrieval choice is
  actually better? (Need a small set of test questions with known correct answers
  from the NCC text.)

## Progress log

- 2026-09-03: Project scoped. Source PDFs and reference example scripts confirmed
  present. This plan file created. No implementation started yet.
- 2026-09-03: Lever #1 (PDF → text) decided and implemented. Wrote
  `scripts/extract_text.py`, ran it on both PDFs, validated output quality
  (header/footer stripped correctly, tables flagged, near-empty pages
  minimal). Extracted JSONL files in `extracted/`.
- 2026-09-03: Lever #2 (chunking strategy) decided and implemented as
  structure-aware clause splitting. Hit and fixed a false-positive detection
  bug (plain-text regex confused TOC listings and table cells with real
  clause headings); fixed by detecting headings via PDF font metadata
  instead. Wrote `scripts/chunk_text.py`, output in `chunks/*.jsonl`
  (1571 + 182 clause chunks). Levers #3 (max chunk size) and #4 (overlap)
  given interim values as a side effect, not yet independently validated.
- 2026-09-03: Contextual chunking (lever #2b) decided as template-based
  (not LLM-generated), since chunking already captures the metadata needed.
  Went back to capture `section_title` in `scripts/extract_text.py`
  (previously discarded during header-stripping), threaded it through
  `scripts/chunk_text.py`, and wrote `scripts/add_context.py` to build the
  final `embedding_text` per chunk. Output in `contextualized_chunks/*.jsonl`.
- 2026-09-03: Embedding model (lever #6) decided: OpenAI
  `text-embedding-3-small`. Set up `OPENAI_API_KEY` in `.env` (gitignored),
  installed `openai`/`tiktoken`, verified connectivity with live test calls
  (initially tested `-large` at 3072 dims, then switched to `-small` at
  1536 dims per final decision). Embedding dimensions (lever #7) leaning
  toward native 1536, truncation experiments deferred.
