# Chunking robustness — Phase 1.5 notes

Hardening notes for PDF parsing + chunking against real, professor-exported PDFs,
captured before Phase 2 (real embedder) freezes chunk quality into vectors. Run the
health gate on any file with `kb chunks <course> <path> --report`.

## Fixes shipped this phase

| # | Fix | Evidence that motivated it |
|---|-----|----------------------------|
| 1 | Drop elements below `min_element_chars` (10) non-whitespace chars | TemplateMethod leaked an 8-char junk chunk from a near-empty page |
| 2 | Reading order: top-down by default, column-major **only on detected two-column pages** | see below — an unguarded column sort reordered 235 of 404 real pages |
| 3 | Documented + tested that `char_range` indexes the chunk **body**, not the title-prepended `text` | invariant `text.endswith(element_text[cs:ce])` |
| 4 | Oversize awareness: `estimate_tokens ≈ chars/4`, warn on ingest, count in `--report` | Final Exam has 7 chunks > ~256 tokens |
| 5 | Page-number/footer lines (`Page N of M`, `n/m`, roman) stripped and never chosen as title | Exam titled every chunk `"Page N of 28"` |
| 6 | Narrow character fold: Latin ligatures + curly quotes → ASCII | `ﬁ` (U+FB01) ×270 in the Graph-Theory notes, so `deﬁnition` never matched a query typed `definition` |

### Fix 2 in detail — why column sorting needs a guard

Two-column scrambling is real: PyMuPDF's raw block order interleaves columns, and
`twocol.pdf` (right column written first) extracts as `Beta1…Beta8 | Alpha1…Alpha8`.

But sorting **every** page by `(column, y)` with the column taken as "bbox centre vs page
midpoint" is far more destructive than the problem it fixes. Real decks are full of
~95%-width blocks whose centres fall either side of the fold by a fraction of a point, so
those pages get shuffled. Measured across 10 real course PDFs (404 pages), comparing
`(column, y)` order against plain `y` order:

| File | Pages | Reordered, unguarded | Reordered, guarded |
|------|-------|---------------------:|-------------------:|
| Enumeration.pdf | 115 | 106 | 0 |
| Graph theory.pdf | 122 | 76 | 0 |
| Sample 2 - Final Exam.pdf | 27 | 27 | 0 |
| lect-EmbeddedSQL-handout.pdf | 44 | 7 | 0 |
| Consumption and Commodities | 37 | 7 | 0 |
| a5.pdf | 6 | 5 | 0 |
| math239_s26_t2_prac.pdf | 4 | 4 | 0 |
| 2_ADT Design.pdf | 37 | 2 | 0 |
| cs247_midterm_cheatsheet.pdf | 2 | 1 | 0 |
| 15_TemplateMethodPattern.pdf | 10 | 0 | 0 |
| **total** | **404** | **235** | **0** |

None of those files is two-column. Handout page 5 — a plainly single-column slide whose
title block has centre `181.4` against a page midpoint of `181.0`:

```
unguarded (column, y):                          guarded (top-down):
  "time. | ⇒some products simply generate a       "Development Process for Embedded SQL
   CLI interface, e.g., ODBC |  | Development      Applications |  | ▶Warning: Not all
   Process for Embedded SQL Applications |  |      RDBMSs communicate with the server to
   ▶Warning: Not all RDBMSs…"                      compile at this |  | time. | ⇒some…"
```

The chunk opened mid-sentence with its title buried in the middle. Worse, because the body
no longer started with the detected title, `_prepend_title` also pasted the title in a
second time — title duplication dropped from 7 to 1 of 44 chunks after the fix.

The guard (`_is_two_column`): split into columns only when no block straddles the midpoint
**and** both sides are occupied. A block spanning the fold means the page is single-column
however its centre happens to land. **Limitation:** this handles two columns only; a 3+
column cheatsheet still falls back to `y` order (same as before, and no file in the corpus
needs it).

### Fix 6 in detail — why a targeted fold, not NFKC

Full NFKC would fold ligatures, but it rewrites more than we want. Measured on the
Graph-Theory notes:

| Rewrite | Count | Wanted? |
|---------|------:|---------|
| `ﬁ` → `fi` | 270 | yes |
| `µ` → `μ` | 12 | no — micro sign is meaningful in notes |
| `¯` → combining macron | 7 | no — changes character count |
| `º` → `o` | 6 | no |
| `ﬂ` → `fl` | 5 | yes |
| `ª` → `a` | 2 | no |

`∑ ∏ ∫` are untouched by NFKC, so the "NFKC damages math" caveat was overstated — but `µ`
and `¯` are real. So `_FOLD` covers the Latin ligature block (`ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ`) and curly
quotes only; dashes and math are preserved deliberately. It is applied in `_page_lines`,
the single point where text enters the pipeline, so the string `char_range` indexes is the
same one every later stage sees. After the fold, Graph theory has 0 ligatures and 0 curly
quotes remaining, 11 matchable `definition`s, and `∑`×58 / `∏`×92 / `—`×16 intact.

## Six-file health snapshot (after fixes)

| File | Course | Mode | Pages | Dropped | Chunks | Titled | Oversize | Median chars |
|------|--------|------|-------|---------|--------|--------|----------|--------------|
| lect-EmbeddedSQL-handout | CS348 | slide | 44 | 0 | 44 | 93% | 0 | 311 |
| Sample 2 - Final Exam | CS247 | slide | 27 | 2 | 25 | 0% | 7 | 355 |
| 15_TemplateMethodPattern | CS247 | slide | 10 | 4 | 6 | 100% | 0 | 459 |
| Consumption and Commodities | PHIL121 | slide | 37 | 6 | 31 | 68% | 0 | 343 |
| a5 | CS240 | prose | 6 | 0 | 11 | 82% | 0 | 800 |
| Graph theory | MATH239 | prose | 122 | 2 | 263 | 73% | 1 | 783 |

## Investigations

### Mixed-mode documents (deferred — now pinned by a test)
Slide-vs-prose mode is chosen once per document from the **median** words/page. A deck of
40 sparse slides + a 5-page dense appendix is classified `slide` (the median is a sparse
slide) → each dense page collapses to **one** chunk. Measured: `is_slide_deck` is True,
45 chunks, the 5 dense pages → 5 chunks, largest **2143 chars (~536 tokens)** — well over a
real embedder's budget, and atomic by construction so oversize warning is the only signal.

**Eventual fix:** per-page mode selection. Deferred this phase, but the current behaviour is
now locked in by `test_mixed_mode_deck_is_a_known_limitation` plus the `mixed.pdf` fixture,
so when per-page selection lands that test fails and states exactly what changed.

### Overlap unit (proposed, not applied)
Overlap is measured in **characters** (`round(0.12 * chunk_size)`), and the next window's
start is already snapped **back to a word boundary** (`sub.rfind(" ", …)`), so overlap does
not begin mid-word today. Window ends snap paragraph → sentence → word. **Proposal:** also
snap the overlap start to a sentence boundary for cleaner overlaps. Small change; not applied.

### Title-detection hit rate & failure modes
Rates from the corpus: slides 93–100% except the exam (0%). Failure modes, measured:

- **A bold-at-line-start fallback does not help — proposal dropped.** Adding "first bold
  line at ≥ body size" as a fallback moves the handout 41/44 → 42/44, PHIL121 21/37 → 21/37,
  and the exam 0/27 → **0/27**: one extra title across 108 pages. The exam's question headers
  are plain 12pt, *not* bold (`(8) Select all classes…` → size 12.0, bold=False). Its only
  large line is a 36pt bold `CROWDMARK` QR watermark, correctly removed as boilerplate. So
  **0% titled is the right answer for that file, not a defect** — it genuinely has no
  per-page titles, only question numbers.
- **Number-only titles** — a book that typesets the section number and name as separate
  pieces yields `4.2` instead of `4.2 Isomorphism` (Graph-Theory notes).
- **Chapter label over name** — the largest line wins, so `Chapter 4` beats the chapter name.
- **Header-as-title residual** — a course/university header that repeats on **< 3 pages**
  (a5's `University of Waterloo`) is not caught by the ≥3-page de-boilerplate rule and can
  still surface as a title / chunk text. Needs semantic header detection; deferred.

## Fixture coverage

`tests/fixtures/_generate.py` regenerates every committed fixture PDF. Each maps to one
behaviour: `slides`/`prose` (mode selection), `notes` (de-boilerplating + page numbers),
`scanned` (near-empty drop), `twocol` (column-major order), `onecol_wide` (the single-column
regression guarded by fix 2), `mixed` (the deferred mixed-mode limitation).

The ligature fold is covered by a string-level test rather than a fixture PDF: PyMuPDF's
base-14 fonts cannot encode these glyphs (`insert_text("deﬁnition … “q” — ∑ ∏", fontname="helv")`
round-trips as `'de·nition … ·q· · · ·'`), and the fonts that can are macOS system TTFs,
which would embed a licensed font in the repo and make regeneration OS-specific.
