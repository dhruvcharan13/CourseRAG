# course-kb

A per-course RAG knowledge base. Each course is an isolated
[LanceDB](https://lancedb.github.io/lancedb/) store in its own folder, fronted by
a single `kb` command-line interface.

**Design principle: zero-config, no API key, no internet required.** Embeddings
run locally — a real sentence-transformers model, or a dependency-free dummy.
Cloud providers (OpenAI/Voyage/Gemini) are opt-in alternates behind the same
interface, added in a later phase. (An Anthropic/Claude API key cannot produce
embeddings — Claude is generation-only — so it is never the embedding default.)

> Status: **working**. Parsing, chunking, per-course stores, real local embeddings
> (384-dim bge), `kb search`/`kb eval`, an opt-in cross-encoder reranker, and an MCP
> server all work. Hybrid retrieval (BM25 + rank fusion) was measured and **rejected**
> — see [The dense-only baseline](#the-dense-only-baseline). A D2L integration comes later.

## Install

```bash
pip install -e ".[dev,local]"   # real local embeddings (pulls torch, ~800MB on disk)
pip install -e ".[dev]"         # lean install: dummy embeddings only
pip install -e ".[local,mcp]"   # + expose the knowledge base to an agent over MCP
```

Runtime dependencies are `lancedb` (which bundles `pyarrow`) and `pymupdf`. The
`local` extra adds `sentence-transformers` and the `mcp` extra adds the MCP SDK; both
are imported lazily — and nothing in the package imports the server module — so the
lean install and `kb --help` never pay for either. A test asserts it.

## Usage

```bash
kb init-course CS240-W26                 # create an isolated course store
kb ingest CS240-W26 sample.txt --category notes   # parse -> chunk -> embed -> store
kb delete CS240-W26 sample.txt           # remove every chunk from one source file
kb info CS240-W26                        # manifest + chunk count
kb list                                  # active and archived courses
kb search CS240-W26 "how do skip lists pick tower height?"   # ranked, cited chunks
kb search CS240-W26 "..." -k 10 --json   # same, as JSON on stdout for piping
kb eval CS240-W26                        # score search against evals/CS240-W26.json
```

`delete` takes the file name exactly as `kb info` lists it. It loads no embedding
model — a vector is a pure function of chunk text, so removing rows costs no inference
— which also means a course can be pruned even when its embedding model is
unavailable. It prunes old table versions to give the disk space back, so a delete
cannot be rolled back; re-ingesting the source file restores it exactly.

## Retrieval

`kb search` embeds the query with the course's own model, then takes an exact cosine
top-k over the stored vectors — brute force, no ANN index, so results are exact. Every
result carries its provenance (`source_file`, `page`, `title`, `module`, `category`) and
its similarity score. Scores are **reported, never thresholded**: a meaningful relevance
floor is model-specific and has to be calibrated from measured data.

A query embedded by a different model than the passages lands in a different vector
space, where scores still look like plausible numbers. `kb search` therefore applies the
same `(model_id, dims)` check `kb ingest` does, and refuses rather than ranking.

### The dense-only baseline

`kb eval <course>` scores search against a hand-labelled question set in
`evals/<course>.json` — questions with the `(file, page)` where the answer actually
lives. `recall@k` is a hit-rate (one intended answer per query, labelled with the pages
it may legitimately appear on); `MRR@k` is truncated, so an answer below rank k scores 0.

Measured on `evals/CS240.json` — 30 questions authored blind from the source PDFs, over
181 chunks from 10 documents (lecture modules 01/03/05, tutorial 01, assignments 0–5):

| recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---:|---:|---:|---:|---:|
| 80.0% | 90.0% | 93.3% | 96.7% | 0.857 |

Correct-chunk scores span 0.684–0.889. **This is the dense-only baseline Phase 4 is
measured against.** An earlier 52-chunk version of this corpus pinned recall@3/@5/@10 at
100%, which measured the corpus, not the retriever; at 181 chunks every cutoff moves.

Two failure modes account for every miss, and both are Phase 4 watch cases:

- **Cross-document — the question outranks the answer.** Asked what move-to-front does,
  the top hit is a4's *"we analyse the move-to-front strategy"* (0.811) while the lecture
  slide that defines it sits third (0.704). Identical at 52 and 181 chunks — same ranks,
  same competing chunk, same scores — so it is a property of the retrieval, not of a
  sparse corpus.
- **Intra-document — the summary outranks the definition.** The only outright miss is
  "difference between o-notation and O-notation": module01 has ~15 order-notation slides
  that paraphrase each other, and the specific defining slide loses to the summary and
  overview slides around it. Same shape for the Θ-definition (rank 6) and the limit rule
  (rank 5).

Notably, two things that were *expected* to break did not: exact-token queries ("Leo",
"Academic Integrity Declaration", "PQ1") all rank 1, and the deliberately confusable pair
"expected run-time of randomized quick-select" vs "…quick-sort" — six slides apart, near
identical phrasing, different answers — is separated cleanly. Dense retrieval's weakness
here is paraphrase-dense clusters, not rare literal tokens.

### A second course: CS247

`evals/CS247.json` is an independent read on the same retriever — 28 blind-authored
questions over 670 chunks from 34 documents (22 lecture decks, 5 assignment specs, 2
project specs, 4 sample finals, 1 midterm). Software design and C++ rather than
algorithms and proofs. No question is ported from CS240; a test asserts the two sets
share none.

| course | chunks | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| CS240 | 181 | 80.0% | 90.0% | 93.3% | 96.7% | 0.857 |
| CS247 | 670 | 71.4% | 92.9% | 92.9% | 96.4% | 0.807 |

The two courses agree on one failure class and disagree on another:

- **Confirmed on both — a question *about* a topic outranks the passage that *answers*
  it.** CS240: an assignment's "we analyse the move-to-front strategy" beats the slide
  defining it. CS247: `A4Q1_Spec` beats the Iterator-pattern definition, and a sample
  final's pImpl *question* beats the pImpl *explanation*. Two independent corpora, so
  this is a property of dense retrieval here, not of one course.
- **New in CS247 — a deck's topic-label chunk outranks its content.** Every deck opens
  with "CS 247: Software Engineering Principles — *Topic*", a short chunk that is 100%
  on-topic and says nothing. Short purely-topical chunks beat long informative ones: it
  accounts for the only outright miss and three of the near-misses.
- **Refuted — "paraphrase clusters fail" does not generalize as stated.** Six decks each
  carry a slide beginning literally "Design Principle:", which looked like a harder
  version of CS240's order-notation cluster. All four such questions ranked 1, with the
  four *highest* scores in the eval (0.877–0.943). Shared *format* is harmless; what
  breaks retrieval is redundant *content* (CS240's slides genuinely restate each other)
  or a query that names a section heading instead of its content (CS247's one miss asks
  for "real-world use cases", which every pattern deck has).

Exact-token queries held up on both: "Weather-O-Rama", "Starbuzz", "Gumball Machine",
"ArcaneMarket Forges" all rank 1, as "Leo" and "PQ1" did on CS240. The code-embedded
identifier `CompositeIterator` came second — and lost to an assignment page, not to
tokenization, so it is the first failure class rather than a code-specific one.

**Token window:** 1 of 670 CS247 chunks exceeds bge's 512 (536 tokens, 24 over); 12 more
sit in 400–512; the median is 97. The one truncation is a printed menu data listing, not
a code block — CS247's code listings chunk smaller than its sample output.

### Reranking (Phase 4)

`--rerank` adds a second stage: dense retrieves the top 20, a local cross-encoder
(`ms-marco-MiniLM-L-6-v2`, ~88MB) rescores those 20 by reading query and passage
*together*, and the top k come back. Off by default; dense-only remains the baseline.

```bash
kb search CS247 "what problem does the Decorator pattern solve" --rerank
kb eval CS247 --rerank      # prints the dense-vs-reranked A/B, net of regressions
```

Measured over all 58 queries on both courses, **net +7 rank-1 (fixed 8, broke 1)**:

| | recall@1 | recall@3 | recall@5 | MRR@10 |
|---|---:|---:|---:|---:|
| CS240 dense | 80.0% | 90.0% | 93.3% | 0.857 |
| CS240 reranked | **93.3%** | 96.7% | 100% | **0.958** |
| CS247 dense | 71.4% | 92.9% | 92.9% | 0.807 |
| CS247 reranked | **82.1%** | 89.3% | 96.4% | **0.873** |

Why it works is specific, and narrower than expected. The diagnostic named two failure
classes; reranking decisively fixes one and is **ambivalent about the other**:

- **Intra-document (C3) — 6 of 9 fixed, 2 improved, 0 regressed.** This was the class no
  category, length or lexical signal could touch even in principle, and it is where
  essentially all the gain comes from. `s7` went 10→1, `n3` 15→1, `n2` 6→1.
- **Question-outranks-answer (C1) — 2 fixed, 2 unchanged, 1 worse.** A cross-encoder
  trained on MS MARCO rewards passages that *look like a response to the query*, and an
  exam question about Decorator resembles a query about Decorator very closely. So it
  fixes this class sometimes and **causes** it others: the single outright regression,
  `c1` (1→5), lost to `Sample 3 - Final Exam p12` and `A3Q1_Spec p1` — both questions
  about Decorator outranking the slide that answers it. `s2` (2→4) failed the same way.

Two honest caveats. CS247's **recall@3 drops** 92.9%→89.3% — the win is concentrated at
rank 1, and at k=3 the two regressions cost more than `s7`'s recovery gains. And `p5`
remains unfixed at >10: it asks for a section heading ("real-world use cases") rather
than its content, which is an eval-authoring artifact, left as-is rather than tuned
against.

Cost: **~64 ms median** per query for 20 candidates warm (p90 80 ms), plus a one-off ~6 s
model load. 2 of 1160 scored pairs exceeded the reranker's 512-token window (worst 521).

### Reproducing the baseline

The source PDFs live in `corpus/CS240/` and are gitignored — they are copyrighted course
material. The eval set records the filenames and pages it labels, so the measurement is
reproducible from the documents without shipping them. A test asserts every labelled page
actually exists in its PDF, so a typo'd label fails loudly instead of masquerading as a
permanent retrieval miss.

## MCP server

Exposes the knowledge base to an agent, so you can ask about your own course materials
and get answers cited back to a file and page. Three read-only tools —
`list_courses`, `course_info`, and `search_course(course, query, k=5, rerank=False)`.
Ingestion stays in the CLI: the agent can read the knowledge base, never rewrite it.

```bash
claude mcp add course-kb \
  -e COURSE_KB_ROOT=/abs/path/to/repo/course-kb \
  -- /abs/path/to/repo/.venv/bin/python -m course_kb.mcp_server
```

**`COURSE_KB_ROOT` is not optional.** A stdio server inherits the *client's* working
directory, and the default `root` is relative — so without an absolute root the server
resolves `course-kb` against whatever directory the editor launched from, finds no
courses, and reports an empty knowledge base as though that were true. It is the one
failure here that looks like success. `load_config` resolves `COURSE_KB_ROOT` (env,
absolute) over `config.toml`'s `root` over the relative default, and a test proves the
tools find the real courses from a foreign cwd — with a negative control showing they
find nothing without it.

Two consequences of the stdio transport shape the module. Stdout *is* the JSON-RPC
channel, so nothing on this path may print — which is why the tools wrap `retrieval`
and the manifest directly rather than reusing `cli.py`'s `cmd_*` functions, all of
which print. And the server sets `HF_HUB_OFFLINE=1` (via `setdefault`, so you can
override it): the models are already cached by the time a course is searchable, so the
Hub round trip only checks for updates nobody asked for. Skipping it takes a cold
search from 3.9s to 2.5s and a cold reranked search from 2.1s to 0.2s, and lets a
server started with no network work rather than stall.

Warm searches are ~13ms. The first one pays the model load.

## On-disk layout

```
course-kb/                     # default root, overridable via config.toml
  courses/<course_id>/
    index.lance/               # LanceDB dataset (one per course)
    raw/                       # original source files
    COURSE.md                  # per-course memory
    manifest.json              # {course, embedding_model, dims, categories, files, last_indexed}
  models/                      # downloaded embedding models (cache_dir)
  archive/                     # retired courses (later phase)
```

## Embeddings

| `embedder` | Model | Dims | Window | Needs |
| --- | --- | --- | --- | --- |
| `auto` (default) | bge if installed, else dummy | 384 or 64 | — | — |
| `local` / `bge` | `BAAI/bge-small-en-v1.5` | 384 | 512 tok | `[local]` extra |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 tok | `[local]` extra |
| `dummy` | SHA-256-seeded PRNG (no semantics) | 64 | — | — |
| `<org>/<model>` | that HuggingFace sentence-transformers model | model's | model's | `[local]` extra |

Vectors are L2-normalized, so cosine similarity is a dot product. The first real
embed downloads ~130MB into `<root>/models`; afterwards it runs fully offline.

**Why bge and not the faster MiniLM.** MiniLM's 256 word-piece window truncates
silently, and it lands on the longest chunks — the atomic proofs and code listings the
chunker deliberately keeps whole, i.e. the ones most likely to be complete answers. On
a real 220-chunk course deck it truncated **90 chunks (40.9%)**, and for queries about
the dropped tail it scored the correct chunk *below a random chunk*. bge's 512-token
window truncates **nothing** in the sampled corpus (largest chunk seen: 479 tokens), at
the cost of ~2.7x slower embedding (4.5ms vs 1.7ms per chunk). Full numbers in
[docs/chunking-robustness.md](docs/chunking-robustness.md). `embedder = "minilm"` still
selects MiniLM if you want the speed.

**Cosine scales are model-specific.** bge's values sit much higher than MiniLM's for the
same pair of texts — two unrelated sentences score 0.476 under bge and 0.054 under
MiniLM. bge is not worse (it rates a near-identical pair 0.978 vs MiniLM's 0.957); its
range is simply compressed. Any relevance threshold must be calibrated per model rather
than carried over — which is why `kb search` reports scores and never cuts on them.
bge also wants a query-side instruction prefix
(`"Represent this sentence for searching relevant passages: "`); passages are embedded
plainly. `kb search` applies it via `embed_query()`, `kb ingest` does not.

**A course's embedding model is fixed at `init-course`.** Vectors from different
models are not comparable, and a table's vector width cannot change, so `kb ingest`
refuses to run when the configured embedder differs from the one recorded in
`manifest.json` — before parsing, embedding, or writing anything. To switch models,
rebuild the course from your source files:

```bash
rm -rf course-kb/courses/CS240-W26
kb init-course CS240-W26
kb ingest CS240-W26 <file> --category <category>   # for each file
```

### Courses built before Phase 2

`CS240`, `CS247`, `CS348`, and `PHIL121` were created before the real embedder existed,
so their manifests record `dummy-hash-64` / `dims: 64` and their vectors carry no
semantic meaning. With the `[local]` extra installed, `embedder = "auto"` now resolves
to a 384-dim model, so **these courses reject further ingestion by design** — the
mismatch guard fires rather than mixing incomparable vectors into one table. That is
expected, not a bug. To bring one up to date, rebuild it from the original source
files using the recipe above (`rm -rf courses/<name>`, `kb init-course <name>`,
re-ingest each PDF). Their `raw/` folders are empty — ingestion has never copied
sources — so the rebuild needs the PDFs you originally ingested from. `kb delete` and
`kb info` keep working on them meanwhile, since neither loads an embedding model.

### Vectors and hardware

Embedding is deterministic: the same text, model, and device produce **bit-identical**
vectors across processes, thread counts, and batch sizes. Across *devices* it is not
exact — measured MPS (Apple GPU) vs CPU: max `1.5e-07` per dimension, cosine
`0.9999999709`. That is far below anything that changes a ranking, but two things
follow. A course embedded on one machine and queried from another (a hosted MCP server,
say) carries a tiny corpus-vs-query device delta, so relevance thresholds should never
be tuned to more precision than that. And tests must not assert bit-equality across
machines — assert cosine thresholds and rank order instead.

## Configuration

Optional. Drop a `config.toml` in the working directory to override defaults;
otherwise sane defaults are used with no file and no environment variables.

One environment variable, `COURSE_KB_ROOT`, overrides `root` with an absolute path and
takes precedence over the file. It exists for processes that don't control their own
working directory — see [MCP server](#mcp-server). Unset, nothing changes.

```toml
root = "course-kb"
embedder = "auto"        # auto | dummy | minilm | <hf-org>/<hf-model>
embed_batch_size = 32
# cache_dir = "~/.cache/course-kb"   # model cache; defaults to <root>/models
chunk_size = 1000
chunk_overlap = 200
min_chunk_chars = 50
```
