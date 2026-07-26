# course-kb

A per-course RAG knowledge base. Each course is an isolated
[LanceDB](https://lancedb.github.io/lancedb/) store in its own folder, fronted by
a single `kb` command-line interface.

**Design principle: zero-config, no API key, no internet required.** Embeddings
run locally — a real sentence-transformers model, or a dependency-free dummy.
Cloud providers (OpenAI/Voyage/Gemini) are opt-in alternates behind the same
interface, added in a later phase. (An Anthropic/Claude API key cannot produce
embeddings — Claude is generation-only — so it is never the embedding default.)

> Status: **Phase 2** — real local embeddings. Parsing, chunking, and per-course
> stores work; embeddings are now semantic (384-dim MiniLM). Retrieval, an MCP
> server, and a D2L integration come in later phases; `kb search` is still a stub.

## Install

```bash
pip install -e ".[dev,local]"   # real local embeddings (pulls torch, ~800MB on disk)
pip install -e ".[dev]"         # lean install: dummy embeddings only
```

Runtime dependencies are `lancedb` (which bundles `pyarrow`) and `pymupdf`. The
`local` extra adds `sentence-transformers`; it is imported lazily, so the lean
install and `kb --help` never pay for it.

## Usage

```bash
kb init-course CS240-W26                 # create an isolated course store
kb ingest CS240-W26 sample.txt --category notes   # parse -> chunk -> embed -> store
kb delete CS240-W26 sample.txt           # remove every chunk from one source file
kb info CS240-W26                        # manifest + chunk count
kb list                                  # active and archived courses
kb search ...                            # not implemented yet (Phase 3)
```

`delete` takes the file name exactly as `kb info` lists it. It loads no embedding
model — a vector is a pure function of chunk text, so removing rows costs no inference
— which also means a course can be pruned even when its embedding model is
unavailable. It prunes old table versions to give the disk space back, so a delete
cannot be rolled back; re-ingesting the source file restores it exactly.

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
than carried over. bge also documents a query-side instruction prefix
(`"Represent this sentence for searching relevant passages: "`) for retrieval; passages
are embedded plainly, and the query path is a Phase 3 concern.

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

```toml
root = "course-kb"
embedder = "auto"        # auto | dummy | minilm | <hf-org>/<hf-model>
embed_batch_size = 32
# cache_dir = "~/.cache/course-kb"   # model cache; defaults to <root>/models
chunk_size = 1000
chunk_overlap = 200
min_chunk_chars = 50
```
