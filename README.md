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
kb info CS240-W26                        # manifest + chunk count
kb list                                  # active and archived courses
kb search ...                            # not implemented yet (Phase 3)
```

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

| `embedder` | Model | Dims | Needs |
| --- | --- | --- | --- |
| `auto` (default) | MiniLM if installed, else dummy | 384 or 64 | — |
| `minilm` / `local` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | `[local]` extra |
| `dummy` | SHA-256-seeded PRNG (no semantics) | 64 | — |
| `<org>/<model>` | that HuggingFace sentence-transformers model | model's | `[local]` extra |

Vectors are L2-normalized, so cosine similarity is a dot product. The first real
embed downloads ~90MB into `<root>/models`; afterwards it runs fully offline.

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
