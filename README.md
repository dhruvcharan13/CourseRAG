# course-kb

A per-course RAG knowledge base. Each course is an isolated
[LanceDB](https://lancedb.github.io/lancedb/) store in its own folder, fronted by
a single `kb` command-line interface.

**Design principle: zero-config, no API key, no internet required.** Embeddings
default to a local implementation. Cloud providers (OpenAI/Voyage/Gemini) are
opt-in alternates behind the same interface, added in a later phase. (An
Anthropic/Claude API key cannot produce embeddings — Claude is generation-only —
so it is never the embedding default.)

> Status: **Phase 0** — project skeleton and the interfaces (contracts) that
> every later phase plugs into. Real document parsing, a local embedding model,
> retrieval, an MCP server, and a D2L integration come in later phases.

## Install

```bash
pip install -e ".[dev]"
```

The only runtime dependency is `lancedb` (which bundles `pyarrow`); `pytest` is
the only dev dependency.

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
  archive/                     # retired courses (later phase)
```

## Configuration

Optional. Drop a `config.toml` in the working directory to override defaults;
otherwise sane defaults are used with no file and no environment variables.

```toml
root = "course-kb"
embedder = "dummy"
chunk_size = 1000
chunk_overlap = 200
min_chunk_chars = 50
```
