# `courserag.embedding`

Turns chunk text into fixed-width vectors. Everything else in the project treats this
package as a black box behind one three-member protocol.

```
__init__.py              the Embedder contract + get_embedder() factory
dummy.py                 DummyEmbedder — deterministic, dependency-free, no semantics
sentence_transformer.py  the real local model (optional [local] extra)
```

## The contract

```python
@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dims: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

A `Protocol`, not an ABC — implementations share no base class and register nothing.
Three members, each load-bearing:

- **`dims` must be readable without doing any work.** `kb init-course` needs the vector
  width to size the Arrow column before anything is embedded. This single requirement
  shapes most of `sentence_transformer.py`.
- **`model_id`** is written into `manifest.json` and compared on every ingest. It is the
  course's identity, not a label.
- **`embed`** is batch-in/batch-out so implementations can batch internally.

`isinstance(x, Embedder)` is a real structural check: a `runtime_checkable` Protocol
verifies *every* member by `hasattr`, not just methods, so a class with `embed()` but no
`dims` fails it.

### Optional capabilities live outside the protocol

The real model also exposes `max_input_tokens`, `count_tokens()`, `query_prefix` and
`embed_query()`. These are deliberately **not** in `Embedder`. Callers discover them
with `getattr`:

```python
count_tokens = getattr(embedder, "count_tokens", None)
if count_tokens is None:
    ...  # fall back to the chars/4 estimate, labelled as an estimate

# retrieval.py — asymmetric models embed queries differently than passages
embed = getattr(embedder, "embed_query", None) or embedder.embed
```

Widening the contract would force the dummy — and any future cloud embedder — to fake a
tokenizer it does not have. Optional capability + graceful degradation keeps the
contract at the size every implementation can actually honour.

## Selection

`get_embedder(name, cfg)` is the only place `Config` is read (`models_dir`,
`embed_batch_size`).

| `embedder` value | resolves to |
|---|---|
| `auto` *(default)* | `local` if sentence-transformers is importable, else `dummy` |
| `local` / `bge` | `BAAI/bge-small-en-v1.5` — 384-dim, 512-token window |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` — 384-dim, 256-token window |
| `dummy` | `DummyEmbedder` — 64-dim, no semantics |
| `<org>/<model>` | any HuggingFace sentence-transformers model |

The `"/" in name` test is how arbitrary models work without an allow-list. Names without
a slash that aren't aliases raise `NotImplementedError`.

## Choices, and the evidence behind them

### `auto` as the default

The only setting where both installs work with no config file: `pip install -e ".[local]"`
gives real 384-dim vectors, and the lean `pip install -e .` still runs init/ingest/chunks
on the dummy. The obvious objection — that vector width now depends on your environment —
is neutralised by the ingest guard, which refuses to mix models into one course and says
exactly why.

### bge-small over the faster MiniLM

Measured on a real 220-chunk course deck (`Enumeration.pdf`), not from spec sheets:

| | all-MiniLM-L6-v2 | bge-small-en-v1.5 |
|---|---:|---:|
| Window | 256 | **512** |
| Chunks truncated | **90 (40.9%)** | **0** |
| Discrimination AUC | **0.864** | 0.843 |
| ms per chunk | **1.69** | 4.50 |
| Cold load | **5.0s** | 8.9s |

MiniLM is faster and marginally better at discrimination. It lost on truncation, which is
not a quality tax but a retrieval hole: truncation is *silent and total* — an oversize
text's vector is bit-identical to the vector of its first ~256 word-pieces — and it lands
on the longest chunks, i.e. the atomic proofs and code listings the chunker deliberately
keeps whole. For queries about content in the dropped tail, MiniLM scored the correct
chunk **below a random chunk** (headroom −0.142 and −0.014 over its own floor). No chunk
in the sampled corpus exceeds 512 tokens, so bge truncates nothing. Full numbers in
[docs/chunking-robustness.md](../../../docs/chunking-robustness.md).

### `DummyEmbedder` stays

It keeps the lean install functional, the test suite fast and offline, and the
no-dependency path honest. It also isolates failures: identical vector *shape*, zero
geometry, so anything that breaks with the dummy is a pipeline bug, not a model issue.

### Unit-norm vectors

`normalize_embeddings=True` makes cosine similarity a bare dot product, so retrieval needs
no per-query division. Magnitude carries no meaning here anyway — the same content
repeated ten times had norm 2.605 versus 6.463 for one copy. bge normalizes internally, so
the flag is redundant *for it*, but it makes the guarantee hold for any model someone
configures by HuggingFace id.

### `dims` from a lookup table, verified on load

`_KNOWN_DIMS` maps model id → width so `init-course` never loads a model. The first time
the model actually loads, its real dimension is asserted against the table — a stale entry
raises rather than writing wrong-width rows. Unknown model ids fall back to loading once
in `__init__` to query the true width.

## Laziness

Torch is imported only when an embed actually happens. Four layers hold that up:

1. `cli.py` imports only this package's `__init__`, never an implementation module.
2. `get_embedder` imports implementations **inside** the matching branch.
3. `sentence_transformer.py` imports `sentence_transformers` inside `_ensure_model()`.
   The type annotation is `TYPE_CHECKING`-only and `self._model` is `Any`, so neither
   reintroduces the import.
4. Availability is probed with `importlib.util.find_spec`, which never executes the module.

| Operation | Loads torch? |
|---|---|
| `kb --help`, `kb list`, `kb delete`, `kb chunks` | no |
| `kb init-course` with a real model | no — `dims` comes from `_KNOWN_DIMS` |
| `kb ingest` where every chunk is already stored | no — early return, and `embed([]) -> []` |
| `kb ingest` with new chunks | **yes, once per process** |

Enforced by [`tests/test_lazy_import.py`](../../../tests/test_lazy_import.py), which asserts
in fresh subprocesses that `torch`, `transformers`, and `sentence_transformers` are absent
from `sys.modules`.

## Gotchas

- **A course's model is fixed at `init-course`.** Ingest compares `(model_id, dims)` to the
  manifest and refuses before parsing or writing. Changing models means rebuilding the
  course from source.
- **Cosine scales are model-specific.** Two unrelated sentences score `0.054` under MiniLM
  and `0.476` under bge; bge rates a near-identical pair higher too (`0.978` vs `0.957`).
  Its range is compressed, not worse. Never carry an absolute threshold across models —
  assert margins, not floors.
- **bge wants a query-side prefix** (`"Represent this sentence for searching relevant
  passages: "`) for retrieval. Passages are embedded plainly. Implemented as
  `embed_query()`, applied only on the search path; `kb ingest` calls `embed()` so
  passages stay prefix-free by construction. A/B'd on the CS240 eval set (30 queries,
  181 chunks): recall@3 `0.900` with the prefix vs `0.867` without, recall@5 `0.933` vs
  `0.900`, MRR `0.857` vs `0.852`; recall@1 and recall@10 are identical. Only 3 of 30
  queries move at all, each by exactly one rank, all three toward the prefix.
  **Directionally consistent and never harmful, but small enough that at this n it is not
  distinguishable from noise** (3/3 same-direction is a sign test at p=0.125). Keep it —
  it is free and it matches how the model was trained — but do not expect it to carry a
  retrieval improvement on its own. `_QUERY_PREFIXES` is keyed by model because MiniLM
  and mpnet are symmetric and prefixing them would only hurt.
- **Determinism is per-device.** Bit-identical across processes, thread counts, and batch
  sizes on one device; `1.5e-07` per dimension between MPS and CPU. Assert cosine
  thresholds and rank order across machines, never bit-equality.
- **`_ensure_model` has a check-then-set race.** Two concurrent first calls can both load —
  wasted seconds, not corruption. A long-lived server should warm the model at startup.
- **Ordering dependency in `__init__`.** `_dims_from_lookup` is assigned *before*
  `_ensure_model()` is called on the unknown-model path, because `_ensure_model` reads it.
  Reordering those lines raises `AttributeError`.
- **Truncation is silent.** The library drops overflow tokens without a warning; the
  visible warning on ingest is ours.

## Adding an embedder

1. Write a class with `model_id`, `dims`, and `embed()`. Do not inherit anything.
2. Import it *inside* a new branch of `get_embedder`, so it costs nothing when unselected.
3. If it has a token budget, add `max_input_tokens` and `count_tokens()` — the CLI will
   pick them up through `getattr` and report exact truncation.
4. Add its width to `_KNOWN_DIMS` if it is a sentence-transformers model, so `init-course`
   stays load-free.
5. New heavy dependencies go in a new `[project.optional-dependencies]` group, guarded by
   `find_spec` with an install hint.

A cloud embedder (OpenAI, Voyage) fits this shape unchanged — an API key check in
`__init__` in place of the `find_spec` guard, and batching against the provider's limit.

## Tests

| File | Covers |
|---|---|
| `test_embedding.py` | dummy determinism, `auto` resolution, factory dispatch, missing-extra `ImportError` |
| `test_sentence_transformer.py` | dims, unit norm, cosine margins, token counts, head-only truncation proof, CLI end-to-end |
| `test_embedder_determinism.py` | bit-identical vectors across processes, thread counts, batch sizes |
| `test_lazy_import.py` | torch stays unimported on every path that should not need it |
| `test_dims_guard.py` | the model/width mismatch refusal, at both store and CLI level |
