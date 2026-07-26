"""Real local embeddings via sentence-transformers (the optional ``[local]`` extra).

Semantics, no API key, no network after the first model download. The default is
``bge-small-en-v1.5``: 384 dimensions, a 512 word-piece window, ~130MB.

The window is why it is the default rather than the faster ``all-MiniLM-L6-v2``.
Measured on a real 220-chunk course deck, MiniLM's 256-piece limit truncated 40.9%
of chunks, and for queries about the dropped tail it ranked the correct chunk *below*
a random one. No chunk in the sampled corpus exceeds 512, so bge truncates nothing.
See docs/chunking-robustness.md.

Everything here is lazy. ``sentence_transformers`` (and therefore torch) is imported
inside :meth:`SentenceTransformerEmbedder._ensure_model`, not at module import, so
``kb --help`` and the DummyEmbedder path never pay for it. ``dims`` comes from
:data:`_KNOWN_DIMS` rather than from the loaded model, because ``init-course`` needs
the vector width to size the Arrow column and ``ingest`` needs it to check the
manifest — neither should trigger a model load.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = ["DEFAULT_MODEL_ID", "MINILM_MODEL_ID", "SentenceTransformerEmbedder", "is_available"]

DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Output width per model, so `dims` is known without importing torch. Verified
# against the real model the first time it loads, so a stale entry here raises
# instead of writing wrong-width vectors.
_KNOWN_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
}

_INSTALL_HINT = (
    "sentence-transformers is not installed. Install the local embedding extra:\n"
    '  pip install -e ".[local]"\n'
    'Or set embedder = "dummy" in config.toml to use the dependency-free embedder.'
)

# Loading a model takes seconds; reuse it across embedders in the same process.
_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}


def is_available() -> bool:
    """Whether sentence-transformers can be imported — checked without importing it."""
    return importlib.util.find_spec("sentence_transformers") is not None


def _model_dims(model: Any) -> int:
    """Output width of a loaded model (renamed in sentence-transformers 5.x)."""
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:
        getter = model.get_sentence_embedding_dimension
    return int(getter())


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder producing unit-norm (cosine-ready) vectors."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        cache_dir: Path | None = None,
        batch_size: int = 32,
    ) -> None:
        if not is_available():
            raise ImportError(_INSTALL_HINT)

        self.model_id = model_id
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._batch_size = max(1, batch_size)
        self._model: Any | None = None

        known = _KNOWN_DIMS.get(model_id)
        if known is not None:
            self.dims = known
            self._dims_from_lookup = True
        else:
            # Unknown model: the only way to learn its width is to load it.
            self._dims_from_lookup = False
            self.dims = _model_dims(self._ensure_model())

    @property
    def max_input_tokens(self) -> int:
        """Word-piece budget per text. Longer texts are truncated at embed time.

        MiniLM's budget is 256, and truncation is silent: the vector for an oversize
        text is *bit-identical* to the vector for its first ~256 word-pieces, so the
        tail contributes nothing at all.
        """
        return int(self._ensure_model().max_seq_length)

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Real word-piece counts (with special tokens), so callers can see truncation.

        The character-based ``estimate_tokens`` proxy under-counts notation-dense text
        (math, SQL, code) by up to 4x, which is exactly the content most likely to be
        truncated. This uses the model's own tokenizer, and is cheap next to the
        forward pass it precedes.
        """
        if not texts:
            return []
        tokenizer = self._ensure_model().tokenizer
        # verbose=False: transformers otherwise logs its own over-length notice, which
        # is the very thing the caller is about to report properly.
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in batches. Vectors are L2-normalized to length 1."""
        if not texts:
            # No work means no model load — keeps a no-op re-ingest torch-free.
            return []
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in vector] for vector in vectors]

    # ----------------------------------------------------------------------- #

    def _ensure_model(self) -> "SentenceTransformer":
        """Load (and memoize) the model. The only place torch gets imported."""
        if self._model is not None:
            return self._model

        cache_folder = str(self._cache_dir) if self._cache_dir is not None else None
        cache_key = (self.model_id, cache_folder)
        model = _MODEL_CACHE.get(cache_key)

        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - is_available() guards this
                raise ImportError(_INSTALL_HINT) from exc

            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            model = SentenceTransformer(self.model_id, cache_folder=cache_folder)

            if self._dims_from_lookup:
                actual = _model_dims(model)
                if actual != self.dims:
                    raise ValueError(
                        f"Model '{self.model_id}' produces {actual}-dim vectors, but this "
                        f"build expected {self.dims}. Course tables are sized from the "
                        f"expected width, so refusing to embed. Fix _KNOWN_DIMS in "
                        f"course_kb/embedding/sentence_transformer.py."
                    )
            _MODEL_CACHE[cache_key] = model

        self._model = model
        return model
