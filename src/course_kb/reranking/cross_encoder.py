"""Local cross-encoder reranking via sentence-transformers (the ``[local]`` extra).

Default is ``ms-marco-MiniLM-L-6-v2``: ~80MB, six layers, trained on MS MARCO passage
ranking. Chosen as the smallest proven option — if it lands within noise, a heavier
reranker is a later probe, not a reflex.

Everything is lazy, for the same reasons as the embedder. ``CrossEncoder`` (and
therefore torch) is imported inside :meth:`CrossEncoderReranker._ensure_model`, never at
module import, so naming a reranker in config costs nothing and only an actual
``--rerank`` pays for it. No extra dependency: ``CrossEncoder`` ships inside
``sentence-transformers``, which the ``[local]`` extra already installs.

Scores are raw logits, roughly -11..+11, higher meaning more relevant. They are **not**
probabilities and **not** comparable to the cosine similarities from the dense stage;
they order candidates and nothing more.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

__all__ = ["DEFAULT_MODEL_ID", "CrossEncoderReranker", "is_available"]

DEFAULT_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_INSTALL_HINT = (
    "Reranking needs sentence-transformers. Install the local extra:\n"
    '  pip install -e ".[local]"\n'
    "Or drop --rerank to use dense retrieval only."
)

# Loading costs seconds; reuse across rerankers in the same process.
_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}


def is_available() -> bool:
    """Whether sentence-transformers can be imported — checked without importing it."""
    return importlib.util.find_spec("sentence_transformers") is not None


class CrossEncoderReranker:
    """Rescores (query, passage) pairs with a local cross-encoder."""

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

    @property
    def max_input_tokens(self) -> int:
        """Word-piece budget for the query and passage *combined*.

        Unlike the embedder, the pair shares one window, so a long passage is truncated
        with the query still in front of it. Optional capability, discovered by
        ``getattr`` like ``count_tokens`` — callers that want to report truncation can,
        and the dummy stays free of it.
        """
        model = self._ensure_model()
        # Renamed in sentence-transformers 5.x; the old name still works but warns.
        return int(getattr(model, "max_seq_length", None) or model.max_length)

    def count_pair_tokens(self, query: str, passages: list[str]) -> list[int]:
        """Real word-piece counts per (query, passage) pair, so truncation is visible."""
        if not passages:
            return []
        tokenizer = self._ensure_model().tokenizer
        encoded = tokenizer(
            [query] * len(passages), passages, add_special_tokens=True,
            truncation=False, verbose=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance logit per passage, in input order. Higher is more relevant."""
        if not passages:
            # No work means no model load, keeping an empty candidate set torch-free.
            return []
        model = self._ensure_model()
        scores = model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]

    # ----------------------------------------------------------------------- #

    def _ensure_model(self) -> "CrossEncoder":
        """Load (and memoize) the model. The only place torch gets imported."""
        if self._model is not None:
            return self._model

        cache_folder = str(self._cache_dir) if self._cache_dir is not None else None
        cache_key = (self.model_id, cache_folder)
        model = _MODEL_CACHE.get(cache_key)

        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - is_available() guards this
                raise ImportError(_INSTALL_HINT) from exc

            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            model = CrossEncoder(self.model_id, cache_folder=cache_folder)
            _MODEL_CACHE[cache_key] = model

        self._model = model
        return model
