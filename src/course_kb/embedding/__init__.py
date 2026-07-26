"""Embedding contract and factory.

An :class:`Embedder` maps texts to fixed-dimension vectors. Two implementations
exist: the real local :class:`~course_kb.embedding.sentence_transformer.SentenceTransformerEmbedder`
(the ``[local]`` extra, defaulting to bge-small-en-v1.5) and the dependency-free,
deterministic :class:`~course_kb.embedding.dummy.DummyEmbedder`. Either way there is
no API key and no network after the model is cached.

The default name is ``"auto"``: the real model when sentence-transformers is
installed, the dummy otherwise — so both installs work with no config file.

Cloud providers (OpenAI/Voyage/Gemini) are opt-in alternates behind this same
protocol, added in later phases. Note: an Anthropic/Claude key cannot produce
embeddings, so Claude is never a default here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from course_kb.config import Config

__all__ = ["Embedder", "get_embedder", "resolve_embedder_name"]


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into fixed-dimension embedding vectors."""

    model_id: str
    dims: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def resolve_embedder_name(name: str) -> str:
    """Resolve ``"auto"`` to a concrete embedder name; pass anything else through.

    ``"auto"`` means "the real local model if it is installed, otherwise the dummy".
    Availability is probed with :func:`importlib.util.find_spec`, which does not
    import sentence-transformers (or torch), so this stays fast.
    """
    if name != "auto":
        return name

    from course_kb.embedding.sentence_transformer import is_available

    return "local" if is_available() else "dummy"


def get_embedder(name: str, cfg: Config) -> Embedder:
    """Return the embedder selected by ``name``.

    Accepted values:

    * ``"auto"`` — the real local model if installed, else the dummy.
    * ``"dummy"`` — the dependency-free :class:`DummyEmbedder`.
    * ``"local"`` / ``"bge"`` — bge-small-en-v1.5 (384-dim, 512-token window).
    * ``"minilm"`` — all-MiniLM-L6-v2 (384-dim, 256-token window; faster, truncates more).
    * any HuggingFace model id (contains ``"/"``) — that sentence-transformers model.

    Implementations are imported inside the branches so selecting the dummy never
    pulls in the model stack.

    Raises:
        NotImplementedError: for an unrecognized name.
        ImportError: when a real model is requested but the ``[local]`` extra is missing.
    """
    resolved = resolve_embedder_name(name)

    if resolved == "dummy":
        from course_kb.embedding.dummy import DummyEmbedder

        return DummyEmbedder()

    from course_kb.embedding.sentence_transformer import (
        DEFAULT_MODEL_ID,
        MINILM_MODEL_ID,
        SentenceTransformerEmbedder,
    )

    aliases = {"local": DEFAULT_MODEL_ID, "bge": DEFAULT_MODEL_ID, "minilm": MINILM_MODEL_ID}

    if resolved in aliases:
        model_id = aliases[resolved]
    elif "/" in resolved:
        model_id = resolved
    else:
        raise NotImplementedError(
            f"Embedder '{name}' is not available. Choose 'auto', 'dummy', "
            f"{', '.join(repr(a) for a in aliases)}, or a HuggingFace model id "
            f"like '{DEFAULT_MODEL_ID}'."
        )

    return SentenceTransformerEmbedder(
        model_id,
        cache_dir=cfg.models_dir,
        batch_size=cfg.embed_batch_size,
    )
