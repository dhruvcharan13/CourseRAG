"""Embedding contract and factory.

An :class:`Embedder` maps texts to fixed-dimension vectors. The default is the
local, deterministic :class:`~course_kb.embedding.dummy.DummyEmbedder` — so the
tool runs zero-config, with no API key and no internet.

Cloud providers (OpenAI/Voyage/Gemini) and a real local model are opt-in
alternates behind this same protocol, added in later phases. Note: an
Anthropic/Claude key cannot produce embeddings, so Claude is never a default here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from course_kb.config import Config

__all__ = ["Embedder", "get_embedder"]


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into fixed-dimension embedding vectors."""

    model_id: str
    dims: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def get_embedder(name: str, cfg: Config) -> Embedder:
    """Return the embedder named ``name``.

    ``"dummy"`` yields the local :class:`DummyEmbedder` (the zero-config default).
    Real local and cloud embedders are added in later phases.

    Raises:
        NotImplementedError: for any name other than ``"dummy"``.
    """
    if name == "dummy":
        from course_kb.embedding.dummy import DummyEmbedder

        return DummyEmbedder()
    raise NotImplementedError(
        f"Embedder '{name}' is not available yet (only 'dummy' exists in Phase 0)."
    )
