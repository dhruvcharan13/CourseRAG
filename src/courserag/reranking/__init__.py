"""Reranking: rescore dense candidates by reading query and passage together.

A bi-encoder embeds the query and the passage separately, so it can only ask "are
these about the same thing?". That is exactly the question our measured failures get
wrong: an assignment *about* move-to-front outranks the slide that *defines* it, and a
deck's overview slide outranks the specific one that answers. A cross-encoder sees the
pair in one forward pass and can distinguish topical overlap from an actual answer.

The trade is cost: no vectors can be precomputed, so every candidate needs its own
forward pass. That is why this is a second stage over a small candidate set rather
than a retriever.

Same shape as :mod:`courserag.embedding` — a minimal protocol, an optional heavy
implementation behind the ``[local]`` extra, and a dependency-free stand-in. Nothing
here imports torch; see :mod:`courserag.reranking.cross_encoder`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from courserag.config import Config

__all__ = ["Reranker", "get_reranker", "resolve_reranker_name"]


@runtime_checkable
class Reranker(Protocol):
    """Scores how well each passage answers a query. Higher is better.

    Two members, for the same reason ``Embedder`` has three: every implementation must
    be able to honour the whole contract. Scales are model-specific and not comparable
    across rerankers (or with cosine), so callers rank by these scores and never
    threshold on them.
    """

    model_id: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


def resolve_reranker_name(name: str) -> str:
    """Resolve ``"auto"`` to a concrete reranker without importing one."""
    if name != "auto":
        return name
    from courserag.reranking.cross_encoder import is_available

    return "local" if is_available() else "dummy"


def get_reranker(name: str, cfg: Config) -> Reranker:
    """Build a reranker by name.

    Imports are inside the branches so that naming a reranker — or having one
    configured but never asking for it — costs nothing. Only ``--rerank`` reaches here.
    """
    resolved = resolve_reranker_name(name)

    if resolved == "dummy":
        from courserag.reranking.dummy import DummyReranker

        return DummyReranker()

    from courserag.reranking.cross_encoder import DEFAULT_MODEL_ID, CrossEncoderReranker

    aliases = {"local": DEFAULT_MODEL_ID, "cross-encoder": DEFAULT_MODEL_ID}
    if resolved in aliases:
        model_id = aliases[resolved]
    elif "/" in resolved:
        model_id = resolved
    else:
        raise NotImplementedError(
            f"Unknown reranker '{name}'. Use 'auto', 'local', 'dummy', or a "
            f"HuggingFace cross-encoder model id."
        )
    return CrossEncoderReranker(model_id, cache_dir=cfg.models_dir)
