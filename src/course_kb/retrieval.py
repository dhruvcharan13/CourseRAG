"""Dense retrieval: turn a query into ranked, cited chunks.

Stored vectors are unit-normalized, so cosine similarity is just their dot product
and ranking by it is ranking by relevance. The actual top-k scan lives in
:meth:`course_kb.store.CourseStore.search`; this module holds everything that is not
the store's business — how a *query* gets embedded, what a result looks like once it
comes back, and how the optional rerank stage composes with the dense one.

:func:`retrieve` is the entry point every front end shares. Keeping the pipeline here
rather than in one of them is what lets the CLI and the MCP server return identical
rankings without either importing the other.

Scores are reported, never thresholded. A relevance floor is model-specific (bge's
unrelated-pair floor sits near 0.63, MiniLM's nowhere near it), so a cutoff has to be
calibrated from measured data rather than guessed. See ``kb eval``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from course_kb.records import ChunkRecord
from course_kb.reranking import get_reranker

if TYPE_CHECKING:
    from course_kb.config import Config
    from course_kb.embedding import Embedder
    from course_kb.reranking import Reranker
    from course_kb.store import CourseStore


def embed_query(embedder: "Embedder", query: str) -> list[float]:
    """Embed one query, using the embedder's query path when it has one.

    Some models are asymmetric: bge expects a retrieval instruction prefix on queries
    that passages must not get, and skipping it costs real accuracy. Rather than widen
    the ``Embedder`` protocol (``model_id``/``dims``/``embed``) for a capability most
    models lack, this discovers ``embed_query`` with ``getattr`` — the same optional
    capability pattern the ingest path uses for ``count_tokens``.

    Passages are never routed through here: ``kb ingest`` calls ``embed`` directly, so
    the query/passage asymmetry holds by construction.
    """
    embed = getattr(embedder, "embed_query", None) or embedder.embed
    return embed([query])[0]


@dataclass
class SearchResult:
    """One retrieved chunk and how similar it was to the query.

    Wraps the :class:`ChunkRecord` rather than flattening it, so provenance travels
    with the text: a result is never a bare vector or a bare string.
    """

    chunk: ChunkRecord
    score: float
    # Set only when a reranker rescored this candidate. Kept beside the dense score
    # rather than replacing it: the two are different scales (cosine in [-1,1] vs a
    # cross-encoder logit), and seeing both is how a reordering gets audited.
    rerank_score: float | None = None

    def to_dict(self, rank: int) -> dict[str, Any]:
        """Flat JSON payload: the scores, the text, and everything needed to cite it.

        The vector is omitted — it is not provenance, and it dwarfs every other field.
        """
        return {
            "rank": rank,
            "score": round(self.score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "id": self.chunk.id,
            "text": self.chunk.text,
            "source_file": self.chunk.source_file,
            "page": self.chunk.page,
            "title": self.chunk.title,
            "module": self.chunk.module,
            "category": self.chunk.category,
        }


def rerank(reranker: "Reranker", query: str, candidates: list[SearchResult]) -> list[SearchResult]:
    """Reorder dense candidates by cross-encoder score, best first.

    Pure: no store, no I/O, and no truncation to k — the caller slices. Each result is
    rebuilt with its ``rerank_score`` attached and its :class:`ChunkRecord` untouched,
    so citations survive the reordering intact.

    Ties keep their dense order: ``sorted`` is stable and candidates arrive in dense
    rank order, so an indifferent reranker degrades to the dense ranking rather than to
    an arbitrary one.
    """
    if not candidates:
        return []
    scores = reranker.score(query, [c.chunk.text for c in candidates])
    scored = [
        SearchResult(chunk=c.chunk, score=c.score, rerank_score=s)
        for c, s in zip(candidates, scores)
    ]
    return sorted(scored, key=lambda r: -r.rerank_score)


def retrieve(
    cfg: "Config", store: "CourseStore", embedder: "Embedder", query: str, k: int, *,
    use_rerank: bool = False, candidates: int | None = None, source_file: str | None = None,
) -> tuple[list[SearchResult], list[SearchResult]]:
    """Run the retrieval pipeline; return ``(final, dense_only)``.

    With reranking off, this is exactly the Phase-3 path — one dense search at depth
    ``k`` and nothing else, so the committed baseline cannot shift underneath us.

    With it on, the dense stage widens to ``candidates`` (never below ``k``, or the
    reranker could not fill the requested page), the cross-encoder rescores those, and
    the top ``k`` come back. The dense-only ordering is returned alongside because
    every caller that reranks also wants the A/B, and stage one already computed it.

    ``source_file`` scopes both stages to one document. It has to reach the *candidate*
    search rather than the final slice: filtering after reranking would rescore
    course-wide candidates and then throw most of them away, returning fewer than ``k``
    results from a stage that had them available.
    """
    vector = embed_query(embedder, query)
    if not use_rerank:
        dense = store.search(vector, k=k, source_file=source_file)
        return dense, dense

    depth = max(candidates or cfg.rerank_candidates, k)
    dense = store.search(vector, k=depth, source_file=source_file)
    reranker = get_reranker(cfg.reranker, cfg)
    return rerank(reranker, query, dense)[:k], dense[:k]
