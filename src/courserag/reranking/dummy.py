"""A dependency-free stand-in reranker, for testing the two-stage wiring.

Deliberately **not** a retrieval strategy. It scores by negated position, so it exactly
reverses whatever order the dense stage produced. That makes it useless for retrieval
and perfect as a test double: if the last dense candidate comes back first, stage two
ran and its ordering won.

Scoring by lexical overlap was considered and rejected. It would reorder more
plausibly, but a keyword signal is precisely the approach the miss diagnostic
falsified for this corpus, and a test double that looks like BM25 invites being
mistaken for one later. This one cannot be mistaken for anything.

Never the default, reachable only via ``reranker = "dummy"``, and excluded from every
measurement.
"""

from __future__ import annotations

__all__ = ["DummyReranker"]


class DummyReranker:
    """Reverses the candidate order. Deterministic, offline, no model."""

    model_id: str = "dummy-reverse"

    def score(self, query: str, passages: list[str]) -> list[float]:
        # Ascending by position, so the LAST candidate scores highest and ranking by
        # score reverses the input. The query is unused — that is the point; this
        # measures wiring, not relevance.
        return [float(i) for i in range(len(passages))]
