"""Retrieval evaluation: does search actually find the right chunk?

A retrieval system without a measured baseline cannot be improved, only changed. This
module defines that baseline — an eval set of questions with hand-labelled answer
locations, and the two metrics scored against it.

Everything here is pure: :func:`score_run` takes already-computed rankings, so the
metric math is testable without a store, a model, or a disk. The CLI does the I/O.

Metric definitions, stated precisely because both names are overloaded:

**recall@k** — the fraction of queries with at least one correct result in the top k.
Each query has exactly *one* intended answer, expressed as a set of acceptable
locations (a concept slide and its pseudocode slide are the same answer, not two).
So this is a hit-rate / success@k, **not** the multi-gold
``|retrieved ∩ relevant| / |relevant|``.

**MRR@k** — mean of ``1 / rank`` of the first correct result, counting 0 when there is
no correct result within the top k. Truncated, so a gold at rank k+1 scores 0 rather
than ``1/(k+1)``.

A result is correct when its ``(source_file, page)`` is one of the labelled locations.
Page granularity rather than chunk id is deliberate: labels stay valid when the chunker
changes, which chunk ids would not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from courserag.retrieval import SearchResult

# Depth every query is run to. Smaller k are prefixes of the same ranking, so one run
# at this depth yields recall@1..@10 and MRR@10 without re-searching.
EVAL_K = 10

# Cutoffs reported in the summary. Must all be <= EVAL_K.
REPORT_KS = (1, 3, 5, 10)

# Where an eval set lives when not given explicitly: <repo>/evals/<course>.json.
EVAL_DIR = "evals"


@dataclass
class EvalQuery:
    """One labelled question: what to ask, and where the answer actually is."""

    id: str
    query: str
    # Acceptable locations of the single intended answer, as (source_file, page).
    gold: set[tuple[str, int]]
    note: str = ""


@dataclass
class EvalSet:
    course: str
    queries: list[EvalQuery]
    default_k: int = 5
    note: str = ""


@dataclass
class QueryOutcome:
    """How one query fared: where its answer landed, and at what score."""

    query: EvalQuery
    results: list[SearchResult]
    # 1-based position of the first correct result, or None if none in the top EVAL_K.
    rank: int | None

    @property
    def hit(self) -> bool:
        """Whether the answer appeared anywhere in the results that were returned."""
        return self.rank is not None

    def hit_at(self, k: int) -> bool:
        """Whether the answer appeared within the top ``k``.

        Distinct from :attr:`hit` because the metrics are truncated: an answer found at
        rank k+1 counts as a miss, and has to be *reported* as one too, or the miss list
        would disagree with the score it produced.
        """
        return self.rank is not None and self.rank <= k

    @property
    def top_score(self) -> float | None:
        """Score of the rank-1 result, correct or not."""
        return self.results[0].score if self.results else None

    @property
    def gold_score(self) -> float | None:
        """Score of the first correct result, or None if it was missed."""
        return self.results[self.rank - 1].score if self.rank is not None else None


@dataclass
class EvalReport:
    outcomes: list[QueryOutcome]
    recall: dict[int, float]
    mrr: float
    k: int

    @property
    def misses(self) -> list[QueryOutcome]:
        return [o for o in self.outcomes if not o.hit_at(self.k)]


def load_eval_set(path: Path) -> EvalSet:
    """Parse an eval set from JSON.

    JSON rather than YAML so the format costs no dependency — the project ships with
    ``lancedb`` and ``pymupdf`` and reads its config with stdlib ``tomllib``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    queries = []
    for raw in data["queries"]:
        gold: set[tuple[str, int]] = set()
        for target in raw["expected"]:
            for page in target["pages"]:
                gold.add((target["source_file"], int(page)))
        if not gold:
            raise ValueError(f"Eval query {raw['id']!r} in {path} labels no expected pages.")
        queries.append(
            EvalQuery(id=raw["id"], query=raw["query"], gold=gold, note=raw.get("note", ""))
        )

    if not queries:
        raise ValueError(f"Eval set {path} contains no queries.")

    return EvalSet(
        course=data["course"],
        queries=queries,
        default_k=int(data.get("default_k", 5)),
        note=data.get("note", ""),
    )


def first_hit_rank(results: list[SearchResult], gold: set[tuple[str, int]]) -> int | None:
    """1-based position of the first result whose (source_file, page) is labelled correct."""
    for position, result in enumerate(results, start=1):
        if (result.chunk.source_file, result.chunk.page) in gold:
            return position
    return None


def score_run(outcomes: list[QueryOutcome], k: int = EVAL_K) -> EvalReport:
    """Compute recall@k and MRR@k over already-executed queries.

    Pure: no searching happens here, so the metric math can be checked against rankings
    written by hand.
    """
    if not outcomes:
        raise ValueError("Cannot score an empty run.")

    total = len(outcomes)
    recall = {
        cutoff: sum(1 for o in outcomes if o.hit_at(cutoff)) / total
        for cutoff in REPORT_KS
        if cutoff <= k
    }
    mrr = sum(1.0 / o.rank for o in outcomes if o.hit_at(k)) / total
    return EvalReport(outcomes=outcomes, recall=recall, mrr=mrr, k=k)


def score_spread(values: list[float]) -> str:
    """``min / median / max`` for a group of scores, or a placeholder when empty.

    Score reporting exists so a relevance threshold can be *calibrated* in a later phase
    instead of guessed. No cutoff is applied anywhere in this module.
    """
    if not values:
        return "(none)"
    return f"{min(values):.3f} / {median(values):.3f} / {max(values):.3f}"
