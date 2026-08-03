"""A deterministic, dependency-free embedder used as the zero-config default.

The vector for a text is derived by seeding a PRNG with the text's SHA-256
digest, so the same text always yields the same vector across runs, processes,
and machines. It carries no semantic meaning — it exists to exercise the
pipeline until a real local model lands in Phase 2.
"""

from __future__ import annotations

import hashlib
import random


class DummyEmbedder:
    """Hash-seeded fixed-dimension embedder. Deterministic, no I/O."""

    model_id: str = "dummy-hash-64"
    dims: int = 64

    def __init__(self, dims: int = 64) -> None:
        self.dims = dims
        self.model_id = f"dummy-hash-{dims}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        # SHA-256 gives a stable per-text seed (unlike the salted builtin hash()).
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest, "big")
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.dims)]
