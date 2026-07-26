"""The real embedder must be deterministic across processes, not just within one.

Phase 3 retrieval tests will assert on rankings, so run-to-run vector drift would make
them flaky in a way that is very hard to attribute. Torch can be nondeterministic across
BLAS/thread configurations, so this pins the property down: same text, fresh processes,
different thread counts and batch sizes, bit-identical vectors.

Skipped without the ``[local]`` extra (the dummy's determinism is covered in
tests/test_embedding.py).
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys

import pytest

from course_kb.embedding.sentence_transformer import is_available

pytestmark = pytest.mark.skipif(
    not is_available(), reason='requires the [local] extra: pip install -e ".[local]"'
)

# One short sentence and one long notation-dense block: more tokens, more matmul,
# more opportunity for reduction order to matter.
TEXTS = [
    "Quicksort has an average-case running time of O(n log n).",
    "Let A = { n in N : n = 1 ( mod 3 ) }. " * 20,
]

_SCRIPT = """
import hashlib, json, struct, sys
from course_kb.config import Config
from course_kb.embedding import get_embedder

texts = json.loads(sys.argv[1])
batch_size = int(sys.argv[2])
vectors = get_embedder("local", Config(embed_batch_size=batch_size)).embed(texts)
# Hash the exact float bit patterns: "identical" must mean identical, not "close".
digests = [
    hashlib.sha256(b"".join(struct.pack("<d", x) for x in v)).hexdigest() for v in vectors
]
print("DIGESTS:" + json.dumps(digests))
"""


def _digests_from_fresh_process(batch_size: int = 32, env: dict | None = None) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, json.dumps(TEXTS), str(batch_size)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("DIGESTS:"))
    return json.loads(line.removeprefix("DIGESTS:"))


def test_vectors_are_bit_identical_across_processes_and_thread_counts():
    import os

    single = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    many = dict(os.environ, OMP_NUM_THREADS="8")

    runs = {
        "default": _digests_from_fresh_process(),
        "default (2nd process)": _digests_from_fresh_process(),
        "1 thread": _digests_from_fresh_process(env=single),
        "8 threads, batch=1": _digests_from_fresh_process(batch_size=1, env=many),
    }

    baseline = runs["default"]
    for label, digests in runs.items():
        assert digests == baseline, f"{label} drifted from the baseline run"


def test_vectors_are_bit_identical_within_a_process():
    from course_kb.config import Config
    from course_kb.embedding import get_embedder

    embedder = get_embedder("local", Config())

    def digest(vector: list[float]) -> str:
        return hashlib.sha256(b"".join(struct.pack("<d", x) for x in vector)).hexdigest()

    first = embedder.embed(TEXTS)
    second = embedder.embed(TEXTS)
    # Also check a text embedded alone matches the same text inside a batch, since a
    # Phase 3 query embeds one string while ingest embeds many.
    alone = embedder.embed([TEXTS[0]])

    assert [digest(v) for v in first] == [digest(v) for v in second]
    assert digest(alone[0]) == digest(first[0])
