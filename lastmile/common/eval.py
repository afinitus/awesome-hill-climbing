"""Initial-state sets, common random numbers and the statistics every chapter reports.

Two rules keep the scoreboard honest (DESIGN.md §3):

1. Search and evaluation use *disjoint* initial states. Tune on ``search``, report on ``eval``.
2. Episode ``i`` of set ``S`` always uses env seed ``INIT_SETS[S][i]`` and policy seed
   ``policy_seed(S, i, salt)``. Two methods scored on the same set therefore face the same
   cube positions *and* the same sampling noise, so their results are paired and can be
   compared with McNemar's test instead of two independent intervals.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import binomtest

INIT_SET_VERSION = "v1"
INIT_SETS: dict[str, list[int]] = {
    "search": list(range(10_000, 10_064)),  # 64 — search, training selection, hyperparameters
    "eval": list(range(20_000, 20_256)),  # 256 — held-out score; never used for selection
    "eval_ext": list(range(30_000, 31_024)),  # 1024 — final claims
    "stress": list(range(40_000, 40_256)),  # 256 — used with variant="hard"/"bias"
}


def policy_seed(init_set: str, index: int, salt: int = 0) -> int:
    """Policy-sampling seed for episode ``index`` of ``init_set``.

    Uses blake2b rather than Python's ``hash()``, which is randomized per process, so the
    seed is identical across workers, runs and machines. Change ``salt`` to get a fresh but
    still paired set of noise draws (e.g. for a second stochastic evaluation).
    """
    key = f"{init_set}:{index}:{salt}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") & (2**63 - 1)  # fits in int64 and JSON


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials.

    Unlike the normal approximation it never leaves [0, 1] and stays sensible at 0/n and n/n,
    which is exactly where a hill-climbing project ends up (30/30 -> [88.6%, 100%]).
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n} (arguments are successes, then trials)")
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value for paired successes ``a`` and ``b``.

    Only the discordant pairs carry information: episodes where exactly one of the two
    methods succeeded. Under "no difference" each discordant pair is a fair coin flip.
    This is the test to quote when claiming "B beats A" on a shared init set.
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match, got {a.shape} and {b.shape}")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    if only_a + only_b == 0:
        return 1.0
    return float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


def bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Paired percentile-bootstrap interval for ``mean(b) - mean(a)``; pass ``(base, new)``.

    Use it to report *how big* the difference is, and ``mcnemar_exact`` to decide *whether* there is
    one. Near the ceiling the two can disagree: with fewer than about 10 discordant pairs (say
    248/256 -> 252/256, four episodes flipped) the percentile bootstrap excludes zero while the exact
    test gives p = 0.125, and the exact test is the one to trust. Episodes are resampled as pairs,
    which keeps the benefit of common random numbers.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError(f"need two non-empty 1-D arrays of equal length, got shapes {a.shape} and {b.shape}")
    diff = b - a
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    chunk = 1_000  # resample in chunks so memory stays small for n_episodes = 1024
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        idx = rng.integers(0, len(diff), size=(size, len(diff)))
        means[start : start + size] = diff[idx].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def min_successes_for_lower_bound(target: float, z: float = 1.96) -> int:
    """All-success fixed sample size whose Wilson lower bound reaches ``target``.

    With zero failures the lower bound is ``n / (n + z^2)``: 73/73 reaches 95%,
    and 381/381 reaches 99%. Use independent, constant-probability trials with a
    fixed evaluation size; these bounds do not justify stopping at a lucky streak.
    """
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")
    n = 1
    while wilson(n, n, z)[0] < target:
        n += 1
    return n


def format_rate(k: int, n: int) -> str:
    """Human-readable success rate with its Wilson interval: ``124/256 = 48.4% [42.4, 54.5]``."""
    lo, hi = wilson(k, n)
    sr = k / n if n else 0.0
    return f"{k}/{n} = {100 * sr:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Per-episode outcomes of one evaluation, in seed order."""

    successes: np.ndarray  # bool[n]
    steps_to_success: np.ndarray  # int[n], -1 on failure
    near_miss: np.ndarray  # float[n], closest horizontal cube-to-cup distance (m)
    env_steps: int  # total simulated steps, for the cost ledger
    seeds: np.ndarray  # int[n] env seeds
    policy_seeds: np.ndarray  # int[n] policy seeds
    init_set: str | None = None
    trajectories: list[dict[str, Any]] | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.successes)

    @property
    def k(self) -> int:
        return int(np.sum(self.successes))

    @property
    def sr(self) -> float:
        return self.k / self.n if self.n else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.k, self.n)

    @property
    def robot_minutes(self) -> float:
        return self.env_steps / 600

    @property
    def summary(self) -> str:
        return format_rate(self.k, self.n)

    def __str__(self) -> str:
        return self.summary
