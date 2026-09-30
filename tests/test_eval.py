import os
import subprocess
import sys

import numpy as np
import pytest

from lastmile.common.eval import (
    INIT_SETS,
    EvalResult,
    bootstrap_diff,
    format_rate,
    mcnemar_exact,
    min_successes_for_lower_bound,
    policy_seed,
    wilson,
)


def test_init_sets_are_disjoint_and_sized():
    sizes = {name: len(seeds) for name, seeds in INIT_SETS.items()}
    assert sizes == {"search": 64, "eval": 256, "eval_ext": 1024, "stress": 256}
    all_seeds = [s for seeds in INIT_SETS.values() for s in seeds]
    assert len(all_seeds) == len(set(all_seeds))


@pytest.mark.parametrize(
    "k, n, lo, hi",
    [
        (15, 30, 0.332, 0.668),
        (27, 30, 0.744, 0.965),
        (30, 30, 0.8865, 1.0),
        (57, 60, 0.863, 0.983),
        (243, 256, 0.915, 0.970),
    ],
)
def test_wilson_reference_numbers(k, n, lo, hi):
    got = wilson(k, n)
    assert got == pytest.approx((lo, hi), abs=1e-3)


def test_streak_lengths_for_the_nines():
    assert min_successes_for_lower_bound(0.95) == 73
    assert min_successes_for_lower_bound(0.99) == 381


def test_policy_seed_is_stable_across_processes():
    # Frozen values: changing them silently breaks common random numbers across versions.
    assert policy_seed("eval", 0) == 6131567021470155164
    assert policy_seed("search", 0, salt=1) == 5103586456224429211
    code = "from lastmile.common.eval import policy_seed; print(policy_seed('eval', 1))"
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": str(h)},
        ).stdout.strip()
        for h in (1, 2)
    }
    assert outs == {str(policy_seed("eval", 1))}
    assert policy_seed("eval", 1) != policy_seed("eval", 1, salt=1)


def test_mcnemar_and_bootstrap():
    assert mcnemar_exact([True] * 10, [False] * 10) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact([True, False], [True, False]) == 1.0
    base = np.array([1] * 50 + [0] * 50, dtype=bool)
    new = np.array([1] * 80 + [0] * 20, dtype=bool)
    lo, hi = bootstrap_diff(base, new)
    assert lo < 0.30 < hi and lo > 0  # true paired difference is +0.30


def test_eval_result_summary():
    res = EvalResult(
        successes=np.array([True] * 124 + [False] * 132),
        steps_to_success=np.full(256, -1),
        near_miss=np.zeros(256),
        env_steps=600,
        seeds=np.arange(256),
        policy_seeds=np.arange(256),
    )
    assert res.summary == "124/256 = 48.4% [42.4, 54.5]" == format_rate(124, 256) == str(res)
    assert res.sr == pytest.approx(124 / 256) and res.robot_minutes == 1.0


def test_wilson_returns_plain_floats_and_rejects_bad_counts():
    lo, hi = wilson(15, 30)
    assert type(lo) is float and type(hi) is float  # no np.float64 reprs in printed intervals
    for k, n in [(11, 10), (-1, 10), (256, 124)]:  # the last one is (n, k) swapped
        with pytest.raises(ValueError):
            wilson(k, n)


def test_bootstrap_rejects_unpaired_input():
    with pytest.raises(ValueError):
        bootstrap_diff([0.5], [1.0] * 256)  # would silently broadcast
    with pytest.raises(ValueError):
        bootstrap_diff([], [])


def test_mcnemar_is_the_significance_test_near_the_ceiling():
    """Four flipped episodes out of 256: the bootstrap interval excludes zero, the exact test does not."""
    base = np.array([True] * 248 + [False] * 8)
    new = np.array([True] * 252 + [False] * 4)
    lo, _ = bootstrap_diff(base, new)
    assert lo > 0
    assert mcnemar_exact(base, new) == pytest.approx(0.125)
