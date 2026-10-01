"""Recheck committed chapter arithmetic; optionally replay saved knobs and tickets.

Run from the repository root with:
    uv run python docs/maintenance/review/verify_chapter_results.py [--replay]

This audit writes no results or media. It verifies existing measurements; it never selects a new policy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from functools import lru_cache, partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lastmile.common.eval import mcnemar_exact, wilson


def arithmetic() -> Counter:
    counts = Counter()

    @lru_cache(None)
    def exact_bootstrap(fixed, broke, n):
        """Exact empirical resampling distribution, independent of the Monte Carlo code."""
        probabilities = np.array([broke, n - fixed - broke, fixed], dtype=float) / n
        pmf = np.array([1.0])
        for _ in range(n):
            pmf = np.convolve(pmf, probabilities)
        cdf = np.cumsum(pmf) / pmf.sum()
        return (np.searchsorted(cdf, [0.025, 0.975]) - n) / n

    def bootstrap_pairs(paired, intervals, n):
        if "fixed" in paired:
            expected = exact_bootstrap(paired["fixed"], paired["broke"], n)
            gap = float(np.max(np.abs(expected - intervals))) * n
            # Saved intervals use 10,000 Monte Carlo draws; exact quantiles need
            # not match bit for bit. More than two count increments is flagged.
            assert gap <= 2.0
            counts["bootstrap_objects"] += 1
            counts["bootstrap_max_endpoint_count_gap"] = max(
                counts["bootstrap_max_endpoint_count_gap"], gap)
        else:
            for key, value in paired.items():
                bootstrap_pairs(value, intervals[key], n)

    def pairs(paired, pvalues):
        if "fixed" in paired:
            a, b = paired["broke"], paired["fixed"]
            assert pvalues == mcnemar_exact([True] * a + [False] * b, [False] * a + [True] * b)
            counts["paired_tests"] += 1
        else:
            for key, value in paired.items():
                pairs(value, pvalues[key])

    def walk(value, eval_n):
        if isinstance(value, dict):
            k, n = value.get("k"), value.get("n")
            if isinstance(k, int) and isinstance(n, int) and n > 0 and 0 <= k <= n:
                counts["count_objects"] += 1
                if "sr" in value:
                    assert np.isclose(value["sr"], k / n, rtol=0, atol=1e-12)
                if "ci" in value:
                    assert np.allclose(value["ci"], wilson(k, n), rtol=0, atol=1e-12)
            if "only_a" in value and "only_b" in value:
                a, b = value["only_a"], value["only_b"]
                key = "p_mcnemar" if "p_mcnemar" in value else "p"
                assert value[key] == mcnemar_exact([True] * a + [False] * b, [False] * a + [True] * b)
                counts["discordant_tests"] += 1
            if "paired" in value and "mcnemar_p" in value:
                pairs(value["paired"], value["mcnemar_p"])
            if "paired" in value and "diff_ci" in value:
                bootstrap_pairs(value["paired"], value["diff_ci"], eval_n)
            for child in value.values():
                walk(child, eval_n)
        elif isinstance(value, list):
            for child in value:
                walk(child, eval_n)

    for path in sorted((ROOT / "results").glob("ch*/*.json")):
        try:
            row = json.loads(path.read_text())
            walk(row, row["final"]["n"])
        except AssertionError as error:
            raise AssertionError(f"Statistical mismatch in {path}") from error
        counts["result_files"] += 1
    counts["bootstrap_unique_distributions"] = exact_bootstrap.cache_info().currsize
    pattern = re.compile(r"(\d[\d,]*)/(\d[\d,]*)\s*=\s*([\d.]+)%?(?:\s*\[([\d.]+),\s*([\d.]+)\])?")
    for path in sorted((ROOT / "docs/chapters").glob("*.md")):
        for match in pattern.finditer(path.read_text()):
            k, n = (int(match[i].replace(",", "")) for i in (1, 2))
            counts["explicit_doc_rates"] += 1
            expected, shown = [100 * k / n], [float(match[3])]
            if match[4]:
                expected += list(100 * np.array(wilson(k, n)))
                shown += [float(match[4]), float(match[5])]
                counts["explicit_doc_intervals"] += 1
            assert all(abs(a - b) <= 0.051 for a, b in zip(expected, shown)), (path, match[0], expected)
    return counts


def replay():
    from lastmile.common.policy_flow import FlowChunkPolicy
    from lastmile.common.rollout import evaluate, shutdown_pool
    from lastmile.envs.knob_controller import KnobController

    rows = []
    ars = json.loads(next((ROOT / "results/ch01").glob("ars_s4_*.json")).read_text())
    rows.append(("ARS seed 4", partial(KnobController, ars["extra"]["best_knobs"], "so101"),
                 ars["final"]["k"], None))
    for prefix in ("cem_near0_s1", "racing_s1"):
        result = json.loads(next((ROOT / "results/ch03").glob(prefix + "_*.json")).read_text())
        make = partial(FlowChunkPolicy.load, ROOT / "checkpoints/base_v1_so101.pt", noise="fixed",
                       ticket=np.asarray(result["extra"]["ticket"], np.float32))
        rows.append((prefix, make, result["final"]["k"], result["extra"]["eval_successes"]))
    try:
        for name, make, expected, outcomes in rows:
            result = evaluate(make, "eval", n_workers=2)
            assert result.k == expected, (name, result.k, expected)
            if outcomes is not None:
                assert "".join("1" if hit else "0" for hit in result.successes) == outcomes
            print(f"{name}: {result.summary}; saved count and available outcome vector match", flush=True)
    finally:
        shutdown_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    print(dict(arithmetic()))
    if args.replay:
        replay()
