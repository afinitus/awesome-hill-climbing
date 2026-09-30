"""Shared library: evaluation sets and statistics, rollouts, policies, cost ledger, plotting, CLI.

Only the light, numpy-only pieces are re-exported here. Import ``lastmile.common.policy_flow``
(torch) and ``lastmile.common.plotting`` (matplotlib) explicitly when you need them.
"""

from lastmile.common.eval import (
    INIT_SET_VERSION,
    INIT_SETS,
    EvalResult,
    bootstrap_diff,
    format_rate,
    mcnemar_exact,
    min_successes_for_lower_bound,
    policy_seed,
    wilson,
)
from lastmile.common.ledger import Ledger, load_results
from lastmile.common.policy import Batched, BatchPolicy, RandomPolicy
from lastmile.common.rollout import evaluate, rollout_seeds

__all__ = [
    "INIT_SETS",
    "INIT_SET_VERSION",
    "BatchPolicy",
    "Batched",
    "EvalResult",
    "Ledger",
    "RandomPolicy",
    "bootstrap_diff",
    "evaluate",
    "format_rate",
    "load_results",
    "mcnemar_exact",
    "min_successes_for_lower_bound",
    "policy_seed",
    "rollout_seeds",
    "wilson",
]
