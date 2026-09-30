from functools import partial

import numpy as np
import pytest
from dummy_env import DummyEnv, GreedyPolicy

from lastmile.common.eval import INIT_SETS, policy_seed
from lastmile.common.ledger import Ledger
from lastmile.common.rollout import evaluate, rollout_seeds


def _same(a, b):
    assert np.array_equal(a.successes, b.successes)
    assert np.array_equal(a.steps_to_success, b.steps_to_success)
    assert np.array_equal(a.near_miss, b.near_miss)
    assert a.env_steps == b.env_steps


def test_parity_across_workers_and_shards():
    one = evaluate(GreedyPolicy, "search", n_workers=1, env_factory=DummyEnv)
    four = evaluate(GreedyPolicy, "search", n_workers=4, env_factory=DummyEnv)
    _same(one, four)
    # A per-env-seeded policy gives the same episodes whatever batch they share.
    _same(one, evaluate(GreedyPolicy, "search", n_workers=1, env_factory=DummyEnv, shard_size=5))
    assert 0 < one.k < one.n  # the fixture policy is neither perfect nor useless
    assert list(one.seeds) == INIT_SETS["search"]
    assert list(one.policy_seeds) == [policy_seed("search", i) for i in range(64)]


def test_results_and_trajectories():
    res = evaluate(
        GreedyPolicy,
        "eval",
        n=24,
        n_workers=1,
        env_factory=partial(DummyEnv, max_steps=30),
        record_trajectories=True,
    )
    assert res.n == 24 and res.init_set == "eval"
    steps = [t["steps"] for t in res.trajectories]
    assert res.env_steps == sum(steps)  # finished envs were never stepped again
    for traj, ok, sts in zip(res.trajectories, res.successes, res.steps_to_success):
        assert traj["obs"].shape == (traj["steps"], 18) and traj["actions"].shape == (traj["steps"], 4)
        assert traj["success"] == ok and traj["rewards"].sum() == float(ok)
        assert sts == (traj["steps"] if ok else -1)
        assert traj["steps"] <= 30
    assert [t["seed"] for t in res.trajectories] == INIT_SETS["eval"][:24]


def test_ledger_is_charged(tmp_path):
    with Ledger("test", "dummy", results_root=tmp_path) as ledger:
        res = evaluate(
            GreedyPolicy, "search", n=8, n_workers=1, env_factory=DummyEnv, ledger=ledger, category="search"
        )
        assert ledger.robot_steps["search"] == res.env_steps
        assert ledger.robot_minutes["search"] == pytest.approx(res.env_steps / 600)


def test_category_follows_the_init_set_and_typos_are_rejected(tmp_path):
    """Search episodes land in the improvement budget without the caller having to remember a flag."""
    kwargs = {"n": 4, "n_workers": 1, "env_factory": DummyEnv}
    ledger = Ledger("test", "dummy", results_root=tmp_path)
    search = evaluate(GreedyPolicy, "search", ledger=ledger, **kwargs)
    held_out = evaluate(GreedyPolicy, "eval", ledger=ledger, **kwargs)
    assert ledger.robot_steps == {"search": search.env_steps, "train": 0, "eval": held_out.env_steps}
    evaluate(GreedyPolicy, "search", ledger=ledger, category="train", **kwargs)
    assert ledger.robot_steps["train"] == search.env_steps
    with pytest.raises(ValueError, match="unknown robot-minute category"):
        evaluate(GreedyPolicy, "search", ledger=ledger, category="serach", **kwargs)


def test_n_must_fit_the_init_set():
    for n in (0, -1, 100):  # search has 64 seeds; slicing would silently return 0, 63 or 64 episodes
        with pytest.raises(ValueError, match="outside 1..64"):
            evaluate(GreedyPolicy, "search", n=n, n_workers=1, env_factory=DummyEnv)


def test_pool_is_reused_when_fewer_workers_are_requested():
    from lastmile.common import rollout

    four = evaluate(GreedyPolicy, "search", n=16, n_workers=4, env_factory=DummyEnv)
    pool = rollout._pool
    two = evaluate(GreedyPolicy, "search", n=16, n_workers=2, env_factory=DummyEnv)
    assert rollout._pool is pool and rollout._pool_size == 4  # no respawn for the smaller request
    _same(four, two)


def test_worker_errors_propagate():
    bad_seed = INIT_SETS["search"][5]
    with pytest.raises(RuntimeError, match="rollout worker failed.*refuses seed"):
        rollout_seeds(
            GreedyPolicy,
            INIT_SETS["search"][:16],
            list(range(16)),
            n_workers=4,
            env_factory=partial(DummyEnv, fail_on_seed=bad_seed),
        )
    # The pool is still usable afterwards.
    assert evaluate(GreedyPolicy, "search", n=8, n_workers=4, env_factory=DummyEnv).n == 8
