"""End to end: the real CupDropEnv, the scripted expert and a FlowChunk policy through ``evaluate()``.

The unit tests use a tiny fake env; these make sure the pieces actually fit together, including in
spawned worker processes. Kept small (a few seeds, short episodes) so the file runs in seconds.
"""

from functools import partial

import numpy as np
import pytest
import torch

from lastmile.common.policy_flow import FlowChunk, FlowChunkConfig, FlowChunkPolicy
from lastmile.common.rollout import evaluate
from lastmile.envs.cupdrop import CupDropEnv, OutcomeTracker, obs_layout
from lastmile.envs.knob_controller import DEFAULT_KNOBS, TUNED_KNOBS, KnobController

ROBOTS = ("so101", "piper")


@pytest.mark.parametrize("robot", ROBOTS)
def test_tuned_expert_through_evaluate(robot):
    res = evaluate(partial(KnobController, TUNED_KNOBS[robot], robot), "search", robot=robot, n=4,
                   n_workers=1, record_trajectories=True)
    assert res.k == res.n == 4
    assert res.env_steps == sum(traj["steps"] for traj in res.trajectories)
    assert np.array_equal(res.steps_to_success, [traj["steps"] for traj in res.trajectories])

    env = CupDropEnv(robot)
    for traj in res.trajectories:
        assert traj["obs"].shape == (traj["steps"], env.obs_dim)
        assert traj["infos"]["grasped"].shape == (traj["steps"],) and traj["infos"]["grasped"].any()
        assert traj["rewards"].sum() == 1.0
        assert OutcomeTracker.label_trajectory(env, traj) == "success"


@pytest.mark.parametrize("robot", ROBOTS)
def test_parity_one_vs_two_workers(robot):
    """Same seeds, same episodes, whether they run in-process or in spawned workers."""
    make_policy = partial(KnobController, DEFAULT_KNOBS, robot)
    kwargs = {"robot": robot, "n": 8, "shard_size": 2}  # 4 shards, so both workers reuse cached envs
    one = evaluate(make_policy, "search", n_workers=1, **kwargs)
    two = evaluate(make_policy, "search", n_workers=2, **kwargs)
    assert np.array_equal(one.successes, two.successes)
    assert np.array_equal(one.steps_to_success, two.steps_to_success)
    assert np.array_equal(one.near_miss, two.near_miss)
    assert one.env_steps == two.env_steps
    assert 0 < one.k < one.n  # mixed outcomes, so the comparison means something


def test_flowchunk_policy_through_evaluate(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "tiny_flow.pt"
    FlowChunk(FlowChunkConfig(obs_dim=obs_layout("so101")[0], hidden=32, depth=2)).save(path)
    res = evaluate(partial(FlowChunkPolicy.load, path), "eval", n=4, n_workers=2,
                   env_kwargs={"max_steps": 20}, record_trajectories=True)
    assert res.n == 4 and 0 < res.env_steps <= 4 * 20
    for traj in res.trajectories:
        assert np.abs(traj["actions"]).max() <= 1.0
        assert traj["last_noise"].shape == (traj["steps"], 32)
        assert traj["last_chunk"].shape == (traj["steps"], 8, 4)
