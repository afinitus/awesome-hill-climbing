from functools import partial

import numpy as np
import pytest
import torch
from dummy_env import DummyEnv

from lastmile.common.policy_flow import (
    FlowChunk,
    FlowChunkConfig,
    FlowChunkPolicy,
    make_chunk_dataset,
    train_flowchunk,
)
from lastmile.common.rollout import evaluate


def _synthetic_dataset(n=512, obs_dim=6, seed=0):
    """Deterministic obs -> chunk mapping, so a trained flow should reproduce it from any noise."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32)
    w = rng.standard_normal((obs_dim, 32)).astype(np.float32) / np.sqrt(obs_dim)
    chunks = np.tanh(obs @ w).reshape(n, 8, 4)
    trajs = [{"obs": obs, "actions": chunks[:, 0], "success": True}]
    ds = make_chunk_dataset(trajs)
    ds.chunks = chunks  # replace the time-shifted chunks with the clean mapping
    return ds


@pytest.fixture(scope="module")
def trained():
    ds = _synthetic_dataset()
    model, losses = train_flowchunk(ds, epochs=40, batch_size=128, lr=2e-3, seed=0)
    return ds, model, losses


def test_training_reduces_loss_and_fits(trained):
    ds, model, losses = trained
    assert losses[-1] < 0.6 * losses[0]
    pred = model.sample(ds.obs[:64], np.zeros((64, 32), np.float32))
    err = np.abs(pred - ds.chunks[:64]).mean()
    baseline = np.abs(ds.chunks[:64] - ds.chunks.mean(0)).mean()
    assert err < 0.3 * baseline


def test_sampling_is_deterministic_and_numpy_matches(trained):
    ds, model, _ = trained
    noise = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    a, b = model.sample(ds.obs[:16], noise), model.sample(ds.obs[:16], noise)
    assert np.array_equal(a, b) and a.shape == (16, 8, 4) and np.abs(a).max() <= 1.0
    fast = model.numpy_sampler()
    assert np.allclose(fast(ds.obs[:16], noise), a, atol=1e-5)
    assert np.array_equal(fast(ds.obs[3:4], noise[3:4])[0], fast(ds.obs[:16], noise)[3])  # batch-invariant


def test_save_load_exact(trained, tmp_path):
    _, model, _ = trained
    path = tmp_path / "flow.pt"
    model.save(path, demos=12)
    loaded = FlowChunk.load(path)
    assert loaded.config == model.config and loaded.meta == {"demos": 12}
    for (k, v), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)
    obs, noise = np.ones((2, 6), np.float32), np.ones((2, 32), np.float32)
    assert np.array_equal(model.sample(obs, noise), loaded.sample(obs, noise))


def test_make_chunk_dataset_pads_with_last_action():
    actions = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
    trajs = [
        {"obs": np.zeros((10, 3)), "actions": actions, "success": True},
        {"obs": np.zeros((5, 3)), "actions": actions[:5], "success": False},
    ]
    ds = make_chunk_dataset(trajs, horizon=8)
    assert len(ds) == 15 and ds.chunks.shape == (15, 8, 4) and ds.n_episodes == 2
    assert np.array_equal(ds.chunks[0], actions[:8])
    assert np.array_equal(ds.chunks[9], np.repeat(actions[9:10], 8, axis=0))
    assert np.array_equal(ds.chunks[6, 3:], np.repeat(actions[9:10], 5, axis=0))
    assert len(make_chunk_dataset(trajs, success_only=True)) == 10


def test_policy_replans_every_four_steps():
    torch.manual_seed(0)
    policy = FlowChunkPolicy(FlowChunk(FlowChunkConfig(obs_dim=6)))
    policy.reset([1, 2, 3])
    obs = np.zeros((3, 6), np.float32)
    noises = []
    for t in range(12):
        action = policy.act(obs + t)
        assert np.array_equal(action, policy.last_chunk[:, t % 4])
        assert (policy.last_step_in_chunk == t % 4).all()
        noises.append(policy.last_noise.copy())
    changed = [t for t in range(1, 12) if not np.array_equal(noises[t], noises[t - 1])]
    assert changed == [4, 8]
    policy.reset([1, 2, 3])  # same seeds -> same noise
    policy.act(obs)
    assert np.array_equal(policy.last_noise, noises[0])

    ticket = np.linspace(-1, 1, 32).astype(np.float32)
    fixed = FlowChunkPolicy(policy.model, noise="fixed", ticket=ticket)
    fixed.reset([0, 1])
    for _ in range(5):
        fixed.act(obs[:2])
        assert np.array_equal(fixed.last_noise, np.stack([ticket, ticket]))


def _scaled_noise(obs, rngs):
    """A callable noise source written to the DESIGN.md §6 signature: one generator per env."""
    return np.stack([0.5 * rng.standard_normal(32) for rng in rngs])


def test_callable_noise_gets_one_generator_per_env():
    torch.manual_seed(0)
    model = FlowChunk(FlowChunkConfig(obs_dim=6))
    obs = np.zeros((3, 6), np.float32)
    batch = FlowChunkPolicy(model, noise=_scaled_noise)
    batch.reset([1, 2, 3])
    batch.act(obs)
    assert batch.last_noise.shape == (3, 32) and batch.last_noise.dtype == np.float32
    alone = FlowChunkPolicy(model, noise=_scaled_noise)
    alone.reset([3])
    alone.act(obs[:1])
    assert np.array_equal(alone.last_noise[0], batch.last_noise[2])  # independent of the batch it shares


def test_policy_in_parallel_rollout(tmp_path):
    torch.manual_seed(0)
    model = FlowChunk(FlowChunkConfig(obs_dim=DummyEnv().obs_dim))
    path = tmp_path / "flow.pt"
    model.save(path)
    make_policy = partial(FlowChunkPolicy.load, path)
    one = evaluate(
        make_policy,
        "search",
        n=16,
        n_workers=1,
        env_factory=partial(DummyEnv, max_steps=12),
        record_trajectories=True,
    )
    four = evaluate(
        make_policy,
        "search",
        n=16,
        n_workers=4,
        env_factory=partial(DummyEnv, max_steps=12),
        record_trajectories=True,
    )
    assert np.array_equal(one.near_miss, four.near_miss)
    for a, b in zip(one.trajectories, four.trajectories):
        assert np.array_equal(a["actions"], b["actions"])
        assert a["last_noise"].shape == (12, 32) and a["last_chunk"].shape == (12, 8, 4)
        assert np.array_equal(a["last_noise"], b["last_noise"])
