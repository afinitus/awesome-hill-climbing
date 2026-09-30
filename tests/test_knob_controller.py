"""Fast checks of the scripted KnobController."""

import numpy as np
import pytest

from lastmile.envs.cupdrop import CupDropEnv
from lastmile.envs.knob_controller import (
    DEFAULT_KNOBS,
    KNOB_NAMES,
    KNOB_SPACE,
    TUNED_KNOBS,
    KnobController,
    knob_dict,
    knob_vector,
)


def _episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    policy.reset([seed])
    for _ in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(policy.act(obs[None])[0])
        if terminated or truncated:
            return info["success"]
    return False


def test_knob_space_is_well_formed():
    assert 8 <= len(KNOB_SPACE) <= 10
    assert len(set(KNOB_NAMES)) == len(KNOB_NAMES)
    for knob in KNOB_SPACE:
        assert knob.low < knob.high and knob.unit and knob.meaning
    for knobs in [DEFAULT_KNOBS, *TUNED_KNOBS.values()]:
        assert set(knobs) == set(KNOB_NAMES)
        assert knob_dict(knobs) == pytest.approx(knobs)  # already inside the bounds
    assert set(TUNED_KNOBS) == {"so101", "piper"}


def test_knob_vector_roundtrip_and_clipping():
    vec = knob_vector(DEFAULT_KNOBS)
    assert knob_dict(vec) == pytest.approx(DEFAULT_KNOBS)
    wild = knob_dict(np.full(len(KNOB_SPACE), 1e3))
    assert all(wild[k.name] == k.high for k in KNOB_SPACE)
    with pytest.raises(ValueError):
        knob_dict({"speed": 1.0})


@pytest.mark.parametrize("robot", ["so101", "piper"])
def test_tuned_controller_solves_a_few_episodes(robot):
    env = CupDropEnv(robot=robot)
    policy = KnobController(TUNED_KNOBS[robot], robot)
    assert all(_episode(env, policy, seed) for seed in (20_000, 20_001))


def test_default_knobs_are_worse_than_tuned():
    env = CupDropEnv(robot="so101")
    seeds = range(10_000, 10_012)
    tuned = sum(_episode(env, KnobController(TUNED_KNOBS["so101"]), s) for s in seeds)
    default = sum(_episode(env, KnobController(DEFAULT_KNOBS), s) for s in seeds)
    assert tuned == len(seeds) and default < tuned


def test_batch_matches_single_env_and_uses_only_obs():
    env = CupDropEnv(robot="so101")
    obs_a, _ = env.reset(seed=1)
    obs_b, _ = env.reset(seed=2)
    batch = KnobController(TUNED_KNOBS["so101"])
    batch.reset([1, 2])
    single = KnobController(TUNED_KNOBS["so101"])
    single.reset([2])
    out = batch.act(np.stack([obs_a, obs_b]))
    assert out.shape == (2, 4) and np.all(np.abs(out) <= 1.0)
    np.testing.assert_array_equal(out[1], single.act(obs_b[None])[0])


def test_noise_is_seeded_per_env():
    noise = {"knob_std": 0.1, "action_std": 0.2}
    obs = CupDropEnv(robot="so101").reset(seed=0)[0][None].repeat(2, axis=0)
    a, b = KnobController(DEFAULT_KNOBS, noise=noise), KnobController(DEFAULT_KNOBS, noise=noise)
    a.reset([5, 6])
    b.reset([5, 6])
    np.testing.assert_array_equal(a.k, b.k)
    np.testing.assert_array_equal(a.act(obs), b.act(obs))
    assert not np.array_equal(a.k[0], a.k[1])  # different seeds -> different jittered knobs
    clean = KnobController(DEFAULT_KNOBS)
    clean.reset([5, 6])
    np.testing.assert_array_equal(clean.k[0], clean.k[1])
    with pytest.raises(ValueError):
        KnobController(DEFAULT_KNOBS, noise={"typo": 1.0})


def test_default_knobs_lower_before_release():
    assert DEFAULT_KNOBS["drop_h"] <= DEFAULT_KNOBS["lift_h"]  # the LOWER phase must not climb


def test_cup_position_is_read_once():
    """The script plans its drop point from the first observation and does not chase a knocked cup."""
    env = CupDropEnv(robot="so101")
    obs, _ = env.reset(seed=1)
    policy = KnobController(TUNED_KNOBS["so101"])
    policy.reset([1])
    policy.act(obs[None])
    np.testing.assert_allclose(policy.cup[0], env.robot.cup_pos, atol=1e-6)
    moved = obs.copy()
    moved[env.obs_slices["cup_pos"]] += 0.05
    twin = KnobController(TUNED_KNOBS["so101"])
    twin.reset([1])
    twin.act(obs[None])
    np.testing.assert_array_equal(policy.act(moved[None]), twin.act(obs[None]))
