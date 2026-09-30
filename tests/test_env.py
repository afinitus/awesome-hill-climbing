"""Fast checks of the CupDropEnv contract (docs/DESIGN.md section 2)."""

import copy
from itertools import pairwise

import mujoco
import numpy as np
import pytest

from lastmile.envs.cupdrop import CUBE_HALF, MAX_DELTA, MAX_LEAD, OBS_KEYS, CupDropEnv, OutcomeTracker, obs_layout
from lastmile.envs.knob_controller import DEFAULT_KNOBS, RELEASE, TUNED_KNOBS, KnobController
from lastmile.envs.robots import get_robot


@pytest.fixture(scope="module")
def env():
    return CupDropEnv(robot="so101")


@pytest.fixture(scope="module")
def env2():
    return CupDropEnv(robot="so101")


def _rollout(env, actions):
    return np.stack([env.step(a)[0] for a in actions])


def _random_actions(n, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (n, 4))


@pytest.mark.parametrize("robot", ["so101", "piper"])
def test_robots_construct_and_step(robot):
    env = CupDropEnv(robot=robot)
    obs, info = env.reset(seed=1)
    assert obs.shape == (env.obs_dim,) and obs.dtype == np.float32
    assert env.n_substeps == round(0.1 / env.model.opt.timestep)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert obs.shape == (env.obs_dim,) and np.isfinite(obs).all()
    assert reward == 0.0 and not terminated and not truncated
    assert set(info) >= {"success", "grasped", "near_miss", "steps", "ee_target"}
    assert info["steps"] == 1 and info["ee_target"].shape == (3,)


def test_obs_slices_are_consistent(env):
    obs, _ = env.reset(seed=3)
    sl = env.obs_slices
    assert list(sl) == list(OBS_KEYS)
    assert sl["joint_pos"].start == 0 and sl["time_frac"].stop == env.obs_dim
    assert all(sl[a].stop == sl[b].start for a, b in pairwise(OBS_KEYS))
    assert obs_layout("so101") == (env.obs_dim, sl)
    assert sl["joint_pos"].stop - sl["joint_pos"].start == len(get_robot("so101").arm_joints) + 1

    obs, *_ = env.step(np.zeros(4))
    np.testing.assert_allclose(obs[sl["cube_to_ee"]], obs[sl["ee_pos"]] - obs[sl["cube_pos"]], atol=1e-6)
    np.testing.assert_allclose(obs[sl["cup_pos"]], get_robot("so101").cup_pos, atol=1e-6)
    np.testing.assert_allclose(obs[sl["ee_pos"]], env.data.site("gripperframe").xpos, atol=1e-6)
    assert obs[sl["time_frac"]][0] == pytest.approx(1 / env.max_steps)
    assert 0.0 <= obs[sl["gripper_open"]][0] <= 1.0
    np.testing.assert_allclose(np.linalg.norm(obs[sl["cube_yaw"]]), 1.0, atol=1e-5)


def test_reset_is_deterministic(env, env2):
    actions = _random_actions(10)
    obs_a, _ = env.reset(seed=20_017)
    obs_b, _ = env2.reset(seed=20_017)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(_rollout(env, actions), _rollout(env2, actions))
    other, _ = env.reset(seed=20_018)
    assert not np.array_equal(other[env.obs_slices["cube_pos"]], obs_a[env.obs_slices["cube_pos"]])


def test_get_set_state_roundtrip(env, env2):
    env.reset(seed=5)
    _rollout(env, _random_actions(15, seed=1))
    state = env.get_state()
    actions = _random_actions(20, seed=2)
    first = _rollout(env, actions)

    env.set_state(state)
    np.testing.assert_array_equal(_rollout(env, actions), first)

    env2.reset(seed=999)  # a different episode in a different env instance
    env2.set_state(state)
    np.testing.assert_array_equal(_rollout(env2, actions), first)


@pytest.mark.parametrize("robot", ["so101", "piper"])
def test_restore_resumes_a_closed_loop_episode_exactly(robot):
    """pass@k and oracles restart a policy from a saved state: the continuation must not drift."""
    env = CupDropEnv(robot=robot)
    policy = KnobController(DEFAULT_KNOBS, robot)
    obs, _ = env.reset(seed=10_003)
    policy.reset([10_003])
    for _ in range(20):
        obs, *_ = env.step(policy.act(obs[None])[0])
    state, saved_policy = env.get_state(), copy.deepcopy(policy)

    def finish(obs, policy):
        rows = []
        for _ in range(40):
            obs, *_ = env.step(policy.act(obs[None])[0])
            rows.append(obs)
        return np.stack(rows)

    first = finish(obs, policy)
    env.set_state(state)
    np.testing.assert_array_equal(env.observe(), obs)  # the restored observation is the one the episode had
    np.testing.assert_array_equal(finish(env.observe(), saved_policy), first)


@pytest.mark.parametrize("robot", ["so101", "piper"])
def test_restore_is_exact_when_the_lead_clamp_binds(robot):
    """A steady full-speed push outruns the arm, so the MAX_LEAD clamp (which reads the measured EE) is active."""
    env = CupDropEnv(robot=robot)
    env.reset(seed=3)
    push = np.array([1.0, -1.0, -1.0, 0.0])
    lo, hi = np.array(env.robot.workspace_lo), np.array(env.robot.workspace_hi)
    for _ in range(40):
        unclamped = np.clip(env.get_state()["ee_target"] + MAX_DELTA * push[:3], lo, hi)
        obs, *_, info = env.step(push)
        if not np.allclose(info["ee_target"], unclamped):
            break
    else:
        pytest.fail("the lead clamp never became active")
    ee = obs[env.obs_slices["ee_pos"]]
    assert np.abs(info["ee_target"] - ee).max() < MAX_LEAD + MAX_DELTA
    state = env.get_state()
    first = _rollout(env, [push] * 10)
    env.set_state(state)
    np.testing.assert_array_equal(env.observe(), obs)
    np.testing.assert_array_equal(_rollout(env, [push] * 10), first)


def test_float64_actions_replay_a_float32_recording(env, env2):
    """Rollouts store actions as float32; the env quantizes too, so a replay matches the original."""
    actions = _random_actions(12, seed=4)
    env.reset(seed=8)
    env2.reset(seed=8)
    np.testing.assert_array_equal(_rollout(env, actions), _rollout(env2, actions.astype(np.float32)))


def test_actions_are_clipped(env, env2):
    env.reset(seed=7)
    env2.reset(seed=7)
    big = np.array([5.0, -7.0, 3.0, 9.0])
    target0 = env.get_state()["ee_target"]
    obs_big, *_, info = env.step(big)
    obs_unit, *_ = env2.step(np.clip(big, -1, 1))
    np.testing.assert_array_equal(obs_big, obs_unit)
    assert np.all(np.abs(info["ee_target"] - target0) <= MAX_DELTA + 1e-9)


def _place_cube(env, xy):
    env.data.joint("cube").qpos[:] = [xy[0], xy[1], env.cup.floor + CUBE_HALF + 0.001, 1, 0, 0, 0]
    env.data.joint("cube").qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_success_detector(env):
    cup = np.array(env.robot.cup_pos)
    env.reset(seed=0)
    _place_cube(env, cup + [0.005, -0.005])
    results = [env.step(np.zeros(4)) for _ in range(3)]
    assert [r[2] for r in results] == [False, False, True]  # terminated after 3 consecutive steps
    assert [r[1] for r in results] == [0.0, 0.0, 1.0]
    assert results[-1][4]["success"]

    env.reset(seed=0)
    outside = cup + [env.cup.inner_radius + env.cup.wall + CUBE_HALF + 0.01, 0.0]
    _place_cube(env, outside)
    results = [env.step(np.zeros(4)) for _ in range(5)]
    assert not any(r[2] for r in results) and not results[-1][4]["success"]
    assert results[-1][4]["near_miss"] == pytest.approx(np.linalg.norm(outside - cup), abs=0.01)


def _run(env, policy, seed, override=None, on_step=None):
    """Roll out a KnobController; `override(policy, action)` may replace actions, `on_step` sees each info."""
    obs, info = env.reset(seed=seed)
    policy.reset([seed])
    for _ in range(env.max_steps):
        action = policy.act(obs[None])[0]
        obs, _, terminated, truncated, info = env.step(action if override is None else override(policy, action))
        if on_step is not None:
            on_step(obs, info)
        if terminated or truncated:
            break
    return obs, info


@pytest.mark.parametrize("robot", ["so101", "piper"])
def test_holding_the_cube_inside_the_cup_is_not_success(robot):
    """The task is to drop the cube in: carry it into the cup, keep the gripper closed, and nothing counts."""
    env = CupDropEnv(robot=robot, max_steps=80)
    sl = env.obs_slices

    def hold(policy, action):  # from the release phase on: lower the cube into the cup and keep squeezing
        if policy.phase[0] >= RELEASE:
            action[2] = np.clip((0.035 - env.observe()[sl["ee_pos"]][2]) / MAX_DELTA, -1.0, 1.0)
            action[3] = 1.0
        return action

    obs, info = _run(env, KnobController(TUNED_KNOBS[robot], robot), 20_000, override=hold)
    cube = obs[sl["cube_pos"]]
    assert info["grasped"] and not info["success"]
    assert np.linalg.norm(cube[:2] - obs[sl["cup_pos"]]) < env.cup.inner_radius - 0.005 and cube[2] < env.cup.height


def test_a_tipped_cup_does_not_hold_the_cube(env):
    env.reset(seed=0)
    cup = np.array(env.robot.cup_pos)
    _place_cube(env, cup)
    tip = np.deg2rad(30.0)
    env.data.joint("cup").qpos[3:] = [np.cos(tip / 2), np.sin(tip / 2), 0, 0]
    mujoco.mj_forward(env.model, env.data)
    assert not env._cube_in_cup(env._cube_cup_dist())
    env.data.joint("cup").qpos[3:] = [1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)
    assert env._cube_in_cup(env._cube_cup_dist())


def test_the_cup_can_be_pushed_and_the_observation_follows_it(env):
    """The cup is a free body: driving the gripper into it moves it, and `cup_pos` reports where it went."""
    obs, _ = env.reset(seed=0)
    sl = env.obs_slices
    home = np.array(env.robot.cup_pos)
    np.testing.assert_allclose(obs[sl["cup_pos"]], home, atol=1e-4)
    for _ in range(60):  # go to rim height beside the cup, then sweep through where it stands
        ee = obs[sl["ee_pos"]]
        target = np.array([home[0], home[1] + (0.10 if ee[2] > 0.05 and abs(ee[1]) < 0.02 else -0.06), 0.04])
        step = np.clip((target - ee) / MAX_DELTA, -1, 1)
        if abs(ee[2] - 0.04) > 0.01:  # first descend at the start position
            step[:2] = 0.0
        obs, *_ = env.step(np.array([*step, 1.0]))
    moved = np.linalg.norm(obs[sl["cup_pos"]] - home)
    assert moved > 0.02
    np.testing.assert_allclose(obs[sl["cup_pos"]], env.data.body("cup").xpos[:2], atol=1e-6)


@pytest.mark.parametrize("robot", ["so101", "piper"])
def test_tuned_drops_cleanly_without_touching_the_cup(robot):
    """The reference controller must not win by leaning on the cup: it stays where it was put."""
    env = CupDropEnv(robot=robot)
    home = np.array(env.robot.cup_pos)
    for seed in (20_000, 20_001, 20_002):
        obs, info = _run(env, KnobController(TUNED_KNOBS[robot], robot), seed)
        assert info["success"]
        assert np.linalg.norm(obs[env.obs_slices["cup_pos"]] - home) < 0.002
        assert env.data.body("cup").xmat[8] > np.cos(np.deg2rad(1.0))


@pytest.mark.parametrize("robot", ["so101", "piper"])
def test_contacts_are_stiff(robot):
    """The squeezed cube must not sink into the table, fingers or cup (soft contacts let policies cheat)."""
    env = CupDropEnv(robot=robot)
    resting, ever = [0.0], [0.0]

    def measure(obs, info):
        c = env.data.contact
        depth = -c.dist[(c.geom1 == env._cube_geom) | (c.geom2 == env._cube_geom)]
        if len(depth):
            ever[0] = max(ever[0], depth.max())
            if np.linalg.norm(env.data.body("cube").cvel[3:]) < 0.1:  # held or resting, not mid-impact
                resting[0] = max(resting[0], depth.max())

    for knobs in (TUNED_KNOBS[robot], DEFAULT_KNOBS):
        for seed in (20_000, 20_001, 20_005, 20_009):
            _run(env, KnobController(knobs, robot), seed, on_step=measure)
    assert 0 < resting[0] < 0.002
    assert ever[0] < env.cup.floor  # even a landing impact never punches through the cup floor


@pytest.mark.parametrize("robot", ["so101", "piper"])
def test_friction_randomization_reaches_the_contacts(robot):
    """MuJoCo takes the max of two geoms' friction, so randomizing the cube alone would do nothing."""
    env = CupDropEnv(robot=robot, dr=True)

    def cube_table_friction(seed):
        env.reset(seed=seed)
        c = env.data.contact
        on_table = (c.geom1 == env.model.geom("table").id) & (c.geom2 == env._cube_geom)
        assert on_table.any()
        return float(c.friction[on_table][0, 0])

    values = {round(cube_table_friction(seed), 6) for seed in range(6)}
    assert len(values) == 6 and 0.7 <= min(values) < 1.0 < max(values) <= 1.3
    jaw_geoms = np.isin(env.model.geom_bodyid, env._jaws) & (env.model.geom_contype > 0)
    scale = env.model.geom_friction[env._cube_geom, 0] / env._friction0[env._cube_geom, 0]
    np.testing.assert_allclose(env.model.geom_friction[jaw_geoms, 0], scale * env._friction0[jaw_geoms, 0])

    plain = CupDropEnv(robot=robot)
    plain.reset(seed=3)
    np.testing.assert_array_equal(plain.model.geom_friction, plain._friction0)


def test_gripper_force_matches_the_robot_spec():
    so101, piper = CupDropEnv(robot="so101"), CupDropEnv(robot="piper")
    np.testing.assert_allclose(so101.model.actuator("gripper").forcerange, [-1.47, 1.47])  # 50% torque
    assert piper.model.actuator("gripper").gainprm[0] == get_robot("piper").gripper_kp


def _label(env, held_at, holding=False, ever=True):
    tracker = OutcomeTracker(env)
    tracker.ever_grasped, tracker.holding, tracker.held_at = ever, holding, held_at
    return tracker.label


def test_outcome_labels_separate_rim_hits_from_off_target_drops(env):
    cup = env.cup
    beside = cup.inner_radius + cup.wall + CUBE_HALF  # cube center when its face scrapes the outer wall
    assert _label(env, None, ever=False) == "missed_grasp"
    assert _label(env, (0.2, 0.09), holding=True) == "still_held"
    assert _label(env, (beside, 0.05)) == "rim_hit"  # lost below the rim, right next to the cup
    assert _label(env, (beside, 0.10)) == "off_target"  # let go above the rim but off-center
    assert _label(env, (0.01, 0.09)) == "off_target"  # let go over the opening, yet not inside at the end
    assert _label(env, (0.20, 0.05)) == "slip"  # lost far from the cup


def test_time_limit_truncates():
    env = CupDropEnv(robot="so101", max_steps=3)
    env.reset(seed=0)
    flags = [env.step(np.zeros(4))[3] for _ in range(3)]
    assert flags == [False, False, True]


def test_bias_variant_offsets_actual_joint_targets(env):
    biased = CupDropEnv(robot="so101", variant="bias")
    env.reset(seed=11)
    obs_b, _ = biased.reset(seed=11)
    bias = np.array(get_robot("so101").bias_q)
    assert np.abs(bias).max() > 0
    arm = [biased.model.actuator(j).id for j in get_robot("so101").arm_joints]
    np.testing.assert_allclose(biased.data.ctrl[arm] - env.data.ctrl[arm], bias, atol=1e-12)
    # The encoders report the believed angles, so the observation hides the offset.
    true_q = np.array([biased.data.joint(j).qpos[0] for j in get_robot("so101").arm_joints])
    np.testing.assert_allclose(obs_b[: len(bias)], true_q - bias, atol=1e-5)


def test_hard_variant_is_harder_and_never_overlaps_the_cup():
    hard = CupDropEnv(robot="so101", variant="hard")
    v1 = CupDropEnv(robot="so101")
    assert hard.cup.inner_radius < v1.cup.inner_radius
    assert np.all(hard.cube_region > v1.cube_region)
    cup = np.array(hard.robot.cup_pos)
    for seed in range(200):
        obs, _ = hard.reset(seed=seed)
        cube = obs[hard.obs_slices["cube_pos"]][:2]
        assert np.linalg.norm(cube - cup) > hard.cup.inner_radius + hard.cup.wall + CUBE_HALF


def test_ik_reaches_the_far_corners_of_the_hard_region():
    """Holding the gripper vertical used to stall the SO-101 4 cm short of these (reachable) corners."""
    hard = CupDropEnv(robot="so101", variant="hard")
    cx, cy = hard.robot.cube_center
    hx, hy = hard.cube_region / 2
    for corner in ([cx + hx, cy + hy, CUBE_HALF], [cx + hx, cy - hy, CUBE_HALF]):
        q = np.array(hard.robot.home_q)
        pos = hard.ik.forward(q)[0]
        for _ in range(40):  # walk the target out in 2 cm steps, warm-starting like the env does
            pos = pos + np.clip(corner - pos, -MAX_DELTA, MAX_DELTA)
            q = hard.ik.solve(pos, q)
        assert np.linalg.norm(hard.ik.forward(q)[0] - corner) < 0.002


def test_renderer_is_lazy(env):
    assert env._renderer is None  # building and stepping never touches OpenGL
