"""KnobController: a scripted pick-and-drop expert with ten named knobs.

This is the "classic robotics" policy of the repo: a phase machine that approaches the cube, grasps it,
carries it over the cup and lets go. It reads only what a real SO-101 setup could measure (cube position
from a camera, EE position from joint encoders, the cup position, the clock), never the simulator state,
so the same code could drive the real arm. The cup position is read once, at the start of the episode:
the script plans its drop point up front and does not chase a cup it has knocked away, so a rim hit is a
failure rather than a lucky shove.

Its behavior is set by `KNOB_SPACE`: offsets, heights, timings and speed. `DEFAULT_KNOBS` are plausible
but mis-tuned (roughly half the episodes fail, for several different reasons); `TUNED_KNOBS` are what a
careful search finds. Chapters 1-2 climb from one to the other, and the tuned controller (plus optional
noise for "sloppy" demos) generates the demonstrations for the learned base policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from lastmile.envs.cupdrop import MAX_DELTA, obs_layout


@dataclass(frozen=True)
class Knob:
    name: str
    low: float
    high: float
    unit: str
    meaning: str


KNOB_SPACE: list[Knob] = [
    Knob("grasp_dx", -0.03, 0.03, "m", "grasp point offset from the cube center along x (base frame)"),
    Knob("grasp_dy", -0.03, 0.03, "m", "grasp point offset from the cube center along y"),
    Knob("grasp_dz", -0.02, 0.03, "m", "gripper-site height relative to the cube center when closing"),
    Knob("approach_h", 0.04, 0.12, "m", "hover height above the table before descending onto the cube"),
    Knob("close_steps", 1, 10, "steps", "control steps to wait with the gripper closing before lifting"),
    Knob("lift_h", 0.04, 0.14, "m", "carry height of the gripper site above the table"),
    Knob("drop_dx", -0.05, 0.05, "m", "release point offset from the cup axis along x"),
    Knob("drop_dy", -0.05, 0.05, "m", "release point offset from the cup axis along y"),
    Knob("drop_h", 0.05, 0.14, "m", "gripper-site height above the table when opening over the cup"),
    Knob("speed", 0.2, 1.0, "x max", "cap on EE speed as a fraction of the 2 cm/step action limit"),
]
KNOB_NAMES = [k.name for k in KNOB_SPACE]

# Plausible first guesses, the kind you write after eyeballing the gripper in the viewer: center the gripper
# frame on the cube "by eye" (about 1 cm off in y, and unaware that the SO-101's fixed jaw wants a small -x
# offset), close a little high to keep the fingertips off the table, and aim 3 cm short of the cup's center.
# On SO-101 v1 this gives a bit under 50% success: missed grasps, a few slips and off-target drops, while
# the successes are clean drops that never touch the cup (tools/validate_env.py prints the measured mix).
# The carry height clears the rim; rim hits start once lift_h goes below about 0.078.
DEFAULT_KNOBS: dict[str, float] = {
    "grasp_dx": 0.0, "grasp_dy": 0.011, "grasp_dz": 0.005, "approach_h": 0.08, "close_steps": 3,
    "lift_h": 0.085, "drop_dx": -0.03, "drop_dy": 0.0, "drop_h": 0.08, "speed": 1.0,
}

# Hand-tuned references (>= 97% on the eval set; see tools/validate_env.py for the measured numbers).
# The grasp offsets differ because the SO-101 has one fixed and one moving jaw while the PiPER's two
# fingers close symmetrically.
TUNED_KNOBS: dict[str, dict[str, float]] = {
    "so101": {"grasp_dx": -0.005, "grasp_dy": 0.0, "grasp_dz": -0.002, "approach_h": 0.06, "close_steps": 3,
              "lift_h": 0.09, "drop_dx": 0.0, "drop_dy": 0.0, "drop_h": 0.09, "speed": 0.8},
    "piper": {"grasp_dx": -0.003, "grasp_dy": 0.0, "grasp_dz": -0.005, "approach_h": 0.06, "close_steps": 3,
              "lift_h": 0.10, "drop_dx": 0.0, "drop_dy": 0.0, "drop_h": 0.10, "speed": 0.8},
}

# Phase machine. Each phase has a target (EE position + gripper command) and an exit test.
APPROACH, DESCEND, CLOSE, LIFT, CARRY, LOWER, RELEASE, RETREAT = range(8)
PHASE_NAMES = ("approach", "descend", "close", "lift", "carry", "lower", "release", "retreat")
POS_TOL = 0.006  # m, "arrived" tolerance in xy
Z_TOL = 0.006  # m, "arrived" tolerance in z
PHASE_TIMEOUT = 30  # steps; a stuck phase moves on rather than freezing the episode
RELEASE_STEPS = 4
GAIN = 1.0  # fraction of the remaining error commanded per step (before the speed cap)


def knob_dict(knobs: Mapping[str, float] | Sequence[float] | np.ndarray) -> dict[str, float]:
    """Accept a {name: value} mapping or a vector in KNOB_SPACE order; clip to the knob bounds."""
    if isinstance(knobs, Mapping):
        missing = set(KNOB_NAMES) - set(knobs)
        if missing:
            raise ValueError(f"missing knobs: {sorted(missing)}")
        values = [knobs[n] for n in KNOB_NAMES]
    else:
        values = list(np.asarray(knobs, dtype=float).reshape(-1))
        if len(values) != len(KNOB_SPACE):
            raise ValueError(f"expected {len(KNOB_SPACE)} knob values, got {len(values)}")
    return {k.name: float(np.clip(v, k.low, k.high)) for k, v in zip(KNOB_SPACE, values)}


def knob_vector(knobs: Mapping[str, float]) -> np.ndarray:
    return np.array([knobs[n] for n in KNOB_NAMES], dtype=float)


class KnobController:
    """Batched scripted policy (BatchPolicy): `reset(seeds)` then `act(obs[B, D]) -> actions[B, 4]`.

    noise (optional, for sloppy demos):
      {"knob_std": s, "action_std": a} -> each episode perturbs every knob by N(0, s * knob range), and
      every action gets N(0, a) added before clipping. Both are drawn from the per-env RNG seeded in reset.
    """

    def __init__(self, knobs: Mapping[str, float] | Sequence[float] | np.ndarray, robot: str = "so101",
                 noise: Mapping[str, float] | None = None):
        self.knobs = knob_dict(knobs)
        self.robot = robot
        self.noise = dict(noise or {})
        unknown = set(self.noise) - {"knob_std", "action_std"}
        if unknown:
            raise ValueError(f"unknown noise keys: {sorted(unknown)}")
        _, self.sl = obs_layout(robot)
        self.reset([0])

    def reset(self, seeds: Sequence[int]) -> None:
        n = len(seeds)
        self.rngs = [np.random.default_rng(s) for s in seeds]
        base = knob_vector(self.knobs)
        span = np.array([k.high - k.low for k in KNOB_SPACE])
        lo = np.array([k.low for k in KNOB_SPACE])
        hi = np.array([k.high for k in KNOB_SPACE])
        knob_std = self.noise.get("knob_std", 0.0)
        self.k = np.stack([
            np.clip(base + (rng.normal(0.0, knob_std, len(base)) * span if knob_std > 0 else 0.0), lo, hi)
            for rng in self.rngs
        ])
        self.phase = np.zeros(n, dtype=int)
        self.phase_steps = np.zeros(n, dtype=int)
        self.hold = np.zeros((n, 3))  # grasp pose, latched when the descent ends
        self.cup = np.full((n, 2), np.nan)  # cup position, latched from the first observation

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(obs)
        if len(obs) != len(self.phase):
            raise ValueError(f"batch size {len(obs)} != number of reset seeds {len(self.phase)}")
        actions = np.zeros((len(obs), 4))
        for i, o in enumerate(obs):
            actions[i] = self._act_one(i, o)
        action_std = self.noise.get("action_std", 0.0)
        if action_std > 0:
            actions += np.stack([rng.normal(0.0, action_std, 4) for rng in self.rngs])
        return np.clip(actions, -1.0, 1.0)

    def _act_one(self, i: int, o: np.ndarray) -> np.ndarray:
        k = dict(zip(KNOB_NAMES, self.k[i]))
        ee = o[self.sl["ee_pos"]].astype(float)
        cube = o[self.sl["cube_pos"]].astype(float)
        if np.isnan(self.cup[i]).any():
            self.cup[i] = o[self.sl["cup_pos"]]
        cup = self.cup[i]
        grasp_xy = cube[:2] + (k["grasp_dx"], k["grasp_dy"])
        drop_xy = cup + (k["drop_dx"], k["drop_dy"])
        p = self.phase[i]

        if p == APPROACH:
            target, grip = np.array([*grasp_xy, k["approach_h"]]), -1.0
            done = _near(ee, target)
        elif p == DESCEND:
            target, grip = np.array([*grasp_xy, cube[2] + k["grasp_dz"]]), -1.0
            done = _near(ee, target)
            self.hold[i] = target  # close where we stopped, not chasing the cube as the jaw pushes it
        elif p == CLOSE:
            target, grip = self.hold[i], 1.0
            done = self.phase_steps[i] >= round(k["close_steps"])
        elif p == LIFT:
            target, grip = np.array([*self.hold[i][:2], k["lift_h"]]), 1.0
            done = _near(ee, target)
        elif p == CARRY:
            target, grip = np.array([*drop_xy, k["lift_h"]]), 1.0
            done = _near(ee, target)
        elif p == LOWER:
            target, grip = np.array([*drop_xy, k["drop_h"]]), 1.0
            done = _near(ee, target)
        elif p == RELEASE:
            target, grip = np.array([*drop_xy, k["drop_h"]]), -1.0
            done = self.phase_steps[i] >= RELEASE_STEPS
        else:  # RETREAT: back up and wait
            target, grip = np.array([*drop_xy, max(k["drop_h"], k["lift_h"]) + 0.02]), -1.0
            done = False

        self.phase_steps[i] += 1
        if done or (p != RETREAT and p != CLOSE and self.phase_steps[i] > PHASE_TIMEOUT):
            self.phase[i] = p + 1
            self.phase_steps[i] = 0

        v = GAIN * (target - ee) / MAX_DELTA
        norm = np.linalg.norm(v)
        if norm > k["speed"]:
            v *= k["speed"] / norm
        return np.array([*v, grip])


def _near(ee: np.ndarray, target: np.ndarray) -> bool:
    return bool(np.linalg.norm(ee[:2] - target[:2]) < POS_TOL and abs(ee[2] - target[2]) < Z_TOL)
