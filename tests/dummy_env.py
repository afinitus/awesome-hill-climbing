"""A tiny stand-in for CupDropEnv with the same API (DESIGN.md §2), for fast library tests.

A point "end effector" moves by ``0.02 * a[0:3]`` per step and must hover within 1 cm
(horizontally) of a seed-dependent target, below 5 cm height, for 3 consecutive steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

MAX_DELTA = 0.02
START = np.array([0.20, 0.0, 0.10])


class DummyEnv:
    def __init__(self, max_steps: int = 40, fail_on_seed: int | None = None):
        self.max_steps = max_steps
        self.fail_on_seed = fail_on_seed  # lets tests check that worker errors propagate
        sizes = {
            "joint_pos": 3,
            "ee_pos": 3,
            "gripper_open": 1,
            "cube_pos": 3,
            "cube_to_ee": 3,
            "cube_yaw": 2,
            "cup_pos": 2,
            "time_frac": 1,
        }
        self.obs_slices: dict[str, slice] = {}
        start = 0
        for name, size in sizes.items():
            self.obs_slices[name] = slice(start, start + size)
            start += size
        self.obs_dim = start
        self.done = True

    def reset(self, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        if seed == self.fail_on_seed:
            raise RuntimeError(f"dummy env refuses seed {seed}")
        rng = np.random.default_rng(seed)
        self.target = np.array([rng.uniform(0.15, 0.27), rng.uniform(-0.08, 0.08), 0.0125])
        self.ee = START.copy()
        self.gripper = -1.0
        self.steps = 0
        self.streak = 0
        self.near_miss = float("inf")
        self.done = False
        return self._obs(), self._info(False)

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("stepped a finished episode (rollout masking is broken)")
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.ee = np.clip(self.ee + MAX_DELTA * a[:3], [0.05, -0.2, 0.0], [0.4, 0.2, 0.3])
        self.gripper = a[3]
        self.steps += 1
        dist = float(np.linalg.norm(self.ee[:2] - self.target[:2]))
        self.near_miss = min(self.near_miss, dist)
        self.streak = self.streak + 1 if (dist < 0.01 and self.ee[2] < 0.05) else 0
        success = self.streak >= 3
        truncated = not success and self.steps >= self.max_steps
        self.done = success or truncated
        return self._obs(), float(success), success, truncated, self._info(success)

    def _obs(self) -> np.ndarray:
        yaw = 0.1
        return np.concatenate(
            [
                self.ee,
                self.ee,
                [(1 - self.gripper) / 2],
                self.target,
                self.target - self.ee,
                [np.sin(yaw), np.cos(yaw)],
                self.target[:2],
                [self.steps / self.max_steps],
            ]
        ).astype(np.float32)

    def _info(self, success: bool) -> dict[str, Any]:
        return {
            "success": success,
            "grasped": False,
            "near_miss": self.near_miss,
            "steps": self.steps,
            "ee_target": self.ee.copy(),
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "ee": self.ee.copy(),
            "target": self.target.copy(),
            "gripper": self.gripper,
            "steps": self.steps,
            "streak": self.streak,
            "near_miss": self.near_miss,
            "done": self.done,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.ee, self.target = state["ee"].copy(), state["target"].copy()
        self.gripper, self.steps, self.streak = state["gripper"], state["steps"], state["streak"]
        self.near_miss, self.done = state["near_miss"], state["done"]


class GreedyPolicy:
    """Noisy proportional controller; about half of its episodes aim slightly off target."""

    def __init__(self, noise: float = 0.3):
        self.noise = noise
        self.rngs: list[np.random.Generator] = []
        self.offsets = np.zeros((0, 3))

    def reset(self, seeds: Sequence[int]) -> None:
        self.rngs = [np.random.default_rng(s) for s in seeds]
        sloppy = np.array([rng.random() < 0.5 for rng in self.rngs])
        self.offsets = np.where(sloppy[:, None], [0.03, 0.0, 0.0], 0.0)

    def act(self, obs: np.ndarray) -> np.ndarray:
        to_target = obs[:, 10:13] + self.offsets  # cube_to_ee slice, plus the sloppy offset
        to_target[:, 2] -= 0.0125  # aim for the table height, not the cube center
        action = np.clip(to_target / MAX_DELTA, -1, 1)
        action += self.noise * np.stack([rng.standard_normal(3) for rng in self.rngs])
        grip = np.ones((len(obs), 1))
        return np.hstack([action, grip]).astype(np.float32)
