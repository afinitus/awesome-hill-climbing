"""The one interface every controller implements: a policy that acts on a *batch* of envs.

Rollout workers step many environments in lockstep and call the policy once per step with
all of their observations stacked, which keeps neural-network inference cheap. Policies own
their randomness: ``reset(seeds)`` hands them one seed per env, so a given episode always
sees the same sampling noise no matter which worker or batch it lands in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

ACT_DIM = 4  # [dx, dy, dz, gripper]


@runtime_checkable
class BatchPolicy(Protocol):
    def reset(self, seeds: Sequence[int]) -> None:
        """Start new episodes; one policy-RNG seed per env in the batch."""
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Map observations ``[B, obs_dim]`` to actions ``[B, 4]`` in [-1, 1]."""
        ...


class RandomPolicy:
    """Uniform random actions from a per-env RNG. Useful for tests and as a sanity floor."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.rngs: list[np.random.Generator] = []

    def reset(self, seeds: Sequence[int]) -> None:
        self.rngs = [np.random.default_rng(s) for s in seeds]

    def act(self, obs: np.ndarray) -> np.ndarray:
        actions = [rng.uniform(-1, 1, ACT_DIM) for rng in self.rngs]
        return self.scale * np.stack(actions).astype(np.float32)


class SingleEnvPolicy(Protocol):
    def reset(self, seed: int) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...  # obs [obs_dim] -> action [4]


class Batched:
    """Run a single-env policy as a ``BatchPolicy`` by keeping one copy per env.

    Handy for stateful controllers that are easier to write for one env at a time.
    ``make_single`` must be picklable (a top-level class or a ``functools.partial``) so the
    wrapper can be built inside a rollout worker.
    """

    def __init__(self, make_single: Callable[[], SingleEnvPolicy]):
        self.make_single = make_single
        self.policies: list[SingleEnvPolicy] = []

    def reset(self, seeds: Sequence[int]) -> None:
        self.policies = [self.make_single() for _ in seeds]
        for policy, seed in zip(self.policies, seeds):
            policy.reset(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.stack([p.act(o) for p, o in zip(self.policies, obs)]).astype(np.float32)
