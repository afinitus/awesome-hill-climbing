"""Batched, multi-process rollouts and the standard ``evaluate()`` call (DESIGN.md §5).

How it works:

* The seed list is cut into fixed-size *shards*. Each shard is one batch: a worker builds
  (and caches) one env per seed, resets them all, and steps them in lockstep so the policy
  is called once per step with ``obs[B, obs_dim]``.
* An env that finishes is masked out: it is never stepped again, while the policy keeps
  receiving its last observation so the batch shape (and any per-row policy state) stays fixed.
* Shards depend only on the seed list, never on ``n_workers``. Together with per-env policy
  seeds this makes results bit-for-bit identical for ``n_workers=1`` and ``n_workers=8``.
* Workers live in a persistent ``spawn`` pool, so repeated evaluations inside a search loop
  do not pay process start-up and env construction every time. The pool only ever grows: a call
  with fewer ``n_workers`` reuses the big pool and simply keeps fewer shards in flight.

Because of ``spawn``, scripts that call ``evaluate`` with ``n_workers > 1`` must guard their
entry point with ``if __name__ == "__main__":``, and ``make_policy`` / ``env_factory`` must be
picklable (top-level functions, classes, or ``functools.partial`` of them — not lambdas).
"""

from __future__ import annotations

import atexit
import math
import multiprocessing as mp
import pickle
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from lastmile.common.eval import INIT_SETS, EvalResult, policy_seed
from lastmile.common.ledger import check_robot_category
from lastmile.common.policy import BatchPolicy

if TYPE_CHECKING:
    from lastmile.common.ledger import Ledger

EnvFactory = Callable[[], Any]
PolicyFactory = Callable[[], BatchPolicy]

MAX_SHARD_SIZE = 32  # largest policy batch
TARGET_SHARDS = 16  # enough shards to keep a 16-worker pool busy on a 256-seed eval


def _make_cupdrop(robot: str, variant: str, **env_kwargs: Any) -> Any:
    """Default env factory. Imports lazily so this module stays light for workers and tests."""
    from lastmile.envs.cupdrop import CupDropEnv

    return CupDropEnv(robot=robot, variant=variant, **env_kwargs)


def default_shard_size(n_seeds: int) -> int:
    """Shard size as a function of the seed count only (never of n_workers, for parity)."""
    return min(MAX_SHARD_SIZE, max(1, math.ceil(n_seeds / TARGET_SHARDS)))


# --------------------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------------------

MAX_CACHED_FACTORIES = 4  # e.g. two robots x two variants alternating inside one script
_env_cache: dict[bytes, list[Any]] = {}  # pickled factory -> envs built in this process, oldest first
_torch_limited = False


def _get_envs(env_factory: EnvFactory, n: int) -> list[Any]:
    """Reuse envs across shards with the same factory; building MuJoCo models is slow.

    A few factories are kept at once, so a script that alternates robots or variants does not rebuild
    its envs on every call.
    """
    try:
        key = pickle.dumps(env_factory)
    except (pickle.PicklingError, AttributeError, TypeError):  # e.g. a lambda, in-process only
        return [env_factory() for _ in range(n)]
    envs = _env_cache.pop(key, [])
    _env_cache[key] = envs  # re-insert: most recently used last
    while len(_env_cache) > MAX_CACHED_FACTORIES:
        _env_cache.pop(next(iter(_env_cache)))
    while len(envs) < n:
        envs.append(env_factory())
    return envs[:n]


def _limit_torch_threads() -> None:
    """One torch thread per worker: many small processes beat one process fighting over cores.

    Only acts if torch is already imported (unpickling a torch policy factory imports it), so workers
    running torch-free policies such as KnobController never pay the ~1.5 s import.
    """
    global _torch_limited
    if not _torch_limited and "torch" in sys.modules and mp.parent_process() is not None:
        sys.modules["torch"].set_num_threads(1)
        _torch_limited = True


def _policy_extras(policy: BatchPolicy, batch: int) -> dict[str, np.ndarray]:
    """Per-env arrays the policy exposes as ``last_*`` attributes (e.g. ``last_noise``)."""
    return {
        name: value
        for name, value in vars(policy).items()
        if name.startswith("last_") and isinstance(value, np.ndarray) and value.shape[:1] == (batch,)
    }


def _run_shard(
    make_policy: PolicyFactory,
    env_factory: EnvFactory,
    seeds: list[int],
    policy_seeds: list[int],
    record: bool,
) -> dict[str, Any]:
    """Roll out one batch of episodes in lockstep and return per-episode results."""
    cpu_start = time.process_time()
    _limit_torch_threads()
    batch = len(seeds)
    envs = _get_envs(env_factory, batch)
    policy = make_policy()
    _limit_torch_threads()  # a factory may import torch only when called
    policy.reset(list(policy_seeds))

    obs = np.stack([env.reset(seed=seed)[0] for env, seed in zip(envs, seeds)]).astype(np.float32)
    active = np.ones(batch, dtype=bool)
    success = np.zeros(batch, dtype=bool)
    steps = np.zeros(batch, dtype=np.int64)
    near_miss = np.full(batch, np.inf)
    logs: list[dict[str, list]] = [{"obs": [], "actions": [], "rewards": []} for _ in range(batch)]
    infos: list[dict[str, list]] = [{} for _ in range(batch)]

    while active.any():
        # float32 is the precision actions are recorded in; CupDropEnv quantizes to it as well, so a
        # recorded action is exactly what the env executed.
        actions = np.clip(np.asarray(policy.act(obs), dtype=np.float32), -1.0, 1.0)
        extras = _policy_extras(policy, batch) if record else {}
        for i in np.flatnonzero(active):
            if record:
                logs[i]["obs"].append(obs[i].copy())
                logs[i]["actions"].append(actions[i])
                for name, value in extras.items():
                    logs[i].setdefault(name, []).append(np.array(value[i]))
            next_obs, reward, terminated, truncated, info = envs[i].step(actions[i])
            obs[i] = next_obs
            steps[i] += 1
            near_miss[i] = info["near_miss"]
            if record:
                logs[i]["rewards"].append(float(reward))
                for key, value in info.items():
                    infos[i].setdefault(key, []).append(np.array(value))
            if terminated or truncated:
                active[i] = False
                success[i] = bool(info["success"])

    trajectories = None
    if record:
        trajectories = []
        for i in range(batch):
            core = ("obs", "actions", "rewards")
            traj = {key: np.asarray(logs[i][key], dtype=np.float32) for key in core}
            traj.update({key: np.stack(values) for key, values in logs[i].items() if key not in core})
            traj.update(
                seed=seeds[i],
                policy_seed=policy_seeds[i],
                success=bool(success[i]),
                steps=int(steps[i]),
                near_miss=float(near_miss[i]),
                final_obs=obs[i].copy(),
                infos={key: np.stack(values) for key, values in infos[i].items()},
            )
            trajectories.append(traj)

    return {
        "successes": success,
        "steps_to_success": np.where(success, steps, -1),
        "near_miss": near_miss,
        "env_steps": int(steps.sum()),
        "trajectories": trajectories,
        "cpu_seconds": time.process_time() - cpu_start,
    }


# --------------------------------------------------------------------------------------
# Persistent process pool
# --------------------------------------------------------------------------------------

_pool: ProcessPoolExecutor | None = None
_pool_size = 0


def _get_pool(n_workers: int) -> ProcessPoolExecutor:
    """The shared pool, restarted only when more workers are needed than it has.

    Respawning costs seconds (process start-up plus rebuilding every worker's envs), so a script that
    searches with 16 workers and evaluates with 8 must not pay it on every switch.
    """
    global _pool, _pool_size
    if _pool is None or _pool_size < n_workers:
        shutdown_pool()
        _pool = ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn"))
        _pool_size = n_workers
    return _pool


def shutdown_pool() -> None:
    """Stop the worker processes (also runs automatically at interpreter exit)."""
    global _pool, _pool_size
    if _pool is not None:
        _pool.shutdown(wait=True, cancel_futures=True)
    _pool, _pool_size = None, 0


atexit.register(shutdown_pool)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def rollout_seeds(
    make_policy: PolicyFactory,
    seeds: Sequence[int],
    policy_seeds: Sequence[int],
    *,
    robot: str = "so101",
    variant: str = "v1",
    env_kwargs: dict[str, Any] | None = None,
    env_factory: EnvFactory | None = None,
    n_workers: int = 8,
    shard_size: int | None = None,
    record_trajectories: bool = False,
    ledger: Ledger | None = None,
    category: str | None = None,
    init_set: str | None = None,
) -> EvalResult:
    """Run one episode per ``(seed, policy_seed)`` pair and return results in seed order.

    ``env_factory`` overrides the default ``CupDropEnv(robot, variant, **env_kwargs)``;
    tests use it to inject a tiny fake env. If ``ledger`` is given, the simulated steps are
    charged to ``ledger.robot_steps[category]`` and worker CPU time to its compute total.
    ``category`` must be "search", "train" or "eval". Left as None it follows the init set: "search"
    for the search set, "eval" otherwise, so search episodes land in the improvement budget by default.

    With ``record_trajectories`` each episode comes back as a dict: ``obs`` / ``actions`` /
    ``rewards`` per step (``obs[t]`` is what the policy saw before ``actions[t]``), ``final_obs``,
    ``infos`` (every env ``info`` key stacked over steps, e.g. ``infos["grasped"]``), the seeds,
    outcome fields, and any per-env ``last_*`` arrays the policy exposes.
    """
    seeds = [int(s) for s in seeds]
    policy_seeds = [int(s) for s in policy_seeds]
    if len(seeds) != len(policy_seeds) or not seeds:
        raise ValueError(f"need matching, non-empty seed lists (got {len(seeds)}, {len(policy_seeds)})")
    if category is None:
        category = "search" if init_set == "search" else "eval"
    check_robot_category(category)  # before any rollout runs, so a typo costs nothing
    if env_factory is None:
        env_factory = partial(_make_cupdrop, robot, variant, **(env_kwargs or {}))
    size = shard_size or default_shard_size(len(seeds))
    shards = [
        (make_policy, env_factory, seeds[i : i + size], policy_seeds[i : i + size], record_trajectories)
        for i in range(0, len(seeds), size)
    ]

    if n_workers <= 1:
        outputs = [_run_shard(*shard) for shard in shards]
        worker_cpu = 0.0  # already counted by this process's own CPU clock
    else:
        outputs = _run_in_pool(shards, n_workers)
        worker_cpu = sum(out["cpu_seconds"] for out in outputs)

    trajectories = None
    if record_trajectories:
        trajectories = [traj for out in outputs for traj in out["trajectories"]]
    result = EvalResult(
        successes=np.concatenate([out["successes"] for out in outputs]),
        steps_to_success=np.concatenate([out["steps_to_success"] for out in outputs]),
        near_miss=np.concatenate([out["near_miss"] for out in outputs]),
        env_steps=sum(out["env_steps"] for out in outputs),
        seeds=np.asarray(seeds),
        policy_seeds=np.asarray(policy_seeds),
        init_set=init_set,
        trajectories=trajectories,
    )
    if ledger is not None:
        ledger.robot_steps[category] += result.env_steps
        ledger.worker_cpu_seconds += worker_cpu
    return result


def _run_in_pool(shards: list[tuple], n_workers: int) -> list[dict[str, Any]]:
    """Run every shard with at most ``n_workers`` in flight, collect results in order, surface errors."""
    pool = _get_pool(n_workers)
    futures: list[Future] = []
    running: set[Future] = set()
    for shard in shards:  # the pool may be larger than n_workers; submitting in a window respects the request
        if len(running) >= n_workers:
            _, running = wait(running, return_when=FIRST_COMPLETED)
        try:
            futures.append(pool.submit(_run_shard, *shard))
        except BrokenProcessPool as exc:
            # A worker can die *between* submissions after earlier futures succeeded. Breaking here
            # would then return a prefix of the episodes beside the full seed list: fail the whole
            # evaluation instead of silently dropping the unsubmitted shards.
            for future in futures:
                future.cancel()
            shutdown_pool()
            seeds = shard[2]
            raise RuntimeError(
                f"rollout worker failed while submitting shard with seeds {seeds[0]}..{seeds[-1]}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        running.add(futures[-1])
    outputs = []
    for shard, future in zip(shards, futures):
        try:
            outputs.append(future.result())
        except Exception as exc:
            for f in futures:
                f.cancel()
            hint = ""
            if isinstance(exc, BrokenProcessPool):
                shutdown_pool()
                # Spawned workers re-import the main script, so unguarded top-level code runs in them.
                hint = "; if workers die at start-up, guard the script's top-level code with if __name__ == '__main__'"
            seeds = shard[2]
            raise RuntimeError(
                f"rollout worker failed on shard with seeds {seeds[0]}..{seeds[-1]}: "
                f"{type(exc).__name__}: {exc} (remote traceback above){hint}"
            ) from exc
    return outputs


def evaluate(
    make_policy: PolicyFactory,
    init_set: str = "eval",
    robot: str = "so101",
    variant: str = "v1",
    n_workers: int = 8,
    ledger: Ledger | None = None,
    category: str | None = None,
    salt: int = 0,
    record_trajectories: bool = False,
    *,
    n: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    env_factory: EnvFactory | None = None,
    shard_size: int | None = None,
) -> EvalResult:
    """Score a policy on a named initial-state set with common random numbers.

    ``n`` keeps only the first ``n`` seeds of the set (for ``--quick`` runs); the paired
    structure is preserved because episode ``i`` keeps the same env and policy seeds.
    ``category`` is the ledger budget to charge; by default "search" for the search set and "eval"
    for every other set (pass "train" or "search" explicitly for other improvement rollouts).
    """
    if n is not None and not 1 <= n <= len(INIT_SETS[init_set]):
        raise ValueError(f"n={n} is outside 1..{len(INIT_SETS[init_set])} for init set {init_set!r}")
    seeds = INIT_SETS[init_set][:n]
    policy_seeds = [policy_seed(init_set, i, salt) for i in range(len(seeds))]
    return rollout_seeds(
        make_policy,
        seeds,
        policy_seeds,
        robot=robot,
        variant=variant,
        env_kwargs=env_kwargs,
        env_factory=env_factory,
        n_workers=n_workers,
        shard_size=shard_size,
        record_trajectories=record_trajectories,
        ledger=ledger,
        category=category,
        init_set=init_set,
    )
