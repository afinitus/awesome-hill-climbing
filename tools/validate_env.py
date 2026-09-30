"""Check that CupDrop is calibrated: the tuned expert solves it, the default knobs fail about half the time.

For each robot this scores TUNED_KNOBS and DEFAULT_KNOBS on the "search" (64) and "eval" (256)
initial-state sets with the standard `evaluate()` harness (the same call every chapter uses), and prints
success with Wilson 95% intervals, failure-mode counts, wall time and "shoved": the number of successful
episodes in which the cup ended more than 1 cm from where it started. A success should be a clean drop,
not a cube bulldozed over the rim, so this must be 0 for the tuned knobs and rare for the default ones.
It then measures the single-core CPU cost of a full 150-step episode in-process and writes media to
media/env/:

    <robot>_scene.png         front / top / wrist cameras at reset
    <robot>_tuned.gif         one successful tuned episode (front camera)
    <robot>_default_fail.gif  one failed default-knob episode

    uv run python tools/validate_env.py              # both robots, about a minute on 8 workers
    uv run python tools/validate_env.py --robots so101 --no-media
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

from lastmile.common.eval import INIT_SETS, EvalResult, wilson
from lastmile.common.plotting import save_gif
from lastmile.common.rollout import evaluate
from lastmile.envs.cupdrop import FAILURE_MODES, CupDropEnv, OutcomeTracker, obs_layout
from lastmile.envs.knob_controller import DEFAULT_KNOBS, TUNED_KNOBS, KnobController

MEDIA = Path(__file__).resolve().parents[1] / "media" / "env"
TARGETS = {  # (robot, knobs) -> (lowest acceptable, highest acceptable) success on "eval"
    ("so101", "tuned"): (0.97, 1.0),
    ("so101", "default"): (0.30, 0.55),
    ("piper", "tuned"): (0.95, 1.0),
}
CPU_BUDGET = 0.25  # seconds per 150-step episode
SHOVE = 0.01  # m; a cup that ends further than this from its start was pushed, not just brushed
MAX_DEFAULT_SHOVED = 0.05  # at most this fraction of default-knob successes may involve a shoved cup


def score(robot: str, knobs: dict, init_set: str, n_workers: int) -> tuple[EvalResult, float]:
    """Evaluate one knob setting on one init set; returns (result with trajectories, wall seconds)."""
    t0 = time.perf_counter()
    res = evaluate(partial(KnobController, knobs, robot), init_set, robot=robot, n_workers=n_workers,
                   record_trajectories=True)
    return res, time.perf_counter() - t0


def shoved_successes(robot: str, res: EvalResult) -> int:
    """Successful episodes in which the cup was pushed away from its starting position."""
    cup = obs_layout(robot)[1]["cup_pos"]
    return sum(
        traj["success"] and np.linalg.norm(traj["final_obs"][cup] - traj["obs"][0][cup]) > SHOVE
        for traj in res.trajectories
    )


def full_episode_cpu(env: CupDropEnv, knobs: dict, seeds: list[int]) -> float:
    """Single-core CPU seconds for a full 150-step episode, measured in-process.

    Successful episodes end early, so this keeps stepping to max_steps to measure the worst case.
    """
    policy = KnobController(knobs, env.robot.name)
    t0 = time.process_time()
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        policy.reset([seed])
        for _ in range(env.max_steps):
            obs, *_ = env.step(policy.act(obs[None])[0])
    return (time.process_time() - t0) / len(seeds)


def render_episode(env: CupDropEnv, knobs: dict, seed: int, every: int = 1) -> list:
    """Replay one episode in-process with the front camera, keeping one frame per `every` steps.

    Same env seed and policy seed as the evaluation, and the env quantizes actions to float32 exactly as
    the rollout harness does, so this reproduces the evaluated episode step for step.
    """
    policy = KnobController(knobs, env.robot.name)
    obs, _ = env.reset(seed=seed)
    policy.reset([seed])
    frames = [env.render("front")]
    for step in range(1, env.max_steps + 1):
        obs, _, terminated, truncated, _ = env.step(policy.act(obs[None])[0])
        if step % every == 0 or terminated or truncated:
            frames.append(env.render("front"))
        if terminated or truncated:
            break
    return frames + [frames[-1]] * (5 // every)  # hold the last frame for about half a second


def write_media(robot: str, env: CupDropEnv, labels: dict[str, list[str]]) -> None:
    from PIL import Image

    MEDIA.mkdir(parents=True, exist_ok=True)
    env.reset(seed=INIT_SETS["eval"][0])
    still = np.concatenate([env.render(cam) for cam in ("front", "top", "wrist")], axis=1)
    Image.fromarray(still).save(MEDIA / f"{robot}_scene.png")

    for name, knobs, want_success in (("tuned", TUNED_KNOBS[robot], True), ("default_fail", DEFAULT_KNOBS, False)):
        eval_labels = labels["tuned" if want_success else "default"]
        match = [i for i, lab in enumerate(eval_labels) if (lab == "success") == want_success]
        if not match:
            print(f"  (no {'successful' if want_success else 'failed'} episode to render for {name})")
            continue
        seed = INIT_SETS["eval"][match[0]]
        every = 1 if want_success else 2  # a failure runs all 150 steps: halve the frames to stay under 3 MB
        path = save_gif(render_episode(env, knobs, seed, every), MEDIA / f"{robot}_{name}.gif", fps=10 // every)
        print(f"  wrote {path.relative_to(MEDIA.parents[1])} (seed {seed}, {eval_labels[match[0]]}, "
              f"{path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robots", nargs="+", default=["so101", "piper"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-media", action="store_true")
    args = parser.parse_args()

    verdicts = []
    for robot in args.robots:
        env = CupDropEnv(robot=robot, variant="v1")  # for labeling, CPU timing and media
        eval_labels: dict[str, list[str]] = {}
        print(f"\n=== {robot} (v1, {args.workers} workers) ===")
        print(f"{'knobs':8s} {'set':7s} {'success':>8s}  {'95% CI':>15s}  {'wall':>6s}  {'shoved':>6s}  failure modes")
        for name, knobs in (("tuned", TUNED_KNOBS[robot]), ("default", DEFAULT_KNOBS)):
            for init_set in ("search", "eval"):
                res, wall = score(robot, knobs, init_set, args.workers)
                labels = [OutcomeTracker.label_trajectory(env, traj) for traj in res.trajectories]
                counts = Counter(labels)
                lo, hi = wilson(res.k, res.n)
                shoved = shoved_successes(robot, res)
                modes = ", ".join(f"{m} {counts[m]}" for m in FAILURE_MODES if counts[m])
                print(f"{name:8s} {init_set:7s} {res.k:3d}/{res.n:<4d} [{lo:6.1%}, {hi:6.1%}]  "
                      f"{wall:5.1f}s  {shoved:6d}  {modes or '-'}")
            eval_labels[name] = labels
            target = TARGETS.get((robot, name))
            if target:
                ok = target[0] <= res.sr <= target[1]
                verdicts.append(f"{robot} {name} eval {res.sr:.1%} (target {target[0]:.0%}-{target[1]:.0%}): "
                                f"{'OK' if ok else 'MISS'}")
            allowed = 0 if name == "tuned" else int(MAX_DEFAULT_SHOVED * res.k)
            verdicts.append(f"{robot} {name} eval successes with a shoved cup: {shoved}/{res.k} "
                            f"(allowed {allowed}): {'OK' if shoved <= allowed else 'MISS'}")
        t0 = time.perf_counter()  # workers and their envs are warm now: the cost of an eval inside a search loop
        evaluate(partial(KnobController, DEFAULT_KNOBS, robot), "eval", robot=robot, n_workers=args.workers)
        verdicts.append(f"{robot} warm 256-episode eval (default knobs, {args.workers} workers): "
                        f"{time.perf_counter() - t0:.1f}s wall")
        cpu = full_episode_cpu(env, DEFAULT_KNOBS, INIT_SETS["search"][:8])
        verdicts.append(f"{robot} full 150-step episode: {cpu:.3f}s CPU (budget {CPU_BUDGET}s): "
                        f"{'OK' if cpu < CPU_BUDGET else 'MISS'}")
        if not args.no_media:
            write_media(robot, env, eval_labels)
        env.close()

    print("\n=== targets ===")
    print("\n".join(verdicts))


if __name__ == "__main__":
    main()
