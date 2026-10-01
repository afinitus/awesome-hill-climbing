# lastmile — design contract (v0.1)

Every module and chapter follows this document. If code and this file disagree, fix one of them in the same change.

The goal of the repo: **one robot, one task, one base policy that works about half the time, and a sequence of methods that climb it toward 95%+**, each reported with its success rate *and* its cost in robot-minutes and human-minutes. Sim first (MuJoCo, runs on a Mac CPU), then a real SO-101.

## 0. Principles

1. **One task, one base policy, one scoreboard.** Chapters start from the same frozen checkpoint (`base_v1`) or the same mis-tuned scripted controller, and are scored on the same held-out initial states.
2. **Cost is part of the result.** Every run writes robot-minutes (search / train / eval kept separate), human-minutes and compute.
3. **Honesty is built into the harness.** Search and evaluation use disjoint initial states. Every success rate carries a Wilson 95% interval. Single-attempt success is reported next to any pass@k. Each chapter shows at least one case where its method fails or regresses.
4. **Single-file algorithms.** `algos/NN_name.py` is one runnable file (~150–450 lines, CleanRL style) that imports only from `lastmile/`. No imports between algorithm files.
5. **Laptop first.** Everything in v0.1 runs on a Mac CPU (torch CPU/MPS). NVIDIA-only things are optional extras.
6. **Robot-agnostic core.** `robot="so101"` is the default and the real-hardware target. `robot="piper"` (AgileX PiPER) is supported in sim through the same API.

## 1. Layout

```
lastmile/
  envs/
    cupdrop.py         # CupDropEnv (the task), both robots
    robots.py          # RobotSpec per embodiment (MJCF path, joints, IK config, home pose, gripper map)
    ik.py              # damped-least-squares IK used by the env
    knob_controller.py # scripted expert with named knobs (demos + chapters 1–2)
    assets/            # vendored Menagerie models (so101/, piper/) — do not edit meshes
  common/
    eval.py            # init sets, policy seeds, Wilson/McNemar/bootstrap
    rollout.py         # batched, multi-process rollouts + evaluate()
    policy.py          # BatchPolicy protocol + adapters
    policy_flow.py     # FlowChunk-S model + FlowChunkPolicy
    ledger.py          # cost accounting + results JSON writer
    plotting.py        # shared matplotlib style + GIF helpers
    cli.py             # dataclass -> argparse helper
algos/                 # one runnable file per chapter/method
tools/                 # validate_env.py, leaderboard.py, ...
tests/                 # fast pytest suite (<60 s total)
results/<chapter>/     # one JSON per run (schema §7)
media/<chapter>/       # plots and GIFs
checkpoints/           # base_v1_so101.pt + manifest json
docs/chapters/chNN.md  # theory, paper map, what went wrong
```

## 2. The task: `CupDropEnv`

A table, a 2.5 cm cube placed at random in a 12 × 16 cm region, and an open cup that starts at a fixed position. The robot must pick up the cube and drop it into the cup. The cup is a light free body (about 125 g): it can be pushed or tipped, so hitting the rim has consequences.

```python
from lastmile.envs.cupdrop import CupDropEnv
env = CupDropEnv(robot="so101", variant="v1", max_steps=150, render_size=(256, 256))
obs, info = env.reset(seed=20_017)        # fully deterministic given seed
obs, reward, terminated, truncated, info = env.step(action)
```

- **Constructor.** `robot: "so101" | "piper"`; `variant: "v1" | "hard" | "bias"`; `max_steps: int = 150` (15 s at 10 Hz); `render_size`; `dr: bool = False` (domain randomization, off in v0.1 chapters: each reset scales the sliding friction of every geom by one factor in [0.7, 1.3], so cube–table and cube–finger friction both change).
  - `v1`: main task. `hard`: smaller cup radius and a wider cube region, including starts outside the demo region. `bias`: constant joint-position offsets injected between commanded and actual joint targets (models servo zero errors; the controller does not know them).
- **Control rate.** 10 Hz. Each `step` runs `round(0.1 / timestep)` physics substeps. One env step = 0.1 s of robot time. **Robot-minutes = env_steps / 600.**
- **Action.** `np.ndarray` shape `(4,)`, float, clipped to `[-1, 1]`:
  - `a[0:3]` = end-effector position delta in the robot base frame, scaled by `max_delta = 0.02 m` per step. The env integrates an internal EE target, clips it to a workspace box, and solves IK for joint targets. The target is also kept within `MAX_LEAD = 0.03 m` (per axis) of the measured EE position, so an unreachable or blocked target does not wind up; when the arm lags (or is pressed on something), a full-scale action therefore moves the target by less than 0.02 m.
  - Actions are quantized to float32 inside `step` (the precision rollouts record), so replaying recorded actions, or stepping in-process with float64 actions, reproduces `evaluate()` bit-for-bit.
  - `a[3]` = absolute gripper command: `-1` fully open, `+1` fully closed (mapped linearly to the robot's gripper range).
  - Orientation is handled by the env: the gripper points down; wrist roll is fixed per robot. (Cube yaw is kept within ±15° so a fixed roll can grasp it.)
- **Observation** (`obs_mode="state"`, float32). `env.obs_dim` and `env.obs_slices: dict[str, slice]` describe it. Required keys, in this order:
  `joint_pos` (arm joints + gripper joint, robot-dependent length), `ee_pos` (3), `gripper_open` (1, in [0, 1]), `cube_pos` (3), `cube_to_ee` (3), `cube_yaw` (2: sin, cos), `cup_pos` (2, the cup's current position: it changes if the cup is pushed), `time_frac` (1). Positions are in the robot base frame, meters.
- **Reward.** Sparse: `1.0` on the step the episode succeeds, else `0.0`.
- **Success.** The cube has been *dropped* into the cup: its center is inside the cup's inner cylinder (horizontal distance to the cup's current axis `< r_inner − 0.005`) and below the rim, the cube touches neither finger, and the cup is upright (tilt < 20°), for 3 consecutive steps. Holding the cube inside the cup does not count. On success `terminated=True`. Time limit gives `truncated=True`.
- **Known metric limitation.** Success uses the cup's current position and final uprightness; it does not reject an earlier cup displacement or tip-and-recovery. The prelaunch replay reproduced 129/256 base successes, including three with final cup displacement greater than 1 cm (seeds 20017, 20047, 20122); seed 20047 moved the cup about 10 cm before completing a drop. The replayed ARS seed-4 policy had 254/256 successes and no such displaced-cup successes. These are scores under CupDrop-v1, not a demonstrated no-shoving benchmark. See [`tools/review_harness.py`](../tools/review_harness.py) and the [review report](maintenance/review/REPORT.md).
- **Contacts.** Every geom uses stiff contact parameters (`solref = [0.01, 1]`, `solimp = [0.99, 0.999, 0.001]`). Soft contacts still permit penetration: the prelaunch probes measured maximum cube-contact depths of 4.6 mm for the base and 5.7 mm for ARS at control-step boundaries. This is not a proof against tunnelling between sampled steps. Gripper strength is set per robot in `RobotSpec`: the SO-101 gripper servo is capped at 50% torque like LeRobot's follower config (about 20 N per jaw on the cube), and the PiPER finger gain is raised so it squeezes with about 2 N per finger instead of 0.2 N.
- **`info`** keys: `success: bool`, `grasped: bool` (cube held between the fingers and off the table), `near_miss: float` (minimum over the episode so far of the horizontal distance from the cube center to the cup axis, meters), `steps: int`, `ee_target: np.ndarray(3)`.
- **Failure labels.** `OutcomeTracker` labels a finished episode from observations and infos alone: `missed_grasp`, `slip`, `rim_hit` (cube lost below rim height right next to the cup), `off_target` (let go over or near the cup above the rim, but not inside), `still_held`.
- **Determinism.** `reset(seed)` samples cube x, y, yaw from `np.random.default_rng(seed)`. Same seed → same initial state, bit-for-bit, on the same machine.
- **Save/restore.** `env.get_state() -> dict` and `env.set_state(state)` capture everything needed to resume exactly (MuJoCo qpos/qvel/act/ctrl/time, EE target, step counter, success streak). After `set_state`, `env.observe()` returns the same observation the original episode had at that step, and both open-loop and closed-loop continuations match bit-for-bit. Used by pass@k, GRPO-style groups and oracles.
- **Rendering.** `env.render(camera="front" | "top" | "wrist") -> np.ndarray[H, W, 3] uint8`. Offscreen; lazily creates the renderer so non-rendering runs pay nothing.
- **Speed budget.** One 150-step episode must take < 0.25 s of CPU including IK and Python overhead on an M-series Mac.

## 3. Initial-state sets and seeds (`lastmile/common/eval.py`)

```python
INIT_SET_VERSION = "v1"
INIT_SETS = {
    "search":   list(range(10_000, 10_064)),   # 64 — search, training selection, hyperparameters
    "eval":     list(range(20_000, 20_256)),   # 256 — held-out score; never used for selection
    "eval_ext": list(range(30_000, 31_024)),   # 1024 — final claims
    "stress":   list(range(40_000, 40_256)),   # 256 — used with variant="hard"/"bias"
}
def policy_seed(init_set: str, index: int, salt: int = 0) -> int  # common random numbers for policy sampling
```

Common random numbers: episode `i` of set `S` always uses env seed `INIT_SETS[S][i]` and policy seed `policy_seed(S, i, salt)`, so two methods evaluated on the same set are paired.

Statistics helpers (same file): `wilson(k, n, z=1.96) -> (lo, hi)`; `mcnemar_exact(a: bool[], b: bool[]) -> p` (the significance test for paired comparisons); `bootstrap_diff(a, b, n=10_000, seed=0) -> (lo, hi)` (an effect-size interval, not a test: unreliable with fewer than ~10 discordant pairs); `min_successes_for_lower_bound(target) -> n` (all-success streak needed, e.g. 0.95 → 73, 0.99 → 381).

## 4. Policies (`lastmile/common/policy.py`)

```python
class BatchPolicy(Protocol):
    def reset(self, seeds: Sequence[int]) -> None: ...          # one seed per env in the batch (policy RNG)
    def act(self, obs: np.ndarray) -> np.ndarray: ...            # obs [B, obs_dim] -> actions [B, 4]
```

- A policy must be constructible inside a worker process from a **picklable factory**: `make_policy: Callable[[], BatchPolicy]` (e.g. `functools.partial(FlowChunkPolicy.load, path, noise="fixed", ticket=w)`).
- Chunked policies handle chunk execution internally (predict H=8, execute 4, replan) and expose `last_chunk` / `last_noise` for logging.
- `KnobController(knobs, robot)` implements `BatchPolicy`.

## 5. Rollouts (`lastmile/common/rollout.py`)

```python
res = evaluate(make_policy, init_set="eval", robot="so101", variant="v1",
               n_workers=8, ledger=None, category=None, salt=0, record_trajectories=False)
res.successes        # bool[n]
res.steps_to_success # int[n], -1 on failure
res.near_miss        # float[n]
res.env_steps        # int, total simulated steps (for the ledger)
res.sr, res.ci       # success rate and Wilson 95% interval
res.trajectories     # list of dicts (obs, actions, rewards, final_obs, infos, success, seed, policy extras) if requested
```

- Seeds are split into shards across a `multiprocessing` pool (`spawn` context — macOS default). Each worker builds its env(s) once and steps its shard **in lockstep** so the policy is called with a batch.
- `n_workers=1` runs in-process (for debugging and tests).
- Workers set `torch.set_num_threads(1)` once torch is imported (torch-free policies never import it).
- If a `ledger` is passed, `res.env_steps` is charged to `ledger.robot_steps[category]`. `category` must be `"search"`, `"train"` or `"eval"` (anything else raises before any rollout runs). With `category=None` it follows the init set: `"search"` for the `search` set, `"eval"` for every other set, so search episodes are in the improvement budget by default.
- `n` (keyword) keeps the first `n` seeds of the set and must be in `1..len(set)`.
- The worker pool is persistent and only grows: pick any `n_workers` per call; a smaller request reuses the existing pool with fewer shards in flight.
- `rollout_seeds(make_policy, seeds, policy_seeds, ...)` is the lower-level call used by `evaluate` and by chapters that need custom seed lists.

## 6. Base policy: FlowChunk-S (`lastmile/common/policy_flow.py`)

- Flow-matching policy over action chunks. MLP 4×256, SiLU, time embedding. Chunk `H = 8`, action dim 4, so the **noise vector is 32-dim**. Execute 4 actions, then replan.
- Training: rectified flow. `x0 ~ N(0, I)` (noise), `x1` = normalized action chunk, `x_t = (1 − t) x0 + t x1`, target velocity `x1 − x0`, `t ~ U(0, 1)`. Observations are normalized with dataset mean/std stored in the checkpoint.
- Sampling: 10 Euler steps from `x0 = noise`. **Deterministic given the noise.** `model.sample(obs[B, D], noise[B, 32]) -> chunk[B, 8, 4]`, clipped to [-1, 1].
- `FlowChunkPolicy(model, noise="gaussian" | "fixed" | callable, ticket: np.ndarray | None)`: `"gaussian"` draws fresh noise per chunk from the per-env policy RNG; `"fixed"` uses `ticket` for every chunk (Golden Ticket, ch03); a callable `noise(obs[B, D], rngs: list[np.random.Generator]) -> [B, 32]` supports steering (ch07) and best-of-N (ch12). `rngs` holds one generator per env, seeded by `reset()`; draw row `i` from `rngs[i]` only, so an episode's noise does not depend on its batch and results stay identical across `n_workers`.
- `FlowChunk.save(path)` / `FlowChunk.load(path)` store weights, normalizer, config and a config hash. Inference runs in numpy or torch-CPU, fast enough for 256-episode evals in under a minute on 8 workers.
- The shared base checkpoint is `checkpoints/base_v1_so101.pt` with `checkpoints/base_v1_so101.json` (dataset hash, demo count, S_eval success + CI, pass@8).
- **Calibration target:** 45–55% success on `eval` (n=256), with pass@8 ≥ 90% (checked with `get_state`/`set_state` resets), so selection and steering methods have headroom.

## 7. Cost ledger and results (`lastmile/common/ledger.py`)

```python
with Ledger(chapter="ch01", method="ars", robot="so101", variant="v1") as L:
    L.robot_steps["search"] += n      # or pass ledger=L to evaluate(..., category="search")
    L.human_minutes["demos"] += 0.0
    L.set_base(id="knobs_default", successes=base_res)      # an EvalResult (preferred) or a bool array
    L.set_final(successes=final_res)                         # must be the same init set, n and seeds as base
    L.extra["best_knobs"] = {...}
# on exit writes results/ch01/<method>_<robot>_<timestamp>.json
```

Results JSON (schema 1):

```json
{
  "schema": 1, "chapter": "ch01", "method": "ars", "robot": "so101", "variant": "v1", "setting": "sim",
  "base":  {"id": "knobs_default", "sr": 0.41, "ci": [0.35, 0.47], "k": 105, "n": 256, "init_set": "eval"},
  "final": {"sr": 0.97, "ci": [0.94, 0.98], "k": 248, "n": 256, "init_set": "eval", "init_set_version": "v1"},
  "budget": {
    "robot_minutes": {"search": 12.3, "train": 0.0, "eval": 0.0},
    "human_minutes": {"demos": 0.0, "labels": 0.0, "takeovers": 0.0, "resets": 0.0, "annotation": 0.0},
    "compute": {"cpu_hours": 0.05, "gpu_hours": 0.0, "wall_minutes": 3.1}
  },
  "extra": {}, "git_sha": "abc123", "timestamp": "2026-09-30T12:00:00", "config": {}
}
```

- The **improvement budget** counts every rollout except the final held-out evaluation. Evaluations of the base and final policy are charged to `eval`. `robot_steps` only has the keys `search`, `train` and `eval`; an unknown category raises instead of creating a bucket the leaderboard would not count.
- `set_base` / `set_final` read `init_set` and the seeds from the `EvalResult` and raise if base and final were not scored on the same episodes (the pairing McNemar's test relies on).
- `tools/leaderboard.py` reads every results JSON and regenerates the README leaderboard table and two plots (success vs log robot-minutes, success vs log human-minutes). Runs with a `variant` other than `v1` are shown as `so101 (hard)` and sorted after the v1 rows; rows whose `final.init_set_version` is not the current one are tagged.

## 8. Chapter conventions

- Top docstring: learning goal, the base RL idea, the 2025–26 papers mirrored, how to run.
- A `@dataclass Config` at the top; `python algos/NN_x.py --help` works via `lastmile.common.cli.parse(Config)`.
- `--quick` flag: finishes in < 2 minutes on a laptop (used by CI smoke tests). Full runs should finish in < 30 minutes on a laptop unless the docstring says otherwise.
- Writes: one results JSON per method run, plots to `media/chNN/`, and a short printed summary with Wilson intervals.
- `docs/chapters/chNN.md`: the idea in plain words, the math, the papers it mirrors (with links), the actual results from our runs (numbers + plots), what went wrong, and the hardware step.

## 9. Plot and media style

- `lastmile.common.plotting.setup()` sets a consistent matplotlib style. Success-rate plots always show Wilson intervals (shaded) and mark the held-out evaluation separately from search-time estimates.
- GIFs: `plotting.save_gif(frames, path, fps=10)`, 256 px, < 3 MB.
