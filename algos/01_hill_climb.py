"""Chapter 1: hill climbing with a success counter (greedy, finite differences, ARS).

Learning goal
    Improve a controller when the only thing you can do is run it and count successes. No gradients of
    the policy, no critic, no reward shaping: a vector of knobs theta and a number R(theta) in [0, 1].

Base RL idea
    Perturb, evaluate, step. R(theta) is a staircase (a success count), so its true gradient is zero
    almost everywhere. Smooth it with Gaussian noise, J(theta) = E_eps[R(theta + sigma*eps)], and the
    gradient of J can be estimated from rollouts alone, with antithetic (mirrored) pairs:

        grad J ~= 1/(2*N*sigma) * sum_i [R(theta + sigma*eps_i) - R(theta - sigma*eps_i)] * eps_i

    (The code drops the 1/2, so its g is twice this estimate; the constant is folded into fd_step.)

    Three methods, one file (--method):
      greedy  perturb the incumbent, keep the candidate only if its score is strictly better.
      fd      g = 1/(N*sigma) * sum_i [R+ - R-] * eps_i, then theta <- theta + step * g.
      ars     Augmented Random Search: keep only the top-b directions (ranked by max(R+, R-)) and divide
              the step by the standard deviation of the 2b rewards that were used:
              theta <- theta + step/(b*sigma_R) * sum_top-b [R+ - R-] * eps.
              (ARS's third trick, observation normalization, does not apply: the knobs are not a
              function of the observation.)

    Two details make this work with few rollouts. (1) Normalized knob space: every knob is mapped to
    [0, 1] by its KNOB_SPACE bounds, so one sigma means "this fraction of the range" for a height in
    meters and for a wait in steps. (2) Common random numbers: all candidates of an iteration run on the
    same minibatch of search start states, so R+ - R- compares knobs, not cube positions.

Papers mirrored (ids from data/papers.csv)
    C000 Kohl & Stone 2004 (finite-difference hill climbing of an Aibo gait on hardware); C012 ARS
    (Mania et al. 2018); and the modern "stage 2" use of the same loop on a policy that already mostly
    works: C186 TD-ES (ES after PPO plateaus), C134 EvolSAC (ES on the true score after SAC), C155 ES at
    Scale (30 perturbed copies, scalar outcome reward).

Honesty
    Every knob choice is made on the `search` set (64 start states). The 256 `eval` start states are run
    twice only: once for the default knobs and once for the knobs each run selected. Every search rollout
    (candidates and the monitoring of the current iterate) is charged to the ledger, and so is the
    hyperparameter tuning that fixed fd_step and the number of iterations (see TUNING below).

Run
    uv run python algos/01_hill_climb.py --quick                  # smoke test, under 2 minutes; all files to runs/ch01_quick/
    uv run python algos/01_hill_climb.py                          # all three methods x 3 seeds (< 30 min)
    uv run python algos/01_hill_climb.py --method fd --fd-step 0.05 --tag _bigstep   # the failure case
    uv run python algos/01_hill_climb.py --method random          # baseline: a random candidate, no search
    uv run python algos/01_hill_climb.py --method ars --seeds 0   # one method, one seed
    uv run python algos/01_hill_climb.py --method slices          # 1-D landscape slices (diagnostic)
    uv run python algos/01_hill_climb.py --seeds 0 1 --iterations 15 --fd-step 0.05 --no-heldout  # tuning

    uv run python algos/01_hill_climb.py --method plot            # rebuild figures from results/ch01
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from lastmile.common.cli import parse
from lastmile.common.eval import INIT_SETS, EvalResult, format_rate, mcnemar_exact, policy_seed
from lastmile.common.ledger import DEFAULT_RESULTS_ROOT, Ledger
from lastmile.common.rollout import evaluate, rollout_seeds
from lastmile.envs.knob_controller import DEFAULT_KNOBS, KNOB_NAMES, KNOB_SPACE, KnobController, knob_vector

MEDIA = Path(__file__).resolve().parents[1] / "media" / "ch01"
QUICK_MEDIA = Path(__file__).resolve().parents[1] / "runs" / "ch01_quick"  # --quick never touches media/ or results/
QUICK_RESULTS = QUICK_MEDIA / "results"
METHODS = ("greedy", "fd", "ars")
LOW = np.array([k.low for k in KNOB_SPACE])
SPAN = np.array([k.high - k.low for k in KNOB_SPACE])
DIM = len(KNOB_SPACE)

# Hyperparameter tuning is robot time too (honesty rule 5). Before the reported runs, fd_step (0.05 ->
# 0.02) and the number of iterations (15 -> 10) were chosen from search-set curves of 15-iteration runs
# on seeds 0 and 1: greedy and ars at their defaults, fd at fd_step 0.05, 0.01 and 0.02 (the --no-heldout
# commands in docs/chapters/ch01.md reproduce them). ars_step, top_b, sigma, n_dirs and batch were set up
# front and never changed. The measured search cost of each method's tuning runs is charged to every
# reported run of that method, so a results file shows what it cost to reach this configuration.
# The reported runs use seeds 2, 3 and 4, which the tuning never saw.
TUNING_ROBOT_MIN = {  # search robot-min of those runs (deterministic, so replaying them gives the same cost)
    "greedy": 454.5 + 576.8,  # seeds 0, 1
    "fd": 683.5 + 732.2 + 497.8 + 459.7 + 452.9 + 441.1,  # fd_step 0.05, 0.01, 0.02 x seeds 0, 1
    "ars": 438.1 + 433.4,
}


@dataclass
class Config:
    method: Literal["all", "greedy", "fd", "ars", "random", "slices", "plot"] = "all"
    robot: str = "so101"
    seeds: tuple[int, ...] = field(default=(2, 3, 4), metadata={"help": "search RNG seeds (0, 1 were used for tuning)"})
    iterations: int = field(default=10, metadata={"help": "each costs 2*n_dirs candidates + 1 monitor eval"})
    n_dirs: int = field(default=6, metadata={"help": "N: antithetic direction pairs per iteration"})
    top_b: int = field(default=3, metadata={"help": "ARS: directions kept per iteration"})
    batch: int = field(default=16, metadata={"help": "search start states every candidate is run on"})
    sigma: float = field(default=0.1, metadata={"help": "perturbation std, as a fraction of each knob's range"})
    fd_step: float = field(default=0.02, metadata={"help": "fd: theta += fd_step * g (0.05 falls off cliffs)"})
    ars_step: float = field(default=0.05, metadata={"help": "ars: theta += ars_step/(b*sigma_R) * sum"})
    tag: str = field(default="", metadata={"help": "suffix for the method name (e.g. a failure run)"})
    n_eval: int = field(default=256, metadata={"help": "held-out eval episodes (256 = the whole set)"})
    heldout: bool = field(default=True, metadata={"help": "eval the pick and write results (off: search-only tuning)"})
    n_workers: int = 6
    gif: bool = True
    quick: bool = False
    QUICK: ClassVar[dict] = {"seeds": (0,), "iterations": 2, "n_dirs": 2, "top_b": 1, "batch": 8, "n_eval": 32}


def to_knobs(x: np.ndarray) -> dict[str, float]:
    """Normalized vector in [0, 1]^10 -> named knobs in physical units."""
    return dict(zip(KNOB_NAMES, LOW + np.clip(x, 0.0, 1.0) * SPAN))


def to_unit(knobs: dict[str, float]) -> np.ndarray:
    return (knob_vector(knobs) - LOW) / SPAN


class Scorer:
    """R(theta): fraction of successes on a list of search start states. Charges every step to the ledger."""

    def __init__(self, cfg: Config, ledger: Ledger):
        self.cfg, self.ledger, self.episodes = cfg, ledger, 0

    def __call__(self, x: np.ndarray, idx: np.ndarray) -> float:
        # Episode i of the search set always uses the same env seed and policy seed (common random numbers).
        res = rollout_seeds(
            partial(KnobController, to_knobs(x), self.cfg.robot),
            [INIT_SETS["search"][i] for i in idx], [policy_seed("search", int(i)) for i in idx],
            robot=self.cfg.robot, n_workers=self.cfg.n_workers, ledger=self.ledger, category="search",
        )
        self.episodes += res.n
        return res.sr

    def monitor(self, x: np.ndarray) -> EvalResult:
        """Score the current iterate on all 64 search states. It costs robot time, so it is charged too."""
        res = evaluate(partial(KnobController, to_knobs(x), self.cfg.robot), "search", robot=self.cfg.robot,
                       n_workers=self.cfg.n_workers, ledger=self.ledger, category="search")
        self.episodes += res.n
        return res


# --------------------------------------------------------------------------------------
# The three update rules. Each spends exactly 2 * n_dirs candidate evaluations on one minibatch.
# --------------------------------------------------------------------------------------


def greedy_step(x, R, rng, cfg):
    """Propose, compare on the same start states, keep only strict improvements."""
    best = R(x)  # the incumbent has to be re-scored: this minibatch is new
    for _ in range(2 * cfg.n_dirs - 1):
        cand = np.clip(x + cfg.sigma * rng.standard_normal(DIM), 0.0, 1.0)
        r = R(cand)
        if r > best:  # a tie is not evidence, so the incumbent stays
            x, best = cand, r
    return x, {"minibatch_sr": best}


def antithetic(x, R, rng, cfg):
    """Evaluate N mirrored pairs theta +- sigma*eps. Candidates are clipped to the knob bounds."""
    eps = rng.standard_normal((cfg.n_dirs, DIM))
    r_plus = np.array([R(np.clip(x + cfg.sigma * e, 0.0, 1.0)) for e in eps])
    r_minus = np.array([R(np.clip(x - cfg.sigma * e, 0.0, 1.0)) for e in eps])
    return eps, r_plus, r_minus


def fd_step(x, R, rng, cfg):
    """Antithetic finite differences on the Gaussian-smoothed success rate."""
    eps, r_plus, r_minus = antithetic(x, R, rng, cfg)
    # Twice the unbiased estimate of grad J (that one divides by 2*sigma); the factor 2 lives in fd_step.
    g = ((r_plus - r_minus)[:, None] * eps).mean(axis=0) / cfg.sigma
    info = {"minibatch_sr": float(np.concatenate([r_plus, r_minus]).mean()), "grad_norm": float(np.linalg.norm(g))}
    return np.clip(x + cfg.fd_step * g, 0.0, 1.0), info


def ars_step(x, R, rng, cfg):
    """ARS (V1-t): use the top-b directions only, and scale the step by the std of their rewards."""
    eps, r_plus, r_minus = antithetic(x, R, rng, cfg)
    top = np.argsort(-np.maximum(r_plus, r_minus), kind="stable")[: cfg.top_b]
    sigma_r = float(np.concatenate([r_plus[top], r_minus[top]]).std())
    info = {"minibatch_sr": float(np.concatenate([r_plus, r_minus]).mean()), "sigma_r": sigma_r}
    if sigma_r < 1e-8:  # every used rollout scored the same: no direction is better, do not divide by zero
        return x, info
    delta = ((r_plus[top] - r_minus[top])[:, None] * eps[top]).sum(axis=0)
    return np.clip(x + cfg.ars_step / (cfg.top_b * sigma_r) * delta, 0.0, 1.0), info


STEP = {"greedy": greedy_step, "fd": fd_step, "ars": ars_step}


def run_method(method: str, seed: int, cfg: Config, base: EvalResult | None) -> dict:
    """One search run from DEFAULT_KNOBS, then one held-out evaluation of the iterate it selected."""
    name = f"{method}{cfg.tag}_s{seed}"
    rng = np.random.default_rng(seed)
    L = Ledger(chapter="ch01", method=name, robot=cfg.robot, config=cfg,
               results_root=QUICK_RESULTS if cfg.quick else DEFAULT_RESULTS_ROOT)
    with L if cfg.heldout else nullcontext(L):  # a --no-heldout (tuning) run writes no results file
        scorer = Scorer(cfg, L)
        x = to_unit(DEFAULT_KNOBS)
        mon = scorer.monitor(x)
        history = [{"iter": 0, "episodes": scorer.episodes, "robot_min": L.robot_minutes["search"],
                    "k": mon.k, "n": mon.n, "x": x.tolist()}]
        for it in range(1, cfg.iterations + 1):
            # A fresh minibatch each iteration, shared by every candidate of that iteration.
            idx = np.sort(rng.choice(len(INIT_SETS["search"]), size=cfg.batch, replace=False))
            x, info = STEP[method](x, lambda cand, idx=idx: scorer(cand, idx), rng, cfg)
            mon = scorer.monitor(x)
            history.append({"iter": it, "episodes": scorer.episodes, "robot_min": L.robot_minutes["search"],
                            "k": mon.k, "n": mon.n, "x": x.tolist(), **info})
            print(f"  {name} it {it:2d}  search {format_rate(mon.k, mon.n)}  "
                  f"episodes {scorer.episodes}  robot-min {L.robot_minutes['search']:.1f}", flush=True)
        # Selection uses search scores only: the iterate with the best 64-state score (latest on ties).
        pick = max(range(len(history)), key=lambda i: (history[i]["k"], i))
        best_x, run_min = np.array(history[pick]["x"]), L.robot_minutes["search"]
        print(f"{name}: picked it {pick} (search {format_rate(history[pick]['k'], history[pick]['n'])}) | "
              f"search {run_min:.1f} robot-min, {scorer.episodes} episodes", flush=True)
        if not cfg.heldout:
            return {"label": f"{method}{cfg.tag}", "seed": seed, "history": history, "robot_min": run_min}
        L.robot_steps["eval"] += base.env_steps  # the shared base evaluation is part of this run's eval cost
        L.set_base(id="knobs_default", successes=base)
        tuning = 0.0 if cfg.quick else TUNING_ROBOT_MIN[method]
        L.robot_steps["search"] += round(tuning * 600)  # 10 Hz: 600 steps per robot-minute
        final = evaluate(partial(KnobController, to_knobs(best_x), cfg.robot), "eval", robot=cfg.robot,
                         n=cfg.n_eval, n_workers=cfg.n_workers, ledger=L, category="eval")
        L.set_final(successes=final)
        p = mcnemar_exact(base.successes, final.successes)
        L.extra.update(seed=seed, history=history, picked_iter=pick, best_knobs=to_knobs(best_x),
                       search_at_pick={"k": history[pick]["k"], "n": history[pick]["n"]},
                       last_iter_search={"k": history[-1]["k"], "n": history[-1]["n"]},
                       search_episodes=scorer.episodes, mcnemar_p_vs_base=p,
                       search_robot_min_this_run=run_min, tuning_robot_min_charged=tuning)
    print(f"{name}: -> eval {final.summary} | base {base.summary} | McNemar p={p:.2g} | "
          f"search budget {run_min:.1f} + tuning {tuning:.1f} robot-min", flush=True)
    return {"method": method, "label": f"{method}{cfg.tag}", "seed": seed, "history": history, "final": final,
            "final_k": final.k, "final_n": final.n, "best_x": best_x, "robot_min": run_min}


# --------------------------------------------------------------------------------------
# Plots, the GIF and the landscape diagnostic
# --------------------------------------------------------------------------------------


def plot_curves(runs: list[dict], base_sr: float, path: Path) -> None:
    """Search-set score of the current iterate vs robot-minutes, one panel per method, eval marked apart."""
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    plotting.setup()
    labels = list(dict.fromkeys(r["label"] for r in runs))  # e.g. greedy, fd, ars (or fd, fd_bigstep)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.4 * len(labels) + 1.8, 3.9), sharex=True, sharey=True,
                             squeeze=False)
    x_max = max(r["history"][-1]["robot_min"] for r in runs)
    for ax, label in zip(axes[0], labels):
        for j, r in enumerate(r for r in runs if r["label"] == label):
            h = r["history"]
            plotting.success_curve(ax, [p["robot_min"] for p in h], [p["k"] for p in h], h[0]["n"],
                                   label=f"seed {r['seed']}: search (n={h[0]['n']})", color=plotting.PALETTE[j])
            plotting.heldout_point(ax, x_max * (1.06 + 0.04 * j), r["final_k"], r["final_n"],
                                   label=f"held-out eval (n={r['final_n']}), selected iterate" if j == 0 else None,
                                   color=plotting.PALETTE[j])
        ax.axhline(base_sr, color=plotting.PALETTE[5], linestyle=":", linewidth=1.2, label="default knobs (eval)")
        ax.set_title(label)
        ax.set_xlabel("search robot-minutes")
    axes[0][0].set_ylabel("success rate")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    plotting.save_fig(fig, path)


def plot_knob_paths(runs: list[dict], path: Path) -> None:
    """Knob trajectories of the first seed of each method: which knobs each method actually moved."""
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    plotting.setup()
    labels = list(dict.fromkeys(r["label"] for r in runs))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.4 * len(labels) + 1.8, 3.9), sharey=True, squeeze=False)
    for ax, label in zip(axes[0], labels):
        r = next(r for r in runs if r["label"] == label)
        xs = np.array([p["x"] for p in r["history"]])
        for d, knob in enumerate(KNOB_NAMES):
            ax.plot(xs[:, d], label=knob, linewidth=1.5, color=plt.cm.tab10(d))
        ax.set_title(f"{label}, seed {r['seed']}")
        ax.set_xlabel("iteration")
        ax.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel("knob value (0 = low, 1 = high bound)")
    axes[0][-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plotting.save_fig(fig, path)


def replot_from_results(cfg: Config) -> None:
    """Rebuild the chapter figures from the results JSONs of full runs (no rollouts)."""
    import json

    newest, base_sr = {}, None  # a rerun writes a new timestamped file: keep only the newest per run name
    for path in sorted((Path(__file__).resolve().parents[1] / "results" / "ch01").glob("*.json")):
        data = json.loads(path.read_text())
        if data["config"].get("quick") or "history" not in data["extra"] or data["robot"] != cfg.robot:
            continue
        old = newest.get(data["method"])
        if old is not None:
            print(f"warning: two results files for {data['method']}: {old[0].name}, {path.name}; using the newer")
        if old is None or data["timestamp"] >= old[1]["timestamp"]:
            newest[data["method"]] = (path, data)
    runs = []
    for path, data in sorted(newest.values(), key=lambda pd: pd[1]["timestamp"]):
        base_sr = data["base"]["sr"]  # identical in every file: the same default knobs on the same 256 states
        runs.append({"label": data["method"].rsplit("_s", 1)[0], "seed": data["extra"]["seed"],
                     "history": data["extra"]["history"], "final_k": data["final"]["k"],
                     "final_n": data["final"]["n"]})
    runs.sort(key=lambda r: r["seed"])
    pick = lambda *labels: [r for lab in labels for r in runs if r["label"] == lab]
    plot_curves(pick(*METHODS), base_sr, MEDIA / "learning_curves.png")
    plot_knob_paths(pick(*METHODS), MEDIA / "knob_paths.png")
    plot_curves(pick("fd", "fd_bigstep"), base_sr, MEDIA / "fd_step_failure.png")
    print(f"rebuilt figures in {MEDIA} from {len(runs)} results files")


def render_episode(knobs: dict[str, float], robot: str, seed: int, every: int) -> tuple[list, bool]:
    """Replay one episode in-process with the front camera (same steps as the evaluated episode)."""
    from lastmile.envs.cupdrop import CupDropEnv

    env, policy = CupDropEnv(robot=robot), KnobController(knobs, robot)
    obs, _ = env.reset(seed=seed)
    policy.reset([seed])
    frames, info = [env.render("front")], {"success": False}
    for step in range(1, env.max_steps + 1):
        obs, _, terminated, truncated, info = env.step(policy.act(obs[None])[0])
        if step % every == 0:
            frames.append(env.render("front"))
        if terminated or truncated:
            break
    return frames, bool(info["success"])


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    """Write a short label in the top-left corner of a frame."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 8 + 6 * len(text), 16], fill=(255, 255, 255))
    draw.text((4, 3), text, fill=(0, 0, 0))
    return np.asarray(img)


def save_before_after(run: dict, base: EvalResult, cfg: Config) -> None:
    """Default knobs failing next to the searched knobs succeeding, on the same held-out start state."""
    from lastmile.common.plotting import save_gif

    flipped = np.flatnonzero(~base.successes & run["final"].successes)
    if len(flipped) == 0:
        print("no eval episode where the default fails and the searched knobs succeed; skipping the GIF")
        return
    seed = int(base.seeds[flipped[0]])
    before, ok_before = render_episode(DEFAULT_KNOBS, cfg.robot, seed, every=3)
    after, ok_after = render_episode(to_knobs(run["best_x"]), cfg.robot, seed, every=3)
    assert not ok_before and ok_after, "replay disagrees with the evaluation"
    n = max(len(before), len(after)) + 3  # pad the shorter clip with its last frame
    pad = lambda f: f + [f[-1]] * (n - len(f))
    frames = [np.concatenate([caption(a, "default knobs: fail"), caption(b, f"{run['label']} knobs: success")],
                             axis=1) for a, b in zip(pad(before), pad(after))]
    path = save_gif(frames, (QUICK_MEDIA if cfg.quick else MEDIA) / "before_after.gif", fps=4, max_size=512)
    print(f"wrote {path} (eval seed {seed}; left: default knobs fail, right: {run['label']} seed "
          f"{run['seed']} knobs succeed; {path.stat().st_size / 1e6:.2f} MB)")


def landscape_slices(cfg: Config, points: int = 9) -> None:
    """Success on the 64 search states along each knob, the others held at DEFAULT_KNOBS.

    A diagnostic for the chapter doc, not part of any method: no method reads these numbers, so the
    rollouts are not in a results file. The cost is printed so the doc can state it.
    """
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    plotting.setup()
    x0, grid, steps = to_unit(DEFAULT_KNOBS), np.linspace(0.0, 1.0, points), 0
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.0), sharey=True, gridspec_kw={"hspace": 0.45})
    for d, (ax, knob) in enumerate(zip(axes.ravel(), KNOB_SPACE)):
        ks = []
        for v in grid:
            x = x0.copy()
            x[d] = v
            res = evaluate(partial(KnobController, to_knobs(x), cfg.robot), "search", robot=cfg.robot,
                           n_workers=cfg.n_workers)
            ks.append(res.k)
            steps += res.env_steps
        plotting.success_curve(ax, LOW[d] + grid * SPAN[d], ks, 64, color=plotting.PALETTE[0])
        ax.axvline(DEFAULT_KNOBS[knob.name], color=plotting.PALETTE[1], linestyle=":", linewidth=1.2)
        ax.set_title(f"{knob.name} ({knob.unit})", fontsize=10)
        print(f"  {knob.name:12s} low -> high bound: " + ", ".join(format_rate(k, 64) for k in ks), flush=True)
    fig.suptitle("Search-set success along each knob, others at DEFAULT_KNOBS (dotted: default value)", fontsize=11)
    plotting.save_fig(fig, MEDIA / "landscape_slices.png")
    print(f"landscape slices: {10 * points * 64} episodes, {steps / 600:.1f} robot-minutes (diagnostic)")


def random_baseline(seed: int, cfg: Config, base: EvalResult) -> EvalResult:
    """Honesty rule 4: one random candidate from the same proposal distribution, with no selection at all.

    If this scored as well as the searched knobs, the search would not have done the work.
    """
    x = np.clip(to_unit(DEFAULT_KNOBS) + cfg.sigma * np.random.default_rng(seed).standard_normal(DIM), 0.0, 1.0)
    with Ledger(chapter="ch01", method=f"random{cfg.tag}_s{seed}", robot=cfg.robot, config=cfg,
                results_root=QUICK_RESULTS if cfg.quick else DEFAULT_RESULTS_ROOT) as L:
        L.robot_steps["eval"] += base.env_steps
        L.set_base(id="knobs_default", successes=base)
        final = evaluate(partial(KnobController, to_knobs(x), cfg.robot), "eval", robot=cfg.robot,
                         n=cfg.n_eval, n_workers=cfg.n_workers, ledger=L, category="eval")
        L.set_final(successes=final)
        L.extra.update(seed=seed, knobs=to_knobs(x), mcnemar_p_vs_base=mcnemar_exact(base.successes, final.successes))
    print(f"random candidate seed {seed} (sigma {cfg.sigma}, no search): eval {final.summary}")
    return final


def main(cfg: Config) -> None:
    if cfg.method == "slices":
        return landscape_slices(cfg)
    if cfg.method == "plot":
        return replot_from_results(cfg)
    methods = METHODS if cfg.method == "all" else (cfg.method,)
    if not cfg.heldout:  # tuning: search-set curves and their cost only, no eval episode is run
        runs = [run_method(m, s, cfg, None) for m in methods for s in cfg.seeds]
        for m in methods:
            print(f"{m}{cfg.tag}: search robot-min per seed " + ", ".join(
                f"s{r['seed']} {r['robot_min']:.1f}" for r in runs if r["label"] == f"{m}{cfg.tag}")
                  + f"; total {sum(r['robot_min'] for r in runs if r['label'] == f'{m}{cfg.tag}'):.1f}")
        return
    make_default = partial(KnobController, DEFAULT_KNOBS, cfg.robot)
    base = evaluate(make_default, "eval", robot=cfg.robot, n=cfg.n_eval, n_workers=cfg.n_workers)
    print(f"default knobs on eval: {base.summary}")
    if cfg.method == "random":
        finals = [random_baseline(s, cfg, base) for s in cfg.seeds]
        print(f"random candidates pooled: {format_rate(sum(f.k for f in finals), sum(f.n for f in finals))}")
        return
    runs = [run_method(m, s, cfg, base) for m in methods for s in cfg.seeds]

    print(f"\n{'method':8s} {'seed':>4s}  {'held-out eval [Wilson 95%]':32s} {'search robot-min':>16s}")
    for m in methods:
        mine = [r for r in runs if r["method"] == m]
        for r in mine:
            print(f"{m:8s} {r['seed']:4d}  {r['final'].summary:32s} {r['robot_min']:16.1f}")
        k, n = sum(r["final"].k for r in mine), sum(r["final"].n for r in mine)
        # Pooled over seeds for a one-line summary. The seeds share their 256 start states, so this
        # interval is narrower than it should be; the per-seed rows are the ones to quote.
        print(f"{m:8s}  all  {format_rate(k, n):32s} {np.mean([r['robot_min'] for r in mine]):16.1f}  (pooled; mean cost)")
    print(f"{'default':8s}    -  {base.summary:32s} {0.0:16.1f}")

    out = QUICK_MEDIA if cfg.quick else MEDIA
    plot_curves(runs, base.sr, out / f"learning_curves{cfg.tag}.png")
    plot_knob_paths(runs, out / f"knob_paths{cfg.tag}.png")
    if cfg.gif:
        from lastmile.common.plotting import optional_media

        with optional_media("the before/after GIF"):
            save_before_after(max(runs, key=lambda r: r["final"].k), base, cfg)


if __name__ == "__main__":
    main(parse(Config))
