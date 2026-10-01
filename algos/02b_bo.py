"""Chapter 2b: Bayesian optimization of four controller knobs with a Gaussian process and logEI.

Learning goal
    When every trial costs robot time, spend computer time instead: fit a model of "knobs -> success" to
    every trial so far and let it choose the next one. Learn how to feed a noisy 0/1 success signal to a
    Gaussian process, and when this beats the sample-and-reweight family of algos/02_cem_cmaes_pi2.py.

The base idea
    Observe  y_t = k_t / n, the success rate of knobs x_t on a batch of n search episodes
             (binomial noise: Var[y_t] ~ p(1 - p) / n, which we pass to the GP as known noise)
    Model    f ~ GP(mu, k_Matern52) fitted to (x_1, y_1) ... (x_t, y_t) by maximizing the marginal likelihood
    Choose   x_{t+1} = argmax_x log EI(x),  EI(x) = E[max(f(x) - f*, 0)],  f* = best posterior mean so far
    Start    with the default knobs plus d + 4 scrambled Sobol points (the defaults recommended by C445)
    Using the best posterior mean as f* (not the best noisy observation) keeps one lucky batch from
    looking like a success the model then refuses to explore past. The incumbent (the recommendation) is
    the observed point with the highest posterior mean; each new incumbent is scored on the full 64-seed
    search set (charged to the search budget) and the best-scoring one is returned. `--acq sobol` replaces
    the GP with more Sobol points and picks the best observed batch score: the random-search baseline.

    The search is over 4 of the 10 knobs (grasp_dx, grasp_dy, grasp_dz, drop_dx), picked from the
    KnobController docstring, which says how the defaults are off. That prior knowledge is free here and is
    not counted in any budget; the review (C445) notes most hardware BO tunes fewer than 10 parameters.

Papers mirrored (ids from data/papers.csv)
    C445 A Decade of Bayesian Optimization for Controller Tuning and Robot Learning (2026): d + 4 initial
         points, logEI, and Sobol as the baseline to beat.
    C057 BOpt-GMM (2024): BO on sparse binary success to improve a policy from few episodes.
    C130 Zero-order optimization primer (2025), for the contrast with algos/02_cem_cmaes_pi2.py.

Run
    uv run python algos/02b_bo.py --quick         # smoke test; results to runs/ch02_quick/
    uv run python algos/02b_bo.py                 # BO and Sobol x 3 seeds, then the comparison plots
    uv run python algos/02b_bo.py --acq logei --batch 2 --n-evals 30 --audit-n 10 --audit-every 30 \
        --tag=-hw --no-media                      # a hardware-sized budget: about 80 episodes per seed
Run algos/02_cem_cmaes_pi2.py first (with and without --tag=-4k) if you want CEM, CMA-ES, PI2 and random
search on the comparison plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from lastmile.common.cli import parse
from lastmile.common.eval import INIT_SETS, format_rate, policy_seed
from lastmile.common.ledger import DEFAULT_RESULTS_ROOT, Ledger, load_results
from lastmile.common.rollout import evaluate, rollout_seeds
from lastmile.envs.knob_controller import DEFAULT_KNOBS, KNOB_SPACE, TUNED_KNOBS, KnobController

MEDIA = Path(__file__).resolve().parents[1] / "media" / "ch02"


@dataclass
class Config:
    robot: str = "so101"
    acq: Literal["all", "logei", "sobol"] = "all"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    knobs: list[str] = field(default_factory=lambda: ["grasp_dx", "grasp_dy", "grasp_dz", "drop_dx"])
    n_evals: int = 36  # total trials, including the default and the d + 4 Sobol points
    batch: int = 16  # search episodes per trial
    audit_n: int = 64  # search episodes used to score each new incumbent
    audit_every: int = 4  # check for a new incumbent every this many trials
    n_workers: int = 6
    heldout: bool = True
    eval_n: int = 256
    media: bool = True
    tag: str = ""  # appended to the method name, e.g. "-hw" for a hardware-sized budget (left off the plots)
    results_root: str = str(DEFAULT_RESULTS_ROOT)
    quick: bool = False
    QUICK: ClassVar[dict] = {"seeds": [0], "n_evals": 10, "batch": 8, "audit_n": 16, "eval_n": 32,
                             "results_root": str(MEDIA.parents[1] / "runs" / "ch02_quick")}  # not results/


def to_knobs(x: np.ndarray, cfg: Config) -> dict[str, float]:
    """A point of [0, 1]^d -> a full knob dict (the other knobs keep their defaults)."""
    space = {k.name: k for k in KNOB_SPACE}
    vals = {n: space[n].low + (space[n].high - space[n].low) * float(np.clip(v, 0, 1))
            for n, v in zip(cfg.knobs, x)}
    return {**DEFAULT_KNOBS, **vals}


def to_unit(knobs: dict[str, float], names: list[str]) -> np.ndarray:
    space = {k.name: k for k in KNOB_SPACE}
    return np.array([(knobs[n] - space[n].low) / (space[n].high - space[n].low) for n in names])


def fit_gp(X: np.ndarray, k: np.ndarray, n: np.ndarray):
    """GP on success rates with known binomial noise.

    The variance uses (k + 1) / (n + 2), so a 0/n or n/n batch does not claim to be noise-free.
    """
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    p_smooth = (k + 1) / (n + 2)
    tx = torch.tensor(X, dtype=torch.double)
    ty = torch.tensor(k / n, dtype=torch.double).unsqueeze(-1)
    tvar = torch.tensor(p_smooth * (1 - p_smooth) / n, dtype=torch.double).unsqueeze(-1)
    gp = SingleTaskGP(tx, ty, train_Yvar=tvar, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


def posterior_mean(gp, X: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return gp.posterior(torch.tensor(X, dtype=torch.double)).mean.squeeze(-1).numpy()


def next_point(gp, best_f: float, d: int, seed: int) -> np.ndarray:
    """Maximize log expected improvement over the unit box (multi-start L-BFGS from random raw samples)."""
    import torch
    from botorch.acquisition import LogExpectedImprovement
    from botorch.optim import optimize_acqf

    torch.manual_seed(seed)
    acq = LogExpectedImprovement(gp, best_f=best_f)
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()
    x, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=8, raw_samples=256)
    return x.squeeze(0).numpy()


def run_bo(acq: str, seed: int, cfg: Config, L: Ledger) -> dict:
    """Trials -> model -> next trial. Returns the trial log, the audit history and the chosen knobs."""
    from torch.quasirandom import SobolEngine

    d, n_search = len(cfg.knobs), len(INIT_SETS["search"])
    sobol = SobolEngine(d, scramble=True, seed=seed)
    X, K = [to_unit(DEFAULT_KNOBS, cfg.knobs)], []  # trial 0 is the controller we start from
    n_init = 1 + d + 4
    rollouts, hist, audited = 0, [], {}

    def trial(x: np.ndarray) -> int:
        nonlocal rollouts
        start = (len(K) * cfg.batch) % n_search  # rotate through the search set
        idx = [(start + j) % n_search for j in range(cfg.batch)]
        res = rollout_seeds(partial(KnobController, to_knobs(x, cfg), cfg.robot),
                            [INIT_SETS["search"][i] for i in idx], [policy_seed("search", i) for i in idx],
                            robot=cfg.robot, n_workers=cfg.n_workers, ledger=L, category="search",
                            init_set="search")
        rollouts += res.n
        return res.k

    def audit(i: int, gp_mean: float | None) -> None:  # score incumbent i on the search audit seeds
        nonlocal rollouts
        if i not in audited:
            res = evaluate(partial(KnobController, to_knobs(X[i], cfg), cfg.robot), "search", robot=cfg.robot,
                           n_workers=cfg.n_workers, ledger=L, category="search", n=cfg.audit_n)
            rollouts += res.n
            audited[i] = res.k
        hist.append({"trial": len(K), "rollouts": rollouts, "robot_min": L.robot_minutes["search"],
                     "incumbent": i, "k": audited[i], "n": cfg.audit_n,
                     "x": list(map(float, X[i])), "gp_mean": gp_mean})

    K.append(trial(X[0]))
    audit(0, None)  # the starting point of the curve, as for the other methods
    while len(K) < cfg.n_evals:
        if len(X) < n_init or acq == "sobol":
            X.append(sobol.draw(1).double().numpy()[0])
            gp = None
        else:
            gp = fit_gp(np.array(X), np.array(K, float), np.full(len(K), float(cfg.batch)))
            mu = posterior_mean(gp, np.array(X))
            X.append(next_point(gp, float(mu.max()), d, seed * 1000 + len(K)))
        K.append(trial(X[-1]))
        if len(K) >= n_init and ((len(K) - n_init) % cfg.audit_every == 0 or len(K) == cfg.n_evals):
            if acq == "sobol":  # random search: the best batch score so far (later wins ties)
                inc, m = max(range(len(K)), key=lambda i: (K[i], i)), None
            else:
                gp = fit_gp(np.array(X), np.array(K, float), np.full(len(K), float(cfg.batch)))
                mu = posterior_mean(gp, np.array(X))
                inc, m = int(np.argmax(mu)), float(mu.max())
            audit(inc, m)
            print(f"  {acq} seed {seed} trial {len(K):2d}: last batch {K[-1]}/{cfg.batch} | incumbent #{inc} "
                  f"({K[inc]}/{cfg.batch}) on search {audited[inc]}/{cfg.audit_n}")
    best = max(audited, key=lambda i: (audited[i], i))
    return {"history": hist, "trials_x": np.array(X).tolist(), "trials_k": K, "best_x": X[best].tolist(),
            "best_search": [audited[best], cfg.audit_n], "rollouts": rollouts}


# ---------------------------------------------------------------------------- Chapter 2 comparison plots


ALL_KNOBS = [k.name for k in KNOB_SPACE]
MAIN_PLOT = ("cem", "cmaes", "pi2", "random", "bo", "sobol")  # the default runs of both scripts
SAME_KNOBS_PLOT = ("cem-4k", "cmaes-4k", "pi2-4k", "random-4k", "bo", "sobol")  # all on BO's 4 knobs


def latest_ch02_runs(cfg: Config) -> dict[str, list[dict]]:
    """The newest full Chapter 2 run with a held-out score for each (variant, knob set, seed).

    Runs are grouped by variant name. A run on a knob set that is not its script's default and has no
    --tag gets the knob count added to its name, so it never silently replaces the default run.
    """
    newest: dict[tuple, dict] = {}
    for r in load_results(cfg.results_root):
        e, c = r.get("extra", {}), r["config"]
        if r.get("chapter") != "ch02" or c.get("quick") or "history" not in e or r["robot"] != cfg.robot:
            continue
        if r.get("final") is None:  # a --no-heldout tuning run: nothing to plot on the held-out axis
            continue
        knobs = list(c.get("knobs", []))
        label = e["variant"]
        if not c.get("tag") and knobs != (ALL_KNOBS if "weights" in c else Config().knobs):
            label += f" ({len(knobs)} knobs)"
        key = (label, tuple(knobs), e["seed"])
        if key not in newest or r["timestamp"] > newest[key]["timestamp"]:
            newest[key] = {**r, "label": label}
    groups: dict[str, list[dict]] = {}
    for key in sorted(newest):
        groups.setdefault(key[0], []).append(newest[key])
    return groups


def plot_comparison(cfg: Config) -> None:
    """All methods on one plot (best-so-far audited search score vs cost), held-out, knobs moved."""
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    runs = latest_ch02_runs(cfg)
    base_color = dict(zip(["cem", "cmaes", "pi2", "bo", "sobol", "random"], plotting.PALETTE))
    colors = {v: base_color[v.removesuffix("-4k")] for v in set(MAIN_PLOT) | set(SAME_KNOBS_PLOT)}
    plotting.setup()
    for wanted, fname, note in ((MAIN_PLOT, "all_methods.png", "CEM/CMA-ES/PI2/random: 10 knobs, BO/Sobol: 4"),
                                (SAME_KNOBS_PLOT, "all_methods_4knobs.png", "every method on the same 4 knobs")):
        groups = {v: runs[v] for v in wanted if v in runs}
        if not groups:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2.5, 2.5, 1.3]})
        for ax, xkey, xlabel in ((axes[0], "rollouts", "search episodes (incl. audits)"),
                                 (axes[1], "robot_min", "search robot-minutes (incl. audits)")):
            for v, rs in groups.items():
                # Step functions: the recommendation only changes at an audit. Pool seeds on a common x grid.
                grid = np.unique(np.concatenate([[h[xkey] for h in r["extra"]["history"]] for r in rs]))
                ks = []
                for r in rs:
                    h = r["extra"]["history"]
                    x, best = np.array([e[xkey] for e in h]), np.maximum.accumulate([e["k"] for e in h])
                    ks.append(best[np.maximum(np.searchsorted(x, grid, side="right") - 1, 0)])
                    ax.step(x, best / h[0]["n"], where="post", color=colors[v], alpha=0.3, lw=1)
                n = rs[0]["extra"]["history"][0]["n"] * len(rs)
                plotting.success_curve(ax, grid, np.sum(ks, axis=0), n, label=f"{v} ({len(rs)} seeds)",
                                       color=colors[v], marker="")
            ax.set_xscale("log")
            ax.set(xlabel=f"{xlabel}, log scale", ylabel="best-so-far success on search (n=64)")
        axes[0].set_title(f"Search: best audited recommendation so far\n({note})", fontsize=10)
        axes[0].legend(loc="lower right", fontsize=8)
        for j, (v, rs) in enumerate(groups.items()):
            for i, r in enumerate(rs):
                f = r["final"]
                plotting.heldout_point(axes[2], j + 0.15 * (i - 1), f["k"], f["n"], color=colors[v], label=None)
        axes[2].set(xticks=range(len(groups)), xticklabels=list(groups), ylim=(0, 1.02),
                    title=f"Held-out eval, one per seed (n={rs[0]['final']['n']})")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        plotting.save_fig(fig, MEDIA / fname)

    groups = {v: runs[v] for v in MAIN_PLOT if v in runs}  # which knobs each method moved
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(11, 3.8))
    width = 0.8 / len(groups)
    default = to_unit(DEFAULT_KNOBS, ALL_KNOBS)
    for j, (v, rs) in enumerate(groups.items()):
        moved = np.array([to_unit(r["extra"]["best_knobs"], ALL_KNOBS) - default for r in rs])
        pos = np.arange(len(ALL_KNOBS)) + (j - (len(groups) - 1) / 2) * width
        ax.bar(pos, moved.mean(axis=0), width, label=v, color=colors[v], alpha=0.75)
        for m in moved:  # one dot per seed: a bar alone would hide seeds that disagree
            ax.plot(pos, m, ".", color="k", ms=3)
    ax.plot(range(len(ALL_KNOBS)), to_unit(TUNED_KNOBS[cfg.robot], ALL_KNOBS) - default, "k*", ms=9,
            label="hand-tuned reference")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(range(len(ALL_KNOBS)), ALL_KNOBS, rotation=30)
    ax.set(ylabel="found - default\n(fraction of knob range)",
           title="Which knobs each method moved (bars: mean of seeds, dots: each seed)")
    ax.legend(ncol=4, fontsize=8)
    plotting.save_fig(fig, MEDIA / "knobs_moved.png")
    print(f"wrote {MEDIA / 'all_methods.png'}, {MEDIA / 'all_methods_4knobs.png'} and {MEDIA / 'knobs_moved.png'}")


def main(cfg: Config) -> None:
    final_label = "search check" if cfg.quick else "held-out"
    base_label = "search check" if cfg.quick else "eval"
    acqs = ["logei", "sobol"] if cfg.acq == "all" else [cfg.acq]
    ev = partial(evaluate, init_set="search" if cfg.quick else "eval", robot=cfg.robot,
                 n_workers=cfg.n_workers, n=cfg.eval_n)
    base = ev(partial(KnobController, DEFAULT_KNOBS, cfg.robot)) if cfg.heldout else None
    finals: dict[str, list] = {}
    for acq in acqs:
        name = ("bo" if acq == "logei" else "sobol") + cfg.tag
        for seed in cfg.seeds:
            with Ledger(chapter="ch02", method=f"{name}_s{seed}", robot=cfg.robot,
                        config=replace(cfg, acq=acq), results_root=cfg.results_root) as L:
                r = run_bo(acq, seed, cfg, L)
                knobs = to_knobs(np.array(r["best_x"]), cfg)
                L.extra.update(method=name, variant=name, seed=seed, best_knobs=knobs,
                               best_search=r["best_search"], history=r["history"],
                               trials_x=r["trials_x"], trials_k=r["trials_k"])
                if base is not None:
                    L.robot_steps["eval"] += base.env_steps  # the shared base evaluation
                    L.set_base(id="knobs_default", successes=base)
                    final = ev(partial(KnobController, knobs, cfg.robot), ledger=L, category="eval")
                    L.set_final(successes=final)
                    finals.setdefault(name, []).append(final)
            held = f" -> {final_label} {final.summary}" if base is not None else ""
            print(f"{name} seed {seed}: search {format_rate(*r['best_search'])}{held} | "
                  f"{L.robot_minutes['search']:.1f} search robot-min, {r['rollouts']} episodes")
    if base is not None:
        print(f"\nsummary (base on {base_label}: {base.summary})")
        for name, fs in finals.items():
            k, n = sum(f.k for f in fs), sum(f.n for f in fs)
            per_seed = ", ".join(f"{f.k}/{f.n}" for f in fs)
            print(f"  {name:6s} {final_label} per seed [{per_seed}]  pooled {format_rate(k, n)}")
    if cfg.media and not cfg.quick and base is not None:
        plot_comparison(cfg)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message=".*torch.jit.script.*")  # printed by gpytorch's import
    main(parse(Config))
