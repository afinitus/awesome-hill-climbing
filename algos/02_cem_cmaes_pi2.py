"""Chapter 2: one update, different weights. CEM, CMA-ES and PI2/MPPI on the knob controller.

Learning goal
    See that the cross-entropy method (CEM), CMA-ES and PI2/MPPI are one loop: sample K candidates
    around a mean, score them, give each a weight, and move the mean to the weighted average. They differ
    only in the weight function and in which statistics they refit afterwards.

The base idea
    Sample   x_i = m + sigma * A z_i,  z_i ~ N(0, I),  C = A A^T          (i = 1..K)
    Score    J_i = success rate of knobs x_i on a batch of search episodes
    Weight   w_i = CEM:    1/k for the top-k ("elites"), 0 otherwise (k = round(0.25 * 10) = 2 by default)
                   CMA-ES: log-rank weights, w ~ ln((K+1)/2) - ln(rank), top half only
                   PI2:    exp(-cost_i / lambda) / sum_j exp(-cost_j / lambda),  cost = 1 - J
    Update   m <- sum_i w_i x_i                    (the one shared step)
    Refit    CEM:    a diagonal variance, var_j = sum_i w_i (x_ij - m_j)^2
             CMA-ES: a full covariance C and the step size sigma, from evolution paths (Hansen 2016)
             PI2:    nothing; the exploration noise is fixed and only decays (MPPI keeps it fixed)
    Candidates with the same score share their weight equally: with a success counter, ties are common.
    `--weights random` is the baseline with no update at all: it keeps sampling from the first
    generation's distribution around the default and keeps the best-scoring candidate (random search).

    The search space is the 10 KnobController knobs scaled to [0, 1] by their KNOB_SPACE bounds, started
    from DEFAULT_KNOBS (about 50% success). Every candidate in a generation sees the same `batch` search
    episodes (common random numbers), and the batch rotates through the 64-seed search set from one
    generation to the next. After each update the mean is scored on all 64 search seeds; the returned
    knobs are the best-scoring mean. That audit is part of the search budget. The held-out eval set (256)
    is used once per run, for the final number.

    Boundary handling: samples outside [0, 1] are clipped, and the clipped point is both scored and
    learned from. This is simple but biased at a bound: DEFAULT speed = 1.0 sits on its upper bound, so
    every clipped candidate has speed <= 1 and any weighted mean pulls speed down, whatever the scores.
    Hansen's tutorial (C005, appendix on boundaries) discusses penalty and resampling alternatives.

Papers mirrored (ids from data/papers.csv)
    C130 An Introduction to Zero-Order Optimization Techniques for Robotics (2025): CEM, CMA-ES and MPPI
         as one sample-weight-update template.
    C005 Hansen, The CMA Evolution Strategy: A Tutorial (2016): the (mu/mu_w, lambda)-CMA-ES implemented
         here, with a full covariance, rank-one plus rank-mu updates and cumulative step-size adaptation.
    C002 Theodorou et al., PI2 (2010): exp(-cost/lambda) path-integral weights, no gradient.
    C445 A Decade of BO for Controller Tuning (2026) and C057 BOpt-GMM: see algos/02b_bo.py.

Run
    uv run python algos/02_cem_cmaes_pi2.py --quick                  # smoke test, under a minute; results to runs/ch02_quick/
    uv run python algos/02_cem_cmaes_pi2.py                          # 3 methods + random, 3 seeds + failures
    uv run python algos/02_cem_cmaes_pi2.py --weights cmaes --seeds 0 1 2 3 4
    uv run python algos/02_cem_cmaes_pi2.py --knobs grasp_dx grasp_dy grasp_dz drop_dx --tag=-4k \
        --no-failures --no-media                                     # BO's 4 knobs, for a fair comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from lastmile.common.cli import parse
from lastmile.common.eval import INIT_SETS, EvalResult, format_rate, policy_seed
from lastmile.common.ledger import DEFAULT_RESULTS_ROOT, Ledger
from lastmile.common.rollout import evaluate, rollout_seeds
from lastmile.envs.knob_controller import DEFAULT_KNOBS, KNOB_SPACE, TUNED_KNOBS, KnobController

MEDIA = Path(__file__).resolve().parents[1] / "media" / "ch02"
METHODS = ("cem", "cmaes", "pi2")
BASELINE = "random"  # same sampler, no weights and no update: the random-selection baseline
SHOWCASE = ("cem", 0)  # the run shown in the before/after GIF, fixed in advance (not picked on eval)


@dataclass
class Config:
    robot: str = "so101"
    weights: Literal["all", "cem", "cmaes", "pi2", "random"] = "all"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    knobs: list[str] = field(default_factory=lambda: [k.name for k in KNOB_SPACE])
    generations: int = 12
    population: int = 10  # K; CMA-ES default for 10 dims: 4 + floor(3 ln 10) = 10
    batch: int = 8  # search episodes per candidate (the same ones for every candidate in a generation)
    audit_n: int = 64  # search episodes used to score each generation's mean
    sigma0: float = 0.1  # initial exploration std, in units of each knob's range
    elite_frac: float = 0.25  # CEM: fraction kept as elites; round(2.5) = 2 of 10 (Python rounds to even)
    cem_smooth: float = 0.5  # CEM: new variance = (1 - a) old + a refit; 1.0 is textbook CEM
    cem_min_std: float = 0.0  # CEM: variance floor (0 lets it collapse)
    pi2_lambda: float = 0.05  # PI2 temperature, in units of success rate
    pi2_decay: float = 0.9  # PI2: exploration std multiplier per generation
    failures: bool = True  # also run the failure cases shown in the chapter doc
    heldout: bool = True  # score the final knobs on the held-out eval set (turn off while tuning)
    eval_n: int = 256
    n_workers: int = 6
    media: bool = True
    tag: str = ""  # appended to every method name, e.g. "-4k" for a run on a knob subset
    results_root: str = str(DEFAULT_RESULTS_ROOT)
    quick: bool = False
    QUICK: ClassVar[dict] = {"seeds": [0], "generations": 2, "population": 6, "batch": 8, "audit_n": 16,
                             "failures": False, "eval_n": 32,  # quick results stay out of results/
                             "results_root": str(MEDIA.parents[1] / "runs" / "ch02_quick")}


# The failure cases: the same loop with one setting changed (see "What went wrong" in docs/chapters/ch02.md).
FAILURES = {
    "cem-greedy": ("cem", {"elite_frac": 0.1, "cem_smooth": 1.0}),  # keep 1 elite of 10, no smoothing
    "pi2-hot": ("pi2", {"pi2_lambda": 1.0}),  # temperature far above the score differences
}


# ---------------------------------------------------------------------------- the knob space


def knob_bounds(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    space = {k.name: k for k in KNOB_SPACE}
    low = np.array([space[n].low for n in cfg.knobs])
    return low, np.array([space[n].high for n in cfg.knobs]) - low


def to_knobs(x: np.ndarray, cfg: Config) -> dict[str, float]:
    """A point of [0, 1]^d -> a full knob dict (knobs outside the search keep their default)."""
    low, span = knob_bounds(cfg)
    return {**DEFAULT_KNOBS, **dict(zip(cfg.knobs, (low + span * np.clip(x, 0, 1)).tolist()))}


def to_unit(knobs: dict[str, float], cfg: Config) -> np.ndarray:
    low, span = knob_bounds(cfg)
    return (np.array([knobs[n] for n in cfg.knobs]) - low) / span


def score_batch(x: np.ndarray, idx: list[int], cfg: Config, L: Ledger) -> EvalResult:
    """Success of knobs x on search episodes `idx` (env seed and policy seed fixed per index)."""
    make = partial(KnobController, to_knobs(x, cfg), cfg.robot)
    return rollout_seeds(make, [INIT_SETS["search"][i] for i in idx], [policy_seed("search", i) for i in idx],
                         robot=cfg.robot, n_workers=cfg.n_workers, ledger=L, category="search",
                         init_set="search")


# ---------------------------------------------------------------------------- the three weight functions


def share_ties(w: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Candidates with equal scores get equal weight (their order among themselves is arbitrary)."""
    w = w.copy()
    for s in np.unique(scores):
        w[scores == s] = w[scores == s].mean()
    return w


def by_rank(raw: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Give raw[r] to the candidate of rank r (0 = best), then share weight among ties."""
    w = np.empty_like(raw)
    w[np.argsort(-scores, kind="stable")] = raw
    return share_ties(w, scores)


def weights_cem(scores: np.ndarray, cfg: Config) -> np.ndarray:
    k = max(1, round(cfg.elite_frac * len(scores)))
    return by_rank(np.r_[np.full(k, 1 / k), np.zeros(len(scores) - k)], scores)


def cmaes_raw_weights(K: int) -> np.ndarray:
    """Hansen's positive weights: w_r ~ ln((K+1)/2) - ln(r) for the best mu = K/2, zero for the rest."""
    raw = np.maximum(0.0, np.log((K + 1) / 2) - np.log(np.arange(1, K + 1)))
    return raw / raw.sum()


def weights_cmaes(scores: np.ndarray, cfg: Config) -> np.ndarray:
    return by_rank(cmaes_raw_weights(len(scores)), scores)


def weights_pi2(scores: np.ndarray, cfg: Config) -> np.ndarray:
    cost = 1.0 - scores
    w = np.exp(-(cost - cost.min()) / cfg.pi2_lambda)  # subtracting the min only avoids underflow
    return w / w.sum()


WEIGHTS = {"cem": weights_cem, "cmaes": weights_cmaes, "pi2": weights_pi2}


# ---------------------------------------------------------------------------- the shared loop


@dataclass
class Search:
    """The sampling distribution N(mean, sigma^2 C) plus CMA-ES's two evolution paths."""

    mean: np.ndarray
    sigma: float
    C: np.ndarray
    p_sigma: np.ndarray
    p_c: np.ndarray
    gen: int = 0

    @classmethod
    def start(cls, mean: np.ndarray, sigma: float) -> Search:
        d = len(mean)
        return cls(mean.copy(), sigma, np.eye(d), np.zeros(d), np.zeros(d))

    def sample(self, K: int, rng: np.random.Generator) -> np.ndarray:
        eigval, B = np.linalg.eigh(self.C)
        z = rng.standard_normal((K, len(self.mean)))
        return self.mean + self.sigma * z @ (B * np.sqrt(np.maximum(eigval, 0))).T


def refit(s: Search, X: np.ndarray, w: np.ndarray, method: str, cfg: Config) -> None:
    """Move to the weighted mean (all methods), then refit what each method adapts."""
    d, K = len(s.mean), len(X)
    y = (X - s.mean) / s.sigma  # each candidate's step, in units of sigma
    y_w = w @ y
    s.mean = np.clip(s.mean + s.sigma * y_w, 0.0, 1.0)  # m <- sum_i w_i x_i (kept inside the box)
    s.gen += 1

    if method == "pi2":  # PI2 / MPPI: only the mean is learned
        s.sigma *= cfg.pi2_decay
    elif method == "cem":  # CEM: diagonal variance of the elites around the new mean
        var = w @ (y - y_w) ** 2
        new = (1 - cfg.cem_smooth) * np.diag(s.C) + cfg.cem_smooth * var
        s.C = np.diag(np.maximum(new, (cfg.cem_min_std / s.sigma) ** 2))
    else:  # CMA-ES, following Hansen's tutorial (C005), section "Summary of the update equations"
        raw = cmaes_raw_weights(K)
        mu_nom = 1 / (raw @ raw)  # nominal mu_eff sets the learning rates
        mu_eff = 1 / (w @ w)  # actual mu_eff (ties spread the weight) normalizes the paths
        c_s = (mu_nom + 2) / (d + mu_nom + 5)
        d_s = 1 + 2 * max(0.0, np.sqrt((mu_nom - 1) / (d + 1)) - 1) + c_s
        c_c = (4 + mu_nom / d) / (d + 4 + 2 * mu_nom / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_nom)
        c_mu = min(1 - c_1, 2 * (mu_nom - 2 + 1 / mu_nom) / ((d + 2) ** 2 + mu_nom))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))  # E||N(0, I)||

        eigval, B = np.linalg.eigh(s.C)
        C_inv_sqrt = B @ np.diag(1 / np.sqrt(np.maximum(eigval, 1e-20))) @ B.T
        s.p_sigma = (1 - c_s) * s.p_sigma + np.sqrt(c_s * (2 - c_s) * mu_eff) * C_inv_sqrt @ y_w
        # h_sigma stalls the rank-one update while the step size is still growing fast
        p_norm = np.linalg.norm(s.p_sigma) / np.sqrt(1 - (1 - c_s) ** (2 * s.gen))
        h_sig = p_norm < (1.4 + 2 / (d + 1)) * chi_n
        s.p_c = (1 - c_c) * s.p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        rank_mu = (w[:, None] * y).T @ y  # sum_i w_i y_i y_i^T (zero-weight candidates drop out)
        s.C = ((1 - c_1 - c_mu) * s.C + c_1 * (np.outer(s.p_c, s.p_c) + (1 - h_sig) * c_c * (2 - c_c) * s.C)
               + c_mu * rank_mu)
        s.C = (s.C + s.C.T) / 2
        s.sigma *= np.exp((c_s / d_s) * (np.linalg.norm(s.p_sigma) / chi_n - 1))


def run_search(method: str, seed: int, cfg: Config, L: Ledger) -> dict:
    """The sample -> weight -> refit loop. Returns the per-generation history and the chosen knobs."""
    rng = np.random.default_rng(seed)
    s = Search.start(to_unit(DEFAULT_KNOBS, cfg), cfg.sigma0)
    n_search = len(INIT_SETS["search"])
    rollouts, hist = 0, []
    incumbent, inc_score = s.mean.copy(), -1.0  # random search only: the best candidate so far

    def audit(tag: str, x: np.ndarray) -> None:  # score point x on the audit seeds (charged to search)
        nonlocal rollouts
        res = evaluate(partial(KnobController, to_knobs(x, cfg), cfg.robot), "search", robot=cfg.robot,
                       n_workers=cfg.n_workers, ledger=L, category="search", n=cfg.audit_n)
        rollouts += res.n
        std = s.sigma * np.sqrt(np.diag(s.C))
        hist.append({"gen": s.gen, "rollouts": rollouts, "robot_min": L.robot_minutes["search"], "k": res.k,
                     "n": res.n, "x": x.tolist(), "mean": s.mean.tolist(),
                     "std": std.tolist(), "sigma": s.sigma, "cov": (s.sigma**2 * s.C).tolist(), "tag": tag})

    audit("start", s.mean)
    for g in range(cfg.generations):
        X = np.clip(s.sample(cfg.population, rng), 0.0, 1.0)  # evaluate (and learn from) the repaired points
        start = (g * cfg.batch) % n_search  # rotate through the search set, same slice for all candidates
        idx = [(start + j) % n_search for j in range(cfg.batch)]
        scores = np.array([score_batch(x, idx, cfg, L).sr for x in X])
        rollouts += cfg.population * cfg.batch
        if method == BASELINE:
            # No weights, no update: the distribution stays N(default, sigma0^2 I). Keep the best batch score
            # seen so far (later wins ties) and audit it, so the budget matches the other methods exactly.
            s.gen += 1
            i = len(scores) - 1 - int(np.argmax(scores[::-1]))
            if scores[i] >= inc_score:
                incumbent, inc_score = X[i].copy(), scores[i]
            audit("gen", incumbent)
        else:
            w = WEIGHTS[method](scores, cfg)
            refit(s, X, w, method, cfg)
            audit("gen", s.mean)
        hist[-1]["scores"] = scores.tolist()
        h = hist[-1]
        print(f"  {method} seed {seed} gen {g + 1:2d}: batch best {scores.max():.2f} mean {scores.mean():.2f}"
              f" | audited point on search {h['k']:2d}/{h['n']} | sigma*std {np.mean(h['std']):.3f}")

    best = max(range(len(hist)), key=lambda i: (hist[i]["k"], i))  # best audited point; later wins ties
    return {"history": hist, "best_gen": hist[best]["gen"], "best_x": hist[best]["x"],
            "best_search": [hist[best]["k"], hist[best]["n"]], "rollouts": rollouts}


# ---------------------------------------------------------------------------- media


def render_episode(knobs: dict, seed: int, robot: str) -> list[np.ndarray]:
    """Replay one eval episode in-process (same env and policy seed as evaluate), front camera."""
    from lastmile.envs.cupdrop import CupDropEnv

    env, policy = CupDropEnv(robot=robot), KnobController(knobs, robot)
    obs, _ = env.reset(seed=seed)
    policy.reset([policy_seed("eval", INIT_SETS["eval"].index(seed))])
    frames = [env.render("front")]
    for _ in range(env.max_steps):
        obs, _, terminated, truncated, _ = env.step(policy.act(obs[None])[0])
        frames.append(env.render("front"))
        if terminated or truncated:
            break
    return frames


def before_after_gif(base: EvalResult, final: EvalResult, knobs: dict, cfg: Config, name: str,
                     label: str) -> None:
    """Default knobs (left) vs found knobs (right) on the first eval episode the default fails and they fix.

    Choosing that episode looks at eval successes, which is fine for an illustration; which run to show is
    fixed in advance (SHOWCASE), so no run is picked because of its held-out score.
    """
    from PIL import Image, ImageDraw

    from lastmile.common.plotting import save_gif

    fixed = np.flatnonzero(~base.successes & final.successes)
    if not len(fixed):
        return
    seed = int(base.seeds[fixed[0]])
    left, right = render_episode(DEFAULT_KNOBS, seed, cfg.robot), render_episode(knobs, seed, cfg.robot)
    T = max(len(left), len(right)) + 8  # pad the shorter episode, then hold the end for a moment
    frames = []
    for t in range(0, T, 2):  # every other step keeps the file small
        pair = np.concatenate([left[min(t, len(left) - 1)], right[min(t, len(right) - 1)]], axis=1)
        img = Image.fromarray(pair)
        draw = ImageDraw.Draw(img)
        draw.text((6, 6), "default knobs", fill=(255, 255, 255))
        draw.text((pair.shape[1] // 2 + 6, 6), f"{label} knobs", fill=(255, 255, 255))
        frames.append(np.asarray(img))
    path = save_gif(frames, MEDIA / f"before_after_{name}.gif", fps=5, max_size=512)
    print(f"wrote {path} ({label}, eval seed {seed}, {path.stat().st_size / 1e6:.2f} MB)")


def plot_curves(runs: dict[str, list[dict]], cfg: Config) -> None:
    """Best-so-far search score vs rollouts (seeds thin, pooled bold with a Wilson band) + held-out."""
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    plotting.setup()
    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 3, 1.3]})
    colors = {**dict(zip(METHODS, plotting.PALETTE)), BASELINE: plotting.PALETTE[5]}
    for j, (name, rs) in enumerate(runs.items()):
        failure = name in FAILURES  # drawn dashed, in the color of the method it breaks
        c, ls = colors[FAILURES[name][0] if failure else name], "--" if failure else "-"
        x, n = [h["rollouts"] for h in rs[0]["history"]], rs[0]["history"][0]["n"]
        now = [np.array([h["k"] for h in r["history"]]) for r in rs]  # the audited point's score each generation
        for axis, ks in ((ax, [np.maximum.accumulate(k) for k in now]), (ax1, now)):
            for k in ks:
                axis.plot(x, k / n, color=c, alpha=0.25, lw=1, ls=ls)
            line = plotting.success_curve(axis, x, np.sum(ks, axis=0), n * len(rs), color=c,
                                          label=f"{name} ({len(rs)} seeds)", marker="." if failure else "o")
            line.set_linestyle(ls)
        for i, r in enumerate(rs):
            if "final" in r:
                plotting.heldout_point(ax2, j + 0.15 * (i - 1), r["final"].k, r["final"].n, color=c,
                                       label=None)
    ax1.set(xlabel="search rollouts", ylabel="success of the audited point on search",
            title="The audited point itself (no best-so-far)")
    for axis in (ax, ax1):
        axis.axhline(rs[0]["base_search"], color="0.4", lw=1, ls=":", label="default knobs")
    ax.set(xlabel="search rollouts (episodes, incl. audits)",
           ylabel="best-so-far success on search", title="Search: best audited point so far")
    ax.legend(loc="lower right", fontsize=8)
    ax2.set(xticks=range(len(runs)), xticklabels=list(runs), ylim=(0, 1.02),
            title=f"Held-out eval (n={cfg.eval_n})")
    ax2.tick_params(axis="x", rotation=45)
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    plotting.save_fig(fig, MEDIA / "search_curves.png")


def ellipse_gif(runs: dict[str, list[dict]], cfg: Config, dims: tuple[str, str] = ("grasp_dy", "drop_dx")):
    """Mean and 2-std ellipse of each search distribution in a 2-knob slice, one frame per generation."""
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    if not set(dims) <= set(cfg.knobs):
        return
    i, j = cfg.knobs.index(dims[0]), cfg.knobs.index(dims[1])
    tuned = to_unit(TUNED_KNOBS[cfg.robot], cfg)
    plotting.setup()
    frames, t = [], np.linspace(0, 2 * np.pi, 60)
    for g in range(cfg.generations + 1):
        fig, axes = plt.subplots(1, len(METHODS), figsize=(9, 3.3))
        for ax, name, color in zip(axes, METHODS, plotting.PALETTE):
            h = runs[name][0]["history"][g]
            cov = np.array(h["cov"])[np.ix_([i, j], [i, j])]
            val, vec = np.linalg.eigh(cov)
            ell = h["mean"][i], h["mean"][j]
            radius = 2 * np.sqrt(np.maximum(val, 0))[:, None]
            pts = np.array(ell)[:, None] + vec @ (radius * [np.cos(t), np.sin(t)])
            ax.plot(*pts, color=color)
            ax.plot(*ell, "o", color=color)
            ax.plot(tuned[i], tuned[j], "*", color="0.3", ms=10)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=dims[0], title=f"{name}, gen {g}: {h['k']}/{h['n']}")
            ax.set_aspect("equal")
        axes[0].set_ylabel(dims[1])
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    path = plotting.save_gif(frames + frames[-1:] * 3, MEDIA / "ellipses.gif", fps=2, max_size=900)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------- main


def main(cfg: Config) -> None:
    final_label = "search check" if cfg.quick else "held-out"
    base_label = "search check" if cfg.quick else "eval"
    names = [*METHODS, BASELINE] if cfg.weights == "all" else [cfg.weights]
    if cfg.failures:
        names += [f for f, (m, _) in FAILURES.items() if m in names]
    ev = partial(evaluate, init_set="search" if cfg.quick else "eval", robot=cfg.robot,
                 n_workers=cfg.n_workers, n=cfg.eval_n)
    base = ev(partial(KnobController, DEFAULT_KNOBS, cfg.robot)) if cfg.heldout else None
    base_search = evaluate(partial(KnobController, DEFAULT_KNOBS, cfg.robot), "search", robot=cfg.robot,
                           n_workers=cfg.n_workers, n=cfg.audit_n)  # for the plot's reference line only
    runs: dict[str, list[dict]] = {}
    for name in names:
        method, overrides = FAILURES.get(name, (name, {}))
        run_cfg = replace(cfg, **overrides)
        for seed in cfg.seeds:
            with Ledger(chapter="ch02", method=f"{name}{cfg.tag}_s{seed}", robot=cfg.robot, config=run_cfg,
                        results_root=cfg.results_root) as L:
                r = run_search(method, seed, run_cfg, L)
                r["base_search"] = base_search.sr
                knobs = to_knobs(np.array(r["best_x"]), cfg)
                L.extra.update(method=method, variant=name + cfg.tag, seed=seed, best_knobs=knobs,
                               best_gen=r["best_gen"], best_search=r["best_search"], history=r["history"])
                if base is not None:
                    L.robot_steps["eval"] += base.env_steps  # the (shared) base evaluation, charged to eval
                    L.set_base(id="knobs_default", successes=base)
                    r["final"] = ev(partial(KnobController, knobs, cfg.robot), ledger=L, category="eval")
                    L.set_final(successes=r["final"])
            k, n = r["best_search"]
            held = f" -> {final_label} {r['final'].summary}" if "final" in r else ""
            print(f"{name} seed {seed}: search {format_rate(k, n)}{held} | "
                  f"{L.robot_minutes['search']:.1f} search robot-min, {r['rollouts']} episodes")
            runs.setdefault(name, []).append(r)

    print(f"\nsummary (base on {base_label}: {base.summary if base else 'not scored'})")
    for name, rs in runs.items():
        if base is None:
            continue
        k, n = sum(r["final"].k for r in rs), sum(r["final"].n for r in rs)
        per_seed = ", ".join(f"{r['final'].k}/{r['final'].n}" for r in rs)
        print(f"  {name:10s} {final_label} per seed [{per_seed}]  pooled {format_rate(k, n)}")
    if cfg.media and not cfg.quick and not cfg.tag and base is not None:  # tagged runs keep the media as is
        plot_curves(runs, cfg)
        if all(m in runs for m in METHODS):
            ellipse_gif(runs, cfg)
        name, seed = SHOWCASE
        if name in runs and seed in cfg.seeds:
            r = runs[name][cfg.seeds.index(seed)]
            from lastmile.common.plotting import optional_media

            with optional_media("the before/after GIF"):
                before_after_gif(base, r["final"], to_knobs(np.array(r["best_x"]), cfg), cfg, name,
                                 f"{name} (seed {seed})")


if __name__ == "__main__":
    main(parse(Config))
