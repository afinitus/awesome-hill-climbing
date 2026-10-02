"""Chapter 6: residual RL on a frozen base. A small TD3 add-on corrects base_v1 without touching it.

Learning goal
    Improve a frozen policy with a correction that starts at exactly the base's performance and moves each
    action at most a little. See what makes it learn safely (a warmed-up, normalized critic), where it is
    easy (a calibration error the base cannot see) and how it breaks (an actor that follows a cold critic
    drags the base below its own success rate while training).

Base idea in plain words
    The residual MDP: the frozen base is part of the environment, and the learner only adds a bounded
    correction to whatever the base does at each step,

        a_t = clip(a_base_t + alpha * tanh(u_theta(s_t, a_base_t)), -1, 1)

    The actor's last layer is zero-initialized, so tanh(0) = 0 and training starts exactly at the base
    (checked bit for bit below). alpha bounds the correction per step: 0.1 is 2 mm of end-effector target
    per step. The env integrates the deltas, so a residual that keeps pushing one way still adds up.
    Training is off-policy TD3 with a critic on the summed action and an n-step target,

        y   = sum_{k<n} gamma^k r_{t+k} + gamma^n (1 - d) min_j Q'_j(s_{t+n}, a'),
        a'  = clip(a_base_{t+n} + alpha * clip(tanh(u'(s_{t+n}, a_base_{t+n})) + eps, -1, 1), -1, 1)
        L_Q = (Q_j(s_t, a_t) - y)^2,        L_pi = -Q_1(s_t, a_base_t + alpha * tanh(u_theta(s_t, a_base_t)))

    The reward is the sparse 0/1 success flag, so Q is a success probability and y is clipped to [0, 1].
    DAWN's two fixes for the critic: (1) warm-up, which fills the replay buffer with pure base rollouts and
    trains the critic alone on them before the actor moves, and (2) LayerNorm in the critic.

Papers mirrored
    ResFiT [C142] (frozen chunked BC policy + per-step TD3 residual, LayerNorm critic, n-step returns;
    ResFiT also mixes demos into every batch, which we replace with the base-rollout warm-up),
    DAWN [C233] (base-data warm-up + critic LayerNorm; its 2x2 ablation is reproduced here), Res-HIL [C432]
    (zero-initialized TD3 residual), PLD [C158] and Policy Decorator [C086] (bounded residuals on frozen
    generative policies), and the classics Residual RL for Robot Control [C017] and Residual Policy
    Learning [C016].

Honesty
    Training episodes use their own env seeds (1_000_000 + 50_000 * run id + i). Every run is scored on
    `search` along the way (charged to search) and the deployed checkpoint is the one with the best search
    score, the untouched base (zero residual) included as a candidate. The held-out `eval` set (n=256) is
    scored once per run, paired with the base on the same seeds and policy noise. The bias variant uses the
    same seeds with constant joint offsets. Every row is charged its own run, the shared base runs, all
    DAWN-ablation and large-bound runs (and the large-bound run's eval passes, as search), and the earlier
    pilot and superseded runs via --prior-search-steps / --prior-train-steps.

Run
    uv run python algos/06_residual_td3.py            # full run: 33-42 min on a busy M5 Pro, 4 processes
    uv run python algos/06_residual_td3.py --quick    # smoke test, < 2 min; writes only to runs/ch06_quick/
The file is longer than the 450-line guideline because it also runs the DAWN ablation, the large-bound
run and the early-training analysis; each is a short block in main().
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from typing import ClassVar

import numpy as np

from lastmile.common.cli import parse
from lastmile.common.eval import bootstrap_diff, format_rate, mcnemar_exact, policy_seed
from lastmile.common.ledger import REPO_ROOT, Ledger
from lastmile.common.plotting import optional_media
from lastmile.common.rollout import evaluate, rollout_seeds, shutdown_pool

# torch is imported inside functions: spawned rollout workers re-import this file.

BASE = REPO_ROOT / "checkpoints/base_v1_so101.pt"
TRAIN_SEED_BASE = 1_000_000  # far from search (10k), eval (20k), eval_ext (30k), stress (40k), ch12 (70k+)
STEPS_PER_MIN = 600


@dataclass
class Config:
    robot: str = "so101"
    variants: list[str] = field(default_factory=lambda: ["v1", "bias"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    bound: float = field(default=0.1, metadata={"help": "alpha, the residual bound per action dimension"})
    total_steps: int = 80_000  # env steps per run, warm-up included (same budget for every DAWN cell)
    warmup_steps: int = 20_000  # base-only transitions before the actor moves (0 = no warm-up)
    layernorm: bool = True  # LayerNorm in the critic
    utd: float = 1.0  # gradient updates per env step
    n_step: int = 3
    gamma: float = 0.99
    batch_size: int = 256
    hidden: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    tau: float = 0.005
    policy_delay: int = 2
    explore_sigma: float = 0.3  # Gaussian noise on tanh(u) during training, so actual noise = bound * sigma
    target_sigma: float = 0.2
    target_clip: float = 0.5
    episodes_per_round: int = 8  # collect this many episodes, then do utd * (their steps) updates
    eval_every: int = 10_000  # env steps between search-set evaluations
    early_evals: list[int] = field(default_factory=lambda: [2000, 5000])  # extra ones, online steps after the actor starts
    early_window: int = 8000  # online steps over which early training-episode success is reported
    ablation: bool = True  # the DAWN 2x2 (warm-up x LayerNorm) on v1
    ablation_seeds: list[int] = field(default_factory=lambda: [0, 1])  # for the three non-default cells
    failure_bound: float = 1.0  # one v1 run with a bound large enough to override the base (0 = skip)
    eval_set: str = "eval"  # the held-out set; --quick uses search so smoke tests never see held-out data
    eval_n: int = 256
    n_workers: int = 4  # parallel training processes, and rollout workers for the held-out evals
    torch_threads: int = 1  # per training process
    prior_search_steps: int = 0  # rollouts spent before this run (pilots), charged to every row
    prior_train_steps: int = 0
    out: str = field(default="", metadata={"help": "output root (default: the repo; --quick: runs/ch06_quick)"})
    replot: str = field(default="", metadata={"help": "redraw the plots from this results JSON and exit"})
    quick: bool = False
    QUICK: ClassVar[dict] = {"seeds": [0], "total_steps": 1600, "warmup_steps": 600, "eval_every": 1000,
                             "early_evals": [], "early_window": 1000,
                             "episodes_per_round": 4, "ablation_seeds": [0], "failure_bound": 0.0,
                             "eval_set": "search", "eval_n": 16, "batch_size": 64, "ablation": False}


# ---- The policy: frozen base + bounded residual, all numpy so rollout workers stay light.


@lru_cache(maxsize=1)
def base_model():
    from lastmile.common.policy_flow import FlowChunk

    return FlowChunk.load(BASE)


def base_policy():
    from lastmile.common.policy_flow import FlowChunkPolicy

    return FlowChunkPolicy(base_model(), noise="gaussian")


class NumpyActor:
    """tanh(u(s, a_base)) with a 2x256 ReLU MLP; one row at a time so results never depend on the batch."""

    def __init__(self, w: dict):
        self.layers = [(w[f"{i}.weight"].T.copy(), w[f"{i}.bias"].copy()) for i in (0, 2, 4)]
        model = base_model()
        self.mu, self.sd = model.obs_mean.numpy(), model.obs_std.numpy()

    def __call__(self, obs: np.ndarray, base: np.ndarray) -> np.ndarray:
        x = np.concatenate([(obs - self.mu) / self.sd, base], 1).astype(np.float32)
        out = np.empty((len(x), 4), np.float32)
        for i, row in enumerate(x):
            h = np.repeat(row[None], 2, 0)  # same 2-row shape for every call, as in NumpyFlow
            for j, (w, b) in enumerate(self.layers):
                h = h @ w + b
                h = np.maximum(h, 0) if j < 2 else h
            out[i] = np.tanh(h[0])
        return out


class ResidualPolicy:
    """BatchPolicy: base_v1 (Gaussian noise, chunk of 8, execute 4) plus bound * (tanh(u) + noise).

    ``weights=None`` is the zero residual. Exploration noise comes from each env's own generator, so an
    episode does not depend on its batch. ``last_base`` and ``last_res`` are recorded by rollouts.
    """

    def __init__(self, weights: dict | None, bound: float, sigma: float = 0.0):
        self.base = base_policy()
        self.actor = NumpyActor(weights) if weights is not None else None
        self.bound, self.sigma = bound, sigma
        self.last_base: np.ndarray | None = None
        self.last_res: np.ndarray | None = None

    def reset(self, seeds) -> None:
        self.base.reset(seeds)
        self.rngs = [np.random.default_rng([int(s), 6]) for s in seeds]

    def act(self, obs: np.ndarray) -> np.ndarray:
        base = self.base.act(obs)
        r = np.zeros_like(base) if self.actor is None else self.actor(obs, base)
        if self.sigma > 0:
            r = np.clip(r + self.sigma * np.stack([g.standard_normal(4) for g in self.rngs]), -1, 1)
        self.last_base, self.last_res = base.copy(), (self.bound * r).astype(np.float32)
        return np.clip(base + self.last_res, -1.0, 1.0)


# ---- TD3 pieces (torch), used only inside the training processes.


def make_nets(cfg: Config, layernorm: bool, seed: int):
    """Seeded here, before any layer is built, so the seed fixes every initial weight (and the run)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    n_in = len(base_model().obs_mean) + 4  # (normalized obs, base or summed action)

    def mlp(n_in: int, n_out: int, ln: bool) -> nn.Sequential:
        layers, w = [], n_in
        for _ in range(2):
            layers += [nn.Linear(w, cfg.hidden)] + ([nn.LayerNorm(cfg.hidden)] if ln else []) + [nn.ReLU()]
            w = cfg.hidden
        return nn.Sequential(*layers, nn.Linear(w, n_out))

    actor = mlp(n_in, 4, False)  # (normalized obs, base action) -> u
    nn.init.zeros_(actor[-1].weight)
    nn.init.zeros_(actor[-1].bias)  # tanh(0) = 0: the residual starts at exactly zero
    critics = nn.ModuleList([mlp(n_in, 1, layernorm) for _ in range(2)])  # (normalized obs, summed action)
    return actor, critics, torch


class Replay:
    """Flat n-step transitions. Every episode end is terminal: success ends the task, and the time limit
    is visible in the observation (time_frac), so there is nothing to bootstrap from after it."""

    def __init__(self, cfg: Config):
        self.n, self.gamma = cfg.n_step, cfg.gamma
        self.cols: dict[str, list] = {k: [] for k in ("s", "b", "a", "R", "done", "s2", "b2")}
        mu, sd = base_model().obs_mean.numpy(), base_model().obs_std.numpy()
        self.norm = lambda o: ((o - mu) / sd).astype(np.float32)
        self.size = 0

    def add(self, trajs: list) -> None:
        for tr in trajs:
            T = len(tr["actions"])
            obs = self.norm(np.vstack([tr["obs"], tr["final_obs"][None]]))
            base = np.vstack([tr["last_base"], np.zeros((1, 4), np.float32)])
            t = np.arange(T)
            R = sum(self.gamma**k * np.where(t + k < T, tr["rewards"][np.minimum(t + k, T - 1)], 0.0)
                    for k in range(self.n))
            nxt = np.minimum(t + self.n, T)
            for key, val in (("s", obs[:T]), ("b", base[:T]), ("a", tr["actions"]), ("R", R),
                             ("done", (t + self.n >= T)),
                             ("s2", obs[nxt]), ("b2", base[nxt])):
                self.cols[key].append(np.asarray(val, np.float32))
            self.size += T
        self.arr = {k: np.concatenate(v) for k, v in self.cols.items()}  # rebuilt per round, cheap here

    def sample(self, rng: np.random.Generator, batch: int, torch):
        idx = rng.integers(0, self.size, batch)
        return {k: torch.as_tensor(v[idx]) for k, v in self.arr.items()}


def train_run(cfg: Config, spec: dict, base_search_k: int) -> dict:
    """One residual-TD3 run in its own process. Returns curves, checkpoints and the rollouts it spent."""
    actor, critics, torch = make_nets(cfg, spec["layernorm"], spec["seed"])  # seeds torch's generator
    import copy

    torch.set_num_threads(cfg.torch_threads)
    rng = np.random.default_rng([spec["seed"], spec["run_id"]])
    t0, cpu0 = time.perf_counter(), time.process_time()
    actor_t, critics_t = copy.deepcopy(actor), copy.deepcopy(critics)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = torch.optim.Adam(critics.parameters(), lr=cfg.lr_critic)
    bound, buf, led = spec["bound"], Replay(cfg), Ledger("ch06", spec["name"])  # led is never written
    episodes, n_updates, debt = 0, 0, 0.0
    weights = lambda: {k: v.detach().numpy().copy() for k, v in actor.state_dict().items()}

    def collect(w: dict | None, sigma: float) -> dict:
        nonlocal episodes
        ids = range(episodes, episodes + cfg.episodes_per_round)
        seeds = [TRAIN_SEED_BASE + 50_000 * spec["run_id"] + i for i in ids]
        pseeds = [policy_seed(f"ch06_train_{spec['run_id']}", i) for i in ids]
        res = rollout_seeds(partial(ResidualPolicy, w, bound, sigma), seeds, pseeds, robot=cfg.robot,
                            variant=spec["variant"], n_workers=1, shard_size=len(seeds),
                            record_trajectories=True, ledger=led, category="train")
        episodes += len(seeds)
        buf.add(res.trajectories)
        res_mag = float(np.mean([np.abs(tr["last_res"]).mean() for tr in res.trajectories]))
        return {"steps": led.robot_steps["train"], "k": res.k, "n": res.n, "res_mag": res_mag,
                "online": w is not None}

    def update(train_actor: bool) -> None:
        nonlocal n_updates
        b = buf.sample(rng, cfg.batch_size, torch)
        with torch.no_grad():
            eps = (torch.randn(cfg.batch_size, 4) * cfg.target_sigma).clamp(-cfg.target_clip, cfg.target_clip)
            r2 = bound * (torch.tanh(actor_t(torch.cat([b["s2"], b["b2"]], 1))) + eps).clamp(-1, 1)
            x2 = torch.cat([b["s2"], (b["b2"] + r2).clamp(-1, 1)], 1)
            q2 = torch.min(critics_t[0](x2), critics_t[1](x2)).squeeze(1)
            y = (b["R"] + cfg.gamma**cfg.n_step * (1 - b["done"]) * q2).clamp(0, 1)
        x = torch.cat([b["s"], b["a"]], 1)
        loss_c = sum(((q(x).squeeze(1) - y) ** 2).mean() for q in critics)
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()
        n_updates += 1
        if train_actor and n_updates % cfg.policy_delay == 0:
            # The actor proposes a new residual on top of the base action that was recorded in state s.
            a_pi = (b["b"] + bound * torch.tanh(actor(torch.cat([b["s"], b["b"]], 1)))).clamp(-1, 1)
            loss_a = -critics[0](torch.cat([b["s"], a_pi], 1)).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
            with torch.no_grad():
                for net, tgt in ((actor, actor_t), (critics, critics_t)):
                    torch._foreach_lerp_(list(tgt.parameters()), list(net.parameters()), cfg.tau)
        elif not train_actor:
            with torch.no_grad():
                torch._foreach_lerp_(list(critics_t.parameters()), list(critics.parameters()), cfg.tau)

    def search_eval(w: dict | None) -> int:
        res = evaluate(partial(ResidualPolicy, w, bound), "search", robot=cfg.robot, variant=spec["variant"],
                       n_workers=1, shard_size=32, ledger=led, category="search")
        return res.k

    rounds, curve, ckpts = [], [], []
    while led.robot_steps["train"] < spec["warmup"]:  # 1. warm-up: pure base data, critic only
        rounds.append(collect(None, 0.0))
    for _ in range(int(cfg.utd * led.robot_steps["train"])):
        update(train_actor=False)
    # The actor has not moved (it is still exactly zero), so these points are the base: reuse its search score.
    for x in sorted({0, led.robot_steps["train"]}):
        curve.append([x, base_search_k, 64])
        ckpts.append(None)
    start = led.robot_steps["train"]  # search evals: early ones after the actor starts, then every eval_every
    marks = sorted({start + e for e in cfg.early_evals} | set(range(start + cfg.eval_every, spec["total"],
                                                                     cfg.eval_every)))
    while led.robot_steps["train"] < spec["total"]:  # 2. online: collect with noise, then update
        before = led.robot_steps["train"]
        rounds.append(collect(weights(), cfg.explore_sigma))
        debt += cfg.utd * (led.robot_steps["train"] - before)
        while debt >= 1 and buf.size >= cfg.batch_size:
            update(train_actor=True)
            debt -= 1
        steps = led.robot_steps["train"]
        if (marks and steps >= marks[0]) or steps >= spec["total"]:
            w = weights()
            curve.append([steps, search_eval(w), 64])
            ckpts.append(w)
            marks = [m for m in marks if m > steps]
    return {"spec": spec, "curve": curve, "ckpts": ckpts, "rounds": rounds, "updates": n_updates,
            "robot_steps": dict(led.robot_steps), "cpu_seconds": time.process_time() - cpu0,
            "wall_minutes": (time.perf_counter() - t0) / 60}


def initial_weights(cfg: Config) -> dict:
    """The actor exactly as training starts (zero last layer), for the zero-residual check."""
    actor, _, _ = make_nets(cfg, cfg.layernorm, cfg.seeds[0])
    return {k: v.detach().numpy().copy() for k, v in actor.state_dict().items()}


def run_specs(cfg: Config) -> list[dict]:
    """Main runs (every variant x seed), the DAWN 2x2 cells on v1, and the large-bound failure run."""
    specs: list[dict] = []

    def add(name: str, variant: str, seed: int, bound: float, warmup: int, ln: bool, group: str) -> None:
        specs.append({"name": name, "variant": variant, "seed": seed, "bound": bound, "warmup": warmup,
                      "layernorm": ln, "total": cfg.total_steps, "group": group, "run_id": len(specs)})

    for v in cfg.variants:
        for s in cfg.seeds:
            add(f"{v}_s{s}", v, s, cfg.bound, cfg.warmup_steps, cfg.layernorm, "main")
    if cfg.ablation and "v1" in cfg.variants:
        default = (cfg.warmup_steps > 0, cfg.layernorm)
        for warm in (True, False):
            for ln in (True, False):
                if (warm, ln) != default:
                    for s in cfg.ablation_seeds:
                        add(f"v1_warm{int(warm)}_ln{int(ln)}_s{s}", "v1", s, cfg.bound,
                            (cfg.warmup_steps or 20_000) if warm else 0, ln, "ablation")
    if cfg.failure_bound > 0 and "v1" in cfg.variants:
        add(f"v1_bound{cfg.failure_bound:g}_s{cfg.seeds[0]}", "v1", cfg.seeds[0], cfg.failure_bound,
            cfg.warmup_steps, cfg.layernorm, "failure")
    return specs


def chosen_index(curve: list) -> int:
    """Checkpoint with the best search score; ties go to the later one (more training)."""
    return max(range(len(curve)), key=lambda i: (curve[i][1], i))


def dawn_cells(cfg: Config, runs: dict) -> dict:
    """(warm-up on, LayerNorm on) -> names of the v1 runs at the default bound in that cell."""
    cells: dict = {}
    for n, r in runs.items():
        sp = r["spec"]
        if sp["variant"] == "v1" and sp["bound"] == cfg.bound and sp["group"] in ("main", "ablation"):
            cells.setdefault((sp["warmup"] > 0, sp["layernorm"]), []).append(n)
    return {key: sorted(v) for key, v in sorted(cells.items(), reverse=True)}


def online_rounds(run: dict) -> list:
    """Training rounds after the actor starts, with x = online env steps since it started."""
    start = max([r["steps"] for r in run["rounds"] if not r["online"]], default=0)
    return [{**r, "x": r["steps"] - start} for r in run["rounds"] if r["online"]]


def cell_label(key) -> str:
    return f"warm-up {'on' if key[0] else 'off'}, LayerNorm {'on' if key[1] else 'off'}"


# ---- Plots and GIFs


def make_plots(cfg: Config, media: Path, s: dict) -> None:
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    plotting.setup()
    P = plotting.PALETTE
    runs = s["runs"]
    final_label = "search" if cfg.eval_set == "search" else f"held-out {cfg.eval_set}"

    def curve(ax, name, color, label=None):
        c = np.array(runs[name]["curve"])
        plotting.success_curve(ax, c[:, 0] / STEPS_PER_MIN, c[:, 1], c[:, 2], label=label, color=color, marker=".")

    def finish(ax, variant, title):
        b = s["base"][variant]["search"]
        ax.axhline(b[0] / b[1], ls="--", color="gray", lw=1.2, label=f"base_v1, search {b[0]}/{b[1]}")
        ax.set(xlabel="training robot-minutes (warm-up included)", ylabel="success (search, n=64)",
               title=title)
        ax.set_ylim(0, 1.02)

    # 1. Learning curves on both variants, per seed, with the held-out evals at the right.
    variants = [v for v in cfg.variants if any(r["spec"]["variant"] == v and r["spec"]["group"] == "main"
                                               for r in runs.values())]
    fig, axes = plt.subplots(1, len(variants), figsize=(6.0 * len(variants), 4.4), squeeze=False)
    for ax, v in zip(axes[0], variants):
        names = [n for n, r in runs.items() if r["spec"]["group"] == "main" and r["spec"]["variant"] == v]
        for i, n in enumerate(names):
            curve(ax, n, P[i % 4], label=f"seed {runs[n]['spec']['seed']}")
        xmax = cfg.total_steps / STEPS_PER_MIN
        if cfg.warmup_steps:
            ax.axvline(cfg.warmup_steps / STEPS_PER_MIN, color="k", lw=0.8, ls=":")
            ax.text(cfg.warmup_steps / STEPS_PER_MIN, 0.04, " actor starts", fontsize=8)
        e = s["eval"][v]
        plotting.heldout_point(ax, xmax * 1.06, *e["pooled"], label=f"{final_label}, residual (pooled)",
                               color=P[1])
        plotting.heldout_point(ax, xmax * 1.06, *e["base"], label=f"{final_label}, base_v1", color="black")
        finish(ax, v, f"Residual TD3 on CupDrop {v}, bound {cfg.bound:g}")
        ax.legend(loc="lower right", fontsize=8)
    plotting.save_fig(fig, media / "learning_curves.png")

    # 2. DAWN 2x2: warm-up on/off x LayerNorm on/off (v1).
    cells = dawn_cells(cfg, runs)
    if len(cells) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True, sharey=True)
        for ax, (key, names) in zip(axes.flat, cells.items()):
            for k, n in enumerate(names):
                curve(ax, n, P[k % 4], label=f"seed {runs[n]['spec']['seed']}")
            b = s["base"]["v1"]["search"]
            ax.axhline(b[0] / b[1], ls="--", color="gray", lw=1.2)
            ax.set_title(cell_label(key), fontsize=10)
            ax.set_ylim(0, 1.02)
            ax.legend(loc="lower right", fontsize=7)
        for ax in axes[1]:
            ax.set_xlabel("training robot-minutes")
        for ax in axes[:, 0]:
            ax.set_ylabel("success (search, n=64)")
        fig.suptitle("DAWN ablation on CupDrop v1 (dashed: base_v1 on search)")
        plotting.save_fig(fig, media / "dawn_grid.png")

        # 2b. What the robot does while it trains: training-episode success after the actor starts,
        # pooled over each cell's seeds in 2000-step bins (exploration noise included).
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for c, (key, names) in enumerate(cells.items()):
            bins: dict = {}
            for n in names:
                for r in online_rounds(runs[n]):
                    kn = bins.setdefault((r["x"] - 1) // 2000, [0, 0])
                    kn[0], kn[1] = kn[0] + r["k"], kn[1] + r["n"]
            idx = sorted(i for i in bins if i < 15)
            x = (np.array(idx) + 0.5) * 2000 / STEPS_PER_MIN
            plotting.success_curve(ax, x, np.array([bins[i][0] for i in idx]), np.array([bins[i][1] for i in idx]),
                                   label=f"{cell_label(key)} ({len(names)} seeds)", color=P[c % 4], marker=".")
        w = s["early"]["warmup_base"]["v1"]
        ax.axhline(w[0] / w[1], ls="--", color="gray", lw=1.2, label=f"base_v1 in the v1 warm-up episodes, {w[0]}/{w[1]}")
        ax.set(xlabel="robot-minutes since the actor started acting", ylabel="training-episode success",
               title="Success while training, v1 (Wilson bands, 2000-step bins)")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=8)
        plotting.save_fig(fig, media / "dawn_early.png")

    # 3. A large bound against the default bound on the same seed.
    fail = [n for n, r in runs.items() if r["spec"]["group"] == "failure"]
    ref = f"v1_s{cfg.seeds[0]}"
    if fail and ref in runs:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        for name, color, label in ((ref, P[0], f"bound {cfg.bound:g}"), (fail[0], P[1], f"bound {cfg.failure_bound:g}")):
            curve(axes[0], name, color, label=label)
            rd = np.array([[r["steps"], r["k"] / r["n"], r["res_mag"]] for r in runs[name]["rounds"]])
            axes[1].plot(rd[:, 0] / STEPS_PER_MIN, rd[:, 2], color=color, label=label)
        finish(axes[0], "v1", f"Search success, bound {cfg.bound:g} vs {cfg.failure_bound:g}")
        axes[0].legend(loc="lower left", fontsize=8)
        axes[1].set(xlabel="training robot-minutes", ylabel="mean |residual| per action dim\n(training, noise included)",
                    title="How far the add-on moves the base")
        axes[1].legend(fontsize=8)
        plotting.save_fig(fig, media / "bound_compare.png")


def make_gif(cfg: Config, media: Path, variant: str, base_trajs: list, res_trajs: list) -> int | None:
    """Same held-out start state side by side: the base fails, base + residual succeeds."""
    from lastmile.common.plotting import replay_frames, save_gif, side_by_side
    from lastmile.envs.cupdrop import CupDropEnv

    pairs = [(b, r) for b, r in zip(base_trajs, res_trajs) if not b["success"] and r["success"]]
    if not pairs:
        return None
    b, r = pairs[0]
    env = CupDropEnv(robot=cfg.robot, variant=variant)
    clips = [replay_frames(env, tr, text, every=3)
             for tr, text in ((b, f"base_v1, {variant} (seed {b['seed']})"), (r, f"base + residual, {variant}"))]
    save_gif(side_by_side(*clips), media / f"{variant}_base_vs_residual.gif", fps=5, max_size=400)
    return int(b["seed"])


# ---- The run


def main(cfg: Config) -> None:
    from concurrent.futures import as_completed

    if cfg.replot:  # every plot is drawn from the summary stored in each results file
        import json

        s = json.loads((REPO_ROOT / cfg.replot).read_text())["extra"]
        saved_cfg = Config(**{k: v for k, v in s["config"].items() if k != "replot"})
        out = REPO_ROOT / (saved_cfg.out or ("runs/ch06_quick" if saved_cfg.quick else "."))
        make_plots(saved_cfg, out / "media/ch06", s)
        return

    out = REPO_ROOT / (cfg.out or ("runs/ch06_quick" if cfg.quick else "."))
    media = out / "media/ch06"
    wall0 = time.perf_counter()
    ev = partial(evaluate, robot=cfg.robot, n_workers=cfg.n_workers)
    specs = run_specs(cfg)
    s: dict = {"base": {}, "runs": {}, "eval": {}, "config": asdict(cfg)}

    with ExitStack() as stack:
        rows = {sp["name"]: stack.enter_context(Ledger("ch06", f"residual-td3_{sp['name']}", robot=cfg.robot,
                                                       variant=sp["variant"], config=cfg,
                                                       results_root=out / "results"))
                for sp in specs if sp["group"] == "main"}

        # 1. The base on each variant: search (the first point of every curve) and held-out, both salt 0.
        base_search, base_eval, shared_search = {}, {}, {}
        for v in cfg.variants:
            base_search[v] = ev(base_policy, "search", variant=v, category="search")
            base_eval[v] = ev(base_policy, cfg.eval_set, variant=v, n=cfg.eval_n, category="eval",
                              record_trajectories=True)
            shared_search[v] = base_search[v].env_steps
            s["base"][v] = {"search": [base_search[v].k, base_search[v].n], "eval": [base_eval[v].k, base_eval[v].n]}
            print(f"base_v1 on {v}: search {base_search[v].summary}, {cfg.eval_set} {base_eval[v].summary}", flush=True)
        # The zero-initialized actor must reproduce the base episode for episode.
        v0 = cfg.variants[0]
        zero = ev(partial(ResidualPolicy, initial_weights(cfg), cfg.bound), "search", variant=v0, category="search")
        assert np.array_equal(zero.successes, base_search[v0].successes) and zero.env_steps == base_search[v0].env_steps
        shared_search[v0] += zero.env_steps
        print("zero-initialized residual reproduces the base on search, episode for episode", flush=True)
        shutdown_pool()  # free the rollout workers while the training processes run

        # 2. Train every run, n_workers processes at a time (each steps its envs in-process).
        runs = {}
        with ProcessPoolExecutor(cfg.n_workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(train_run, cfg, sp, base_search[sp["variant"]].k) for sp in specs]
            for fut in as_completed(futures):
                r = fut.result()
                runs[r["spec"]["name"]] = r
                c = r["curve"]
                print(f"[{(time.perf_counter() - wall0) / 60:5.1f} min] {r['spec']['name']}: search curve "
                      f"{[k for _, k, _ in c]}/64, {r['wall_minutes']:.1f} min", flush=True)
        runs = {sp["name"]: runs[sp["name"]] for sp in specs}  # spec order

        # 3. Choose a checkpoint per run on search; score main runs once on the held-out set.
        finals, extra_evals, fail_eval_steps = {}, {}, 0
        for name, r in runs.items():
            sp, i = r["spec"], chosen_index(r["curve"])
            s["runs"][name] = {"spec": sp, "curve": r["curve"], "rounds": r["rounds"], "updates": r["updates"],
                               "chosen": {"index": i, "train_steps": r["curve"][i][0], "search": r["curve"][i][1:]},
                               "robot_steps": r["robot_steps"], "wall_minutes": r["wall_minutes"]}
            if sp["group"] == "main":
                make = partial(ResidualPolicy, r["ckpts"][i], sp["bound"])
                finals[name] = ev(make, cfg.eval_set, variant=sp["variant"], n=cfg.eval_n, ledger=rows[name],
                                  category="eval", record_trajectories=sp["seed"] == cfg.seeds[0])
            elif sp["group"] == "failure":  # what deploying it would do, chosen on search and unchosen
                for tag, j in (("chosen", i), ("last", len(r["curve"]) - 1)):
                    res = ev(partial(ResidualPolicy, r["ckpts"][j], sp["bound"]), cfg.eval_set, variant="v1",
                             n=cfg.eval_n, category="eval")
                    fail_eval_steps += res.env_steps  # not a final evaluation: charged to every row as search
                    b0 = base_eval["v1"].successes
                    extra_evals[f"{name}_{tag}"] = {"k": res.k, "n": res.n, "train_steps": r["curve"][j][0],
                                                    "mcnemar_p": mcnemar_exact(b0, res.successes),
                                                    "fixed": int((~b0 & res.successes).sum()),
                                                    "broke": int((b0 & ~res.successes).sum())}
        s["failure_eval"] = extra_evals

        # The robot's view of training: training-episode success over the first early_window online steps.
        s["early"] = {"warmup_base": {}, "per_run": {}, "cells": {}}  # warmup_base: pure-base training episodes
        for v in cfg.variants:
            warm = [r for x in runs.values() if x["spec"]["variant"] == v for r in x["rounds"] if not r["online"]]
            s["early"]["warmup_base"][v] = [sum(r["k"] for r in warm), sum(r["n"] for r in warm)]
        for n, r in runs.items():
            sel = [x for x in online_rounds(r) if x["x"] <= cfg.early_window]
            s["early"]["per_run"][n] = [sum(x["k"] for x in sel), sum(x["n"] for x in sel)]
        for key, names in dawn_cells(cfg, runs).items():
            s["early"]["cells"][cell_label(key)] = [sum(s["early"]["per_run"][n][i] for n in names) for i in (0, 1)]

        # 4. Paired statistics, failure modes, costs.
        from lastmile.envs.cupdrop import CupDropEnv, OutcomeTracker

        for v in cfg.variants:
            names = [n for n in finals if runs[n]["spec"]["variant"] == v]
            b0 = base_eval[v].successes
            e = s["eval"][v] = {"base": [base_eval[v].k, base_eval[v].n], "per_seed": {}, "mcnemar_p": {},
                                "diff_ci": {}, "paired": {},
                                "pooled": [sum(finals[n].k for n in names), sum(finals[n].n for n in names)]}
            for n in names:
                f = finals[n].successes
                e["per_seed"][n] = [finals[n].k, finals[n].n]
                e["mcnemar_p"][n] = mcnemar_exact(b0, f)
                e["diff_ci"][n] = bootstrap_diff(b0, f)
                e["paired"][n] = {"fixed": int((~b0 & f).sum()), "broke": int((b0 & ~f).sum()),
                                  "both_fail": int((~b0 & ~f).sum())}
            env = CupDropEnv(robot=cfg.robot, variant=v)
            first = finals[f"{v}_s{cfg.seeds[0]}"]
            e["failure_modes"] = {}
            for tag, r in (("base", base_eval[v]), (f"residual_s{cfg.seeds[0]}", first)):
                labels = [OutcomeTracker.label_trajectory(env, t) for t in r.trajectories]
                e["failure_modes"][tag] = {m: labels.count(m) for m in sorted(set(labels))}

        # Every row pays for its own run, the shared base runs, all ablation and large-bound runs (and the
        # large-bound eval passes) of this experiment, and the earlier pilot and superseded runs.
        side = {n: r["robot_steps"] for n, r in runs.items() if r["spec"]["group"] != "main"}
        side_train = sum(c["train"] for c in side.values())
        side_search = sum(c["search"] for c in side.values()) + fail_eval_steps
        s["cost_steps"] = {"shared_search": shared_search, "ablation_and_failure_runs": side,
                           "failure_eval_passes": fail_eval_steps,
                           "prior": {"search": cfg.prior_search_steps, "train": cfg.prior_train_steps}}
        for name, row in rows.items():
            r, v = runs[name], runs[name]["spec"]["variant"]
            row.robot_steps["train"] += r["robot_steps"]["train"] + side_train + cfg.prior_train_steps
            row.robot_steps["search"] += (r["robot_steps"]["search"] + shared_search[v] + side_search
                                          + cfg.prior_search_steps)
            row.robot_steps["eval"] += base_eval[v].env_steps
            row.worker_cpu_seconds += r["cpu_seconds"]
            row.set_base(id="base_v1", successes=base_eval[v])
            row.set_final(successes=finals[name])
            s["cost_steps"][name] = dict(row.robot_steps)

        s["gif_seed"] = {}
        with optional_media("the base vs residual GIFs"):
            for v in cfg.variants:
                s["gif_seed"][v] = make_gif(cfg, media, v, base_eval[v].trajectories,
                                            finals[f"{v}_s{cfg.seeds[0]}"].trajectories)
        make_plots(cfg, media, s)
        for row in rows.values():
            row.extra = s

    print(f"\n=== Chapter 6 summary ({cfg.eval_set}, n={cfg.eval_n}, bound {cfg.bound:g}) ===")
    for v in cfg.variants:
        e = s["eval"][v]
        print(f"[{v}] base_v1: {format_rate(*e['base'])}")
        for n in e["per_seed"]:
            ch = s["runs"][n]["chosen"]
            print(f"[{v}] {n}: {format_rate(*e['per_seed'][n])}  McNemar p={e['mcnemar_p'][n]:.2g}  "
                  f"{e['paired'][n]}  chosen at {ch['train_steps'] / STEPS_PER_MIN:.0f} train robot-min "
                  f"(search {ch['search'][0]}/64)")
        print(f"[{v}] pooled: {format_rate(*e['pooled'])}  failure modes: {e['failure_modes']}")
    for n, r in s["runs"].items():
        c = r["curve"][-1]
        print(f"{n}: search curve {[k for _, k, _ in r['curve']]}/64, last {format_rate(c[1], c[2])}; "
              f"training episodes in the first {cfg.early_window} online steps {format_rate(*s['early']['per_run'][n])}")
    for v, kn in s["early"]["warmup_base"].items():
        print(f"warm-up training episodes on {v} (pure base): {format_rate(*kn)}")
    for cell, kn in s["early"]["cells"].items():
        print(f"early training success, {cell}: {format_rate(*kn)}")
    for n, f in s["failure_eval"].items():
        print(f"{n}: {format_rate(f['k'], f['n'])}  McNemar p={f['mcnemar_p']:.2g}  fixed {f['fixed']} broke {f['broke']}")
    for name, row in rows.items():
        print(f"row {name} robot-min:", {c: round(m, 1) for c, m in row.robot_minutes.items()})
    print(f"wall {(time.perf_counter() - wall0) / 60:.1f} min; results in {out / 'results/ch06'}")


if __name__ == "__main__":
    main(parse(Config))
