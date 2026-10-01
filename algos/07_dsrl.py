"""Chapter 7: DSRL. Reinforcement learning in the noise space of a frozen flow policy.

Learning goal
    Turn Chapter 3's golden ticket (one fixed noise vector) into a state-dependent policy: RL picks the noise,
    the frozen base_v1 decodes it into actions. Measure what that buys over the best fixed ticket, what it
    costs in robot-minutes, whether it collapses behavior modes, and where it stops: on start states the base
    cannot solve, steering has nothing to select, and only something that edits actions (a residual) can move.

The base idea
    FlowChunk-S maps (observation s, noise w in R^32) to a chunk of 8 actions with 10 deterministic Euler steps.
    DSRL treats w as the action of a new MDP, the *latent-noise MDP*: at every chunk decision a small actor
    outputs w = pi_W(s), the frozen sampler decodes chunk = flow(s, w), 4 actions run, and the reward is task
    success (1 on the success step, else 0). We train pi_W with SAC on that MDP; the critic sees (s, w):

        w = c * tanh(u),  u ~ N(mu_phi(s), sigma_phi(s)^2)                  (a bounded box, c = noise_scale)
        critic:  Q_j(s_t, w_t) -> G_t = gamma^(T - t) * success              (Monte Carlo return, j = 1, 2)
                 (or the n-step SAC target r + ... + gamma^n [min_j Q'_j(s', w') - alpha log pi(w'|s')], --n-step n)
        actor:   max E_{w ~ pi(.|s)} [min_j Q_j(s, w) - alpha * log pi(w|s)]   (alpha fixed)

    The actor's mean head starts at zero, so before any training the deterministic policy is exactly the zero
    ticket of Chapter 3 (75.8% on eval), and its exploration (sigma = 1) is close to the Gaussian base. The
    first warmup_episodes train only the critic. A fixed ticket is the special case pi_W(s) = w0 for every s;
    DSRL is its state-dependent generalization.
    Arms (the full run trains the grid; one flag runs one arm):
      dsrl            all 32 noise dimensions.
      pss             --subspace k: steer only the k noise directions that move the decoded chunk most, found
                      by finite-difference probing of the frozen network at 64 base-visited states (w = c V_k z).
      hybrid          PSS steering plus a bounded residual, as in RFS: chunk = clip(flow(s, w) + beta * tanh(v)),
                      v in R^4 added to all 8 actions, so the policy can leave the base's support. --hybrid alone
                      gives the full-32 version (rfs).
      residual        --residual-only: the residual alone on top of the zero ticket (w = 0), no steering.
    Support limit (CONTRIBUTING rule 8): steering only reweights chunks the base can already produce. A noise
    actor re-picks at every chunk, so it can beat any finite pass@k, but it cannot solve a start from which no
    sequence of base chunks succeeds. On the "hard" variant (smaller cup, cube region 1.5x wider, many starts
    outside the demo region) the script groups held-out states by how often the Gaussian base solves them in 16
    tries and compares steering (pss) with the arms that can edit actions (hybrid, residual).

Papers mirrored (ids from data/papers.csv)
    C120 DSRL (2025-06): SAC over the input noise of a frozen diffusion/flow policy.
    C413 PSS (2026-09): steer only in a top-k principal steering subspace found by finite differences.
    C228 RFS (2026-02): one policy outputs both the noise and an additive residual.
    C324 SCORE (2026-06): support-constrained steering; the base's support is the ceiling.
    C304 BMD (2026-05): warns that noise steering collapses behavior modes; we measure grasp modes.
    C236 Golden Ticket (2026-03): the state-independent special case (Chapter 3).

Honesty
    Training episodes use their own env seeds (TRAIN_SEED_BASE + ...), charged to "train". Checkpoints are
    scored on the search set (charged to "search") and each run keeps the best one (ties: the later one).
    The held-out set (eval for v1, stress for hard; n = 256) is scored once per run. Baselines on both
    variants: the Gaussian base, the zero ticket and an untrained random state-dependent actor (plus Chapter
    3's tickets on v1). The 14 pilot runs that chose the recipe (PILOT_ROBOT_MINUTES, runs/ch07_pilots/) are
    charged to every trained row and to the headline row. Each run's chosen actor (weights + steering basis)
    is saved to runs/ch07_full/actors/<method>.npz (load_steer rebuilds the policy).

Length
    About 850 lines, over CONTRIBUTING's 150-450 guideline: an explicit exception. The algorithm (policy, SAC,
    PSS probe, training loop) is lines ~160-450; the rest is the chapter's measurements (support groups, grasp
    modes, random-selection controls, paired tests) and their plots, which the rules also require.

Run
    uv run python algos/07_dsrl.py --quick            # smoke test, < 2 min, writes only to runs/ch07_quick/
    uv run python algos/07_dsrl.py                    # full grid (4 v1 arms + 3 hard arms) x 3 seeds, ~40 min
    uv run python algos/07_dsrl.py --subspace 8       # one arm only (also --hybrid, ...): writes to runs/ch07_arm/
    uv run python algos/07_dsrl.py --replot results/ch07/noise-steering-best_so101_<stamp>.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from lastmile.common.cli import parse
from lastmile.common.eval import bootstrap_diff, format_rate, mcnemar_exact, policy_seed, wilson
from lastmile.common.ledger import DEFAULT_RESULTS_ROOT, REPO_ROOT, Ledger
from lastmile.common.plotting import optional_media
from lastmile.common.policy_flow import FlowChunk, FlowChunkPolicy
from lastmile.common.rollout import evaluate, rollout_seeds
from lastmile.envs.cupdrop import obs_layout

CKPT = str(REPO_ROOT / "checkpoints" / "base_v1_so101.pt")
MEDIA = REPO_ROOT / "media" / "ch07"
DIM = 32  # noise size: 8 actions x 4 dims
TRAIN_SEED_BASE = 110_000  # + 50_000 for hard, + 10_000 per seed; far from search/eval/eval_ext/stress
HELDOUT = {"v1": "eval", "hard": "stress"}  # DESIGN.md section 3: stress is the held-out set for "hard"
LOG_STD = (-5.0, 1.0)
# The 14 pilot runs made while choosing the critic target, temperature, box and residual size (search set only,
# seed 0, before the full run): the sum of runs/ch07_pilots/pilot_costs.jsonl, listed in docs/chapters/ch07.md.
# Charged to every trained row (v1 and hard) and to the headline row (--no-pilot-costs turns it off).
PILOT_ROBOT_MINUTES = {"search": 735.24, "train": 1440.42}


@dataclass
class Config:
    robot: str = "so101"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    arms: list[str] = field(default_factory=lambda: ["dsrl", "pss", "hybrid", "residual"],
                            metadata={"help": "arms trained on v1: dsrl, pss, hybrid, residual, rfs"})
    hard_arms: list[str] = field(default_factory=lambda: ["pss", "hybrid", "residual"],
                                 metadata={"help": "arms trained on the hard variant (support-limit experiment)"})
    subspace: int = field(default=0, metadata={"help": "one arm only: steer in the top-k PSS directions"})
    hybrid: bool = field(default=False, metadata={"help": "one arm only: noise steering + bounded residual (RFS)"})
    residual_only: bool = field(default=False, metadata={"help": "one arm only: residual on the zero ticket"})
    variant: str = "v1"  # variant for a one-arm run
    pss_k: int = 8  # k of the grid's pss and hybrid arms (fixed in advance, not tuned)
    noise_scale: float = 1.5  # c: the actor steers w in the box [-c, c]^32
    res_scale: float = 0.2  # beta: bound of the residual, a 4-dim offset added to every action of the chunk
    train_episodes: int = 512
    episodes_per_round: int = 64  # collect this many episodes with the current actor, then update
    utd: float = 0.5  # gradient updates per new transition (one transition = one chunk decision)
    eval_every: int = 128  # training episodes between search-set checkpoints
    hidden: int = 256
    batch_size: int = 256
    lr: float = 3e-4  # critic and temperature
    actor_lr: float = 1e-4
    gamma: float = 0.97  # per chunk decision (4 env steps)
    n_step: int = 0  # critic target: n-step TD with a target network; 0 = Monte Carlo return (no bootstrap)
    tau: float = 0.005
    alpha: float = 0.003  # entropy temperature, fixed (see the chapter doc for what auto-tuning did)
    tune_alpha: bool = False  # SAC's automatic temperature with target entropy -d, starting from alpha
    warmup_episodes: int = 128  # critic-only updates on the first episodes (the actor stays at the zero ticket)
    search_n: int = 64
    eval_n: int = 256
    support_salts: int = 16  # Gaussian tries per hard held-out state, to group states by base support
    pss_states: int = 64
    n_workers: int = 4
    pilot_costs: bool = True  # charge PILOT_ROBOT_MINUTES to every trained row and the headline row
    media: bool = True
    replot: str = ""
    quick: bool = False
    QUICK: ClassVar[dict] = {"seeds": [0], "arms": ["pss"], "hard_arms": ["hybrid"], "train_episodes": 64,
                             "episodes_per_round": 32, "eval_every": 32, "warmup_episodes": 32, "search_n": 16, "eval_n": 32,
                             "support_salts": 2, "pss_states": 8, "pilot_costs": False}


RES_DIM = 4  # the residual: one bounded offset per action dimension, added to all 8 actions of the chunk


def make_arm(k: int, residual: bool) -> dict:
    """An arm = how many noise directions the actor steers (32 = all, 0 = none) and whether it adds a residual."""
    name = {(DIM, False): "dsrl", (DIM, True): "rfs", (0, True): "residual"}.get(
        (k, residual), "hybrid" if residual else "pss")
    method = "residual" if k == 0 else "dsrl" + (f"-pss{k}" if k < DIM else "") + ("-res" if residual else "")
    return {"name": name, "k": k, "residual": residual, "method": method}


def grid_arm(name: str, cfg: Config) -> dict:
    k, res = {"dsrl": (DIM, False), "pss": (cfg.pss_k, False), "hybrid": (cfg.pss_k, True),
              "residual": (0, True), "rfs": (DIM, True)}[name]
    return make_arm(k, res)


# ---------------------------------------------------------------------------- the policy (rollout workers)


def actor_forward(P: dict, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numpy copy of the torch Actor below: normalized obs -> (mean, log_std) of the pre-tanh latent."""
    h = np.clip((obs - P["mu"]) / P["sd"], -10, 10).astype(np.float32)
    for w, b in P["trunk"]:
        h = np.maximum(h @ w + b, 0.0)
    return h @ P["wm"] + P["bm"], np.clip(h @ P["ws"] + P["bs"], *LOG_STD)


class SteerPolicy(FlowChunkPolicy):
    """Frozen base_v1 driven by a noise actor through FlowChunkPolicy's callable-noise hook.

    The actor's output a in [-1, 1]^d is split into a steering part (decoded to w = c * V a_steer, with V the
    identity or the PSS basis; no steering means w = 0) and, for hybrid/residual arms, a 4-dim residual
    beta * a_res added to every action of the decoded chunk.
    ``last_latent`` (the executed a) is what the replay buffer stores. Stochastic mode samples
    u ~ N(mu, sigma^2) from env i's own generator, so episodes do not depend on their batch.
    """

    def __init__(self, model, P: dict, basis, residual: bool, noise_scale: float, res_scale: float,
                 stochastic: bool):
        super().__init__(model, noise=self._noise)
        self.P, self.stochastic, self.basis, self.residual = P, stochastic, basis, residual
        self.c, self.beta = noise_scale, res_scale
        self.last_latent: np.ndarray | None = None

    def _noise(self, obs: np.ndarray, rngs: list) -> np.ndarray:
        mean, log_std = actor_forward(self.P, obs)
        if self.stochastic:
            eps = np.stack([rng.standard_normal(mean.shape[1], dtype=np.float32) for rng in rngs])
            mean = mean + np.exp(log_std) * eps
        self.last_latent = np.tanh(mean).astype(np.float32)
        z = self.last_latent[:, : self.basis.shape[1]]  # no steering: basis [32, 0], so w = 0 (zero ticket)
        return self.c * z @ self.basis.T

    def act(self, obs: np.ndarray) -> np.ndarray:
        replan = self.t % self.execute == 0
        action = super().act(obs)  # on a replan this calls _noise, then decodes the chunk
        if replan and self.residual:
            delta = self.beta * self.last_latent[:, None, -RES_DIM:]
            self.last_chunk = np.clip(self.last_chunk + delta, -1.0, 1.0)
            action = self.last_chunk[:, 0]
        return action


def steer_policy(P: dict, basis, residual: bool, noise_scale: float, res_scale: float, stochastic: bool):
    """Picklable factory (use with functools.partial)."""
    return SteerPolicy(FlowChunk.load(CKPT), P, basis, residual, noise_scale, res_scale, stochastic)


def fixed(w) -> partial:
    return partial(FlowChunkPolicy.load, CKPT, noise="fixed", ticket=np.asarray(w, dtype=np.float32))


GAUSS = partial(FlowChunkPolicy.load, CKPT)  # the unsteered base: fresh N(0, I) noise per chunk


def save_steer(path: Path, P: dict, basis: np.ndarray, residual: bool, cfg: Config) -> None:
    """Everything needed to rebuild a chosen (deterministic) steering policy, in one .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trunk = {f"trunk{i}_{n}": a for i, wb in enumerate(P["trunk"]) for n, a in zip("wb", wb)}
    np.savez(path, **trunk, **{k: v for k, v in P.items() if k != "trunk"}, basis=basis, residual=residual,
             noise_scale=cfg.noise_scale, res_scale=cfg.res_scale)


def load_steer(path) -> partial:
    """Inverse of save_steer: a picklable factory for the deterministic policy (e.g. for evaluate)."""
    z = dict(np.load(path))
    P = {k: z[k] for k in ("mu", "sd", "wm", "bm", "ws", "bs")}
    P["trunk"] = [(z[f"trunk{i}_w"], z[f"trunk{i}_b"]) for i in range(2)]
    return partial(steer_policy, P, z["basis"], bool(z["residual"]), float(z["noise_scale"]),
                   float(z["res_scale"]), False)


# ---------------------------------------------------------------------------- SAC in the latent-noise MDP


class Actor(nn.Module):
    def __init__(self, obs_dim: int, d: int, hidden: int, mu, sd, zero_init: bool = True):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean, self.log_std = nn.Linear(hidden, d), nn.Linear(hidden, d)
        if zero_init:  # mean 0 everywhere: the deterministic policy starts as the zero ticket
            nn.init.zeros_(self.mean.weight), nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight), nn.init.zeros_(self.log_std.bias)  # sigma = 1 at the start
        self.register_buffer("mu", torch.as_tensor(mu)), self.register_buffer("sd", torch.as_tensor(sd))

    def norm(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.mu) / self.sd).clamp(-10, 10)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized tanh-Gaussian sample and its log-density (with the tanh correction)."""
        h = self.trunk(self.norm(obs))
        mean, log_std = self.mean(h), self.log_std(h).clamp(*LOG_STD)
        u = mean + log_std.exp() * torch.randn_like(mean)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(1)
        return a, logp - torch.log(1 - a**2 + 1e-6).sum(1)

    def export(self) -> dict:
        def np_(t):
            return t.detach().numpy().astype(np.float32).copy()
        return {"mu": np_(self.mu), "sd": np_(self.sd),
                "trunk": [(np_(self.trunk[i].weight.T), np_(self.trunk[i].bias)) for i in (0, 2)],
                "wm": np_(self.mean.weight.T), "bm": np_(self.mean.bias),
                "ws": np_(self.log_std.weight.T), "bs": np_(self.log_std.bias)}


class Critic(nn.Module):
    """Two Q heads on (normalized s, latent a), LayerNorm after each hidden layer (as in RLPD)."""

    def __init__(self, obs_dim: int, d: int, hidden: int):
        super().__init__()

        def head():
            return nn.Sequential(nn.Linear(obs_dim + d, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q1, self.q2 = head(), head()

    def forward(self, s_n: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s_n, a], 1)
        return self.q1(x).squeeze(1), self.q2(x).squeeze(1)


class SAC:
    def __init__(self, cfg: Config, d: int, mu, sd, seed: int):
        torch.manual_seed(seed)
        self.cfg, self.d = cfg, d
        self.actor = Actor(len(mu), d, cfg.hidden, mu, sd)
        self.critic, self.target = Critic(len(mu), d, cfg.hidden), Critic(len(mu), d, cfg.hidden)
        self.target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(np.log(cfg.alpha), dtype=torch.float32, requires_grad=True)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.opt_t = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = -float(d)
        self.gen = np.random.default_rng(seed)

    def update(self, B: dict[str, torch.Tensor], n_updates: int, train_actor: bool = True) -> dict:
        cfg, n = self.cfg, len(B["g"])
        disc = cfg.gamma ** cfg.n_step
        for _ in range(n_updates):
            i = torch.as_tensor(self.gen.integers(0, n, cfg.batch_size))
            s, a, y, sn, boot = B["s"][i], B["a"][i], B["g"][i], B["sn"][i], B["boot"][i]
            alpha = self.log_alpha.exp().detach()
            if cfg.n_step:  # bootstrap from the target critic n decisions later (soft value, SAC)
                with torch.no_grad():
                    a2, logp2 = self.actor.sample(sn)
                    q1t, q2t = self.target(self.actor.norm(sn), a2)
                    y = y + disc * boot * (torch.min(q1t, q2t) - alpha * logp2)
                    y = y.clamp(0.0, 1.0)  # 0/1 reward, gamma < 1: the value is a success probability
            q1, q2 = self.critic(self.actor.norm(s), a)
            loss_c = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
            self.opt_c.zero_grad(), loss_c.backward(), self.opt_c.step()
            a_new, logp = self.actor.sample(s)
            if train_actor:  # during the warm-up only the critic learns, from base-like exploration data
                q1n, q2n = self.critic(self.actor.norm(s), a_new)
                loss_a = (alpha * logp - torch.min(q1n, q2n)).mean()
                self.opt_a.zero_grad(), loss_a.backward(), self.opt_a.step()
                if cfg.tune_alpha:
                    loss_t = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                    self.opt_t.zero_grad(), loss_t.backward(), self.opt_t.step()
            with torch.no_grad():  # (unused when n_step = 0)
                for p, pt in zip(self.critic.parameters(), self.target.parameters()):
                    pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
        return {"q": q1.mean().item(), "alpha": alpha.item(), "entropy": -logp.mean().item()}


def transitions(trajs: list[dict], gamma: float, n_step: int) -> dict[str, np.ndarray]:
    """One sample per chunk decision j: (s_j, a_j, n-step return g_j, s_{j+n}, bootstrap mask).

    r_j is the reward of the decision's 4 env steps (1 on the success step). With n_step = 0 the return
    runs to the end of the episode (Monte Carlo, no bootstrap). The episode's last decision has nothing
    after it, success or time limit: time_frac is in the observation, so the time limit is part of the state.
    """
    cols: dict[str, list] = {k: [] for k in ("s", "a", "g", "sn", "boot")}
    for tr in trajs:
        t = np.flatnonzero(tr["last_step_in_chunk"] == 0)
        m, n = len(t), n_step or len(t)
        cum = np.concatenate([[0.0], np.cumsum(tr["rewards"])])
        r = np.diff(cum[np.append(t, len(tr["actions"]))])
        end = np.minimum(np.arange(m) + n, m)  # first decision not summed into g_j
        g = np.array([np.sum(gamma ** np.arange(e - j) * r[j:e]) for j, e in enumerate(end)])
        obs = np.vstack([tr["obs"], tr["final_obs"][None]])
        cols["s"].append(tr["obs"][t]), cols["a"].append(tr["last_latent"][t]), cols["g"].append(g)
        cols["sn"].append(obs[np.append(t, len(tr["actions"]))[end]])
        cols["boot"].append((end < m).astype(np.float32))
    return {k: np.concatenate(v).astype(np.float32) for k, v in cols.items()}


# ---------------------------------------------------------------------------- PSS: probing the frozen network


def pss_basis(cfg: Config, variant: str, ledger: Ledger) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvectors of M = mean_s J(s)^T J(s), J = d chunk / d w at w = 0, by central finite differences.

    The probe states are chunk-decision observations from base rollouts on training seeds (charged to "train");
    probing itself only evaluates the network, it needs no robot time. Returns (eigenvalues, eigenvectors),
    largest first.
    """
    n_eps = max(4, cfg.pss_states // 8)
    seeds = [TRAIN_SEED_BASE + 50_000 * (variant == "hard") + 9_000 + i for i in range(n_eps)]
    res = rollout_seeds(GAUSS, seeds, [policy_seed("ch07_pss", i) for i in range(n_eps)], robot=cfg.robot,
                        variant=variant, n_workers=cfg.n_workers, record_trajectories=True, ledger=ledger,
                        category="train")
    obs = np.concatenate([tr["obs"][tr["last_step_in_chunk"] == 0] for tr in res.trajectories])
    obs = obs[np.random.default_rng(0).choice(len(obs), min(cfg.pss_states, len(obs)), replace=False)]
    sampler, eps = FlowChunk.load(CKPT).numpy_sampler(), 0.05
    probes = np.concatenate([eps * np.eye(DIM), -eps * np.eye(DIM)]).astype(np.float32)
    M = np.zeros((DIM, DIM))
    for o in obs:
        out = sampler(np.repeat(o[None], 2 * DIM, 0), probes).reshape(2 * DIM, DIM)
        J = (out[:DIM] - out[DIM:]).T / (2 * eps)  # [chunk entries, noise dims]
        M += J.T @ J / len(obs)
    vals, vecs = np.linalg.eigh(M)
    return vals[::-1].copy(), vecs[:, ::-1].copy()


# ---------------------------------------------------------------------------- one training run


def rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "sr": k / n if n else 0.0, "ci": list(wilson(int(k), int(n)))}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """McNemar's exact test and a paired bootstrap interval for b - a on the same episodes."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return {"p": mcnemar_exact(a, b), "only_a": int(np.sum(a & ~b)), "only_b": int(np.sum(~a & b)),
            "diff_ci": list(bootstrap_diff(a, b))}


def kn(res) -> dict:
    return rate(res.k, res.n)


def bits(a) -> str:
    return "".join("1" if x else "0" for x in a)


def train_run(cfg: Config, arm: dict, variant: str, seed: int, basis: np.ndarray, L: Ledger) -> dict:
    """SAC in rounds: collect episodes with the stochastic actor, update, score the deterministic actor.

    ``basis`` [32, k] maps the actor's k steering numbers to noise (identity for k = 32, PSS directions for
    k < 32, nothing for k = 0).
    """
    d = basis.shape[1] + (RES_DIM if arm["residual"] else 0)
    base = FlowChunk.load(CKPT)
    agent = SAC(cfg, d, base.obs_mean.numpy(), base.obs_std.numpy(), seed=1000 * seed + 7)
    make = partial(steer_policy, basis=basis, residual=arm["residual"], noise_scale=cfg.noise_scale,
                   res_scale=cfg.res_scale)
    offset = TRAIN_SEED_BASE + 50_000 * (variant == "hard") + 10_000 * seed  # shared by arms: paired starts
    buf: dict[str, list] = {k: [] for k in ("s", "a", "g", "sn", "boot")}
    curve, snaps, logs, done_eps, train_k = [], [], [], 0, 0

    def checkpoint():
        P = agent.actor.export()
        r = evaluate(partial(make, P, stochastic=False), "search", robot=cfg.robot, variant=variant,
                     n_workers=cfg.n_workers, ledger=L, category="search", n=cfg.search_n)
        curve.append({"episodes": done_eps, "train_minutes": L.robot_minutes["train"], "k": r.k, "n": r.n})
        snaps.append((P, r.successes))
        print(f"    {arm['name']}/{variant} s{seed} ep {done_eps:4d} | train {L.robot_minutes['train']:6.1f} robot-min "
              f"| search {r.summary}" + (f" | {logs[-1]}" if logs else ""), flush=True)

    checkpoint()  # before any update: the zero ticket (the actor's mean head is zero)
    while done_eps < cfg.train_episodes:
        n = min(cfg.episodes_per_round, cfg.train_episodes - done_eps)
        idx = range(done_eps, done_eps + n)
        res = rollout_seeds(partial(make, agent.actor.export(), stochastic=True), [offset + i for i in idx],
                            [policy_seed(f"ch07_train_{variant}", 10_000 * seed + i) for i in idx],
                            robot=cfg.robot, variant=variant, n_workers=cfg.n_workers,
                            record_trajectories=True, ledger=L, category="train")
        done_eps, train_k = done_eps + n, train_k + res.k
        new = transitions(res.trajectories, cfg.gamma, cfg.n_step)
        for k, v in new.items():
            buf[k].append(v)
        B = {k: torch.as_tensor(np.concatenate(v)) for k, v in buf.items()}
        stats = agent.update(B, int(cfg.utd * len(new["g"])), train_actor=done_eps > cfg.warmup_episodes)
        logs.append(f"rollouts {res.k}/{res.n} | Q {stats['q']:.2f} alpha {stats['alpha']:.4f} "
                    f"H {stats['entropy']:.1f}")
        if done_eps % cfg.eval_every == 0 or done_eps >= cfg.train_episodes:
            checkpoint()
    best = max(range(len(curve)), key=lambda j: (curve[j]["k"], j))  # best search score, ties: later
    return {"arm": arm["name"], "variant": variant, "seed": seed, "curve": curve, "chosen": best, "P": snaps[best][0],
            "train_rollouts": rate(train_k, done_eps), "make": partial(make, snaps[best][0], stochastic=False)}


# ---------------------------------------------------------------------------- analysis helpers

SIDE_TOL = 0.004  # m: grasp offsets within +-4 mm of the cube center count as "center"
TIMING = (12, 16)  # grasp step: <= 12 early, <= 16 mid, later = late


def grasp_modes(trajs: list[dict], sl: dict) -> dict:
    """Which side of the cube (base-frame y) the gripper closes on, and when, per grasped episode.

    The base was cloned from clean and sloppy demos whose grasp offsets and timings differ, so these 3 x 3
    cells are the behavior "modes" we can count; BMD [C304] warns that noise steering collapses them.
    """
    cells = {}
    for tr in trajs:
        g = tr["infos"]["grasped"]
        if not g.any():
            key = "no grasp"
        else:
            t = int(np.argmax(g))
            after = tr["obs"][t + 1] if t + 1 < len(tr["obs"]) else tr["final_obs"]
            dy = float(after[sl["cube_to_ee"]][1])
            side = "-y" if dy < -SIDE_TOL else "+y" if dy > SIDE_TOL else "center"
            when = "early" if t + 1 <= TIMING[0] else "mid" if t + 1 <= TIMING[1] else "late"
            key = f"{side}/{when}"
        cells[key] = cells.get(key, 0) + 1
    grasped = {k: v for k, v in cells.items() if k != "no grasp"}
    p = np.array(list(grasped.values()), float) / max(1, sum(grasped.values()))
    return {"cells": cells, "entropy_bits": float(-(p * np.log2(p)).sum()) if len(p) else 0.0,
            "modes_5pct": int(np.sum(p >= 0.05))}


def ch03_tickets() -> dict[str, np.ndarray]:
    """Chapter 3's chosen tickets (its headline and its peek-free pick), read from its results JSON."""
    files = sorted((DEFAULT_RESULTS_ROOT / "ch03").glob("golden_ticket_*.json"))
    if not files:
        return {}
    A = json.loads(files[-1].read_text())["extra"]["analysis"]
    return {"ch03_headline": np.array(A["headline"]["ticket"], np.float32),
            "ch03_peek_free": np.array(A["peek_free"]["ticket"], np.float32)}


# ---------------------------------------------------------------------------- media


def make_plots(A: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    plotting.setup()
    P = plotting.PALETTE
    colors = {"dsrl": P[0], "pss": P[2], "hybrid": P[1], "residual": P[3], "rfs": P[4]}
    variants = [v for v in ("v1", "hard") if any(r["variant"] == v for r in A["runs"])]

    # 1. learning curves: search success of the deterministic actor vs train robot-minutes (+ held-out)
    fig, axes = plt.subplots(1, len(variants), figsize=(6.4 * len(variants), 4.4), squeeze=False)
    for ax, v in zip(axes[0], variants):
        for arm, col in colors.items():
            runs = [r for r in A["runs"] if r["variant"] == v and r["arm"] == arm]
            if not runs:
                continue
            m = min(len(r["curve"]) for r in runs)
            x = np.mean([[c["train_minutes"] for c in r["curve"][:m]] for r in runs], 0)
            k = np.sum([[c["k"] for c in r["curve"][:m]] for r in runs], 0)
            n = sum(r["curve"][0]["n"] for r in runs)
            plotting.success_curve(ax, x, k, n, label=f"{arm} (search, {len(runs)} seeds pooled)", color=col)
            for r in runs:
                ax.plot([c["train_minutes"] for c in r["curve"]], [c["k"] / c["n"] for c in r["curve"]],
                        color=col, lw=0.6, alpha=0.35)
            kf, nf = sum(r["final"]["k"] for r in runs), sum(r["final"]["n"] for r in runs)
            plotting.heldout_point(ax, x[-1] * 1.03, kf, nf, color=col, label=None)
        b = A["baselines"][v]
        for key, ls, name in (("gauss_search", ":", "Gaussian base"), ("zero_search", "--", "zero ticket")):
            ax.axhline(b[key]["sr"], color="0.35", ls=ls, lw=1.1, label=f"{name} (search)")
        ax.set(xlabel="train robot-minutes (per run)", ylabel="success",
               title=f"{v}: search set (Wilson bands); diamonds = held-out {HELDOUT[v]}")
        ax.legend(fontsize=7.5, loc="lower right")
    plotting.save_fig(fig, out / "learning_curves.png")

    # 2. held-out success of every arm and baseline, per variant (per-seed points, pooled diamond)
    fig, axes = plt.subplots(1, len(variants), figsize=(6.4 * len(variants), 4.2), squeeze=False)
    for ax, v in zip(axes[0], variants):
        b, labels, j = A["baselines"][v], [], 0
        for key, name in (("gauss", "Gaussian\nbase"), ("zero", "zero\nticket"), ("ch03_peek_free", "ch03\nticket"),
                          ("random_actor", "random\nactor")):
            if key not in b:
                continue
            items = b[key] if isinstance(b[key], list) else [b[key]]
            for i, t in enumerate(items):
                plotting.heldout_point(ax, j + 0.12 * (i - (len(items) - 1) / 2), t["k"], t["n"], color="0.4",
                                       label=None)
            labels.append(name)
            j += 1
        for arm, col in colors.items():
            runs = [r for r in A["runs"] if r["variant"] == v and r["arm"] == arm]
            if not runs:
                continue
            for i, r in enumerate(runs):
                plotting.heldout_point(ax, j + 0.12 * (i - 1), r["final"]["k"], r["final"]["n"], color=col,
                                       label=None)
            labels.append(arm)
            j += 1
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, 1.02), ylabel="success (Wilson 95%)",
               title=f"{v}: held-out {HELDOUT[v]} (n={A['eval_n']}), one point per seed")
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    plotting.save_fig(fig, out / "heldout.png")

    # 3. support limit: hard held-out states grouped by Gaussian successes in support_salts tries
    if "support" in A:
        S = A["support"]
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        series = [("gauss", f"Gaussian base (a fresh try, salt {S['salts']})", "0.3"), ("zero", "zero ticket", P[4])]
        series += [(arm, f"{arm} (seeds pooled)", colors[arm]) for arm in colors if arm in S["groups"][0]]
        for j, (key, name, col) in enumerate(series):
            p, err = plotting.wilson_err([g[key][0] for g in S["groups"]], [g[key][1] for g in S["groups"]])
            ax.errorbar(np.arange(len(S["groups"])) + 0.1 * (j - len(series) / 2), p, yerr=err, fmt="o",
                        color=col, capsize=3, label=name)
        ax.set(xticks=range(len(S["groups"])), ylim=(-0.02, 1.02), ylabel="success on those states (Wilson 95%)",
               xticklabels=[f"{g['name']}\n(n={g['n']})" for g in S["groups"]],
               xlabel=f"hard held-out states by Gaussian successes in {S['salts']} tries (salts 0-{S['salts'] - 1})",
               title="Steering needs states the base can sometimes solve")
        ax.legend(fontsize=7.5, loc="upper left")
        plotting.save_fig(fig, out / "support_limit.png")

    # 4. grasp modes before and after
    M = A["modes"]
    names = [n for n in ("gauss", "zero", "ch03_peek_free", "dsrl", "pss", "hybrid", "residual") if n in M]
    cells = [f"{s}/{w}" for s in ("-y", "center", "+y") for w in ("early", "mid", "late")]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        c = M[name]["cells"]
        tot = sum(v for k, v in c.items() if k != "no grasp")
        ax.bar(np.arange(len(cells)) + width * (j - len(names) / 2 + 0.5), [c.get(k, 0) / tot for k in cells],
               width, label=f"{name} ({M[name]['modes_5pct']} cells >= 5%, H = {M[name]['entropy_bits']:.2f} bits)",
               color=colors.get(name, {"gauss": "0.65", "zero": P[4]}.get(name, "#8c6d31")))
    ax.set(xticks=range(len(cells)), xticklabels=cells, ylabel="share of grasped episodes",
           xlabel="grasp side (EE y offset from cube) / grasp timing", title="Grasp modes on v1 eval")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, loc="upper right")
    plotting.save_fig(fig, out / "grasp_modes.png")

    # 5. PSS spectrum
    if A.get("pss_eigenvalues"):
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        for v, ev in A["pss_eigenvalues"].items():
            ev = np.array(ev)
            ax.plot(np.arange(1, len(ev) + 1), np.cumsum(ev) / ev.sum(), marker="o", ms=3, label=f"{v} states")
        ax.axvline(A["pss_k"], color=P[1], ls="--", lw=1, label=f"k = {A['pss_k']}")
        ax.set(xlabel="noise directions (largest sensitivity first)", ylabel="share of total sensitivity",
               title="How many noise directions move the chunk (PSS probe)", ylim=(0, 1.02))
        ax.legend()
        plotting.save_fig(fig, out / "pss_spectrum.png")


def make_gifs(A: dict, trajs: dict, pick: tuple | None, cfg: Config, out: Path) -> None:
    """Replays of recorded held-out episodes (illustration only, nothing is selected with them).

    v1: the Gaussian base vs the headline pick, on the first eval state where the base fails and the pick
    succeeds. hard: pss vs residual (seed 0), preferring a state the base never solved in 16 tries.
    """
    from lastmile.common.plotting import replay_frames, save_gif, side_by_side
    from lastmile.envs.cupdrop import CupDropEnv

    s0 = cfg.seeds[0]
    for left, right, name in ((("v1", "gauss"), pick, "base_vs_dsrl.gif"),
                              (("hard", "pss", s0), ("hard", "residual", s0), "hard_pss_vs_residual.gif")):
        if left not in trajs or right not in trajs:
            continue
        v, L, R = left[0], trajs[left], trajs[right]
        never = set(A.get("support", {}).get("never_idx", [])) if v == "hard" else set()
        fixed_ = [i for i in range(len(L)) if not L[i]["success"] and R[i]["success"]]
        cand = [i for i in fixed_ if i in never] + fixed_ + sorted(never)
        if not cand:
            continue
        i = cand[0]
        env = CupDropEnv(robot=cfg.robot, variant=v, render_size=(200, 200))
        clips = [replay_frames(env, tr[i], f"{'/'.join(map(str, key[1:]))} ({v}, seed {tr[i]['seed']})", every=3)
                 for tr, key in ((L, left), (R, right))]
        path = save_gif(side_by_side(*clips), out / name, fps=5, max_size=404)
        A.setdefault("gifs", {})[name] = {"seed": int(L[i]["seed"]), "index": i, "left": list(map(str, left)),
                                          "right": list(map(str, right)), "never_solved": i in never}
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------- main


def main(cfg: Config) -> None:
    if cfg.replot:
        make_plots(json.loads(Path(cfg.replot).read_text())["extra"]["analysis"], MEDIA)
        return
    torch.set_num_threads(2)
    t0 = time.time()
    single = cfg.subspace > 0 or cfg.hybrid or cfg.residual_only
    # Only the full grid writes results/ch07 and media/ch07; --quick and one-arm runs stay under runs/.
    out = REPO_ROOT / "runs" / ("ch07_quick" if cfg.quick else "ch07_arm" if single else "ch07_full")
    root, media = (out / "results", out / "media" / "ch07") if cfg.quick or single else (DEFAULT_RESULTS_ROOT, MEDIA)
    new_ledger = partial(Ledger, chapter="ch07", robot=cfg.robot, config=cfg, results_root=root)
    if single:
        jobs = [(cfg.variant, make_arm(0 if cfg.residual_only else cfg.subspace or DIM,
                                       cfg.hybrid or cfg.residual_only))]
    else:
        jobs = [("v1", grid_arm(a, cfg)) for a in cfg.arms] + [("hard", grid_arm(a, cfg)) for a in cfg.hard_arms]
    variants = sorted({v for v, _ in jobs}, key=["v1", "hard"].index)
    final_sets = {v: "search" if cfg.quick else HELDOUT[v] for v in variants}
    ev = partial(evaluate, robot=cfg.robot, n_workers=cfg.n_workers)
    shared = Ledger("ch07", "shared")  # never written: baselines and analysis rollouts (no selection)
    steering = ("dsrl", "pss", "hybrid", "rfs")  # the headline row picks among these arms' v1 runs
    # The headline row is a pick among dsrl / pss / hybrid runs (named so it is not read as full-noise DSRL).
    head = new_ledger(method="noise-steering-best") if any(v == "v1" and a["name"] in steering for v, a in jobs) else None
    if head:
        head.__enter__()  # opened now so its compute covers the whole run; written at the end
    A: dict = {"eval_n": cfg.eval_n, "search_n": cfg.search_n, "pss_k": cfg.subspace or cfg.pss_k, "baselines": {},
               "runs": [], "modes": {}}
    sl = obs_layout(cfg.robot)[1]
    finals, trajs, base_res = {}, {}, {}

    # 1. Baselines, fixed in advance: Gaussian base (salt 0 = the paired base), zero ticket, Chapter 3 tickets.
    tickets = ch03_tickets() if "v1" in variants else {}
    for v in variants:
        hs = final_sets[v]
        g = ev(GAUSS, hs, variant=v, n=cfg.eval_n, ledger=shared, category="eval", record_trajectories=True)
        z = ev(fixed(np.zeros(DIM)), hs, variant=v, n=cfg.eval_n, ledger=shared, category="eval",
               record_trajectories=True)
        b = A["baselines"][v] = {
            "gauss": rate(g.k, g.n), "zero": rate(z.k, z.n),
            "gauss_search": kn(ev(GAUSS, "search", variant=v, n=cfg.search_n, ledger=shared, category="search")),
            "zero_search": kn(ev(fixed(np.zeros(DIM)), "search", variant=v, n=cfg.search_n, ledger=shared,
                                 category="search"))}
        base_res[v], finals[v, "gauss"], finals[v, "zero"] = g, g.successes, z.successes
        trajs[v, "gauss"], trajs[v, "zero"] = g.trajectories, z.trajectories
        b["zero_vs_gauss"] = paired(g.successes, z.successes)
        if v == "v1":
            A["modes"]["gauss"], A["modes"]["zero"] = grasp_modes(g.trajectories, sl), grasp_modes(z.trajectories, sl)
            for name, w in tickets.items():
                r = ev(fixed(w), hs, variant=v, n=cfg.eval_n, ledger=shared, category="eval",
                       record_trajectories=True)
                b[name], finals[v, name] = rate(r.k, r.n), r.successes
                A["modes"][name] = grasp_modes(r.trajectories, sl)
        print(" | ".join([f"[{v}] Gaussian base {g.summary}", f"zero ticket {z.summary}"]
                         + [f"{k} {format_rate(b[k]['k'], b[k]['n'])}" for k in tickets if k in b]), flush=True)

    # 2. Hard: group held-out states by how often the Gaussian base solves them (the base's support).
    if "hard" in variants:
        salts = [base_res["hard"].successes] + [
            ev(GAUSS, final_sets["hard"], variant="hard", n=cfg.eval_n, salt=s, ledger=shared, category="eval").successes
            for s in range(1, cfg.support_salts)]
        A["support_counts"] = np.sum(salts, 0).tolist()
        # The groups are defined by those tries, so the table's Gaussian column is one more, independent try.
        fresh = ev(GAUSS, final_sets["hard"], variant="hard", n=cfg.eval_n, salt=cfg.support_salts, ledger=shared,
                   category="eval")
        finals["hard", "gauss_fresh"] = fresh.successes
        A["baselines"]["hard"]["gauss_fresh_salt"] = {"salt": cfg.support_salts, **rate(fresh.k, fresh.n)}
        print(f"[hard] pass@{cfg.support_salts}: {format_rate(int(np.any(salts, 0).sum()), cfg.eval_n)} | "
              f"fresh Gaussian try (salt {cfg.support_salts}) {fresh.summary}", flush=True)

    # 3. PSS directions (only if an arm steers a subspace), probed once per variant and shared by its runs.
    pss, pss_steps = {}, {}
    for v in variants:
        ks = [a["k"] for vv, a in jobs if vv == v and 0 < a["k"] < DIM]
        if ks:
            probe = Ledger("ch07", "probe")  # never written: its rollouts are charged to every run that uses it
            vals, pss[v] = pss_basis(cfg, v, probe)
            pss_steps[v] = probe.robot_steps["train"]
            A.setdefault("pss_eigenvalues", {})[v] = vals.tolist()
            print(f"[{v}] PSS: top {ks[0]} of 32 directions carry {vals[:ks[0]].sum() / vals.sum():.1%} "
                  "of the chunk's sensitivity", flush=True)

    def basis_for(v: str, k: int) -> np.ndarray:
        return np.eye(DIM, dtype=np.float32) if k == DIM else pss[v][:, :k].astype(np.float32) if k else \
            np.zeros((DIM, 0), np.float32)

    # 4. The training runs, one results JSON each.
    for v, arm in jobs:
        hs = final_sets[v]
        for seed in cfg.seeds:
            method = f"{arm['method']}{'-hard' if v == 'hard' else ''}_s{seed}"
            with new_ledger(method=method, variant=v) as L:
                L.robot_steps["train"] += pss_steps[v] if 0 < arm["k"] < DIM else 0
                r = train_run(cfg, arm, v, seed, basis_for(v, arm["k"]), L)
                if r["curve"][0]["k"] != A["baselines"][v]["zero_search"]["k"]:
                    raise AssertionError("the untrained actor should reproduce the zero ticket on search")
                final = ev(r["make"], hs, variant=v, n=cfg.eval_n, ledger=L, category="eval",
                           record_trajectories=True)
                L.robot_steps["eval"] += base_res[v].env_steps  # the shared base evaluation
                L.set_base(id="base_v1_gaussian", successes=base_res[v])
                L.set_final(successes=final)
                actor_file = out / "actors" / f"{method}.npz"
                save_steer(actor_file, r["P"], basis_for(v, arm["k"]), arm["residual"], cfg)
                run = {k: r[k] for k in ("arm", "variant", "seed", "curve", "chosen", "train_rollouts")}
                run.update(method=method, final=rate(final.k, final.n), eval_successes=bits(final.successes),
                           vs_base=paired(base_res[v].successes, final.successes),
                           vs_zero=paired(finals[v, "zero"], final.successes),
                           robot_minutes=dict(L.robot_minutes),  # this run's own rollouts (no pilots)
                           eval_steps=final.env_steps, actor_file=str(actor_file.relative_to(REPO_ROOT)))
                if cfg.pilot_costs:  # the pilots chose this run's recipe: charge them to its row too
                    for c, minutes in PILOT_ROBOT_MINUTES.items():
                        L.robot_steps[c] += round(minutes * 600)
                    run["pilot_robot_minutes"] = PILOT_ROBOT_MINUTES
                if (v, "ch03_peek_free") in finals:
                    run["vs_ch03_peek_free"] = paired(finals[v, "ch03_peek_free"], final.successes)
                if v == "v1":
                    run["modes"] = grasp_modes(final.trajectories, sl)
                L.extra.update(run)
                A["runs"].append(run)
                if head and v == "v1" and arm["name"] in steering:
                    head.worker_cpu_seconds += L.worker_cpu_seconds
                trajs[v, arm["name"], seed] = final.trajectories
                if seed == cfg.seeds[0] and v == "v1":
                    A["modes"][arm["name"]] = run["modes"]
            print(f"  {method}: search {format_rate(r['curve'][r['chosen']]['k'], cfg.search_n)} (checkpoint "
                  f"{r['chosen']}) -> {hs} {final.summary}; McNemar vs base p = {run['vs_base']['p']:.2g}, "
                  f"vs zero ticket p = {run['vs_zero']['p']:.2g} | {run['robot_minutes']['train']:.0f} train + "
                  f"{run['robot_minutes']['search']:.0f} search robot-min (own, before pilots)", flush=True)

    # 5. Random-selection control (honesty rule 4), on both variants: an untrained actor with random weights,
    #    i.e. a random state-dependent noise map in the same box, deterministic. No search, no training.
    for v in [] if single else variants:
        b = A["baselines"][v]
        b["random_actor"] = []
        for seed in cfg.seeds:
            torch.manual_seed(5000 + seed)  # the same three random actors on both variants
            mu, sd = FlowChunk.load(CKPT).obs_mean.numpy(), FlowChunk.load(CKPT).obs_std.numpy()
            P = Actor(len(mu), DIM, cfg.hidden, mu, sd, zero_init=False).export()
            with new_ledger(method=f"random_actor{'-hard' if v == 'hard' else ''}_s{seed}", variant=v) as L:
                r = ev(partial(steer_policy, P, basis_for(v, DIM), False, cfg.noise_scale, cfg.res_scale, False),
                       final_sets[v], variant=v, n=cfg.eval_n, ledger=L, category="eval")
                L.robot_steps["eval"] += base_res[v].env_steps
                L.set_base(id="base_v1_gaussian", successes=base_res[v])
                L.set_final(successes=r)
                L.extra.update(note="untrained actor, random weights, deterministic; no search, no training",
                               vs_base=paired(base_res[v].successes, r.successes),
                               vs_zero=paired(finals[v, "zero"], r.successes))
            b["random_actor"].append(rate(r.k, r.n) | {"vs_zero": paired(finals[v, "zero"], r.successes)})
        print(f"[{v}] random actor (no training):", [x["k"] for x in b["random_actor"]], f"of {cfg.eval_n}",
              flush=True)

    # 6. Support-limit table: success per group of hard states, arms pooled over seeds.
    if "support_counts" in A:
        c = np.array(A["support_counts"])
        groups = [(f"0 of {cfg.support_salts}", c == 0), ("1-4", (c >= 1) & (c <= 4)), ("5+", c >= 5)]
        A["support"] = {"salts": cfg.support_salts, "never_idx": np.flatnonzero(c == 0).tolist(), "groups": []}
        for name, m in groups:
            g = {"name": name, "n": int(m.sum()), "gauss": [int(finals["hard", "gauss_fresh"][m].sum()), int(m.sum())],
                 "zero": [int(finals["hard", "zero"][m].sum()), int(m.sum())]}
            for arm in {a["name"] for vv, a in jobs if vv == "hard"}:
                s = np.array([[ch == "1" for ch in r["eval_successes"]] for r in A["runs"]
                              if r["variant"] == "hard" and r["arm"] == arm])
                g[arm] = [int(s[:, m].sum()), int(m.sum()) * len(s)]
            A["support"]["groups"].append(g)
        print("[hard] success by base support group:", {g["name"]: {k: f"{v[0]}/{v[1]}" for k, v in g.items()
                                                                   if isinstance(v, list)}
                                                          for g in A["support"]["groups"]}, flush=True)

    # 7. Headline row: the best v1 noise-steering run (dsrl, pss, hybrid) by search score, paying for all of them.
    dsrl_v1 = [r for r in A["runs"] if r["arm"] in steering and r["variant"] == "v1"]
    A["wall_minutes"] = (time.time() - t0) / 60
    if head:  # the GIFs are made before the JSON is written, so the GIF's episode is in the analysis
        L = head
        best = max(dsrl_v1, key=lambda r: (r["curve"][r["chosen"]]["k"], -r["seed"]))
        L.worker_cpu_seconds += shared.worker_cpu_seconds
        for r in dsrl_v1:
            for c in ("train", "search"):
                L.robot_steps[c] += round(r["robot_minutes"][c] * 600)
        if cfg.pilot_costs:
            for c, minutes in PILOT_ROBOT_MINUTES.items():
                L.robot_steps[c] += round(minutes * 600)
            L.extra["pilot_robot_minutes"] = PILOT_ROBOT_MINUTES
        L.robot_steps["eval"] += base_res["v1"].env_steps + best["eval_steps"]
        L.set_base(id="base_v1_gaussian", successes=base_res["v1"])
        L.set_final(successes=np.array([ch == "1" for ch in best["eval_successes"]]), init_set=final_sets["v1"])
        L.extra.update(note=f"best of {len(dsrl_v1)} v1 noise-steering runs by search score ({best['method']}); "
                            "pays for the training and search rollouts of all of them",
                       pick=best["method"], analysis=A)
        if cfg.media:
            with optional_media("the DSRL GIFs"):
                make_gifs(A, trajs, ("v1", best["arm"], best["seed"]), cfg, media)
        L.__exit__(None, None, None)
        print(f"best-by-search pick: {best['method']} -> eval {format_rate(best['final']['k'], best['final']['n'])}; JSON {L.path}")
    if cfg.media:
        make_plots(A, media)

    print("\n=== Chapter 7 summary (held-out; Wilson 95%) ===")
    for v in variants:
        b = A["baselines"][v]
        print(f"[{v}] Gaussian base {format_rate(b['gauss']['k'], b['gauss']['n'])} | zero ticket "
              f"{format_rate(b['zero']['k'], b['zero']['n'])}")
        for arm in dict.fromkeys(a["name"] for vv, a in jobs if vv == v):
            rs = [r for r in A["runs"] if r["variant"] == v and r["arm"] == arm]
            k, n = sum(r["final"]["k"] for r in rs), sum(r["final"]["n"] for r in rs)
            print(f"  {arm:9s} per seed {[r['final']['k'] for r in rs]} of {cfg.eval_n}, pooled {format_rate(k, n)}; "
                  f"train robot-min per run {np.mean([r['robot_minutes']['train'] for r in rs]):.0f}")
    print(f"wall {A['wall_minutes']:.1f} min")


if __name__ == "__main__":
    main(parse(Config))
