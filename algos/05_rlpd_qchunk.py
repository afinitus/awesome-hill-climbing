"""Chapter 5: off-policy critics. SAC -> RLPD -> Q-chunking, on top of the frozen base.

Learning goal
    Learn a critic that reuses every transition it has seen (demos, failures, old rollouts) through
    Bellman backups on a replay buffer, then compare three ways of turning it into a policy: a
    Gaussian actor trained from scratch (SAC), the same actor with RLPD's tricks, and Q-chunking's
    best-of-N extraction from the frozen base_v1 (QC). Sparse 0/1 reward on CupDrop-v1.

Base idea in plain words
    Q(s, a) predicts the discounted chance of success after doing a in s. Temporal-difference (TD)
    learning fits Q to its own prediction one decision later (a Bellman backup), on transitions drawn
    from a replay buffer, so every transition is reused thousands of times, whoever collected it.
    All methods here decide once per chunk: the 4 actions base_v1 executes per plan (h = 4, 0.4 s).
    The critic scores that 16-number chunk, and the target is the h-step backup

        y = sum_{k<h} gamma^k r_{t+k} + gamma^h (1 - done) V(s_{t+h})

    It is unbiased because all h actions were chosen at time t, so no intermediate decision of the
    data's policy differs from the target policy (Q-chunking's point). That holds for base_v1, QC
    and actor rollouts. The scripted demos chose their actions step by step (closed loop); the target
    is still valid for them because the simulator is deterministic, so replaying the same 4 actions
    from the same state gives the same outcome (only approximately, since the observation leaves out
    velocities and other hidden state). With a sparse reward the target also carries the success
    signal h steps back per backup instead of one.
        SAC   V(s') = min(Q'_1, Q'_2)(s', a'), a' ~ pi(.|s'); a tanh-Gaussian actor over the chunk
              maximizes Q + alpha * entropy. The entropy bonus stays out of the backup, so Q remains a
              discounted success probability in [0, 1] like QC's. Prior data just sits in the buffer.
        RLPD  SAC + every batch half prior data, half online data + LayerNorm in the critics + an
              ensemble of 10 critics (target: min over a random pair) + 4 critic updates per new
              transition (update-to-data ratio, UTD, of 4).
        QC    no actor at all: the policy is best-of-N from the frozen base. Decode N chunks from
              base_v1, execute argmax_j mean_e Q_e(s, a_j) (mean over all 10 members). The target
              selects j* = argmax_j mean_e Q_e(s', a'_j), the same rule as acting, then values that
              one chunk with V(s') = min_pair Q'(s', a'_{j*}). Selection uses the online critic;
              pessimistic evaluation uses the target critic (RLPD/REDQ). Same critic as RLPD.
    Offline to online: QC's critic first fits the prior data, then keeps learning from its own
    rollouts. The "three regimes" check compares the base policy's success with the data's. The
    "dip" check: QC-CQL pretrains its critic on the demos alone with a conservative (CQL-style)
    penalty and goes online with no warm-up, against QC's WSRL-style warm-up with base rollouts.
    QC only reranks base_v1's own proposals: it can recombine chunks from different samples, but it
    cannot solve a start state from which no sequence of base chunks succeeds.

Papers mirrored
    Q-chunking [C133] (chunk critic, h-step target, best-of-N from a flow BC policy), RLPD [C037],
    Three Regimes of offline-to-online RL [C172], IPE [C348] (pretrained critics help little),
    WSRL [C088] (warm up with frozen-policy rollouts), Cal-QL [C041] (why conservative pretraining
    dips), Decoupled Q-chunking [C191] (critic chunk longer than the executed one; not used here).

Honesty
    Demos, warm-up and online rollouts use chapter-5 env seeds (5_000_000+), disjoint from search,
    eval, eval_ext and stress. Hyperparameters were fixed after one pilot (seed 0, trained on those
    seeds, scored on `search`); nothing is chosen on `eval`. Two settings were added after the pilot
    without a new pilot: cql_cands (QC-CQL's penalty) and actor_rounds (see PRIOR_STEPS). Every demo, warm-up and online rollout is charged to `train` (demos also to human
    minutes); pilot and superseded full-run rollouts are charged through PRIOR_STEPS. Each method and seed is scored once on
    the held-out `eval` set (n=256), paired with base_v1 (same env and policy seeds), McNemar's test.

Run
    uv run python algos/05_rlpd_qchunk.py            # full configuration (4 workers; MPS when available)
    uv run python algos/05_rlpd_qchunk.py --quick    # smoke test, < 1 min (51 s MPS / 55 s CPU); writes only to runs/ch05_quick/
"""

from __future__ import annotations

import copy
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lastmile.common.cli import parse
from lastmile.common.eval import bootstrap_diff, format_rate, mcnemar_exact, policy_seed
from lastmile.common.ledger import REPO_ROOT, Ledger
from lastmile.common.plotting import optional_media
from lastmile.common.policy_flow import FlowChunk, FlowChunkPolicy
from lastmile.common.rollout import evaluate, rollout_seeds
from lastmile.envs.knob_controller import TUNED_KNOBS, KnobController

BASE = REPO_ROOT / "checkpoints/base_v1_so101.pt"
H = 4  # actions per decision: base_v1 plans 8 and executes 4, so the critic's chunk is the executed 4
ADIM = 4 * H  # the chunk as one 16-number action
GAMMA = 0.99  # per env step; one chunk backup discounts by GAMMA ** H
SEEDS0 = 5_000_000  # chapter-5 env seeds (demos, warm-up, online), far from every init set
METHODS = ("sac", "rlpd", "qc", "qc-cql")
# Rollouts spent before the full run (env steps, each rollout counted once, from runs/ch05_pilots/ and
# runs/ch05_quick/ results files). Pilot 1 (seed 0, all four methods: SAC, RLPD and QC 10 rounds, QC-CQL 6)
# was scored on `search` and fixed the round budgets; the first --quick run showed the entropy bonus
# inflating SAC's Q. After the pilot, without re-piloting: QC went to 20 rounds while actor_rounds kept
# SAC/RLPD at the pilot's 10; QC-CQL went to 10 rounds, and cql_cands = 8 (new) restricted its CQL penalty
# to 8 of the 32 candidates per state (4x fewer critic passes in the penalty). The original full run
# reproduced the pilot's SAC, RLPD and QC rounds. The corrected QC target deliberately changes its trajectory.
PRIOR_STEPS: dict[str, dict[str, int]] = {"pilot_1": {"search": 44_515, "train": 160_368},
                                          "first_quick_run": {"search": 20_759, "train": 10_363},
                                          "prelaunch_invalid_full_run": {"search": 0, "train": 572_459}}
# The first full run used a different target-policy argmax. Its development cost is retained:
# sum the 12 rows' train steps, subtract each row's already-counted pilots, then count each seed's
# shared demos and warm-up once (not four and three times). See the archived run manifest.
PRIOR_HUMAN_MINUTES = {"prelaunch_invalid_full_run": {"demos": 4_999 / 600}}


@dataclass
class Config:
    robot: str = "so101"
    methods: list[str] = field(default_factory=lambda: list(METHODS), metadata={"help": "of " + ", ".join(METHODS)})
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_demos: int = 20  # half clean, half sloppy: the same recipe base_v1 was cloned from (ch00)
    warmup_episodes: int = 256  # base_v1 rollouts that seed the buffer (WSRL-style warm-up)
    rounds: int = 20  # QC's online rounds; each collects round_episodes, then trains
    actor_rounds: int = 10  # SAC and RLPD (4-5x slower per round on this laptop; see the chapter doc)
    dip_rounds: int = 10  # QC-CQL: the start of online training is where a dip would show
    round_episodes: int = 32
    utd_sac: int = 1  # critic updates per new transition (update-to-data ratio)
    utd_rlpd: int = 4
    utd_qc: int = 2  # QC and QC-CQL
    offline_updates: int = 5000  # QC critic updates on the prior data before the first online round
    n_cand: int = 32  # base candidates per decision for QC (acting and target)
    ensemble: int = 10
    hidden: int = 256
    batch: int = 256
    lr: float = 3e-4
    tau: float = 0.005
    init_alpha: float = 0.1  # SAC entropy temperature at the start (auto-tuned)
    cql_alpha: float = 1.0
    cql_cands: int = 8  # base candidates per state in the CQL penalty (of n_cand)
    eval_n: int = 256
    final_set: str = "eval"  # pilots scored on "search" (with --eval-n 64), never on eval
    n_workers: int = 4
    threads: int = 2  # torch threads in this process
    device: str = "auto"  # "mps" when available, else "cpu"
    out: str = ""  # output root (default: the repo; --quick: runs/ch05_quick); pilots used runs/ch05_pilots/*
    quick: bool = False
    # Smoke test: small enough for CI's CPU-only runners (a 4-member ensemble, UTD 2 for RLPD).
    QUICK: ClassVar[dict] = {"seeds": [0], "n_demos": 4, "warmup_episodes": 16, "rounds": 2, "actor_rounds": 2,
                             "round_episodes": 4, "dip_rounds": 2, "offline_updates": 50, "n_cand": 8, "eval_n": 16,
                             "ensemble": 4, "utd_rlpd": 2, "batch": 128, "final_set": "search"}


# ---- Prior data: scripted demos (like base_v1's) and warm-up rollouts of base_v1, cut into chunk decisions.


def record_demos(cfg: Config, seed: int, ledger: Ledger) -> list[dict]:
    """Half TUNED, half sloppy scripted demos on chapter-5 seeds, every attempt kept (ch00's recipe)."""
    trajs = []
    for sloppy in (False, True):
        seeds = [SEEDS0 + 1_000_000 * seed + 900_000 + 50_000 * sloppy + j for j in range(cfg.n_demos // 2)]
        noise = {"knob_std": 0.15, "action_std": 0.3} if sloppy else None
        # Hashed controller seeds, so the sloppy jitter is independent of the cube pose (ch00 caveat 5).
        res = rollout_seeds(partial(KnobController, TUNED_KNOBS[cfg.robot], cfg.robot, noise), seeds,
                            [policy_seed("ch05_demos", s) for s in seeds], robot=cfg.robot,
                            n_workers=cfg.n_workers, record_trajectories=True, ledger=ledger, category="train")
        ledger.human_minutes["demos"] += res.env_steps / 600  # an operator teleoperates at 10 Hz
        trajs += res.trajectories
    return trajs


def add_candidates(trajs: list[dict], model: FlowChunk, n: int, seed: int) -> None:
    """Base candidates at each decision state of episodes that did not log them (the scripted demos)."""
    obs = np.concatenate([t["obs"][::H] for t in trajs])
    noise = np.random.default_rng(seed).standard_normal((len(obs) * n, 32), dtype=np.float32)
    cands = model.sample(np.repeat(obs, n, 0), noise)[:, :H].reshape(len(obs), n, ADIM).astype(np.float16)
    i = 0
    for t in trajs:
        full = np.zeros((len(t["actions"]), n, ADIM), np.float16)
        k = len(full[::H])
        full[::H], i = cands[i:i + k], i + k
        t["last_cands"] = full


def transitions(trajs: list[dict], cands: bool) -> dict[str, np.ndarray]:
    """One transition per decision (every H steps): s, a (16), R = sum gamma^k r, s', done, candidates.

    Time is part of the observation, so a time-out is a true terminal state (no chance left) and both
    success and time-out end the backup. A chunk cut short by the episode's end repeats its last action.
    """
    cols: dict[str, list] = {k: [] for k in ("obs", "act", "rew", "nobs", "done", "cand", "ncand")}
    for tr in trajs:
        T = len(tr["actions"])
        ts = np.arange(0, T, H)
        idx = ts[:, None] + np.arange(H)
        cols["obs"].append(tr["obs"][ts])
        cols["act"].append(np.concatenate([tr["actions"], np.repeat(tr["actions"][-1:], H, 0)])[idx].reshape(-1, ADIM))
        cols["rew"].append(np.concatenate([tr["rewards"], np.zeros(H)])[idx] @ GAMMA ** np.arange(H))
        cols["nobs"].append(np.vstack([tr["obs"], tr["final_obs"][None]])[np.minimum(ts + H, T)])
        cols["done"].append(ts + H >= T)
        if cands:  # candidates at s, and at s' (the next decision's candidates; unused when done)
            c = tr["last_cands"][ts]
            cols["cand"].append(c)
            cols["ncand"].append(np.concatenate([c[1:], np.zeros_like(c[:1])]))
    return {k: np.concatenate(v).astype(np.float16 if "cand" in k else np.float32) for k, v in cols.items() if v}


def cat(*parts: dict) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class Replay:
    """Transitions as tensors on the training device; grows by whole rounds, samples uniformly."""

    def __init__(self, dev: str):
        self.dev, self.data = dev, {}

    def add(self, tr: dict) -> None:
        for k, v in tr.items():
            t = torch.as_tensor(v).to(self.dev)
            self.data[k] = torch.cat([self.data[k], t]) if k in self.data else t

    def __len__(self) -> int:
        return len(self.data["obs"]) if self.data else 0

    def sample(self, n: int) -> dict:
        idx = torch.randint(len(self), (n,), device=self.dev)
        return {k: v[idx] for k, v in self.data.items()}


# ---- Networks (torch, for learning) and their numpy twins (for the rollout workers).


class VLinear(nn.Module):
    """E independent linear layers in one batched matmul: x [E or 1, B, i] -> [E, B, o]."""

    def __init__(self, E: int, i: int, o: int):
        super().__init__()
        bound = 1 / math.sqrt(i)  # nn.Linear's default initialization, per member
        self.w = nn.Parameter(torch.empty(E, i, o).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(E, 1, o).uniform_(-bound, bound))


class Critic(nn.Module):
    """An ensemble of E critics Q_e(s, a) = MLP([normalized s, a]); RLPD adds LayerNorm after each layer."""

    def __init__(self, E: int, hidden: int, layernorm: bool, mu: torch.Tensor, sd: torch.Tensor):
        super().__init__()
        self.layers = nn.ModuleList([VLinear(E, len(mu) + ADIM, hidden), VLinear(E, hidden, hidden),
                                     VLinear(E, hidden, 1)])
        self.layernorm = layernorm
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, members: list[int] | None = None) -> torch.Tensor:
        """obs [..., D], act [..., 16] -> Q [E (or len(members)), ...]."""
        x = torch.cat([(obs - self.mu) / self.sd, act], -1)
        shape = x.shape[:-1]
        x = x.reshape(1, -1, x.shape[-1])
        for i, layer in enumerate(self.layers):
            w, b = (layer.w, layer.b) if members is None else (layer.w[members], layer.b[members])
            x = torch.baddbmm(b, x.expand(len(w), -1, -1), w)
            if i < 2:
                x = torch.relu(F.layer_norm(x, x.shape[-1:]) if self.layernorm else x)
        return x.reshape(len(x), *shape)

    def export(self, path) -> None:
        arrays = {f"{p}{i}": getattr(layer, p).detach().cpu().numpy() for i, layer in enumerate(self.layers)
                  for p in ("w", "b")}
        np.savez(path, mu=self.mu.cpu().numpy(), sd=self.sd.cpu().numpy(), ln=self.layernorm, **arrays)


class NumpyCritic:
    """The exported ensemble in numpy: one obs [D] and N chunks [N, 16] -> mean Q over members [N]."""

    def __init__(self, path):
        z = np.load(path)
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(3)]
        self.mu, self.sd, self.ln = z["mu"], z["sd"], bool(z["ln"])

    def __call__(self, obs: np.ndarray, chunks: np.ndarray) -> np.ndarray:
        x = np.concatenate([np.broadcast_to((obs - self.mu) / self.sd, (len(chunks), len(obs))), chunks], 1)[None]
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < 2:
                if self.ln:
                    x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
                x = np.maximum(x, 0)
        return x[..., 0].mean(0)


class Actor(nn.Module):
    """SAC's tanh-Gaussian policy over the 16-number chunk."""

    def __init__(self, hidden: int, mu: torch.Tensor, sd: torch.Tensor):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(mu), hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * ADIM))
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized a = tanh(m + s * eps) and its log-probability (with the tanh correction)."""
        m, log_s = self.net((obs - self.mu) / self.sd).chunk(2, -1)
        log_s = log_s.clamp(-5, 2)
        u = m + log_s.exp() * torch.randn_like(m)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - m) / log_s.exp()) ** 2 - log_s - 0.5 * math.log(2 * math.pi)
                - torch.log(1 - a**2 + 1e-6)).sum(-1)
        return a, logp

    def export(self, path) -> None:
        lin = [m for m in self.net if isinstance(m, nn.Linear)]
        np.savez(path, mu=self.mu.cpu().numpy(), sd=self.sd.cpu().numpy(),
                 **{f"w{i}": m.weight.detach().cpu().numpy().T for i, m in enumerate(lin)},
                 **{f"b{i}": m.bias.detach().cpu().numpy() for i, m in enumerate(lin)})


# ---- Policies the rollout workers run (picklable through functools.partial).


class ActorPolicy:
    """BatchPolicy for a SAC/RLPD actor: a new chunk every H steps, sampled (training) or its mean (eval)."""

    def __init__(self, path: str, stochastic: bool):
        z = np.load(path)
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(3)]
        self.mu, self.sd, self.stochastic = z["mu"], z["sd"], stochastic
        self.last_chunk: np.ndarray | None = None

    def reset(self, seeds) -> None:
        self.rngs, self.t = [np.random.default_rng(s) for s in seeds], 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.t % H == 0:
            x = (obs - self.mu) / self.sd
            for i, (w, b) in enumerate(self.layers):
                x = x @ w + b
                x = np.maximum(x, 0) if i < 2 else x
            m, log_s = np.split(x, 2, axis=1)
            eps = np.stack([rng.standard_normal(ADIM) for rng in self.rngs]) if self.stochastic else 0.0
            self.last_chunk = np.tanh(m + np.exp(np.clip(log_s, -5, 2)) * eps).reshape(len(obs), H, 4)
        self.t += 1
        return self.last_chunk[:, (self.t - 1) % H]


class QCPolicy(FlowChunkPolicy):
    """base_v1 choosing among N decoded candidates: "first" (its own draw: the base), "random" or "critic".

    Logs the executed half of every candidate as `last_cands`, so the TD target at s' can use exactly the
    base proposals the policy saw there.
    """

    def __init__(self, n: int, pick: str, critic_path: str | None = None):
        super().__init__(FlowChunk.load(BASE), noise=self.choose)
        self.n, self.pick = n, pick
        self.critic = NumpyCritic(critic_path) if critic_path else None
        self.prev = self.last_cands = None

    def reset(self, seeds) -> None:
        super().reset(seeds)
        self.prev = self.last_cands = None

    def choose(self, obs: np.ndarray, rngs: list) -> np.ndarray:
        out = np.zeros((len(obs), 32), np.float32)
        if self.last_cands is None:
            self.last_cands = np.zeros((len(obs), self.n, ADIM), np.float16)
        for i, (o, rng) in enumerate(zip(obs, rngs)):
            if self.prev is not None and np.array_equal(o, self.prev[i]):
                continue  # a finished env keeps receiving its last obs; its choice is never executed
            noise = rng.standard_normal((self.n, 32), dtype=np.float32)
            cands = self.sampler(np.repeat(o[None], self.n, 0), noise)[:, :H].reshape(self.n, ADIM)
            j = 0
            if self.pick == "random":
                j = int(rng.integers(self.n))
            elif self.pick == "critic":
                j = int(np.argmax(self.critic(o, cands)))
            out[i], self.last_cands[i] = noise[j], cands
        self.prev = obs.copy()
        return out


# ---- The learner: one critic update rule with switches for SAC, RLPD and QC.


def qc_bootstrap_value(online_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """Select with the acting ensemble mean, then value that same chunk with the target-pair minimum.

    Both inputs have shape [critics, batch, candidates]. Maximizing the target minimum instead would
    back up a different candidate-selection policy from the one QC actually executes.
    """
    chosen = online_q.mean(0).argmax(-1, keepdim=True)
    return target_q.min(0).values.gather(1, chosen).squeeze(1)


class Learner:
    def __init__(self, cfg: Config, method: str, seed: int, dev: str, mu: torch.Tensor, sd: torch.Tensor):
        torch.manual_seed(seed)
        self.cfg, self.qc, self.rlpd = cfg, method.startswith("qc"), method != "sac"  # QC uses RLPD's critic
        self.rng = np.random.default_rng(seed)
        self.E = cfg.ensemble if self.rlpd else 2
        self.q = Critic(self.E, cfg.hidden, self.rlpd, mu, sd).to(dev)
        self.q_targ = copy.deepcopy(self.q).requires_grad_(False)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        if not self.qc:
            self.actor = Actor(cfg.hidden, mu, sd).to(dev)
            self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
            self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=dev, requires_grad=True)
            self.al_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        # RLPD keeps prior and online data apart and samples half of every batch from each.
        self.buffers = [Replay(dev), Replay(dev)] if method == "rlpd" else [Replay(dev)]

    def batch(self) -> dict:
        parts = [b.sample(self.cfg.batch // len(self.buffers)) for b in self.buffers]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    def update(self, cql: bool = False, actor: bool = False) -> torch.Tensor:
        b = self.batch()
        with torch.no_grad():
            pair = self.rng.choice(self.E, 2, replace=False).tolist()  # min over a random pair (REDQ/RLPD)
            if self.qc:  # the acting argmax, valued pessimistically by the target critic pair
                B, N, _ = b["ncand"].shape
                obs, cand = b["nobs"][:, None].expand(B, N, -1), b["ncand"].float()
                v = qc_bootstrap_value(self.q(obs, cand), self.q_targ(obs, cand, pair))
            else:
                a, _ = self.actor.sample(b["nobs"])
                v = self.q_targ(b["nobs"], a, pair).min(0).values  # no entropy term in the backup (see doc)
            y = b["rew"] + GAMMA**H * (1 - b["done"]) * v
        q = self.q(b["obs"], b["act"])  # [E, B]
        loss = ((q - y) ** 2).mean(1).sum()
        if cql:  # push down the base's candidates at s relative to the action in the data (CQL-style)
            cand = b["cand"][:, : self.cfg.cql_cands].float()
            qc = self.q(b["obs"][:, None].expand(*cand.shape[:2], -1), cand)
            loss = loss + self.cfg.cql_alpha * (torch.logsumexp(qc, -1) - q).mean(1).sum()
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.lerp_(p, self.cfg.tau)
        if actor:
            self.update_actor(b["obs"])
        return q.detach().mean()  # a tensor: converting to float here would stall the GPU every update

    def update_actor(self, obs: torch.Tensor) -> None:
        a, logp = self.actor.sample(obs)
        q = self.q(obs, a)
        q = q.mean(0) if self.rlpd else q.min(0).values  # RLPD's actor uses the ensemble mean
        loss = (self.log_alpha.exp().detach() * logp - q).mean()
        self.a_opt.zero_grad()
        loss.backward()
        self.a_opt.step()
        alpha_loss = -(self.log_alpha * (logp.detach() - ADIM / 2)).mean()  # target entropy -|A|/2
        self.al_opt.zero_grad()
        alpha_loss.backward()
        self.al_opt.step()

    def policy(self, path, train: bool):
        """Export the current weights and return a picklable policy factory for the rollout workers."""
        if self.qc:
            self.q.export(path)
            return partial(QCPolicy, self.cfg.n_cand, "critic", str(path))
        self.actor.export(path)
        return partial(ActorPolicy, str(path), train)


def run_method(cfg: Config, method: str, seed: int, prior: dict, dev: str, mu, sd, ledger: Ledger, work) -> dict:
    """Offline phase (QC only), then online rounds: roll out the current policy, add the data, train."""
    t0 = time.perf_counter()
    learner = Learner(cfg, method, seed, dev, mu, sd)
    learner.buffers[0].add(prior)
    if learner.qc:
        for _ in range(cfg.offline_updates):
            learner.update(cql=method == "qc-cql")
    curve, utd = [], {"sac": cfg.utd_sac, "rlpd": cfg.utd_rlpd}.get(method, cfg.utd_qc)
    n_rounds = {"qc": cfg.rounds, "qc-cql": cfg.dip_rounds}.get(method, cfg.actor_rounds)
    for r in range(n_rounds):
        make = learner.policy(work / f"{method}_s{seed}.npz", train=True)
        # Every method sees the same online env seeds (common random numbers across methods).
        idx = [cfg.round_episodes * r + i for i in range(cfg.round_episodes)]
        res = rollout_seeds(make, [SEEDS0 + 1_000_000 * seed + 100_000 + i for i in idx],
                            [policy_seed(f"ch05_online_s{seed}", i) for i in idx], robot=cfg.robot,
                            n_workers=cfg.n_workers, record_trajectories=True, ledger=ledger, category="train")
        tr = transitions(res.trajectories, cands=learner.qc)
        learner.buffers[-1].add(tr)
        n_up = utd * len(tr["obs"])
        mean_q = float(torch.stack([learner.update(actor=not learner.qc and u % utd == utd - 1)
                                    for u in range(n_up)]).mean())
        curve.append({"k": res.k, "n": res.n, "train_steps": ledger.robot_steps["train"], "mean_q": mean_q})
        print(f"  {method} seed {seed} round {r + 1}/{n_rounds}: {format_rate(res.k, res.n)}, "
              f"{n_up} updates, mean Q {mean_q:.3f}, {time.perf_counter() - t0:.0f}s", flush=True)
    final = work / f"{method}_s{seed}_final.npz"
    return {"curve": curve, "make_eval": learner.policy(final, train=False), "minutes": (time.perf_counter() - t0) / 60,
            "make_stochastic": None if learner.qc else partial(ActorPolicy, str(final), True)}


# ---- Plots and GIF


def make_plots(cfg: Config, media, s: dict) -> None:
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    plotting.setup()
    P = plotting.PALETTE
    colors = {"sac": P[3], "rlpd": P[4], "qc": P[0], "qc-cql": P[2]}
    labels = {"sac": "SAC + prior data", "rlpd": "RLPD", "qc": "QC (best-of-N, warm-up)",
              "qc-cql": "QC-CQL (demos only, no warm-up)"}
    base = s["eval"]["base"]
    final_label = "search" if cfg.final_set == "search" else f"held-out {cfg.final_set}"

    def pooled_curve(m):
        curves = [s["curves"][m][str(seed)] for seed in cfg.seeds]
        k = [sum(c[r]["k"] for c in curves) for r in range(len(curves[0]))]
        n = [sum(c[r]["n"] for c in curves) for r in range(len(curves[0]))]
        x = [np.mean([c[r]["train_steps"] for c in curves]) / 600 for r in range(len(curves[0]))]
        return x, k, n

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in cfg.methods:
        x, k, n = pooled_curve(m)
        plotting.success_curve(ax, x, k, n, label=f"{labels[m]}: training rollouts", color=colors[m])
        ev = s["eval"]["pooled"][m]
        plotting.heldout_point(ax, x[-1] * 1.04, ev[0], ev[1], label=f"{labels[m]}: {final_label}", color=colors[m])
    ax.axhline(base[0] / base[1], ls="--", color="black", lw=1, label=f"base_v1, {final_label}")
    ax.set(xlabel="fresh train robot-minutes (demos + warm-up + online, mean over seeds)", ylabel="success",
           title=f"Learning curves, {len(cfg.seeds)} seeds pooled")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plotting.save_fig(fig, media / "learning_curves.png")

    if {"qc", "qc-cql"} <= set(cfg.methods):
        fig, ax = plt.subplots()
        for m in ("qc", "qc-cql"):
            _, k, n = pooled_curve(m)
            r = min(len(k), cfg.dip_rounds)
            plotting.success_curve(ax, range(1, r + 1), k[:r], n[:r], label=labels[m], color=colors[m])
        ax.axhline(base[0] / base[1], ls="--", color="black", lw=1, label=f"base_v1, {final_label}")
        ax.set(xlabel="online round (round 1 = the offline critic, before any online update)",
               ylabel="success (training rollouts)", title="Offline to online: is there a dip?")
        ax.legend(fontsize=8, loc="lower right")
        plotting.save_fig(fig, media / "dip.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs, ticks = 0, []
    for m in cfg.methods:
        for j, seed in enumerate(cfg.seeds):
            k, n = s["eval"]["per_seed"][m][str(seed)]
            p, err = plotting.wilson_err([k], n)
            ax.errorbar([xs], p, yerr=err, fmt="o", color=colors[m], capsize=3,
                        label=labels[m] if j == 0 else None)
            xs += 1
        ticks.append(xs - (len(cfg.seeds) + 1) / 2)
        xs += 1
    ax.axhline(base[0] / base[1], ls="--", color="black", lw=1, label="base_v1")
    ax.set_xticks(ticks, cfg.methods)
    ax.set(ylabel=f"success ({cfg.final_set}, n={cfg.eval_n} per seed)", ylim=(0, 1.02),
           title=f"{final_label.capitalize()}, one point per seed")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plotting.save_fig(fig, media / "eval_per_seed.png")


def make_gif(cfg: Config, media, base_trajs: list, qc_trajs: list) -> int | None:
    """The first eval seed the base fails and QC (first seed) solves, side by side."""
    from lastmile.common.plotting import replay_frames, save_gif, side_by_side
    from lastmile.envs.cupdrop import CupDropEnv

    pairs = [(b, q) for b, q in zip(base_trajs, qc_trajs) if not b["success"] and q["success"]]
    if not pairs:
        return None
    b, q = pairs[0]
    env = CupDropEnv(robot=cfg.robot)
    clips = [replay_frames(env, t, text, every=3) for t, text in
             ((b, f"base_v1 (seed {b['seed']})"), (q, f"QC best-of-{cfg.n_cand}"))]
    save_gif(side_by_side(*clips), media / "base_vs_qc.gif", fps=5, max_size=400)
    return int(b["seed"])


# ---- The run


def main(cfg: Config) -> None:
    torch.set_num_threads(cfg.threads)
    dev = ("mps" if torch.backends.mps.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    out = REPO_ROOT / (cfg.out or ("runs/ch05_quick" if cfg.quick else ""))
    media, work = out / "media/ch05", out / "runs/ch05_work"
    work.mkdir(parents=True, exist_ok=True)
    model = FlowChunk.load(BASE)
    mu, sd = model.obs_mean.clone(), model.obs_std.clone()
    s: dict = {"device": dev, "curves": {m: {} for m in cfg.methods}, "regime": {}, "minutes": {}}
    wall0 = time.perf_counter()

    with ExitStack() as stack:
        # All rows are open for the whole run, so each row's cpu_hours = this process's CPU for the WHOLE run
        # (every method's training and eval, shared by all rows) + that row's own rollout workers. Do not sum
        # cpu_hours over rows; the qc-random row (no training) shows the shared part.
        led = partial(Ledger, chapter="ch05", robot=cfg.robot, config=cfg, results_root=out / "results")
        rows = {(m, seed): stack.enter_context(led(method=f"{m}_s{seed}")) for m in cfg.methods for seed in cfg.seeds}
        rnd_row = stack.enter_context(led(method="qc-random")) if "qc" in cfg.methods else None
        ev = partial(evaluate, init_set=cfg.final_set, robot=cfg.robot, n_workers=cfg.n_workers, n=cfg.eval_n)
        base_eval = ev(partial(FlowChunkPolicy.load, BASE, noise="gaussian"), record_trajectories=True)
        print(f"base_v1 ({cfg.final_set}): {base_eval.summary}", flush=True)

        finals, shared = {}, {}
        for seed in cfg.seeds:
            # 1. Prior data for this seed: demos and warm-up rollouts of base_v1 (both charged to `train`).
            shared[seed] = {"demos": Ledger("ch05", "demos"), "warmup": Ledger("ch05", "warmup")}  # never written
            demos = record_demos(cfg, seed, shared[seed]["demos"])
            add_candidates(demos, model, cfg.n_cand, seed)
            wseeds = [SEEDS0 + 1_000_000 * seed + i for i in range(cfg.warmup_episodes)]
            warm = rollout_seeds(partial(QCPolicy, cfg.n_cand, "first"), wseeds,
                                 [policy_seed(f"ch05_warmup_s{seed}", i) for i in range(len(wseeds))],
                                 robot=cfg.robot, n_workers=cfg.n_workers, record_trajectories=True,
                                 ledger=shared[seed]["warmup"], category="train")
            d_demo, d_warm = transitions(demos, True), transitions(warm.trajectories, True)
            s["regime"][str(seed)] = {"demos": [int(sum(t["success"] for t in demos)), len(demos)],
                                      "warmup_base": [warm.k, warm.n]}
            s.setdefault("shared_steps", {})[str(seed)] = {k: v.robot_steps["train"] for k, v in shared[seed].items()}
            print(f"seed {seed}: demos {format_rate(*s['regime'][str(seed)]['demos'])}, "
                  f"warm-up base rollouts {warm.summary}", flush=True)
            no_cands = {k: v for k, v in cat(d_demo, d_warm).items() if "cand" not in k}
            priors = {"sac": no_cands, "rlpd": no_cands, "qc": cat(d_demo, d_warm), "qc-cql": d_demo}

            # 2. Each method: train, then score the final policy once on eval (paired with the base).
            for m in cfg.methods:
                row = rows[m, seed]
                for part in ("demos",) if m == "qc-cql" else ("demos", "warmup"):  # the prior data it used
                    row.robot_steps["train"] += shared[seed][part].robot_steps["train"]
                    row.human_minutes["demos"] += shared[seed][part].human_minutes["demos"]
                    row.worker_cpu_seconds += shared[seed][part].worker_cpu_seconds
                res = run_method(cfg, m, seed, priors[m], dev, mu, sd, row, work)
                s["curves"][m][str(seed)], s["minutes"][f"{m}_s{seed}"] = res["curve"], res["minutes"]
                finals[m, seed] = ev(res["make_eval"], ledger=row, category="eval",
                                     record_trajectories=(m == "qc" and seed == cfg.seeds[0]))
                print(f"{m} seed {seed} ({cfg.final_set}): {finals[m, seed].summary}", flush=True)
                if res["make_stochastic"]:  # an extra, not the row's result: SAC is scored with its mean action
                    st = ev(res["make_stochastic"], ledger=row, category="eval")
                    s.setdefault("eval_stochastic_actor", {}).setdefault(m, {})[str(seed)] = [st.k, st.n]
                    print(f"{m} seed {seed} sampled actor ({cfg.final_set}, extra): {st.summary}", flush=True)

        # 3. Controls: QC's proposal distribution with a random pick (the base again, drawn differently).
        rnd_eval = ev(partial(QCPolicy, cfg.n_cand, "random"), ledger=rnd_row, category="eval") if rnd_row else None

        # 4. Statistics, the regime check, failure modes, costs.
        b0 = base_eval.successes
        s["eval"] = {"base": [base_eval.k, base_eval.n], "per_seed": {}, "pooled": {}, "mcnemar_p": {},
                     "diff_ci": {}, "paired": {}}
        for m in cfg.methods:
            fs = {str(seed): finals[m, seed] for seed in cfg.seeds}
            s["eval"]["per_seed"][m] = {k: [f.k, f.n] for k, f in fs.items()}
            s["eval"]["pooled"][m] = [sum(f.k for f in fs.values()), sum(f.n for f in fs.values())]
            s["eval"]["mcnemar_p"][m] = {k: mcnemar_exact(b0, f.successes) for k, f in fs.items()}
            s["eval"]["diff_ci"][m] = {k: bootstrap_diff(b0, f.successes) for k, f in fs.items()}
            s["eval"]["paired"][m] = {k: {"fixed": int((~b0 & f.successes).sum()), "broke": int((b0 & ~f.successes).sum()),
                                          "both_fail": int((~b0 & ~f.successes).sum())} for k, f in fs.items()}
        if rnd_eval is not None:
            s["eval"]["random"] = [rnd_eval.k, rnd_eval.n]
            s["eval"]["random_mcnemar_p"] = mcnemar_exact(b0, rnd_eval.successes)
        reg = s["regime"]
        s["regime"]["pooled"] = {k: [sum(reg[str(x)][k][i] for x in cfg.seeds) for i in (0, 1)]
                                 for k in ("demos", "warmup_base")}
        from lastmile.envs.cupdrop import CupDropEnv, OutcomeTracker

        env = CupDropEnv(robot=cfg.robot)
        s["failure_modes"] = {}
        if "qc" in cfg.methods:
            for name, r in (("base", base_eval), (f"qc_s{cfg.seeds[0]}", finals["qc", cfg.seeds[0]])):
                labels = [OutcomeTracker.label_trajectory(env, t) for t in r.trajectories]
                s["failure_modes"][name] = {k: labels.count(k) for k in sorted(set(labels))}
        prior = PRIOR_STEPS if not cfg.quick else {}
        for row in [*rows.values(), *([rnd_row] if rnd_row else [])]:
            for item in prior.values():
                for c in ("search", "train"):
                    row.robot_steps[c] += item.get(c, 0)
            if not cfg.quick:
                for item in PRIOR_HUMAN_MINUTES.values():
                    for c, minutes in item.items():
                        row.human_minutes[c] += minutes
            row.robot_steps["eval"] += base_eval.env_steps
            row.set_base(id="base_v1", successes=base_eval)
        for (m, seed), row in rows.items():
            row.set_final(successes=finals[m, seed])
            s["cost_steps_" + f"{m}_s{seed}"] = dict(row.robot_steps)
        if rnd_row:
            rnd_row.set_final(successes=rnd_eval)
        s["prior_steps"] = prior
        s["prior_human_minutes"] = PRIOR_HUMAN_MINUTES if not cfg.quick else {}

        s["gif_seed"] = None
        if "qc" in cfg.methods:
            with optional_media("the base vs QC GIF"):
                s["gif_seed"] = make_gif(cfg, media, base_eval.trajectories, finals["qc", cfg.seeds[0]].trajectories)
        make_plots(cfg, media, s)
        s["wall_minutes"] = (time.perf_counter() - wall0) / 60
        for row in [*rows.values(), *([rnd_row] if rnd_row else [])]:
            row.extra = s

    print(f"\n=== Chapter 5 summary ({cfg.final_set}, n = {cfg.eval_n} per seed) ===")
    print(f"base_v1:          {base_eval.summary}")
    for m in cfg.methods:
        for seed in cfg.seeds:
            f = finals[m, seed]
            print(f"{m:7s} seed {seed}:   {f.summary}  McNemar p={s['eval']['mcnemar_p'][m][str(seed)]:.2g}  "
                  f"vs base: {s['eval']['paired'][m][str(seed)]}")
        print(f"{m:7s} pooled:   {format_rate(*s['eval']['pooled'][m])}")
    if rnd_eval is not None:
        print(f"random pick of {cfg.n_cand}: {rnd_eval.summary}  McNemar p={s['eval']['random_mcnemar_p']:.2g}")
    reg = s["regime"]["pooled"]
    print(f"three regimes (both on chapter-5 seeds): demos {format_rate(*reg['demos'])} vs base_v1 warm-up "
          f"rollouts {format_rate(*reg['warmup_base'])} (base_v1 on {cfg.final_set}: {base_eval.summary})")
    for (m, seed), row in rows.items():
        print(f"row {m}_s{seed} robot-min:", {c: round(v, 1) for c, v in row.robot_minutes.items()},
              f"(method wall {s['minutes'][f'{m}_s{seed}']:.1f} min)")
    print(f"wall {s['wall_minutes']:.1f} min on {dev}; results in {out / 'results/ch05'}")


if __name__ == "__main__":
    main(parse(Config))
