"""Chapter 8: propose, edit, select. An EXPO-style edit policy with Q selection on the frozen base.

Learning goal
    Use the frozen base as a *proposer*, a small bounded *editor* as the only thing RL trains (besides the
    critic), and a Q ensemble as the *selector*. This is the recipe behind EXPO-FT's real-robot fine-tuning.
    See what the edit adds over plain best-of-N with the same learned critic, what the edit bound beta does,
    what it costs in robot-minutes and in latency, and where it breaks.

Base idea in plain words
    At every chunk decision the frozen base proposes N = 8 chunks (8 noise vectors). The editor adds a
    bounded correction to each one, and the robot executes the best of the 2N candidates under the critic:

        a_j ~ base(s),   a~_j = clip(a_j + beta * tanh(u_j)),   u_j ~ N(mu(s, a_j), sigma(s, a_j))
        a*  = argmax over {a_1..a_N, a~_1..a~_N} of  mean_i Q_i(s, a)

    The critic's "action" is the 4 executed steps of the chunk (16 numbers), with the 4-step chunk target
    of Q-chunking. It learns by TD on replay, and the TD target uses the SAME selection rule at the next
    state (EMaQ / EXPO style: the policy being evaluated is "propose, edit, take the argmax"):

        y = r + gamma^k (1 - done) * min_{i in M} Q'_i(s', a'*),   a'* = argmax over the 2N candidates at s'

    where M is a random pair of the target ensemble (REDQ: pessimism against overestimation). The editor
    is trained SAC-style to raise Q, with an entropy term whose weight alpha is tuned automatically:

        L_edit = E[ alpha * log pi(u | s, a) - mean_i Q_i(s, clip(a + beta * tanh(u))) ]

    No RL gradient enters the base. beta = 0 turns the editor off: best-of-N with a TD-learned critic
    (Chapter 12 used Monte-Carlo labels instead). Small beta keeps every executed chunk within beta of a
    base proposal. beta = 2 lets the editor reach any action in [-1, 1] (a stand-in for MiDAS's full-chunk
    editor, which can leave the base's support). Co-trained variant: the base also keeps training with its
    flow-matching loss, but only on the chunks of successful episodes (success-filtered BC, the Chapter 9
    idea). That is a deviation from EXPO, which trains the base on all replay mini-batches.

    Controls that split the gain between editing and selecting (all on eval, paired with the base):
    beta = 0 with N = 16 (as many candidates as beta > 0, trained), a uniform random pick among the 2N
    candidates, and the deterministic edit of a random proposal with no critic pick.

    Selection and small edits can only reach behavior near what the base already proposes: the ceiling of
    beta = 0 is the base's support, and a beta-bounded edit widens it by beta per action, no more.

Papers mirrored
    EXPO (Dong et al. 2025, arXiv 2507.07986; its code is in data/resources.csv), EXPO-FT [C276],
    Real-Time EXPO-FT [C412], DICE-RL [C235] (residual on a frozen BC sampler, best-Q sample), MiDAS [C402]
    (full-chunk editor), FASTER [C263] (pick the noise seed before denoising, to save compute), RL Token
    [C264] (tiny actor-critic that edits a frozen VLA's chunk). Classic pieces: EMaQ [N436] (max over N
    behavior samples in the backup), REDQ (arXiv 2101.05982), RLPD [C037], Q-chunking [C133].

Honesty
    Training episodes use their own env seeds (200_000 + 100_000 * seed + i), charged to `train`. Learning
    curves and the final selection score each run on the `search` set (charged to `search`). beta is chosen
    by the pooled final search score; every row of the chosen beta (and the co-trained rows built on it)
    pays for all the beta-sweep runs, and each row says whether it was the one chosen (extra["role"]).
    The held-out `eval` set (n = 256, salt 0, paired with base_v1) is scored once per run, after every
    choice was made.

Run
    uv run python algos/08_expo_edit.py            # full run: about 65 min on an M5 Pro (4 processes)
    uv run python algos/08_expo_edit.py --quick    # smoke test, < 2 min; writes only to runs/ch08_quick/
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np

from lastmile.common.cli import parse
from lastmile.common.eval import INIT_SETS, bootstrap_diff, format_rate, mcnemar_exact, policy_seed, wilson
from lastmile.common.ledger import REPO_ROOT, Ledger
from lastmile.common.plotting import optional_media
from lastmile.common.rollout import evaluate, rollout_seeds

# torch, policy_flow and matplotlib are imported inside functions: spawned rollout workers re-import this
# file, and only the ones that build a policy should pay for torch.

BASE = REPO_ROOT / "checkpoints/base_v1_so101.pt"
EXECUTE = 4  # the base executes 4 actions of each 8-action chunk, then replans
ACT = EXECUTE * 4  # the critic's action: the 4 executed steps x [dx, dy, dz, grip]
TRAIN_SEED_BASE = 200_000  # far from search (10k), eval (20k), eval_ext (30k), stress (40k), ch12 (70k-95k)


@dataclass
class Config:
    robot: str = "so101"
    betas: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 2.0],
                               metadata={"help": "edit bounds; 0 = best-of-N with the learned critic"})
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    cotrain: bool = True  # phase 2: co-train the base (flow-matching BC on successes) at the chosen beta
    n_props: int = 8  # N base proposals per decision (2N candidates with the edits)
    wide_n: int = 16  # control: beta = 0 trained with this N (as many candidates as beta > 0); 0 = skip
    n_envs: int = 16  # envs stepped in lockstep by each training process
    train_steps: int = 60_000  # env steps of online interaction per run (100 robot-minutes)
    warmup: int = 1_000  # transitions in replay before the first gradient step
    updates_per_tick: int = 16  # gradient steps per round of n_envs chunk decisions (UTD ~1 per chunk)
    batch: int = 256
    lr: float = 3e-4
    hidden: int = 256
    ensemble: int = 5  # REDQ-style ensemble (10 in REDQ; 5 halves the cost of scoring 2N candidates)
    subset: int = 2  # target = min over this many random target critics
    gamma: float = 0.99  # per env step; a chunk of k steps discounts by gamma^k
    tau: float = 0.005  # Polyak rate of the target critics
    target_entropy: float = -16.0  # SAC target for the editor's (tanh-corrected) entropy, -dim(edit)
    bc_lr: float = 1e-4  # co-trained base: flow-matching fine-tuning step size
    bc_steps_per_tick: int = 1
    refresh_every: int = 100  # co-trained base: re-propose the replay's next-state chunks every k ticks
    curve_points: int = 6  # search-set evaluations along each run
    final_salts: int = 2  # search-set salts for the final score that picks beta
    search_n: int = 64  # search states per search-set evaluation (all 64 in full runs)
    eval_n: int = 256
    support_salts: int = 16  # base rollouts on each eval state the chosen run fails: does the base solve it?
    latency_ns: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    n_workers: int = 4  # training processes and rollout workers
    torch_threads: int = 1  # per training process
    quick: bool = False
    QUICK: ClassVar[dict] = {"betas": [0.0, 0.1], "seeds": [0], "train_steps": 1_200, "warmup": 200,
                             "updates_per_tick": 4, "curve_points": 1, "final_salts": 1, "search_n": 16,
                             "eval_n": 32, "refresh_every": 10, "latency_ns": [1, 8], "n_envs": 8,
                             "support_salts": 4,
                             "ensemble": 3}


def run_name(beta: float, cotrain: bool = False, wide: int = 0) -> str:
    return (("bon-q" if beta == 0 else f"expo-b{beta:g}") + ("-cotrain" if cotrain else "")
            + (f"-n{wide}" if wide else ""))


# ---- Networks (torch for training, plain numpy for the rollout workers)


def make_critic(n: int, d_in: int, hidden: int):
    """n independent LayerNorm MLP critics evaluated in one broadcast matmul (RLPD-style critics)."""
    import torch
    from torch import nn

    class Ensemble(nn.Module):
        def __init__(self):
            super().__init__()
            shapes = [(d_in, hidden), (hidden, hidden), (hidden, 1)]
            self.w = nn.ParameterList([nn.Parameter(torch.empty(n, i, o).uniform_(-i**-0.5, i**-0.5))
                                       for i, o in shapes])
            self.b = nn.ParameterList([nn.Parameter(torch.zeros(n, 1, o)) for _, o in shapes])

        def forward(self, x, members=None):
            """x [B, d_in] -> Q [n, B] (or [len(members), B])."""
            ws = [w if members is None else w[members] for w in self.w]
            bs = [b if members is None else b[members] for b in self.b]
            h = x
            for i in range(2):
                h = torch.relu(nn.functional.layer_norm(h @ ws[i] + bs[i], (hidden,)))
            return (h @ ws[2] + bs[2]).squeeze(-1)

    return Ensemble()


def make_editor(d_in: int, hidden: int):
    """Gaussian edit policy in tanh space. Zero-initialized mean: it starts by proposing no edit."""
    from torch import nn

    class Editor(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
            self.mu, self.log_std = nn.Linear(hidden, ACT), nn.Linear(hidden, ACT)
            for layer, bias in ((self.mu, 0.0), (self.log_std, -1.0)):
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, bias)

        def forward(self, x):
            h = self.body(x)
            return self.mu(h), self.log_std(h).clamp(-5.0, 1.0)

    return Editor()


class NumpyNets:
    """The trained critic ensemble and editor in plain numpy, for the rollout workers."""

    def __init__(self, ckpt: dict):
        c = ckpt["critic"]
        self.cw = [c[f"w.{i}"].numpy() for i in range(3)]
        self.cb = [c[f"b.{i}"].numpy() for i in range(3)]
        e = ckpt["editor"]
        self.ew = None if e is None else [(e[f"{k}.weight"].numpy().T, e[f"{k}.bias"].numpy())
                                          for k in ("body.0", "body.2", "mu")]
        self.mu, self.sd, self.beta = ckpt["obs_mean"], ckpt["obs_std"], ckpt["beta"]

    def q(self, x: np.ndarray) -> np.ndarray:
        """x [R, d_in] -> Q [n_critics, R]."""
        h = x
        for i in range(2):
            h = h @ self.cw[i] + self.cb[i]
            h = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-5)
            h = np.maximum(h, 0)
        return (h @ self.cw[2] + self.cb[2])[..., 0]

    def edit(self, x: np.ndarray, props: np.ndarray) -> np.ndarray:
        """Deterministic edit at evaluation time: the mean of the edit policy."""
        h = x
        for w, b in self.ew[:2]:
            h = np.maximum(h @ w + b, 0)
        mu = h @ self.ew[2][0] + self.ew[2][1]
        return np.clip(props + self.beta * np.tanh(mu), -1.0, 1.0)


# ---- The deployable policy: propose N, edit each, execute the argmax-Q candidate (a BatchPolicy)


class ExpoPolicy:
    """Built inside rollout workers from a saved checkpoint (use functools.partial(expo_policy, ...)).

    Proposals come from env i's own generator, so an episode's result does not depend on its batch or
    worker (at the first decision, proposal 0 uses the same noise as the base's own first chunk; later
    the streams differ, since this policy draws N noise vectors per decision). ``pick`` is the selection
    rule: "q" (argmax of the critic mean, the method), or a control: "random-base" (a random proposal:
    the base distribution again), "random-all" (a random one of the 2N candidates) and "edit-random"
    (the edit of a random proposal: the editor alone, no critic pick).
    """

    def __init__(self, ckpt_path: str, n: int | None = None, pick: str = "q", record: bool = False):
        import torch

        from lastmile.common.policy_flow import FlowChunk

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.sampler = FlowChunk.load(REPO_ROOT / ckpt["base"]).numpy_sampler()
        self.nets, self.n, self.pick, self.record = NumpyNets(ckpt), n or ckpt["n"], pick, record
        self.rngs: list = []
        self.t = 0

    def reset(self, seeds) -> None:
        self.rngs = [np.random.default_rng(s) for s in seeds]
        self.t, self.prev = 0, None

    def decide(self, o: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, int]:
        """One env's decision: (candidates [C, 16], scores [C], chosen index)."""
        noise = rng.standard_normal((self.n, 32), dtype=np.float32)
        props = self.sampler(np.repeat(o[None], self.n, 0), noise)
        props = props[:, :EXECUTE].reshape(self.n, ACT)
        on = np.broadcast_to((o - self.nets.mu) / self.nets.sd, (self.n, len(o)))
        cands = props
        if self.nets.beta > 0:
            cands = np.concatenate([props, self.nets.edit(np.concatenate([on, props], 1), props)])
        x = np.concatenate([np.broadcast_to(on[:1], (len(cands), len(o))), cands], 1).astype(np.float32)
        scores = self.nets.q(x).mean(0)
        j = {"q": lambda: int(np.argmax(scores)), "random-base": lambda: int(rng.integers(self.n)),
             "random-all": lambda: int(rng.integers(len(cands))),
             "edit-random": lambda: self.n + int(rng.integers(self.n))}[self.pick]()
        return cands, scores, j

    def act(self, obs: np.ndarray) -> np.ndarray:
        k = self.t % EXECUTE
        if k == 0:
            B, C = len(obs), self.n * (2 if self.nets.beta > 0 else 1)
            if self.prev is None:
                self.plan = np.zeros((B, EXECUTE, 4), np.float32)
                self.last_q, self.last_choice = np.zeros(B, np.float32), np.zeros(B, np.int64)
                if self.record:
                    self.last_cands = np.zeros((B, C, ACT), np.float32)
                    self.last_scores = np.zeros((B, C), np.float32)
            for i, (o, rng) in enumerate(zip(obs, self.rngs)):
                # A finished env keeps receiving its last obs; skip it (its choice is never executed).
                if self.prev is not None and np.array_equal(o, self.prev[i]):
                    continue
                cands, scores, j = self.decide(o, rng)
                self.plan[i] = cands[j].reshape(EXECUTE, 4)
                self.last_q[i], self.last_choice[i] = scores[j], j
                if self.record:
                    self.last_cands[i], self.last_scores[i] = cands, scores
            self.prev = obs.copy()
        self.last_step_in_chunk = np.full(len(obs), k)
        self.t += 1
        return self.plan[:, k]


def expo_policy(ckpt_path: str, pick: str = "q") -> ExpoPolicy:
    return ExpoPolicy(ckpt_path, pick=pick)


def base_policy():
    from lastmile.common.policy_flow import FlowChunkPolicy

    return FlowChunkPolicy.load(BASE, noise="gaussian")


# ---- One training run (runs in its own process): online RL with replay, lockstep envs, chunk decisions


def train_run(cfg: Config, beta: float, seed: int, cotrain: bool, wide: int, work: str) -> dict:
    """One run. wide > 0: the beta = 0 control trained with N = wide proposals instead of cfg.n_props."""
    import torch

    from lastmile.common.policy_flow import FlowChunk
    from lastmile.envs.cupdrop import CupDropEnv

    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(seed)
    t_wall, cpu0 = time.perf_counter(), time.process_time()
    name = f"{run_name(beta, cotrain, wide)}_s{seed}"
    rng = np.random.default_rng([seed, 8, int(cotrain), round(beta * 1000)] + ([wide] if wide else []))
    base = FlowChunk.load(BASE)
    mu, sd = base.obs_mean.clone(), base.obs_std.clone()  # the frozen normalizer, also for co-training
    d_obs, N, B, E = base.config.obs_dim, wide or cfg.n_props, cfg.n_envs, cfg.ensemble
    critic, target = make_critic(E, d_obs + ACT, cfg.hidden), make_critic(E, d_obs + ACT, cfg.hidden)
    target.load_state_dict(critic.state_dict())
    q_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    editor = make_editor(d_obs + ACT, cfg.hidden) if beta > 0 else None
    if editor is not None:
        e_opt = torch.optim.Adam(editor.parameters(), lr=cfg.lr)
        log_alpha = torch.tensor(np.log(0.1), requires_grad=True)
        a_opt = torch.optim.Adam([log_alpha], lr=cfg.lr)
    bc_opt = torch.optim.AdamW(base.parameters(), lr=cfg.bc_lr, weight_decay=1e-4) if cotrain else None
    base.requires_grad_(cotrain)

    def norm(o):
        return (torch.as_tensor(o, dtype=torch.float32) - mu) / sd

    @torch.no_grad()
    def propose(obs_np: np.ndarray) -> np.ndarray:
        """N base chunks per state, executed part only: [R, N, 16]."""
        o = torch.as_tensor(np.repeat(obs_np, N, 0), dtype=torch.float32)
        w = torch.as_tensor(rng.standard_normal((len(o), 32), dtype=np.float32))
        return base.sample_tensor(o, w)[:, :EXECUTE].reshape(len(obs_np), N, ACT).numpy()

    def candidates(on, props, stochastic: bool):
        """on [R, D], props [R, N, 16] (tensors) -> candidates [R, C, 16], plus (u, mu, log_std)."""
        if editor is None:
            return props, None
        m, ls = editor(torch.cat([on[:, None].expand(-1, N, -1), props], -1))
        u = m + ls.exp() * torch.randn_like(m) if stochastic else m
        return torch.cat([props, (props + beta * torch.tanh(u)).clamp(-1, 1)], 1), (u, m, ls)

    def q_all(net, on, cands, members=None):
        """Q of every candidate: [n_critics, R, C]."""
        R, C = cands.shape[:2]
        x = torch.cat([on[:, None].expand(-1, C, -1), cands], -1).reshape(R * C, -1)
        return net(x, members).reshape(-1, R, C)

    # Replay, one row per chunk decision. The next state's proposals are the next decision's own (EMaQ:
    # the backup maxes over fresh base samples at s'), so they are stored once and never re-sampled.
    cap = cfg.train_steps + B
    buf = {"obs": np.zeros((cap, d_obs), np.float32), "act": np.zeros((cap, ACT), np.float32),
           "rew": np.zeros(cap, np.float32), "k": np.zeros(cap, np.float32),
           "done": np.zeros(cap, np.float32),
           "nxt": np.full(cap, -1, np.int64), "props": np.zeros((cap, N, ACT), np.float32),
           "ready": np.zeros(cap, bool)}
    ptr, steps, episodes, n_updates, tick = 0, 0, 0, 0, 0
    envs = [CupDropEnv(robot=cfg.robot) for _ in range(B)]

    def new_episode(i):
        nonlocal episodes
        s = TRAIN_SEED_BASE + 100_000 * seed + episodes
        episodes += 1
        return envs[i].reset(seed=s)[0]

    obs = np.stack([new_episode(i) for i in range(B)])
    last = np.full(B, -1)
    ep_log = [{"obs": [], "actions": []} for _ in range(B)]
    bc_obs, bc_chunks = [], []  # co-training data: (obs_t, actions[t:t+8]) from successful episodes
    online, curve, diag, search_steps = [], [], {"edited_chosen": [], "edit_size": [], "q_loss": []}, 0
    marks = [int(cfg.train_steps * (i + 1) / cfg.curve_points) for i in range(cfg.curve_points)]
    ckpt_path = Path(work) / f"{name}.pt"

    def save():
        path = ckpt_path
        base_rel = BASE.relative_to(REPO_ROOT)
        if cotrain:
            base_rel = Path(work).relative_to(REPO_ROOT) / f"{name}_base.pt"
            base.save(REPO_ROOT / base_rel, note=f"base_v1 co-trained in ch08 run {name}")
        torch.save({"critic": critic.state_dict(), "editor": None if editor is None else editor.state_dict(),
                    "obs_mean": mu.numpy(), "obs_std": sd.numpy(), "beta": beta, "n": N,
                    "base": str(base_rel)}, path)
        return str(path)

    def search_eval(salts) -> list:
        nonlocal search_steps
        out = []
        for s in salts:
            seeds = INIT_SETS["search"][: cfg.search_n]
            r = rollout_seeds(partial(expo_policy, save()), seeds,
                              [policy_seed("search", i, s) for i in range(len(seeds))],
                              robot=cfg.robot, n_workers=1, init_set="search")
            search_steps += r.env_steps
            out.append(r.successes.tolist())
        return out

    def update():
        nonlocal n_updates
        ready = np.flatnonzero(buf["ready"][:ptr])
        b = ready[rng.integers(len(ready), size=cfg.batch)]
        on, a = norm(buf["obs"][b]), torch.as_tensor(buf["act"][b])
        nx = np.where(buf["nxt"][b] >= 0, buf["nxt"][b], b)  # done rows: any index, masked out below
        with torch.no_grad():
            on2, p2 = norm(buf["obs"][nx]), torch.as_tensor(buf["props"][nx])
            c2, _ = candidates(on2, p2, stochastic=True)
            j = q_all(critic, on2, c2).mean(0).argmax(-1)  # the acting rule, used inside the target
            a2 = c2[torch.arange(len(b)), j]
            members = torch.as_tensor(rng.choice(E, cfg.subset, replace=False))
            q2 = target(torch.cat([on2, a2], -1), members).min(0).values
            not_done = torch.as_tensor(1.0 - buf["done"][b])
            y = torch.as_tensor(buf["rew"][b]) + not_done * cfg.gamma ** torch.as_tensor(buf["k"][b]) * q2
        q = critic(torch.cat([on, a], -1))
        loss = ((q - y) ** 2).mean(1).sum()
        q_opt.zero_grad()
        loss.backward()
        q_opt.step()
        diag["q_loss"].append(loss.item() / E)
        if editor is not None:  # SAC-style editor step on one random stored proposal per state
            pa = torch.as_tensor(buf["props"][b, rng.integers(N, size=len(b))])
            m, ls = editor(torch.cat([on, pa], -1))
            u = m + ls.exp() * torch.randn_like(m)
            logp = (torch.distributions.Normal(m, ls.exp()).log_prob(u)
                    - torch.log(1 - torch.tanh(u) ** 2 + 1e-6)).sum(-1)
            q_new = critic(torch.cat([on, (pa + beta * torch.tanh(u)).clamp(-1, 1)], -1)).mean(0)
            e_loss = (log_alpha.exp().detach() * logp - q_new).mean()
            e_opt.zero_grad()
            e_loss.backward()
            e_opt.step()
            a_loss = -(log_alpha * (logp.detach() + cfg.target_entropy)).mean()
            a_opt.zero_grad()
            a_loss.backward()
            a_opt.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), target.parameters()):
                pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
        n_updates += 1

    def bc_step():
        o, c = np.concatenate(bc_obs), np.concatenate(bc_chunks)
        i = rng.integers(len(o), size=cfg.batch)
        loss = base.loss(torch.as_tensor(o[i]), torch.as_tensor(c[i]))
        bc_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        bc_opt.step()

    while steps < cfg.train_steps:
        # 1. Every env decides at once: N proposals, edits, argmax under the critic (stochastic edits).
        idx = np.arange(ptr, ptr + B)
        props = propose(obs)
        buf["obs"][idx], buf["props"][idx] = obs, props
        for i in np.flatnonzero(last >= 0):
            buf["nxt"][last[i]], buf["ready"][last[i]] = idx[i], True
        with torch.no_grad():
            on = norm(obs)
            cands, _ = candidates(on, torch.as_tensor(props), stochastic=True)
            j = q_all(critic, on, cands).mean(0).argmax(-1).numpy()
            chosen = cands[torch.arange(B), torch.as_tensor(j)].numpy()
        diag["edited_chosen"].append(float((j >= N).mean()))
        if editor is not None:
            diag["edit_size"].append(float(np.abs(cands[:, N:].numpy() - props).mean()))
        # 2. Execute the 4 actions of each chosen chunk (an env that finishes early idles until next tick).
        for i in range(B):
            r, done, k = 0.0, False, 0
            for a in chosen[i].reshape(EXECUTE, 4):
                ep_log[i]["obs"].append(obs[i].copy())
                ep_log[i]["actions"].append(a)
                o2, rew, term, trunc, info = envs[i].step(a)
                obs[i], r, k, steps = o2, r + rew, k + 1, steps + 1
                if term or trunc:
                    done = True
                    break
            buf["act"][idx[i]], buf["rew"][idx[i]], buf["k"][idx[i]] = chosen[i], r, k
            buf["done"][idx[i]] = float(done)  # time is in the observation, so a timeout is terminal too
            if done:
                buf["ready"][idx[i]], last[i] = True, -1
                online.append((steps, bool(info["success"])))
                if cotrain and info["success"]:
                    acts = np.asarray(ep_log[i]["actions"], np.float32)
                    t = np.minimum(np.arange(len(acts))[:, None] + np.arange(8)[None], len(acts) - 1)
                    bc_obs.append(np.asarray(ep_log[i]["obs"], np.float32))
                    bc_chunks.append(acts[t])
                ep_log[i] = {"obs": [], "actions": []}
                obs[i] = new_episode(i)
            else:
                last[i] = idx[i]
        ptr += B
        tick += 1
        # 3. Learn: critic + editor steps; the co-trained base takes flow-matching steps on successes.
        if buf["ready"][:ptr].sum() >= cfg.warmup:
            for _ in range(cfg.updates_per_tick):
                update()
        if cotrain and bc_obs:
            for _ in range(cfg.bc_steps_per_tick):
                bc_step()
            if tick % cfg.refresh_every == 0:  # the stored next-state proposals came from an older base
                for s0 in range(0, ptr, 2048):
                    buf["props"][s0: min(ptr, s0 + 2048)] = propose(buf["obs"][s0: min(ptr, s0 + 2048)])
        # 4. Learning curve on the search set (deterministic edits, the deployable policy).
        if marks and steps >= marks[0]:
            marks.pop(0)
            res = search_eval([0])[0]
            curve.append({"train_steps": steps, "k": int(sum(res)), "n": len(res)})
            print(f"  [{name}] {steps / 600:6.1f} robot-min: search {format_rate(int(sum(res)), len(res))}"
                  f"  ({(time.perf_counter() - t_wall) / 60:.1f} min)", flush=True)
    final_search = search_eval(range(cfg.final_salts))
    w = max(1, len(diag["edited_chosen"]) // 10)
    return {"name": name, "beta": beta, "seed": seed, "cotrain": cotrain, "wide": wide, "ckpt": save(),
            "train_steps": steps, "search_steps": search_steps, "episodes": episodes, "updates": n_updates,
            "curve": curve, "final_search": final_search,
            # training episodes (stochastic edits) in 10 bins: [last env step of the bin, successes, episodes]
            "online": [[int(seg[-1][0]), int(sum(x[1] for x in seg)), len(seg)]
                       for seg in np.array_split(np.array(online, dtype=object), 10) if len(seg)],
            "edited_chosen": [float(np.mean(diag["edited_chosen"][i: i + w]))
                              for i in range(0, len(diag["edited_chosen"]), w)],
            "edit_size": float(np.mean(diag["edit_size"][-200:])) if diag["edit_size"] else 0.0,
            "q_loss_last": float(np.mean(diag["q_loss"][-200:])) if diag["q_loss"] else None,
            "alpha": log_alpha.exp().item() if editor is not None else None,
            "bc_samples": int(sum(len(o) for o in bc_obs)), "wall_s": time.perf_counter() - t_wall,
            "cpu_s": time.process_time() - cpu0}


def run_all(cfg: Config, jobs: list[tuple[float, int, bool, int]], work: Path) -> list[dict]:
    with ProcessPoolExecutor(min(cfg.n_workers, len(jobs)), mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(train_run, cfg, *job, str(work)) for job in jobs]
        return [f.result() for f in futs]


# ---- Diagnostics on held-out trajectories


def q_calibration(trajs: list, gamma: float) -> dict:
    """Critic's score of the executed candidate vs the return it is trained toward. The TD target counts
    the reward undiscounted inside a chunk and discounts gamma^4 per chunk boundary, so a decision at step
    t of an episode that succeeds at its last step T - 1 is worth gamma^(4 * floor((T - 1 - t) / 4))."""
    pred, real = [], []
    for tr in trajs:
        t = np.flatnonzero(tr["last_step_in_chunk"] == 0)
        pred.append(tr["last_q"][t])
        real.append(gamma ** (EXECUTE * ((tr["steps"] - 1 - t) // EXECUTE)) * tr["success"])
    pred, real = np.concatenate(pred), np.concatenate(real)
    bins = np.clip((pred * 10).astype(int), 0, 9)
    return {"mean_pred": float(pred.mean()), "mean_real": float(real.mean()),
            "first_decision_pred": float(np.mean([tr["last_q"][0] for tr in trajs])),
            "bins": [[float(pred[bins == i].mean()), float(real[bins == i].mean()), int((bins == i).sum())]
                     for i in range(10) if (bins == i).any()]}


def latency_ms(ckpt: str, n: int, obs: np.ndarray, reps: int = 50) -> float:
    """Wall time of one chunk decision for one env (N proposals, N edits, 2N critic scores), in ms."""
    pol = ExpoPolicy(ckpt, n)
    rng = np.random.default_rng(0)
    pol.decide(obs, rng)
    t0 = time.perf_counter()
    for _ in range(reps):
        pol.decide(obs, rng)
    return 1e3 * (time.perf_counter() - t0) / reps


# ---- Plots and the proposal-fan GIF


def make_plots(cfg: Config, media: Path, s: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from lastmile.common import plotting

    plotting.setup()
    P = plotting.PALETTE + ["#8c6d31", "#17becf"]
    names = list(s["configs"])
    colors = {nm: P[i % len(P)] for i, nm in enumerate(names)}

    def reference(ax, k, n, label, ls="--"):  # a reference rate as a line inside its Wilson band
        lo, hi = wilson(k, n)
        ax.axhspan(lo, hi, color="gray", alpha=0.15, lw=0)
        ax.axhline(k / n, ls=ls, color="gray", label=f"{label}: {format_rate(k, n)}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for nm in names:
        runs = [s["runs"][r] for r in s["configs"][nm]]
        x = np.mean([[c["train_steps"] / 600 for c in r["curve"]] for r in runs], 0)
        k = np.sum([[c["k"] for c in r["curve"]] for r in runs], 0)
        n = np.sum([[c["n"] for c in r["curve"]] for r in runs], 0)
        plotting.success_curve(ax, x, k, n, label=f"{nm} (pooled over {len(runs)} seeds)", color=colors[nm])
        for r in runs:
            ax.plot(x, [c["k"] / c["n"] for c in r["curve"]], color=colors[nm], alpha=0.3, lw=0.8)
    reference(ax, *s["base_search"], "base_v1 (search)")
    nm = s["chosen"]
    plotting.heldout_point(ax, x[-1] * 1.03, *s["eval"][nm]["pooled"], label=f"held-out eval, {nm} (chosen)",
                           color="black")
    ax.set(xlabel="training robot-minutes (online interaction)", ylabel="success", xlim=(0, None),
           title="Learning curves on the search set (thin = each seed)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    plotting.save_fig(fig, media / "learning_curves.png")

    # Held-out eval: methods (colored) and controls (gray), pooled with Wilson 95%, dots = seeds.
    groups = [(nm, s["eval"][nm]["per_seed"], colors[nm]) for nm in names]
    groups += [(lab, ps, "gray") for lab, ps in s["controls"].items()]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    reference(ax, *s["eval_base"], "base_v1")
    for i, (nm, per_seed, color) in enumerate(groups):
        k, n = (int(sum(v[j] for v in per_seed.values())) for j in (0, 1))
        p, err = plotting.wilson_err([k], [n])
        ax.errorbar([i], p, yerr=err, fmt="D", color=color, capsize=4)
        ax.scatter([i] * len(per_seed), [a / b for a, b in per_seed.values()], color=color, alpha=0.5, s=14)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], rotation=30, ha="right", fontsize=7.5)
    ax.set(ylabel=f"held-out success (eval, n = {cfg.eval_n} per seed)", ylim=(0, 1.02),
           title="Held-out eval: methods and controls (diamond = pooled, Wilson 95%; dots = seeds)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right", fontsize=8)
    plotting.save_fig(fig, media / "eval_by_config.png")

    fig, ax = plt.subplots()
    ns = sorted(int(n) for n in s["latency_ms"])
    ax.plot(ns, [s["latency_ms"][str(n)] for n in ns], marker="o")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set(xlabel="N base proposals (2N candidates)", ylabel="ms per chunk decision (one env, numpy, 1 core)",
           title=f"Latency of propose-edit-select ({s['chosen']})", xlim=(0, None), ylim=(0, None))
    plotting.save_fig(fig, media / "latency.png")

    fig, ax = plt.subplots()
    for nm in names:
        bins = np.array(s["q_calibration"][nm]["bins"])
        ax.plot(bins[:, 0], bins[:, 1], marker="o", color=colors[nm], label=nm)
    ax.plot([0, 1.2], [0, 1.2], ls="--", color="gray", lw=1, label="calibrated")
    ax.set(xlabel="critic's Q of the executed candidate (binned)", ylabel="realized return (TD definition)",
           title="Critic vs realized gamma^(4 floor((T-1-t)/4)) * success (eval, seed 0)")
    ax.legend(fontsize=8)
    plotting.save_fig(fig, media / "q_calibration.png")


def make_fan_gif(cfg: Config, media: Path, ckpt: str, seed: int, pseed: int, label: str) -> bool:
    """The proposal fan, replayed on one eval episode: at each decision, the N base proposals (gray), their
    edits (blue) and the executed candidate (red), drawn as commanded end-effector paths seen from above."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from lastmile.common.plotting import save_gif
    from lastmile.envs.cupdrop import CupDropEnv

    env = CupDropEnv(robot=cfg.robot)
    pol = ExpoPolicy(ckpt, record=True)
    sl, n = env.obs_slices, pol.n
    pol.reset([pseed])
    obs, info, frames, t, done = env.reset(seed=seed)[0], {"success": False}, [], 0, False
    while not done:
        a = pol.act(obs[None])[0]
        if t % EXECUTE == 0:  # new decision: redraw the fan (commanded path = ee + 2 cm x cumulative action)
            ee, cube, j = obs[sl["ee_pos"]], obs[sl["cube_pos"]], int(pol.last_choice[0])
            steps = pol.last_cands[0].reshape(-1, EXECUTE, 4)
            paths = 100 * (ee[:2] + 0.02 * np.concatenate([np.zeros((len(steps), 1, 2)),
                                                           np.cumsum(steps[:, :, :2], 1)], 1))
            fig = Figure(figsize=(2.56, 2.56), dpi=100)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            for c, path in enumerate(paths):
                ax.plot(path[:, 0], path[:, 1], color="#2f6690" if c >= n else "gray", alpha=0.7, lw=0.9)
            ax.plot(paths[j][:, 0], paths[j][:, 1], color="#d1495b", lw=2.5)
            ax.scatter(*(cube[:2] * 100), marker="s", color="#edae49", s=60, zorder=3)
            half = max(1.5, 1.2 * np.abs(paths - 100 * ee[:2]).max())  # zoom to the fan, at least +-1.5 cm
            cx, cy = 100 * ee[:2]
            ax.set(xlim=(cx - half, cx + half), ylim=(cy - half, cy + half))
            ax.set_aspect("equal")
            ax.tick_params(labelsize=5)
            q = pol.last_scores[0]
            grip = "close" if steps[j, -1, 3] > 0 else "open"
            ax.set_title(f"t={t}: {'edit' if j >= n else 'base'} #{j % n}, Q {q[j]:.2f} "
                         f"(best base {q[:n].max():.2f}), {grip}", fontsize=5.5)
            ax.set_xlabel("x (cm); gray = base, blue = edits, red = executed, square = cube", fontsize=4.5)
            fig.tight_layout(pad=0.3)
            fig.canvas.draw()
            fan = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        obs, _, term, trunc, info = env.step(a)
        t, done = t + 1, term or trunc
        cam = env.render("front")[:, :, :3]
        frames.append(np.concatenate([cam, np.full((cam.shape[0], 4, 3), 255, np.uint8), fan], 1))
    frames += [frames[-1]] * 6
    save_gif(frames, media / "proposal_fan.gif", fps=5, max_size=516)
    print(f"proposal fan GIF: eval seed {seed}, {label}, success={info['success']}")
    return bool(info["success"])


# ---- The run


def main(cfg: Config) -> None:
    import torch

    from lastmile.envs.cupdrop import CupDropEnv, OutcomeTracker

    torch.set_num_threads(2)
    out = REPO_ROOT / "runs/ch08_quick" if cfg.quick else REPO_ROOT
    media, work = out / "media/ch08", out / "runs/ch08_work"
    work.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    s: dict = {"runs": {}, "configs": {}, "controls": {}, "control_stats": {}}
    wide = [cfg.wide_n] if cfg.wide_n else []

    with ExitStack() as stack:
        # One results file per run, opened now so wall time and main-process CPU are the whole run's.
        led = partial(Ledger, chapter="ch08", robot=cfg.robot, config=cfg, results_root=out / "results")
        names = [run_name(b) for b in cfg.betas] + [run_name(0.0, False, w) for w in wide]
        rows = {f"{nm}_s{sd}": stack.enter_context(led(method=f"{nm}_s{sd}"))
                for nm in names for sd in cfg.seeds}
        co_rows = {sd: stack.enter_context(led(method=f"cotrain_s{sd}")) for sd in cfg.seeds if cfg.cotrain}
        ctrl_row = stack.enter_context(led(method=f"random-pick-{cfg.n_props}"))  # carries every control
        shared = Ledger("ch08", "shared")  # never written: base rollouts that every row pays for

        # 1. Phase 1: the beta sweep (frozen base, every seed) and the N = 16 control. Slow jobs first.
        jobs = ([(b, sd, False, 0) for b in sorted(cfg.betas, reverse=True) if b > 0 for sd in cfg.seeds]
                + [(0.0, sd, False, w) for w in wide for sd in cfg.seeds]
                + [(0.0, sd, False, 0) for sd in cfg.seeds if 0.0 in cfg.betas])
        phase1 = run_all(cfg, jobs, work)

        def record(results):
            for r in results:
                s["runs"][r["name"]] = r
                s["configs"].setdefault(run_name(r["beta"], r["cotrain"], r["wide"]), []).append(r["name"])
                fs = np.array(r["final_search"])
                print(f"{r['name']}: final search {format_rate(int(fs.sum()), fs.size)}, "
                      f"{r['train_steps'] / 600:.0f} train robot-min, {r['wall_s'] / 60:.1f} min wall",
                      flush=True)

        def pooled_search(nm):
            fs = np.concatenate([np.ravel(s["runs"][r]["final_search"]) for r in s["configs"][nm]])
            return [int(fs.sum()), int(fs.size)]

        record(phase1)

        # 2. Choose beta on search (the N = 16 control is not a candidate): best pooled final search
        # score, the smaller beta on ties.
        s["search_by_config"] = {nm: pooled_search(nm) for nm in s["configs"]}
        rates = {b: s["search_by_config"][run_name(b)] for b in cfg.betas}
        beta_star = -max((k / n, -b) for b, (k, n) in rates.items())[1]
        s["beta_star"], s["chosen"] = beta_star, run_name(beta_star)
        print("pooled final search:", {nm: format_rate(*v) for nm, v in s["search_by_config"].items()},
              f"-> beta* = {beta_star:g}", flush=True)

        # 3. Phase 2: the co-trained base at beta*.
        if cfg.cotrain:
            record(run_all(cfg, [(beta_star, sd, True, 0) for sd in cfg.seeds], work))
            s["search_by_config"][run_name(beta_star, True)] = pooled_search(run_name(beta_star, True))
            for sd, row in co_rows.items():
                row.method = f"{run_name(beta_star, True)}_s{sd}"
                rows[row.method] = row

        # 4. Held-out eval, once per run, paired with the base (same env and policy seeds, salt 0).
        ev = partial(evaluate, init_set="eval", robot=cfg.robot, n_workers=cfg.n_workers, n=cfg.eval_n,
                     category="eval")
        base_eval = ev(base_policy, ledger=shared, record_trajectories=True)
        base_search = [evaluate(base_policy, "search", robot=cfg.robot, n_workers=cfg.n_workers, salt=k,
                                n=cfg.search_n, ledger=shared, category="search")
                       for k in range(cfg.final_salts)]
        s["base_search"] = [sum(r.k for r in base_search), sum(r.n for r in base_search)]
        s["eval_base"] = [base_eval.k, base_eval.n]
        finals, s["eval"], s["q_calibration"] = {}, {}, {}
        b0 = base_eval.successes
        for nm, runs in s["configs"].items():
            e = s["eval"][nm] = {"per_seed": {}, "mcnemar_p": {}, "diff_ci": {}, "paired": {}}
            for rn in runs:
                r = s["runs"][rn]
                f = finals[rn] = ev(partial(expo_policy, r["ckpt"]), ledger=rows[rn],
                                    record_trajectories=True)
                e["per_seed"][rn] = [f.k, f.n]
                e["mcnemar_p"][rn] = mcnemar_exact(b0, f.successes)
                e["diff_ci"][rn] = bootstrap_diff(b0, f.successes)
                e["paired"][rn] = {"fixed": int((~b0 & f.successes).sum()),
                                   "broke": int((b0 & ~f.successes).sum())}
                picks = np.concatenate([tr["last_choice"][tr["last_step_in_chunk"] == 0]
                                        for tr in f.trajectories])
                r["edited_chosen_eval"] = float(np.mean(picks >= (r["wide"] or cfg.n_props)))
            e["pooled"] = [int(np.sum([v[i] for v in e["per_seed"].values()])) for i in (0, 1)]
            s["q_calibration"][nm] = q_calibration(finals[runs[0]].trajectories, cfg.gamma)
            print(f"eval {nm}: pooled {format_rate(*e['pooled'])}; per seed "
                  f"{ {k: format_rate(*v) for k, v in e['per_seed'].items()} }", flush=True)

        # 5. Controls on the chosen runs, paired with the base and with the method (honesty rule 4): which
        # part of the gain is the editor and which is the critic's pick?
        chosen_runs = s["configs"][s["chosen"]]
        n2 = 2 * cfg.n_props if beta_star > 0 else cfg.n_props
        ctrl_specs = [(f"control: random of {cfg.n_props} proposals", "random-base", chosen_runs[:1]),
                      (f"control: random of {n2} candidates", "random-all", chosen_runs)]
        if beta_star > 0:
            ctrl_specs.append(("control: edit of a random proposal", "edit-random", chosen_runs))
        ctrl_evals = {}
        for label, pick, runs in ctrl_specs:
            per, st = s["controls"].setdefault(label, {}), s["control_stats"].setdefault(label, {})
            for rn in runs:
                c = ctrl_evals[pick, rn] = ev(partial(expo_policy, s["runs"][rn]["ckpt"], pick),
                                              ledger=ctrl_row)
                per[rn] = [c.k, c.n]
                st[rn] = {"mcnemar_p_vs_base": mcnemar_exact(b0, c.successes),
                          "mcnemar_p_vs_method": mcnemar_exact(finals[rn].successes, c.successes)}
            print(f"{label}: { {k: format_rate(*v) for k, v in per.items()} }", flush=True)

        # 6. Diagnostics: failure labels, whether the base ever solves the chosen run's eval failures
        # (its support), and latency vs N.
        first = s["runs"][chosen_runs[0]]
        env = CupDropEnv(robot=cfg.robot)
        s["failure_modes"] = {}
        for nm, r in (("base", base_eval), (first["name"], finals[first["name"]])):
            labels = [OutcomeTracker.label_trajectory(env, t) for t in r.trajectories]
            s["failure_modes"][nm] = {m: labels.count(m) for m in sorted(set(labels))}
        fails, K = np.flatnonzero(~finals[first["name"]].successes), cfg.support_salts
        s["base_support_on_failures"] = {"salts": K, "solved_by_seed": {}}
        if len(fails):
            sup = rollout_seeds(base_policy, [int(base_eval.seeds[i]) for i in fails for _ in range(K)],
                                [policy_seed("eval", int(i), k) for i in fails for k in range(K)],
                                robot=cfg.robot, n_workers=cfg.n_workers, init_set="eval", ledger=ctrl_row,
                                category="eval")
            s["base_support_on_failures"]["solved_by_seed"] = {
                int(base_eval.seeds[i]): int(k) for i, k in zip(fails, sup.successes.reshape(-1, K).sum(1))}
        obs0 = base_eval.trajectories[0]["obs"][0]
        s["latency_ms"] = {str(n): latency_ms(first["ckpt"], n, obs0) for n in cfg.latency_ns}

        # 7. Costs. Each row pays its own interaction and search rollouts (and its training process's CPU).
        # The chosen rows and the co-trained rows (built on beta*) pay the whole beta sweep, which shaped
        # the choice; the N = 16 control did not shape it. Every row pays the base's search and eval rollouts.
        sweep_runs = [r for r in phase1 if not r["wide"]]
        sweep = {k: sum(r[k] for r in sweep_runs) for k in ("train_steps", "search_steps", "cpu_s")}
        s["cost_steps"] = {"beta_sweep_total": sweep, "shared": dict(shared.robot_steps)}
        roles = {}
        for rn, row in rows.items():
            r = s["runs"][rn]
            role = roles[rn] = ("cotrained-ablation" if r["cotrain"] else "control" if r["wide"]
                                else "chosen" if r["beta"] == beta_star else "ablation")
            parts = ([sweep] if role in ("chosen", "cotrained-ablation") else []) + (
                [r] if role != "chosen" else [])  # a chosen run is already inside the sweep total
            for p in parts:
                row.robot_steps["train"] += p["train_steps"]
                row.robot_steps["search"] += p["search_steps"]
                row.worker_cpu_seconds += p["cpu_s"]
            for c in ("search", "eval"):
                row.robot_steps[c] += shared.robot_steps[c]
            row.worker_cpu_seconds += shared.worker_cpu_seconds
            row.set_base(id="base_v1", successes=base_eval)
            row.set_final(successes=finals[rn])
            s["cost_steps"][rn] = dict(row.robot_steps)
        ctrl_row.robot_steps["eval"] += shared.robot_steps["eval"]
        ctrl_row.worker_cpu_seconds += shared.worker_cpu_seconds
        ctrl_row.set_base(id="base_v1", successes=base_eval)
        ctrl_row.set_final(successes=ctrl_evals["random-base", chosen_runs[0]])

        # 8. Media: the proposal fan on the first eval episode the base failed and the chosen policy solved.
        s["gif_seed"] = None
        pairs = [i for i, tr in enumerate(finals[first["name"]].trajectories) if tr["success"] and not b0[i]]
        if beta_star > 0 and pairs:
            i = pairs[0]
            with optional_media("the proposal-fan GIF"):
                ok = make_fan_gif(cfg, media, first["ckpt"], int(base_eval.seeds[i]),
                                  int(base_eval.policy_seeds[i]), first["name"])
                assert ok == bool(finals[first["name"]].successes[i]), "GIF episode diverged from eval"
                s["gif_seed"] = int(base_eval.seeds[i])
        make_plots(cfg, media, s)
        s["wall_minutes"] = (time.perf_counter() - wall0) / 60
        for rn, row in [*rows.items(), ("controls", ctrl_row)]:  # every file carries the whole summary
            role = roles.get(rn, "control")
            row.extra = {**s, "role": role, "selected_on_search": role == "chosen"}

    print(f"\n=== Chapter 8 summary (held-out eval, n = {cfg.eval_n} per seed, unless marked) ===")
    print(f"base_v1: {base_eval.summary}")
    for label, per in s["controls"].items():
        print(f"{label}: { {k: format_rate(*v) for k, v in per.items()} }  {s['control_stats'][label]}")
    for nm in s["configs"]:
        e = s["eval"][nm]
        print(f"{nm:24s} pooled {format_rate(*e['pooled'])}  "
              f"search {format_rate(*s['search_by_config'][nm])}")
        for rn in s["configs"][nm]:
            print(f"    {rn}: {format_rate(*e['per_seed'][rn])}  McNemar p={e['mcnemar_p'][rn]:.2g}  "
                  f"{e['paired'][rn]}  edited picks {s['runs'][rn].get('edited_chosen_eval', 0):.3f}")
    print(f"chosen on search: beta* = {beta_star:g} ({s['chosen']})")
    print(f"base support on {first['name']}'s eval failures (base successes of {cfg.support_salts} salts):",
          s["base_support_on_failures"]["solved_by_seed"])
    print(f"latency ms/decision: { {n: round(x, 2) for n, x in s['latency_ms'].items()} }")
    for rn, row in [*rows.items(), ("controls", ctrl_row)]:
        print(f"row {rn} ({roles.get(rn, 'control')}) robot-min:",
              {c: round(m, 1) for c, m in row.robot_minutes.items()}, f"cpu-h {row.cpu_hours:.2f}")
    print(f"wall {s['wall_minutes']:.1f} min; results in {out / 'results/ch08'}")


if __name__ == "__main__":
    main(parse(Config))
