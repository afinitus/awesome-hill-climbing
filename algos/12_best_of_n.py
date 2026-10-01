"""Chapter 12: test-time search with a verifier. Best-of-N on the frozen base, no weight updates.

Learning goal
    Spend compute at deployment instead of changing weights. At every chunk decision, sample N candidate
    chunks from the frozen flow policy (N noise vectors), score each with a learned verifier, execute the
    best one. See how far that climbs, what it costs (labeled rollouts to train the verifier, latency per
    decision) and where it breaks (a verifier trained on too few rollouts, a verifier that gets worse as N grows).

Base idea in plain words
    Best-of-N with a critic is one greedy policy-improvement step. If Q(s, a) is the chance of success
    when the base takes chunk a in state s and then keeps acting as usual, then picking

        a* = argmax_{j <= N} Q(s, a_j),   a_j = flow(s, w_j),  w_j ~ N(0, I)

    at every decision is at least as good as the base (the policy-improvement theorem), *if* Q is right.
    We learn Q from the base's own rollouts with Monte-Carlo labels, two ways:
        classifier   P(success | s, a), trained with cross-entropy on the episode's outcome
        MC value     Q(s, a) = gamma^(T - t) * success, regressed with squared error (T = success step)
    Selection only chooses behavior the base already has: the ceiling is the base's support (pass@k as k
    grows without bound), not a finite pass@k. Reranking every chunk recombines good chunks from different
    samples, so it can solve a start state that 16 whole base episodes never solved, but not one where no
    sequence of base chunks succeeds. Controls: a random pick among N (the base distribution again), and a
    simulator lookahead that rolls each candidate out from a saved state (not deployable: it needs resets).

Papers mirrored
    UF-OPS [C259] (a verifier trained on the policy's own rollouts, best-of-N at deployment), Q-Planning
    [C392] and SeeQ [C438] (sample-then-rank with a learned Q), V-GPS [C075] (a Q-function reranks samples
    of a frozen generalist), IDQL [C042] (a critic picks among diffusion samples), VLA-ATTC [C288].

Honesty
    Labeled rollouts use their own env seeds (70_000 + 10_000 * verifier_seed + i). (verifier, N) is chosen
    on `search`, pooled over verifier seeds, so every row pays for all seeds' labels and sweeps and for the
    simulator lookahead. `--prior-costs FILE` adds rollouts spent before this run (pilots, earlier runs).
    The held-out `eval` set (n=256) is scored once for the chosen setting, per verifier seed.

Run
    uv run python algos/12_best_of_n.py            # full run, about 25 min on an M5 Pro, 6 workers
    uv run python algos/12_best_of_n.py --quick    # smoke test, < 2 min; writes only to runs/ch12_quick/
"""

from __future__ import annotations

import json
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
from lastmile.common.eval import INIT_SETS, bootstrap_diff, format_rate, mcnemar_exact, policy_seed
from lastmile.common.ledger import REPO_ROOT, Ledger
from lastmile.common.plotting import optional_media
from lastmile.common.rollout import evaluate, rollout_seeds

# torch, policy_flow and matplotlib are imported inside functions: spawned rollout workers re-import
# this file, and only the ones that build a flow policy should pay for torch.

BASE = REPO_ROOT / "checkpoints/base_v1_so101.pt"
TRAIN_SEED_BASE = 70_000  # labeled rollouts: far from search (10k), eval (20k), eval_ext (30k), stress (40k)
EXECUTE = 4  # the base executes 4 actions of each 8-action chunk, then replans


@dataclass
class Config:
    robot: str = "so101"
    ns: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32], metadata={"help": "best-of-N sizes"})
    pass_k: int = 16  # diagnostic: policy salts per search seed
    search_salts: int = 4  # salts per search-set point (64 seeds each), paired with the base's salts
    verifier_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    train_episodes: int = 4096  # labeled base rollouts per verifier seed (chosen in pilots on dev seeds)
    data_sweep: list[int] = field(default_factory=lambda: [128, 512, 2048])  # fewer labeled episodes
    gamma: float = 0.99  # per env step, for the MC value target
    hidden: int = 256
    train_steps: int = 3000
    batch_size: int = 512
    lr: float = 1e-3
    oracle_ns: list[int] = field(default_factory=lambda: [2, 4, 8])
    oracle_rollouts: int = 4  # simulator continuations per candidate (shared across candidates)
    oracle_seeds: int = 64  # first search seeds scored by the simulator lookahead
    eval_n: int = 256
    n_workers: int = 6
    prior_costs: str = field(default="", metadata={"help": "JSON manifest of rollouts spent before this run"})
    quick: bool = False
    QUICK: ClassVar[dict] = {"ns": [1, 4], "pass_k": 4, "search_salts": 1, "verifier_seeds": [0],
                             "train_episodes": 96, "data_sweep": [32], "train_steps": 300, "oracle_ns": [2],
                             "oracle_rollouts": 2, "oracle_seeds": 6, "eval_n": 64}


# ---- Labeled experience: roll out the base, keep (obs, chunk) at every chunk decision with its outcome.


def base_policy():
    from lastmile.common.policy_flow import FlowChunkPolicy

    return FlowChunkPolicy.load(BASE, noise="gaussian")


def collect(cfg: Config, vseed: int, n_eps: int, ledger: Ledger) -> dict:
    """Base rollouts on the verifier seed's own env seeds, cut into one sample per chunk decision."""
    start = 10_000 * vseed
    seeds = [TRAIN_SEED_BASE + start + i for i in range(n_eps)]
    pseeds = [policy_seed("ch12_train", start + i) for i in range(n_eps)]
    res = rollout_seeds(base_policy, seeds, pseeds, robot=cfg.robot, n_workers=cfg.n_workers,
                        record_trajectories=True, ledger=ledger, category="train")
    cols = {"obs": [], "chunk": [], "success": [], "q": [], "episode": []}
    for e, tr in enumerate(res.trajectories):
        t = np.flatnonzero(tr["last_step_in_chunk"] == 0)  # the steps where a new chunk was sampled
        cols["obs"].append(tr["obs"][t])
        cols["chunk"].append(tr["last_chunk"][t].reshape(len(t), -1))
        cols["success"].append(np.full(len(t), float(tr["success"])))
        # Monte-Carlo value: discounted by the steps still needed to succeed, 0 for a failed episode.
        cols["q"].append(cfg.gamma ** (tr["steps"] - t) * tr["success"])
        cols["episode"].append(np.full(len(t), e))
    data = {k: np.concatenate(v).astype(np.float32 if k != "episode" else np.int64) for k, v in cols.items()}
    data["episode_success"] = res.successes
    return data


# ---- Verifiers: a small MLP on (normalized obs, flattened chunk) -> one logit.


def train_verifier(cfg: Config, data: dict, kind: str, n_eps: int, seed: int, path: Path) -> dict:
    """kind = "clf" (success classifier), "q" (MC value) or "state" (classifier that never sees the chunk).

    The last 10% of episodes are held out for early stopping, split by episode so that decisions from one
    episode (which share its label) never sit on both sides.
    """
    import torch
    from torch import nn

    from lastmile.common.policy_flow import FlowChunk

    torch.manual_seed(seed)
    base = FlowChunk.load(BASE)
    mu, sd = base.obs_mean.numpy(), base.obs_std.numpy()
    keep = data["episode"] < n_eps
    obs = (data["obs"][keep] - mu) / sd
    chunk = data["chunk"][keep] * (0.0 if kind == "state" else 1.0)
    x = torch.as_tensor(np.concatenate([obs, chunk], 1), dtype=torch.float32)
    y = torch.as_tensor(data["q" if kind == "q" else "success"][keep])
    val = torch.as_tensor(data["episode"][keep] >= int(0.9 * n_eps))
    net = nn.Sequential(nn.Linear(x.shape[1], cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, cfg.hidden),
                        nn.ReLU(), nn.Linear(cfg.hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-3)

    def loss_fn(logit, target):
        if kind == "q":  # regression onto the discounted return; sigmoid keeps the output in [0, 1]
            return ((torch.sigmoid(logit) - target) ** 2).mean()
        return nn.functional.binary_cross_entropy_with_logits(logit, target)

    xt, yt, xv, yv = x[~val], y[~val], x[val], y[val]
    gen = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for step in range(cfg.train_steps):
        idx = torch.randint(0, len(xt), (min(cfg.batch_size, len(xt)),), generator=gen)
        loss = loss_fn(net(xt[idx]).squeeze(1), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 99 or step == cfg.train_steps - 1:
            with torch.no_grad():
                v = loss_fn(net(xv).squeeze(1), yv).item()
            if v < best:  # early stopping: keep the weights with the lowest held-out loss
                best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        score = net(xv).squeeze(1).numpy()
    ok = data["success"][keep][val.numpy()]
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "kind": kind}, path)
    return {"val_loss": best, "val_auc": auc(score, ok), "n_samples": len(x), "n_episodes": n_eps}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Chance that a random success decision scores above a random failure decision (ties count half)."""
    pos, neg = score[label > 0.5], score[label <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


class NumpyVerifier:
    """The trained MLP in plain numpy, so rollout workers score candidates without torch overhead."""

    def __init__(self, path: str | Path):
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        self.layers = [(sd[f"{i}.weight"].numpy().T, sd[f"{i}.bias"].numpy()) for i in (0, 2, 4)]
        self.mu, self.sd, self.blind = ckpt["mu"], ckpt["sd"], ckpt["kind"] == "state"

    def __call__(self, obs: np.ndarray, chunks: np.ndarray) -> np.ndarray:
        """One obs ``[D]`` and N chunks ``[N, 8, 4]`` -> N logits (higher = better)."""
        o = np.broadcast_to((obs - self.mu) / self.sd, (len(chunks), len(obs)))
        c = chunks.reshape(len(chunks), -1) * (0.0 if self.blind else 1.0)
        h = np.concatenate([o, c], 1).astype(np.float32)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            h = np.maximum(h, 0) if i < 2 else h
        return h[:, 0]


# ---- Best-of-N through FlowChunkPolicy's callable-noise hook: pick which noise the base decodes.


class PickNoise:
    """``noise(obs[B, D], rngs) -> [B, 32]``: draw N candidate noises per env, return the chosen one.

    Candidates come from env i's own generator, so N=1 draws exactly the noise the base would have drawn
    (N=1 reproduces the base bit for bit), and results do not depend on the batch or worker.
    """

    def __init__(self, sampler, n: int, pick: str, verifier_path: str | None):
        self.sampler, self.n, self.pick = sampler, n, pick
        self.verifier = NumpyVerifier(verifier_path) if verifier_path else None
        self.prev: np.ndarray | None = None

    def __call__(self, obs: np.ndarray, rngs: list) -> np.ndarray:
        out = np.zeros((len(obs), 32), np.float32)
        for i, (o, rng) in enumerate(zip(obs, rngs)):
            # A finished env keeps receiving its last obs; skip it (its choice is never executed).
            if self.prev is not None and np.array_equal(o, self.prev[i]):
                continue
            cand = rng.standard_normal((self.n, 32), dtype=np.float32)
            j = 0
            if self.n > 1 and self.pick == "random":
                j = int(rng.integers(self.n))  # the control: the base distribution with extra steps
            elif self.n > 1:
                chunks = self.sampler(np.repeat(o[None], self.n, 0), cand)
                j = int(np.argmax(self.verifier(o, chunks)))
            out[i] = cand[j]
        self.prev = obs.copy()
        return out


def bon_policy(n: int, pick: str, verifier_path: str | None = None):
    """Picklable factory (use with functools.partial) for the best-of-N policy."""
    from lastmile.common.policy_flow import FlowChunk, FlowChunkPolicy

    model = FlowChunk.load(BASE)
    chooser = PickNoise(model.numpy_sampler(), n, pick, verifier_path)
    return FlowChunkPolicy(model, noise=chooser)


def search_score(cfg: Config, make, ledger: Ledger, salts: range | None = None) -> list:
    """One `evaluate` per salt on the search set (paired with the base's salts)."""
    salts = salts if salts is not None else range(cfg.search_salts)
    return [evaluate(make, "search", robot=cfg.robot, n_workers=cfg.n_workers, ledger=ledger,
                     category="search", salt=s) for s in salts]


def kn(results: list) -> list[int]:
    return [sum(r.k for r in results), sum(r.n for r in results)]


def rates(d: dict) -> dict:
    """{key: [k, n]} -> {key: "k/n = p% [lo, hi]"} for progress prints."""
    return {key: format_rate(*v) for key, v in d.items()}


# ---- Simulator lookahead (NOT deployable): score each candidate by rolling it out in a scratch simulator.


def oracle_episode(robot: str, n: int, m: int, seed: int, pseed: int, gamma: float) -> tuple[bool, int, int]:
    """One episode where each decision tries all N candidates from the saved state, m times each (the
    candidate chunk, then the base), and executes the candidate with the best mean discounted outcome.
    The m continuation noises are shared by all candidates (common random numbers), so candidates differ
    only in their own chunk. If every rollout fails, argmax keeps candidate 0, the base's own noise."""
    from lastmile.common.policy_flow import FlowChunk
    from lastmile.envs.cupdrop import CupDropEnv

    sampler = FlowChunk.load(BASE).numpy_sampler()
    env, scratch = CupDropEnv(robot=robot), CupDropEnv(robot=robot)
    rng, look = np.random.default_rng(pseed), np.random.default_rng([pseed, 1])
    obs, _ = env.reset(seed=seed)
    look_steps, done, info = 0, False, {"success": False}
    while not done:
        cand = rng.standard_normal((n, 32), dtype=np.float32)
        chunks = sampler(np.repeat(obs[None], n, 0), cand)
        cont = look.standard_normal((m, 40, 32), dtype=np.float32)  # 150 steps / 4 < 40 later chunks
        state, scores = env.get_state(), np.zeros(n)
        for j, k in np.ndindex(n, m):
            scratch.set_state(state)
            t, c, o, fin, inf = 0, 0, None, False, {}
            while not fin:
                plan = chunks[j] if c == 0 else sampler(o[None], cont[k, c - 1: c])[0]
                for a in plan[:EXECUTE]:
                    o, _, term, trunc, inf = scratch.step(a)
                    t, fin = t + 1, term or trunc
                    if fin:
                        break
                c += 1
            look_steps += t
            scores[j] += (gamma**t if inf["success"] else 0.0) / m
        for a in chunks[int(np.argmax(scores))][:EXECUTE]:
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            if done:
                break
    return bool(info["success"]), int(env._steps), look_steps


def run_oracle(cfg: Config, n: int) -> dict:
    seeds = INIT_SETS["search"][: cfg.oracle_seeds]
    pseeds = [policy_seed("search", i, 0) for i in range(len(seeds))]  # salt 0, paired with the base
    with ProcessPoolExecutor(cfg.n_workers, mp_context=mp.get_context("spawn")) as pool:
        out = list(pool.map(partial(oracle_episode, cfg.robot, n, cfg.oracle_rollouts, gamma=cfg.gamma),
                            seeds, pseeds))
    succ = np.array([o[0] for o in out])
    return {"k": int(succ.sum()), "n": len(succ), "successes": succ.tolist(),
            "executed_steps": sum(o[1] for o in out), "lookahead_steps": sum(o[2] for o in out)}


def latency_ms(obs: np.ndarray, n: int, verifier_path: str, reps: int = 60) -> float:
    """Wall time of one chunk decision (sample N chunks + score them) for a single env obs [1, D], in ms."""
    from lastmile.common.policy_flow import FlowChunk

    chooser = PickNoise(FlowChunk.load(BASE).numpy_sampler(), n, "verifier", verifier_path)
    rngs = [np.random.default_rng(0)]
    t0 = time.perf_counter()
    for _ in range(reps):
        chooser.prev = None  # otherwise the repeated obs looks like a finished env and is skipped
        noise = chooser(obs, rngs)
        chooser.sampler(obs, noise)  # the policy then decodes the chosen noise once more
    return 1e3 * (time.perf_counter() - t0) / reps


# ---- Plots and GIF


def make_plots(cfg: Config, media: Path, s: dict) -> None:
    """All figures from the summary dict `s` (the same dict goes into the results JSON as `extra`)."""
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    plotting.setup()
    P, ns, kind, n_star = plotting.PALETTE, cfg.ns, s["chosen"]["kind"], s["chosen"]["n"]
    final_label = "search" if cfg.quick else "held-out eval"

    def log2(ax, ticks, label):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlabel(label)

    fig, ax = plt.subplots()
    ks = sorted(int(k) for k in s["pass_at_k"])
    plotting.success_curve(ax, ks, [s["pass_at_k"][str(k)][0] for k in ks], 64,
                           label="pass@k: any of k tries succeeds (search, sim resets)")
    ax.axhline(s["pass_at_k"]["1"][0] / 64, ls="--", color="gray", label="pass@1 (search)")
    log2(ax, ks, "k (independent tries per start state)")
    ax.set(ylabel="success", title="The base succeeds from most starts sometimes")
    ax.legend(loc="lower right")
    plotting.save_fig(fig, media / "pass_at_k.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    small = s["curve_episodes"]
    for key, label, color, marker in [("q", f"MC-value verifier ({cfg.train_episodes} eps/seed)", P[0], "o"),
                                      ("clf", f"success classifier ({cfg.train_episodes} eps/seed)", P[2], "o"),
                                      (f"{kind}_{small}", f"{kind} verifier, only {small} eps/seed", P[3], "o"),
                                      ("random", "random pick of N (control)", P[5], "x")]:
        c = s["curves"][key]
        plotting.success_curve(ax, ns, [c[str(n)][0] for n in ns], [c[str(n)][1] for n in ns],
                               label=label, color=color, marker=marker)
    on, ob = [1] + [int(n) for n in s["oracle"]], s["oracle_base"]
    plotting.success_curve(ax, on, [ob["k"]] + [s["oracle"][str(n)]["k"] for n in on[1:]], ob["n"],
                           color=P[4], marker="s", label=f"sim lookahead, {cfg.oracle_rollouts} rollouts per "
                           f"candidate, {ob['n']} seeds (not deployable)")
    plotting.heldout_point(ax, n_star * 1.12, *s["eval"]["pooled"], label=f"{final_label}, {kind} N={n_star}",
                           color=P[1])
    plotting.heldout_point(ax, 1.12, *s["eval"]["base"], label=f"{final_label}, base", color="black")
    log2(ax, ns, "N (candidate chunks per decision)")
    ax.set(ylabel="success", title="Best-of-N at every chunk decision (search set unless marked)")
    ax.legend(loc="lower right", fontsize=8)
    plotting.save_fig(fig, media / "success_vs_n.png")

    fig, ax = plt.subplots()
    sweep = s["data_sweep"]
    sizes = sorted(int(x) for x in sweep["pooled"])
    for per in sweep["per_seed"].values():
        ax.plot(sizes, [per[str(x)][0] / per[str(x)][1] for x in sizes], color="gray", alpha=0.5, lw=1)
    plotting.success_curve(ax, sizes, [sweep["pooled"][str(x)][0] for x in sizes],
                           [sweep["pooled"][str(x)][1] for x in sizes],
                           label=f"{kind} verifier, N={n_star} (search, pooled; gray = each seed)")
    b = s["curves"]["random"]["1"]
    ax.axhline(b[0] / b[1], ls="--", color=P[1], label="base (search)")
    log2(ax, sizes, "labeled base rollouts used to train the verifier")
    ax.set(ylabel="success", title="Few labeled rollouts: small, seed-dependent gains")
    ax.legend(loc="lower right", fontsize=9)
    plotting.save_fig(fig, media / "data_sweep.png")


def make_gif(cfg: Config, media: Path, base_trajs: list, bon_trajs: list, label: str) -> int | None:
    """Same eval seed side by side: the base fails, best-of-N succeeds. Returns that seed."""
    from lastmile.common.plotting import replay_frames, save_gif, side_by_side
    from lastmile.envs.cupdrop import CupDropEnv

    pairs = [(b, v) for b, v in zip(base_trajs, bon_trajs) if not b["success"] and v["success"]]
    if not pairs:
        return None
    b, v = pairs[0]
    env = CupDropEnv(robot=cfg.robot)
    clips = [replay_frames(env, tr, text, every=3) for tr, text in ((b, f"base (seed {b['seed']})"), (v, label))]
    save_gif(side_by_side(*clips), media / "base_vs_best_of_n.gif", fps=5, max_size=400)
    return int(b["seed"])


# ---- The run


def main(cfg: Config) -> None:
    import torch

    torch.set_num_threads(4)
    out = REPO_ROOT / "runs/ch12_quick" if cfg.quick else REPO_ROOT
    media, work = out / "media/ch12", out / "runs/ch12_work"
    work.mkdir(parents=True, exist_ok=True)
    curve_eps = min(cfg.data_sweep, key=lambda x: abs(x - 512))  # this small-data verifier gets an N curve
    shared = Ledger("ch12", "shared")  # never written: counts the rollouts every row pays for
    s: dict = {"curve_episodes": curve_eps, "curves": {}, "per_seed": {}, "verifiers": {}}
    wall0 = time.perf_counter()

    with ExitStack() as stack:
        led = partial(Ledger, chapter="ch12", robot=cfg.robot, config=cfg, results_root=out / "results")
        rows = {v: stack.enter_context(led(method=f"bon_s{v}")) for v in cfg.verifier_seeds}
        rnd_row = stack.enter_context(led(method="bon-random"))

        # 1. Diagnostic: pass@k on search. Salts 0..search_salts-1 double as the base (N=1) in the sweeps.
        diag = search_score(cfg, base_policy, shared, range(cfg.pass_k))
        base_hits = np.sum([r.successes for r in diag], 0)  # per search state: successes out of pass_k
        ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= cfg.pass_k]
        s["pass_at_k"] = {str(k): [int(np.any([r.successes for r in diag[:k]], 0).sum()), 64] for k in ks}
        base_search = diag[: cfg.search_salts]
        print("pass@k (search):", rates(s["pass_at_k"]), flush=True)

        # 2. Random-pick control: the base's own distribution, drawn in a different order. First a check
        # that the wrapper is transparent: N=1 must reproduce the base episode for episode.
        n1 = search_score(cfg, partial(bon_policy, 1, "random"), shared, range(1))[0]
        assert np.array_equal(n1.successes, diag[0].successes) and n1.env_steps == diag[0].env_steps
        s["curves"]["random"] = {"1": kn(base_search)}
        for n in cfg.ns[1:]:
            s["curves"]["random"][str(n)] = kn(search_score(cfg, partial(bon_policy, n, "random"), shared))
        print("random pick (search):", rates(s["curves"]["random"]), flush=True)

        # 3. Per verifier seed: labeled rollouts, two verifiers (+ a chunk-blind one), a sweep over N.
        datas, hits = {}, {}  # hits[(v, kind, n)]: per search state, successes over the search salts
        for v in cfg.verifier_seeds:
            datas[v] = collect(cfg, v, cfg.train_episodes, rows[v])
            print(f"seed {v}: {cfg.train_episodes} labeled episodes, base success "
                  f"{format_rate(int(datas[v]['episode_success'].sum()), cfg.train_episodes)}", flush=True)
            s["per_seed"][str(v)] = {}
            for kind in ("clf", "q", "state"):
                path = work / f"{kind}_s{v}_{cfg.train_episodes}.pt"
                info = train_verifier(cfg, datas[v], kind, cfg.train_episodes, v, path)
                s["verifiers"][f"{kind}_s{v}"] = info
                print(f"  {kind}: held-out AUC {info['val_auc']:.3f}", flush=True)
                if kind == "state":  # its AUC is the point: it ranks states, so it cannot rank candidates
                    continue
                s["per_seed"][str(v)][kind] = {"1": kn(base_search)}
                for n in cfg.ns[1:]:
                    res = search_score(cfg, partial(bon_policy, n, "verifier", str(path)), rows[v])
                    s["per_seed"][str(v)][kind][str(n)] = kn(res)
                    hits[v, kind, n] = np.sum([r.successes for r in res], 0)
                print(f"  {kind} success vs N: {rates(s['per_seed'][str(v)][kind])}", flush=True)
        selection_steps = {v: dict(rows[v].robot_steps) for v in rows}  # rollouts that shaped the choice

        # 4. Choose (verifier, N) on search: best pooled success over seeds, the smaller N on ties. Also
        # record what each seed alone would have chosen: the rows share one choice, so they are replicates
        # of "train a verifier, deploy it at the pooled choice", not of the whole recipe.
        def best(curves: dict) -> list:
            """[kind, N] with the highest success over N > 1, the smaller N on ties."""
            _, neg_n, kind = max((c[0] / c[1], -int(n), kind) for kind in ("clf", "q")
                                 for n, c in curves[kind].items() if int(n) > 1)
            return [kind, -neg_n]

        for kind in ("clf", "q"):
            s["curves"][kind] = {str(n): [sum(s["per_seed"][str(v)][kind][str(n)][i] for v in cfg.verifier_seeds)
                                          for i in (0, 1)] for n in cfg.ns}
        kind, n_star = best(s["curves"])
        s["chosen"] = {"kind": kind, "n": n_star, "search": s["curves"][kind][str(n_star)],
                       "per_seed_choice": {str(v): best(s["per_seed"][str(v)]) for v in cfg.verifier_seeds}}
        print(f"chosen on search: {kind} N={n_star} {format_rate(*s['chosen']['search'])}; "
              f"each seed alone: {s['chosen']['per_seed_choice']}", flush=True)
        # Hard start states: the base solved them in at most 2 of pass_k tries. How does the choice do there?
        chosen_hits = sum(hits[v, kind, n_star] for v in cfg.verifier_seeds)
        tries = len(cfg.verifier_seeds) * cfg.search_salts
        s["hard_states"] = {str(i): {"seed": int(INIT_SETS["search"][i]), "base": [int(base_hits[i]), cfg.pass_k],
                                     "chosen": [int(chosen_hits[i]), tries]}
                            for i in np.flatnonzero(base_hits <= 2)}
        print("hard search states, env seed: base successes -> chosen successes:",
              {h["seed"]: f"{h['base'][0]}/{h['base'][1]} -> {h['chosen'][0]}/{h['chosen'][1]}"
               for h in s["hard_states"].values()}, flush=True)

        # 5. Data sweep for the chosen verifier: fewer labeled rollouts (nested subsets of the same pool).
        sweep = {"per_seed": {}, "pooled": {}}
        small_curve = {str(n): [0, 0] for n in cfg.ns}
        for v in cfg.verifier_seeds:
            per = sweep["per_seed"][str(v)] = {str(cfg.train_episodes): s["per_seed"][str(v)][kind][str(n_star)]}
            for ne in cfg.data_sweep:
                path = work / f"{kind}_s{v}_{ne}.pt"
                s["verifiers"][f"{kind}_s{v}_{ne}"] = train_verifier(cfg, datas[v], kind, ne, v, path)
                for n in cfg.ns[1:] if ne == curve_eps else [n_star]:
                    kn_ = kn(search_score(cfg, partial(bon_policy, n, "verifier", str(path)), rows[v]))
                    if ne == curve_eps:
                        small_curve[str(n)] = [a + b for a, b in zip(small_curve[str(n)], kn_)]
                    if n == n_star:
                        per[str(ne)] = kn_
            print(f"seed {v} data sweep at N={n_star}: {rates(per)}", flush=True)
        small_curve["1"] = [len(cfg.verifier_seeds) * x for x in kn(base_search)]
        s["curves"][f"{kind}_{curve_eps}"] = small_curve
        sweep["pooled"] = {ne: [sum(p[ne][i] for p in sweep["per_seed"].values()) for i in (0, 1)] for ne in per}
        s["data_sweep"] = sweep

        # 6. Simulator lookahead (not deployable; its rollouts are search budget like any other) and latency.
        s["oracle_base"] = {"k": int(diag[0].successes[: cfg.oracle_seeds].sum()), "n": cfg.oracle_seeds}
        s["oracle"] = {}
        for n in cfg.oracle_ns:
            t0 = time.perf_counter()
            o = s["oracle"][str(n)] = run_oracle(cfg, n)
            print(f"sim lookahead N={n}: {format_rate(o['k'], o['n'])} ({time.perf_counter() - t0:.0f}s, "
                  f"{o['executed_steps'] + o['lookahead_steps']} sim steps)", flush=True)
        vpath = str(work / f"{kind}_s{cfg.verifier_seeds[0]}_{cfg.train_episodes}.pt")
        s["latency_ms"] = {str(n): latency_ms(datas[cfg.verifier_seeds[0]]["obs"][:1], n, vpath) for n in cfg.ns}

        # 7. Held-out eval, once: the base, the chosen setting for each verifier seed, the random control.
        ev = partial(evaluate, init_set="search" if cfg.quick else "eval", robot=cfg.robot,
                     n_workers=cfg.n_workers, n=cfg.eval_n)
        base_eval = ev(base_policy, record_trajectories=True)
        finals = {}
        for v in cfg.verifier_seeds:
            make = partial(bon_policy, n_star, "verifier", str(work / f"{kind}_s{v}_{cfg.train_episodes}.pt"))
            finals[v] = ev(make, ledger=rows[v], category="eval", record_trajectories=not finals)
        rnd_eval = ev(partial(bon_policy, n_star, "random"), ledger=rnd_row, category="eval")
        pooled = [sum(f.k for f in finals.values()), sum(f.n for f in finals.values())]
        b0 = base_eval.successes
        s["eval"] = {"base": [base_eval.k, base_eval.n], "pooled": pooled, "random": [rnd_eval.k, rnd_eval.n],
                     "per_seed": {str(v): [f.k, f.n] for v, f in finals.items()},
                     "mcnemar_p": {str(v): mcnemar_exact(b0, f.successes) for v, f in finals.items()},
                     "diff_ci": {str(v): bootstrap_diff(b0, f.successes) for v, f in finals.items()},
                     "paired": {str(v): {"fixed": int((~b0 & f.successes).sum()),
                                         "broke": int((b0 & ~f.successes).sum()),
                                         "both_fail": int((~b0 & ~f.successes).sum())} for v, f in finals.items()},
                     "random_mcnemar_p": mcnemar_exact(b0, rnd_eval.successes)}
        from lastmile.envs.cupdrop import CupDropEnv, OutcomeTracker

        env = CupDropEnv(robot=cfg.robot)
        first = finals[cfg.verifier_seeds[0]]
        s["failure_modes"] = {}
        for name, r in (("base", base_eval), (f"best_of_n_s{cfg.verifier_seeds[0]}", first)):
            labels = [OutcomeTracker.label_trajectory(env, t) for t in r.trajectories]
            s["failure_modes"][name] = {m: labels.count(m) for m in sorted(set(labels))}

        # 8. Costs (robot steps). Every row pays the shared search (diagnostic + random sweep). Because the
        # choice pooled all verifier seeds, each verifier row also pays the other seeds' labeled rollouts and
        # sweeps, and the simulator lookahead. A --prior-costs manifest adds rollouts spent before this run.
        oracle_steps = sum(o["executed_steps"] + o["lookahead_steps"] for o in s["oracle"].values())
        prior = json.loads((REPO_ROOT / cfg.prior_costs).read_text()) if cfg.prior_costs else {"items": []}
        s["cost_steps"] = {"shared_search": shared.robot_steps["search"], "sim_lookahead_search": oracle_steps,
                           "selection_per_seed": selection_steps, "prior": prior}
        for key, row in [*rows.items(), ("random", rnd_row)]:
            row.robot_steps["search"] += shared.robot_steps["search"]
            if key != "random":
                row.robot_steps["search"] += oracle_steps
                for w in rows:
                    for c in ("search", "train"):
                        row.robot_steps[c] += selection_steps[w][c] if w != key else 0
            for item in prior["items"]:
                if item["rows"] == "all" or key != "random":
                    for c in ("search", "train"):
                        row.robot_steps[c] += item.get(c, 0)
            row.robot_steps["eval"] += base_eval.env_steps
            row.worker_cpu_seconds += shared.worker_cpu_seconds
            row.set_base(id="base_v1", successes=base_eval)
        for v, row in rows.items():
            row.method = f"bon-{kind}-n{n_star}_s{v}"
            row.set_final(successes=finals[v])
            s["cost_steps"][f"seed{v}"] = dict(row.robot_steps)
        rnd_row.method = f"bon-random-n{n_star}"
        rnd_row.set_final(successes=rnd_eval)
        s["cost_steps"]["random"] = dict(rnd_row.robot_steps)

        s["gif_seed"] = None
        with optional_media("the base vs best-of-N GIF"):
            s["gif_seed"] = make_gif(cfg, media, base_eval.trajectories, first.trajectories,
                                     f"best-of-{n_star}, {kind} verifier") if first.trajectories else None
        make_plots(cfg, media, s)
        for row in [*rows.values(), rnd_row]:
            row.extra = s

    final_label = "search" if cfg.quick else "held-out eval"
    print(f"\n=== Chapter 12 summary ({final_label} unless marked) ===")
    print(f"base_v1:            {base_eval.summary}")
    for v, f in finals.items():
        print(f"{kind} N={n_star} seed {v}: {f.summary}  McNemar p={s['eval']['mcnemar_p'][str(v)]:.2g}  "
              f"vs base per episode: {s['eval']['paired'][str(v)]}")
    print(f"pooled over seeds:  {format_rate(*pooled)}")
    print(f"random pick N={n_star}: {rnd_eval.summary}  McNemar p={s['eval']['random_mcnemar_p']:.2g}")
    print(f"chosen on search: {kind} N={n_star}, search {format_rate(*s['chosen']['search'])}")
    for key, row in [*rows.items(), ("random", rnd_row)]:
        print(f"row {key} robot-min:", {c: round(m, 1) for c, m in row.robot_minutes.items()})
    sel_min = {v: {c: round(d[c] / 600, 1) for c in ("search", "train")} for v, d in selection_steps.items()}
    print(f"this run only (robot-min): shared search {shared.robot_steps['search'] / 600:.1f}, "
          f"sim lookahead {oracle_steps / 600:.1f}, each seed's labels + sweeps {sel_min}")
    print(f"latency ms/decision: { {n: round(x, 2) for n, x in s['latency_ms'].items()} }")
    print(f"wall {(time.perf_counter() - wall0) / 60:.1f} min; results in {out / 'results/ch12'}")


if __name__ == "__main__":
    main(parse(Config))
