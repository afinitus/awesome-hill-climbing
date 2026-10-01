"""Chapter 0: the base policy. Behavior-clone a flow-matching chunk policy from mixed-quality demos.

Learning goal
    Build the frozen starting point every later chapter climbs from: `base_v1`, a FlowChunk-S policy
    (DESIGN.md §6) cloned from scripted demos, calibrated on the search set to succeed about half the
    time while still succeeding *sometimes* from almost every start (pass@8 >= 90%), so that selection,
    steering and RL methods have headroom to climb.

Base idea in plain words
    Behavior cloning is supervised learning on (observation, action) pairs: copy what the demonstrator
    did. A flow-matching policy copies the whole *distribution* of demonstrated action chunks, not just
    their average, by learning a velocity field that carries Gaussian noise to a chunk. Rectified flow:

        x0 ~ N(0, I),  x1 = normalized action chunk,  x_t = (1 - t) x0 + t x1,  t ~ U(0, 1)
        loss = || v_theta(obs, x_t, t) - (x1 - x0) ||^2

    Sampling integrates dx/dt = v_theta from the noise in 10 Euler steps. The clone reproduces the
    mixture it was shown: at a fixed training budget, the fraction of sloppy demos is the main dial on
    its success rate, which is how we set the base rate. More demos of the same mix still help once
    training grows with the data (the matched-epoch ablation below measures by how much).

Papers mirrored
    Rectified flow (Liu et al. 2022, arXiv 2209.03003) and flow matching (Lipman et al. 2022, 2210.02747)
    for the training objective; action chunking from ACT (Zhao et al. 2023, 2304.13705) and Diffusion
    Policy (Chi et al. 2023, 2303.04137); the noise-as-a-knob view that later chapters exploit comes from
    Golden Ticket [C236]; mixed-quality demo data as the bottleneck: RINSE [C274], JITI [C189], PACL [C448].

Honesty
    Every knob (demo count, sloppy fraction, training steps, which training seed becomes base_v1) is
    chosen on the `search` set. The held-out `eval` set (n=256) is scored once at the end. Demo rollouts
    are charged to robot-minutes (`train`) and to human-minutes (`demos`) at their real-arm time (10 Hz);
    calibration rollouts, and the pilot and earlier runs that shaped this script, are charged to `search`.

Run
    uv run python algos/00_bc_flow.py            # full run, 11-16 min on an M5 Pro with 6 workers
    uv run python algos/00_bc_flow.py --quick    # smoke test, < 2 min; every file (results JSON included)
                                                 # goes to runs/ch00_quick/
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np

from lastmile.common.cli import parse
from lastmile.common.eval import format_rate, wilson
from lastmile.common.ledger import REPO_ROOT, Ledger, git_sha
from lastmile.common.rollout import evaluate, rollout_seeds
from lastmile.envs.cupdrop import CupDropEnv, OutcomeTracker
from lastmile.envs.knob_controller import TUNED_KNOBS, KnobController

# torch, policy_flow and matplotlib are imported inside the functions that need them. Rollout workers
# are spawned and re-import this file; the ones that only run the scripted controller then never load
# torch or matplotlib, which keeps worker start-up fast.

DEMO_SEED_BASE = 50_000  # far from search (10k), eval (20k), eval_ext (30k) and stress (40k) seeds

# Search rollouts spent *before* this run, charged here so the ledger shows what the recipe really cost
# (CONTRIBUTING rule 5). Pilots: about 46 configurations tried by hand before this script existed
# (keep-all vs successes-only, sloppy noise 0.1/0.2 vs 0.15/0.3, training length), about 23,500 search
# episodes. Their lengths were not logged, so each is charged 100 steps (this script averages ~91).
# Earlier full runs of this script: exact search steps from their results files (runs/ch00_superseded/).
PRIOR_SEARCH_STEPS = {
    "pilots_estimate_23500_episodes": 23_500 * 100,
    "full_run_20260930-133515": 716_358,
    "full_run_20260930-135247": 1_228_713,
}


@dataclass
class Config:
    robot: str = "so101"
    n_demos: list[int] = field(default_factory=lambda: [10, 20, 40, 80], metadata={"help": "demo counts"})
    sloppy_fracs: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75],
                                      metadata={"help": "fractions of sloppy demos"})
    sloppy_knob_std: float = 0.15  # per-episode knob jitter (fraction of each knob's range)
    sloppy_action_std: float = 0.3  # per-step Gaussian action noise
    train_steps: int = 4000  # gradient steps per model in the sweep, fixed from the pilots
    steps_ablation: list[int] = field(default_factory=lambda: [1000, 2000, 8000, 16000])
    epoch_ablation: bool = True  # retrain every n at the chosen fraction with base_v1's epoch count
    batch_size: int = 256
    lr: float = 1e-3
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2], metadata={"help": "pipeline seeds"})
    search_salts: int = 2  # policy-noise salts per search evaluation (64 episodes each)
    target: tuple[float, float] = (0.45, 0.55)  # calibration band for search success
    pass_k: int = 8
    min_pass_k: float = 0.90
    eval_n: int = 256
    n_workers: int = 6
    train_threads: int = 2  # torch threads per training process (one process per seed)
    quick: bool = False
    QUICK: ClassVar[dict] = {"n_demos": [10, 20], "sloppy_fracs": [0.5], "train_steps": 1500,
                             "steps_ablation": [], "epoch_ablation": False, "seeds": [0, 1],
                             "search_salts": 1, "pass_k": 2, "eval_n": 64}


# --------------------------------------------------------------------------------------------------
# Demos: the scripted TUNED controller is our "operator"; with noise it becomes a sloppy operator.
# --------------------------------------------------------------------------------------------------


def demo_seeds(seed: int, sloppy: bool, count: int) -> list[int]:
    """Env seeds for pipeline `seed`: clean demos at 50_000 + 2000*seed + j, sloppy ones 1000 higher."""
    start = DEMO_SEED_BASE + 2_000 * seed + (1_000 if sloppy else 0)
    return list(range(start, start + count))


def record_demos(cfg: Config, seed: int, sloppy: bool, count: int, ledger: Ledger) -> list[dict]:
    """Record `count` demos. We keep every attempt, failed or not (see the chapter doc for why)."""
    if count == 0:
        return []
    noise = {"knob_std": cfg.sloppy_knob_std, "action_std": cfg.sloppy_action_std} if sloppy else None
    seeds = demo_seeds(seed, sloppy, count)
    # Known flaw, kept so base_v1 stays frozen: the controller seed equals the env seed, and both seed
    # np.random.default_rng, so a sloppy demo's knob jitter is a (weak) function of its cube pose.
    # A base_v2 should pass [policy_seed("demos", s) for s in seeds] instead (see ch00.md).
    res = rollout_seeds(partial(KnobController, TUNED_KNOBS[cfg.robot], cfg.robot, noise), seeds, seeds,
                        robot=cfg.robot, n_workers=cfg.n_workers, record_trajectories=True,
                        ledger=ledger, category="train")  # the arm is busy while someone teleoperates it
    ledger.human_minutes["demos"] += res.env_steps / 600  # ...and so is the operator, at 10 Hz
    return res.trajectories


def n_sloppy_of(n: int, frac: float) -> int:
    # Python rounds halves to even: round(2.5) = 2 and round(7.5) = 8, so at n = 10 the "25%" and
    # "75%" cells are really 2/10 and 8/10 sloppy. The results JSON stores the real count per cell.
    return round(n * frac)


def build_dataset(pool: dict[str, list[dict]], n: int, frac: float):
    """The first n_sloppy sloppy and n - n_sloppy clean demos of a pool, so datasets nest as n grows."""
    from lastmile.common.policy_flow import make_chunk_dataset

    n_sloppy = n_sloppy_of(n, frac)
    trajs = pool["clean"][: n - n_sloppy] + pool["sloppy"][:n_sloppy]
    return make_chunk_dataset(trajs, horizon=8), trajs


# --------------------------------------------------------------------------------------------------
# Training (one process per pipeline seed) and scoring
# --------------------------------------------------------------------------------------------------


def epochs_for(n_samples: int, steps: int, batch_size: int) -> int:
    """train_flowchunk counts epochs; we fix gradient steps so small datasets are not under-trained."""
    return max(1, math.ceil(steps / math.ceil(n_samples / batch_size)))


def train_job(obs, chunks, steps: int, seed: int, batch_size: int, lr: float, threads: int, path: str):
    """Train one FlowChunk-S and save it. Top-level so a spawned worker can run it."""
    import torch

    from lastmile.common.policy_flow import ChunkDataset, train_flowchunk

    torch.set_num_threads(threads)
    ds = ChunkDataset(obs, chunks)
    epochs = epochs_for(len(ds), steps, batch_size)
    model, losses = train_flowchunk(ds, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    model.save(path)
    return losses[-1]


def train_all(cfg: Config, pool_exec, datasets: dict, steps: int | dict, tag: str, work: Path) -> dict:
    """Train one model per pipeline seed in parallel; `steps` may differ per seed. Returns {seed: path}."""
    paths = {s: str(work / f"{tag}_seed{s}.pt") for s in datasets}
    jobs = [pool_exec.submit(train_job, ds.obs, ds.chunks, steps[s] if isinstance(steps, dict) else steps,
                             s, cfg.batch_size, cfg.lr, cfg.train_threads, paths[s])
            for s, ds in datasets.items()]
    for job in jobs:
        job.result()
    return paths


def score(cfg: Config, path: str, init_set: str, salts: range, ledger: Ledger, category: str,
          n: int | None = None, record: bool = False) -> list:
    """One `evaluate` per policy-noise salt. Salt s gives every episode fresh (but paired) noise."""
    from lastmile.common.policy_flow import FlowChunkPolicy

    return [evaluate(partial(FlowChunkPolicy.load, path, noise="gaussian"), init_set, robot=cfg.robot,
                     n_workers=cfg.n_workers, ledger=ledger, category=category, salt=salt, n=n,
                     record_trajectories=record and salt == 0) for salt in salts]


def sr(d: dict) -> dict:
    """{"k", "n"} -> the same plus the rate and its Wilson interval, for the JSON files."""
    return {"k": d["k"], "n": d["n"], "sr": d["k"] / d["n"], "ci": wilson(d["k"], d["n"])}


def flat(seed_runs: dict) -> list:
    return [r for rs in seed_runs.values() for r in rs]


def pooled(results: list) -> tuple[int, int]:
    return sum(r.k for r in results), sum(r.n for r in results)


def pass_at_k(results: list, k: int) -> tuple[int, int]:
    """Episodes (env seeds) solved by at least one of the first k noise salts."""
    hits = np.any([r.successes for r in results[:k]], axis=0)
    return int(hits.sum()), len(hits)


def cluster_ci(results: list, n_boot: int = 10_000) -> tuple[float, float]:
    """95% bootstrap interval that resamples *start states*, not episodes.

    Pooling seeds and salts reuses the same 64 (or 256) start states, so a Wilson interval on the pooled
    k/n treats repeated draws as independent and is conditional on these states. Here each state keeps
    all of its draws together, which gives an interval for the start-state distribution.
    """
    assert all(np.array_equal(r.seeds, results[0].seeds) for r in results), "results must share states"
    per_state = np.mean([r.successes for r in results], axis=0)
    idx = np.random.default_rng(0).integers(0, len(per_state), size=(n_boot, len(per_state)))
    lo, hi = np.quantile(per_state[idx].mean(axis=1), [0.025, 0.975])
    return float(lo), float(hi)


def summarize(cfg: Config, seed_runs: dict) -> dict:
    """Pooled pass@1 (Wilson and cluster intervals), pass@k and per-seed counts for {seed: [results]}."""
    k, m = pooled(flat(seed_runs))
    hk = [pass_at_k(rs, cfg.pass_k) for rs in seed_runs.values()]
    pk = (sum(h[0] for h in hk), sum(h[1] for h in hk))
    return {**sr({"k": k, "n": m}), "ci_cluster": cluster_ci(flat(seed_runs)), "pass_k": pk,
            "pass_k_ci": wilson(*pk), "per_seed": {s: pooled(rs) for s, rs in seed_runs.items()}}


# --------------------------------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------------------------------


def log2_axis(ax, ticks: list[int], xlabel: str) -> None:
    from matplotlib.ticker import NullLocator

    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(xlabel)


def make_plots(cfg, media, grid, ablation, epoch_abl, chosen, final, eval_runs, modes) -> None:
    import matplotlib.pyplot as plt

    from lastmile.common import plotting

    plotting.setup()
    n_sel, f_sel = chosen

    def curve(ax, xs, rows, **kw):  # rows: dicts with pooled "k" and "n" (search estimates)
        plotting.success_curve(ax, xs, [r["k"] for r in rows], [r["n"] for r in rows], **kw)

    fig, ax = plt.subplots()
    for frac in cfg.sloppy_fracs:
        ns = [n for n in cfg.n_demos if (n, frac) in grid]
        curve(ax, ns, [grid[(n, frac)] for n in ns], label=f"{frac:.0%} sloppy (search)")
    ax.axhspan(*cfg.target, color="0.85", zorder=0, label="target band")
    plotting.heldout_point(ax, n_sel * 1.1, final.k, final.n, label="base_v1, held-out eval")
    log2_axis(ax, cfg.n_demos, f"demos (all attempts kept), {cfg.train_steps} gradient steps each")
    ax.set(ylabel="success rate",
           title=f"Calibration on search ({len(cfg.seeds)} seeds x {cfg.search_salts} salts x 64)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plotting.save_fig(fig, media / "calibration.png")

    if len(ablation) > 1:
        fig, ax = plt.subplots()
        steps = sorted(ablation)
        curve(ax, steps, [ablation[s] for s in steps],
              label=f"pass@1, n={n_sel}, {f_sel:.0%} sloppy (search)")
        plotting.success_curve(ax, steps, [ablation[s]["pass_k"][0] for s in steps],
                               [ablation[s]["pass_k"][1] for s in steps], label=f"pass@{cfg.pass_k} (search)",
                               marker="s")
        plotting.heldout_point(ax, cfg.train_steps * 1.1, final.k, final.n, label="base_v1, held-out eval")
        ax.axhspan(*cfg.target, color="0.85", zorder=0)
        log2_axis(ax, steps, "gradient steps")
        ax.set(ylabel="success rate", title="Training length (search set)")
        ax.legend(fontsize=9, loc="best")
        plotting.save_fig(fig, media / "train_steps.png")

    if epoch_abl:
        fig, ax = plt.subplots()
        ns, ne, top = [n for n in cfg.n_demos if (n, f_sel) in grid], sorted(epoch_abl), max(ablation)
        curve(ax, ns, [grid[(n, f_sel)] for n in ns], label=f"fixed {cfg.train_steps} steps (the sweep)")
        curve(ax, ne, [epoch_abl[n] for n in ne], marker="s",
              label=f"epochs matched to each seed's n={n_sel} model")
        a = ablation[top]  # a single search point: draw its Wilson interval as an error bar
        lo_hi = np.abs(np.array(wilson(a["k"], a["n"])) - a["sr"])[:, None]
        ax.errorbar([n_sel], [a["sr"]], yerr=lo_hi, fmt="^", capsize=3, color=plotting.PALETTE[2],
                    label=f"n={n_sel} trained {top} steps")
        plotting.heldout_point(ax, n_sel * 1.1, final.k, final.n, label="base_v1, held-out eval")
        log2_axis(ax, ns, f"demos ({f_sel:.0%} sloppy, all attempts kept)")
        ax.set(ylabel="success rate", title="More demos of the same mix: it depends on training (search)")
        ax.legend(fontsize=8, loc="lower right")
        plotting.save_fig(fig, media / "data_scaling.png")

    fig, ax = plt.subplots()
    ks = list(range(1, len(eval_runs) + 1))
    hits = [pass_at_k(eval_runs, k) for k in ks]
    plotting.success_curve(ax, ks, [h[0] for h in hits], hits[0][1], label="base_v1 pass@k (held-out eval)")
    ax.axhline(cfg.min_pass_k, color="0.5", ls="--", lw=1, label=f"headroom target {cfg.min_pass_k:.0%}")
    ax.set_xticks(ks)
    ax.set(xlabel="k (noise draws per start state, sim resets)", ylabel="solved start states",
           title="pass@k is a ceiling for selection, not a success rate")
    ax.legend(fontsize=9, loc="lower right")
    plotting.save_fig(fig, media / "pass_at_k.png")

    fig, ax = plt.subplots()
    labels = [m for m, _ in modes.most_common()]
    colors = [plotting.PALETTE[2] if m == "success" else plotting.PALETTE[1] for m in labels]
    ax.bar(labels, [modes[m] for m in labels], color=colors)
    for i, m in enumerate(labels):
        ax.text(i, modes[m], str(modes[m]), ha="center", va="bottom")
    ax.set(ylabel=f"episodes (of {sum(modes.values())})", title="base_v1 outcomes on held-out eval (salt 0)")
    ax.grid(axis="x", visible=False)
    plotting.save_fig(fig, media / "failure_modes.png")


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------


def main(cfg: Config) -> None:
    out = REPO_ROOT / ("runs/ch00_quick" if cfg.quick else "")  # quick runs never touch the real files
    media, ckpt_dir, work = out / "media/ch00", out / "checkpoints", out / "runs/ch00_work"
    work.mkdir(parents=True, exist_ok=True)
    trainers = ProcessPoolExecutor(len(cfg.seeds), mp_context=mp.get_context("spawn"))
    grid_keys = [(n, f) for n in cfg.n_demos for f in cfg.sloppy_fracs]

    with Ledger(chapter="ch00", method="bc_flow", robot=cfg.robot, config=cfg, results_root=out / "results") as L:
        # 1. Record one demo pool per pipeline seed; every dataset in the sweep is a prefix of it.
        n_clean = max(n - n_sloppy_of(n, f) for n, f in grid_keys)
        n_sloppy = max(n_sloppy_of(n, f) for n, f in grid_keys)
        pools = {s: {"clean": record_demos(cfg, s, False, n_clean, L),
                     "sloppy": record_demos(cfg, s, True, n_sloppy, L)} for s in cfg.seeds}
        demo_quality = {kind: format_rate(sum(t["success"] for s in cfg.seeds for t in pools[s][kind]),
                                          sum(len(pools[s][kind]) for s in cfg.seeds))
                        for kind in ("clean", "sloppy")}
        print(f"demo pool: clean {demo_quality['clean']}, sloppy {demo_quality['sloppy']}")

        # 2. Calibration sweep on the search set: train each (n, frac) for every seed, score a few salts.
        grid, runs = {}, {}
        for n, frac in grid_keys:
            t0 = time.time()
            paths = train_all(cfg, trainers, {s: build_dataset(pools[s], n, frac)[0] for s in cfg.seeds},
                              cfg.train_steps, f"n{n}_f{frac}", work)
            runs[(n, frac)] = {s: score(cfg, p, "search", range(cfg.search_salts), L, "search")
                               for s, p in paths.items()}
            g = grid[(n, frac)] = summarize(cfg, runs[(n, frac)])
            print(f"n={n:3d} sloppy={frac:.2f}  search {format_rate(g['k'], g['n'])}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

        # 3. Pick the smallest setting (fewest demos, then least sloppiness) inside the band with headroom.
        lo, hi = cfg.target
        chosen, fallback = None, "no setting landed in the target band with enough headroom"
        for key in sorted(grid_keys):
            if lo <= grid[key]["sr"] <= hi:
                for s in cfg.seeds:  # top up to pass_k salts to measure headroom (search set only)
                    runs[key][s] += score(cfg, str(work / f"n{key[0]}_f{key[1]}_seed{s}.pt"), "search",
                                          range(cfg.search_salts, cfg.pass_k), L, "search")
                spk = grid[key]["search_pass_k"] = summarize(cfg, runs[key])["pass_k"]
                print(f"  candidate {key}: search pass@{cfg.pass_k} {format_rate(*spk)}")
                if spk[0] / spk[1] >= cfg.min_pass_k:
                    chosen, fallback = key, None
                    break
        if chosen is None:  # say so, and take the setting closest to 50%
            chosen = min(grid_keys, key=lambda c: abs(grid[c]["sr"] - 0.5))
            print(f"WARNING: {fallback}; using the closest setting {chosen}")
        seed_sr = {s: pooled(runs[chosen][s]) for s in cfg.seeds}
        base_seed = min(cfg.seeds, key=lambda s: abs(seed_sr[s][0] / seed_sr[s][1] - 0.5))  # on search
        n_sel, f_sel = chosen
        print(f"chosen n={n_sel} sloppy={f_sel}; base_v1 = seed {base_seed} "
              f"(search {format_rate(*seed_sr[base_seed])})")

        # 4. Ablations on the search set (reported, never used to select). Every point uses all pass_k
        # salts, so pass@1 (pooled over salts) and pass@k come from the same episodes.
        # 4a. Training length for the chosen data.
        ablation = {cfg.train_steps: summarize(cfg, runs[chosen])}
        for steps in cfg.steps_ablation:
            paths = train_all(cfg, trainers, {s: build_dataset(pools[s], n_sel, f_sel)[0] for s in cfg.seeds},
                              steps, f"abl{steps}", work)
            ablation[steps] = summarize(cfg, {s: score(cfg, p, "search", range(cfg.pass_k), L, "search")
                                              for s, p in paths.items()})
        for steps, a in sorted(ablation.items()):
            print(f"ablation {steps:5d} steps: search pass@1 {format_rate(a['k'], a['n'])}, "
                  f"pass@{cfg.pass_k} {format_rate(*a['pass_k'])}", flush=True)
        # 4b. Demo count at a matched number of epochs. The sweep holds gradient steps fixed, so a bigger
        # dataset is seen fewer times; here each seed trains every n for as many epochs as its n_sel model.
        epoch_abl = {}
        if cfg.epoch_ablation:
            sel_len = {s: len(build_dataset(pools[s], n_sel, f_sel)[0]) for s in cfg.seeds}
            epochs = {s: epochs_for(sel_len[s], cfg.train_steps, cfg.batch_size) for s in cfg.seeds}
            epoch_abl[n_sel] = {**ablation[cfg.train_steps], "epochs": epochs[base_seed],
                                "steps": {s: cfg.train_steps for s in cfg.seeds}}
            for n in [m for m in cfg.n_demos if m != n_sel]:
                ds = {s: build_dataset(pools[s], n, f_sel)[0] for s in cfg.seeds}
                steps = {s: epochs[s] * math.ceil(len(ds[s]) / cfg.batch_size) for s in cfg.seeds}
                paths = train_all(cfg, trainers, ds, steps, f"ep_n{n}_f{f_sel}", work)
                runs_n = {s: score(cfg, p, "search", range(cfg.pass_k), L, "search")
                          for s, p in paths.items()}
                epoch_abl[n] = {**summarize(cfg, runs_n), "epochs": epochs[base_seed], "steps": steps}
            for n, a in sorted(epoch_abl.items()):
                print(f"matched epochs n={n:3d} ({max(a['steps'].values())} steps): search pass@1 "
                      f"{format_rate(a['k'], a['n'])}, pass@{cfg.pass_k} {format_rate(*a['pass_k'])}")
        trainers.shutdown()

        # 5. Held-out evaluation, once. Base = the scripted TUNED controller that produced the clean demos.
        final_set = "search" if cfg.quick else "eval"
        scripted = evaluate(partial(KnobController, TUNED_KNOBS[cfg.robot], cfg.robot), final_set,
                            robot=cfg.robot, n_workers=cfg.n_workers, ledger=L, category="eval", n=cfg.eval_n)
        L.set_base(id="knobs_tuned_scripted", successes=scripted)
        per_seed_eval, eval_runs, eval_seconds = {}, [], 0.0
        for s in cfg.seeds:
            path = str(work / f"n{n_sel}_f{f_sel}_seed{s}.pt")
            t0 = time.time()
            res = score(cfg, path, final_set, range(1), L, "eval", n=cfg.eval_n, record=s == base_seed)[0]
            per_seed_eval[s] = res
            if s == base_seed:
                eval_seconds = time.time() - t0
                eval_runs = [res] + score(cfg, path, final_set, range(1, cfg.pass_k), L, "eval", n=cfg.eval_n)
        final = per_seed_eval[base_seed]
        L.set_final(successes=final)
        env = CupDropEnv(robot=cfg.robot)
        modes = Counter(OutcomeTracker.label_trajectory(env, t) for t in final.trajectories)
        pk, mean1 = pass_at_k(eval_runs, cfg.pass_k), pooled(eval_runs)
        pooled_eval = pooled(list(per_seed_eval.values()))
        this_run_search = L.robot_minutes["search"]
        if not cfg.quick:  # the pilots and earlier full runs were real search rollouts too
            L.robot_steps["search"] += sum(PRIOR_SEARCH_STEPS.values())

        # 6. Save base_v1 with its manifest (full runs only write to checkpoints/).
        from lastmile.common.policy_flow import FlowChunk

        p_sel = pools[base_seed]
        ds, trajs = build_dataset(p_sel, n_sel, f_sel)
        demo_steps = int(sum(len(t["actions"]) for t in trajs))
        n_sl = n_sloppy_of(n_sel, f_sel)
        top, big = max(ablation), (max(epoch_abl) if epoch_abl else None)
        manifest = {
            "name": "base_v1", "robot": cfg.robot, "variant": "v1",
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_sha": git_sha(), "dataset_hash": ds.hash(), "n_samples": len(ds),
            "n_demos": n_sel, "sloppy_fraction": f_sel, "n_sloppy": n_sl,
            "keep_rule": "all attempts (successes and failures)",
            "demo_successes": int(sum(t["success"] for t in trajs)),
            "sloppy_noise": {"knob_std": cfg.sloppy_knob_std, "action_std": cfg.sloppy_action_std},
            "demo_seeds": {"clean": [int(t["seed"]) for t in trajs[: n_sel - n_sl]],
                           "sloppy": [int(t["seed"]) for t in trajs[n_sel - n_sl :]]},
            "demo_policy_seeds": "same as the demo env seeds (see known_caveats)",
            "demo_env_steps": demo_steps, "real_arm_demo_minutes": demo_steps / 600,
            "training": {"steps": cfg.train_steps, "epochs": epochs_for(len(ds), cfg.train_steps,
                                                                        cfg.batch_size),
                         "batch_size": cfg.batch_size, "lr": cfg.lr, "seed": base_seed,
                         "optimizer": "AdamW+cosine"},
            "search": {**sr(dict(zip("kn", seed_sr[base_seed]))), "salts": len(runs[chosen][base_seed]),
                       "ci_cluster": cluster_ci(runs[chosen][base_seed])},
            "eval": {"k": final.k, "n": final.n, "sr": final.sr, "ci": final.ci, "salt": 0},
            f"eval_pass_at_{cfg.pass_k}": sr(dict(zip("kn", pk))),
            f"eval_pass_at_1_mean_over_{cfg.pass_k}_salts": {**sr(dict(zip("kn", mean1))),
                                                               "ci_cluster": cluster_ci(eval_runs)},
            "failure_modes_eval": dict(modes), "eval_seconds": eval_seconds,
            "eval_seconds_n_episodes": cfg.eval_n, "calibration_fallback": fallback,
            "known_caveats": [  # later chapters: read these before claiming a gain over base_v1
                "Under-trained on purpose: more steps on the same demos score higher (plateau_search_sr).",
                "More demos of the same mix, trained as many epochs, score higher (more_demos_search_sr).",
                "Sloppy demos: controller seed = env seed, so knob jitter is weakly tied to the cube pose.",
                "Failures are mostly missed grasps (failure_modes_eval).",
            ],
            "plateau_search_sr": {"steps": top, "n_demos": n_sel, **sr(ablation[top])},
            "more_demos_search_sr": {"n_demos": big, "epochs": epoch_abl[big]["epochs"], **sr(epoch_abl[big])}
            if big else None,
            "mandatory_baselines": {"chapters": [4, 9, 10], "baselines": [
                "train longer on the same demos", "train longer on more demos of the same mix"]},
        }
        model = FlowChunk.load(work / f"n{n_sel}_f{f_sel}_seed{base_seed}.pt")
        ckpt = ckpt_dir / f"base_v1_{cfg.robot}.pt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")  # write, then rename, so no reader ever sees half a checkpoint
        model.save(tmp, **{k: v for k, v in manifest.items() if isinstance(v, (str, int, float))})
        os.replace(tmp, ckpt)
        manifest["config_hash"] = model.config.hash()
        manifest["arch"] = asdict(model.config)
        (ckpt_dir / f"base_v1_{cfg.robot}.json").write_text(json.dumps(manifest, indent=2) + "\n")

        L.extra.update({
            "checkpoint": str(ckpt.relative_to(REPO_ROOT)), "manifest": manifest,
            "demo_pool_quality": demo_quality,
            "calibration": [{"n_demos": n, "sloppy_frac": f, "n_sloppy": n_sloppy_of(n, f), **g}
                            for (n, f), g in grid.items()],
            "chosen": {"n_demos": n_sel, "sloppy_frac": f_sel, "base_seed": base_seed},
            "steps_ablation": {str(s): v for s, v in sorted(ablation.items())},
            "epoch_ablation": {str(n): v for n, v in sorted(epoch_abl.items())},
            "eval_per_seed": {str(s): {"k": r.k, "n": r.n, "ci": r.ci} for s, r in per_seed_eval.items()},
            "eval_pooled_seeds": {**sr(dict(zip("kn", pooled_eval))),
                                  "ci_cluster": cluster_ci(list(per_seed_eval.values()))},
            "eval_pass_at_k": {str(k): sr(dict(zip("kn", pass_at_k(eval_runs, k))))
                               for k in range(1, cfg.pass_k + 1)},
            "failure_modes": dict(modes),
            "search_robot_minutes_this_run": this_run_search,
            "prior_search_steps_charged": {} if cfg.quick else PRIOR_SEARCH_STEPS,
        })

        # 7. Media: plots, then GIFs of demos and of base_v1 (first success and first failure, in seed order).
        make_plots(cfg, media, grid, ablation, epoch_abl, chosen, final, eval_runs, modes)
        from lastmile.common.plotting import optional_media, replay_frames, save_gif, side_by_side

        def clip(traj, label):
            return replay_frames(CupDropEnv(robot=cfg.robot), traj, label)

        wins = [t for t in final.trajectories if t["success"]]
        fails = [t for t in final.trajectories if not t["success"]]
        with optional_media("the ch00 GIFs"):
            save_gif(side_by_side(clip(p_sel["clean"][0], "clean demo"), clip(p_sel["sloppy"][0], "sloppy demo")),
                     media / "demos_clean_vs_sloppy.gif", fps=5, max_size=400)
            if wins and fails:
                save_gif(side_by_side(clip(wins[0], f"base_v1 success (seed {wins[0]['seed']})"),
                                      clip(fails[0], f"base_v1 failure (seed {fails[0]['seed']})")),
                         media / "base_v1_success_vs_failure.gif", fps=5, max_size=400)
        L.extra["gif_episodes"] = {
            "success": int(wins[0]["seed"]) if wins else None,
            "failure": int(fails[0]["seed"]) if fails else None,
            "failure_label": OutcomeTracker.label_trajectory(env, fails[0]) if fails else None,
            "sloppy_demo_success": bool(p_sel["sloppy"][0]["success"]) if p_sel["sloppy"] else None,
        }

    rm, c = L.robot_minutes, cluster_ci(eval_runs)
    print("\n=== Chapter 0 summary ===")
    print(f"scripted TUNED (eval):         {scripted.summary}")
    print(f"base_v1 pass@1 (eval, salt 0): {final.summary}   <- held-out")
    print(f"base_v1 pass@1 mean {cfg.pass_k} salts:  {format_rate(*mean1)}  "
          f"(start-state cluster bootstrap [{100 * c[0]:.1f}, {100 * c[1]:.1f}])")
    print(f"base_v1 pass@{cfg.pass_k} (eval):         {format_rate(*pk)}  (sim resets: a ceiling)")
    print("\n".join(f"  pass@{k}: {format_rate(*pass_at_k(eval_runs, k))}" for k in range(1, cfg.pass_k + 1)))
    for s, r in per_seed_eval.items():
        print(f"  seed {s} eval: {r.summary}{'  <- base_v1 (picked on search)' if s == base_seed else ''}")
    print(f"  pooled seeds eval: {format_rate(*pooled_eval)}")
    print(f"failure modes: {dict(modes)}")
    print(f"{cfg.eval_n}-episode eval took {eval_seconds:.1f}s with {cfg.n_workers} workers")
    print(f"robot-minutes: search {rm['search']:.1f} (this run {this_run_search:.1f}), train (demos) "
          f"{rm['train']:.1f}, eval {rm['eval']:.1f}; human demo minutes {L.human_minutes['demos']:.1f}; "
          f"base_v1's own demos {demo_steps / 600:.1f} min")
    print(f"results: {L.path}\ncheckpoint: {ckpt}")


if __name__ == "__main__":
    main(parse(Config))
