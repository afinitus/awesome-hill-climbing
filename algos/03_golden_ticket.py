"""Chapter 3: the golden ticket. Black-box search over one noise vector of a frozen flow policy.

Learning goal
    Improve the shared base policy base_v1 without touching a single weight, by searching the 32 numbers it
    turns into actions, and learn to report such a search honestly: pick on the search set, score on eval,
    and look at how far the search score overstates the held-out one.

The base idea
    FlowChunk-S turns an observation s and a noise vector w (8 actions x 4 dims = 32 numbers) into an action
    chunk with 10 deterministic Euler steps. Normally w ~ N(0, I) is drawn fresh for every chunk. A *ticket*
    is one fixed w used for every chunk of every episode, which turns the policy into a deterministic
    controller a = pi(s, w) with 32 knobs. Search then maximizes a success count with common random numbers:

        J(w) = (1/n) sum_{i=1..n} success(s_i, pi(., w)),      w* = argmax over the tickets we tried of J(w)

    on the n = 64 search start states. Four ways to spend the same budget of 2048 search episodes:
      random     32 tickets from N(0, I), each on all 64 search states; keep the best.
      cem        cross-entropy method from the prior N(0, I): 4 generations of 8, mean <- average of the
                 top 3, std refit and smoothed; keep the best candidate ever scored.
      cem_near0  the same CEM started with std 0.5 around w = 0. Added after w = 0 scored 45/64 on search
                 and after seeing quick-mode eval numbers (the first 64 eval seeds), so it is flagged, and
                 the report also names the best run among the other three methods ("peek-free").
      racing     random search with sequential halving: 512 tickets on 1 state, keep the best half, double
                 the states (1, 2, 4, ..., 64), so 8 finalists end up scored on all 64.
    J(w*) is a maximum of noisy estimates, so it is biased upward ("winner's curse", regression to the mean).
    That is why the ticket is chosen on search and reported on the held-out eval set (256 states), and why we
    also print the *optimistic* number you would get by picking the best ticket on eval itself.

    Support limit: a ticket only selects chunks the frozen network can already produce, pi(s, w) for some w;
    it cannot teach the policy anything new (the same holds for noise-space RL in Ch 7 and best-of-N in
    Ch 12). The Gaussian base's pass@k estimates that ceiling, and pass@8 undercounts it per state: states
    picked because all 8 Gaussian tries failed are partly states where the base was unlucky, the same
    regression to the mean as above. The script reruns those states with more Gaussian tries (pass@64).

Baselines (honesty rule 4): the unsteered base (Gaussian noise, eval salt 0), a random ticket with no
selection (the first ticket of each random-search pool), and the zero ticket w = 0, fixed in advance.

Papers mirrored (ids from data/papers.csv)
    C236 Golden Ticket (2026-03): one fixed noise vector for a frozen diffusion/flow policy, found by random
         search; real Franka pick 80 -> 98% on 50 held-out episodes after 150 search episodes.
    C120 DSRL (2025-06): the noise as an action space, learned instead of searched (our Ch 7).
    C055 Diffusion-ES (2024-02): evolutionary search through a frozen diffusion model at test time.
    C208 FPO++ (2026-02): zero-noise sampling (our zero ticket) alone lifts a flow policy on Can 10% -> 71%.
    INSPO and SDN (named in the curriculum) are not in data/papers.csv yet, so they are not cited here.

Run
    uv run python algos/03_golden_ticket.py --quick     # smoke test, < 1 min, writes to runs/ch03_quick
    uv run python algos/03_golden_ticket.py             # 4 methods x 3 seeds + baselines + media, ~25 min
    uv run python algos/03_golden_ticket.py --replot results/ch03/golden_ticket_so101_<stamp>.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np

from lastmile.common.cli import parse
from lastmile.common.eval import INIT_SETS, EvalResult, bootstrap_diff, format_rate, mcnemar_exact, wilson
from lastmile.common.ledger import DEFAULT_RESULTS_ROOT, Ledger
from lastmile.common.policy_flow import FlowChunkPolicy
from lastmile.common.rollout import evaluate, rollout_seeds

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "base_v1_so101.pt")
MEDIA = ROOT / "media" / "ch03"
METHODS = ("random", "cem", "cem_near0", "racing")
PEEKED = ("cem_near0",)  # methods designed after looking at results (see the docstring)
DIM = 32  # 8 actions x 4 dims


@dataclass
class Config:
    robot: str = "so101"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])  # search seeds (which tickets get drawn)
    search_n: int = 64  # search start states per full score (a power of two, for racing)
    n_random: int = 32  # random search: tickets, each on all search_n states
    cem_pop: int = 8
    cem_gens: int = 4
    cem_elites: int = 3
    cem_smooth: float = 0.5  # new var = (1 - a) old var + a elite var
    near0_sigma: float = 0.5  # cem_near0: start std around w = 0 (cem starts at the prior's std, 1)
    race_tickets: int = 512  # racing: tickets in round 0 (on one state each)
    eval_n: int = 256
    pass_k: int = 8  # Gaussian eval salts, for pass@k and the per-state difficulty of each start state
    pass_more: int = 64  # Gaussian tries in total on the eval states that failed all pass_k tries
    ext: bool = True  # also score the chosen and the eval-picked tickets on eval_ext (1024)
    n_workers: int = 6
    media: bool = True
    replot: str = ""  # redraw the plots from a golden_ticket results JSON and exit
    results_root: str = ""  # default: results/ (full) or runs/ch03_quick/ (quick)
    quick: bool = False
    QUICK: ClassVar[dict] = {"seeds": [0], "search_n": 16, "n_random": 4, "cem_pop": 4, "cem_gens": 2,
                             "cem_elites": 2, "race_tickets": 16, "eval_n": 64, "pass_k": 2, "pass_more": 8,
                             "ext": False}


class TicketPolicy(FlowChunkPolicy):
    """Many tickets in one batch: each episode's policy seed is the index of its ticket.

    A fixed ticket uses no random numbers, so the policy-seed slot is free. Using it for the ticket id lets
    one rollout call score hundreds of (ticket, start state) pairs in full batches, which is much faster than
    one call per ticket. Results match FlowChunkPolicy(noise="fixed", ticket=w) exactly (asserted below).
    """

    def __init__(self, model, tickets: np.ndarray):
        tickets = np.asarray(tickets, dtype=np.float32)
        super().__init__(model, noise="fixed", ticket=tickets[0])
        self.tickets = tickets

    def reset(self, seeds) -> None:
        super().reset(seeds)
        self.ticket = self.tickets[np.asarray(seeds, dtype=np.int64)]  # [B, 32], one row per episode


def score(tickets: np.ndarray, init_set: str, idx, cfg: Config, L: Ledger, category: str) -> np.ndarray:
    """Success of every ticket on start states ``idx`` of ``init_set``: bool [n_tickets, len(idx)]."""
    seeds = [INIT_SETS[init_set][i] for i in idx]
    env_seeds = [s for _ in range(len(tickets)) for s in seeds]
    ticket_ids = [t for t in range(len(tickets)) for _ in seeds]
    res = rollout_seeds(partial(TicketPolicy.load, CKPT, tickets=tickets), env_seeds, ticket_ids,
                        robot=cfg.robot, n_workers=cfg.n_workers, ledger=L, category=category,
                        init_set=init_set)
    return res.successes.reshape(len(tickets), len(seeds))


def fixed(w: np.ndarray):
    return partial(FlowChunkPolicy.load, CKPT, noise="fixed", ticket=np.asarray(w, dtype=np.float32))


GAUSS = partial(FlowChunkPolicy.load, CKPT)  # the unsteered base: fresh N(0, I) noise for every chunk


# ---------------------------------------------------------------------------- the searches
# Each returns the chosen ticket, its search score, and a best-so-far curve [(search episodes, best k)].


def random_search(rng, cfg, L):
    pool = rng.standard_normal((cfg.n_random, DIM)).astype(np.float32)
    k = score(pool, "search", range(cfg.search_n), cfg, L, "search")  # [tickets, states], kept for the replay
    best = int(np.argmax(k.sum(1)))  # ties: the earliest ticket
    curve = [((j + 1) * cfg.search_n, int(k[: j + 1].sum(1).max())) for j in range(len(k))]
    return {"ticket": pool[best], "search_k": int(k[best].sum()), "curve": curve, "pool": pool,
            "pool_k": k.sum(1), "pool_search": k}


def cem(rng, cfg, L, sigma0: float = 1.0):
    mu, sd = np.zeros(DIM), np.full(DIM, sigma0)  # sigma0 = 1 is the prior N(0, I) the base was trained on
    seen, seen_k = [], []
    for g in range(cfg.cem_gens):
        X = mu + sd * rng.standard_normal((cfg.cem_pop, DIM))
        if g > 0:
            X[0] = mu  # from generation 1 on, the current mean is itself a candidate
        k = score(X.astype(np.float32), "search", range(cfg.search_n), cfg, L, "search").sum(1)
        elites = X[np.argsort(-k, kind="stable")[: cfg.cem_elites]]  # ties: earlier candidate
        mu = elites.mean(0)
        sd = np.sqrt((1 - cfg.cem_smooth) * sd**2 + cfg.cem_smooth * elites.var(0))
        seen += list(X)
        seen_k += list(k)
        print(f"    cem gen {g}: best {k.max()}/{cfg.search_n}, mean {k.mean():.1f}, "
              f"|mu| {np.linalg.norm(mu):.2f}, mean std {sd.mean():.2f}")
    seen_k = np.array(seen_k)
    curve = [((j + 1) * cfg.search_n, int(seen_k[: j + 1].max())) for j in range(len(seen_k))]
    best = int(np.argmax(seen_k))  # the best candidate ever scored (means included), not the last mean
    return {"ticket": seen[best].astype(np.float32), "search_k": int(seen_k[best]), "curve": curve}


def racing(rng, cfg, L):
    """Sequential halving: survivors are ranked by total successes on the states seen so far."""
    tickets = rng.standard_normal((cfg.race_tickets, DIM)).astype(np.float32)
    order = rng.permutation(cfg.search_n)  # each seed screens on different early states
    alive = np.arange(len(tickets))
    wins = np.zeros(len(tickets), dtype=int)
    used, rounds, spent = 0, [], 0
    while True:
        n_states = max(1, 2 * used)  # 1, 2, 4, ..., search_n
        new = score(tickets[alive], "search", order[used:n_states], cfg, L, "search")
        wins[alive] += new.sum(1)
        spent += new.size
        used = n_states
        rounds.append({"tickets": len(alive), "states": used, "leader_k": int(wins[alive].max())})
        if used >= cfg.search_n:
            break
        keep = np.argsort(-wins[alive], kind="stable")[: max(1, len(alive) // 2)]  # ties: lower index
        alive = alive[np.sort(keep)]
    best = alive[int(np.argmax(wins[alive]))]
    print("    racing rounds: " + ", ".join(f"{r['tickets']}@{r['states']}" for r in rounds))
    return {"ticket": tickets[best], "search_k": int(wins[best]), "curve": [(spent, int(wins[best]))],
            "rounds": rounds}


SEARCH = {"random": random_search, "cem": cem, "racing": racing,
          "cem_near0": lambda rng, cfg, L: cem(rng, cfg, L, sigma0=cfg.near0_sigma)}


# ---------------------------------------------------------------------------- statistics helpers


def rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "sr": k / n, "ci": list(wilson(int(k), int(n)))}


def bits(a: np.ndarray) -> list[str]:
    """Per-episode successes as '0110...' strings (one per ticket): compact in JSON, easy to reload."""
    return ["".join("1" if x else "0" for x in row) for row in np.atleast_2d(a)]


def paired(a: EvalResult, b: EvalResult) -> dict:
    """McNemar's exact test and a paired bootstrap interval for b - a on the same episodes."""
    return {"p_mcnemar": mcnemar_exact(a.successes, b.successes),
            "only_a": int(np.sum(a.successes & ~b.successes)),
            "only_b": int(np.sum(~a.successes & b.successes)),
            "diff_ci": list(bootstrap_diff(a.successes, b.successes))}


BUDGETS = ((10, 5), (10, 16), (32, 16), (32, 64))  # (tickets, search states); 10 x 5 is the SO-101 step


def small_budgets(search_mat: np.ndarray, eval_mat: np.ndarray, reps: int = 4000) -> list[dict]:
    """Replay smaller searches offline from tickets that were scored on every search and eval state.

    Draw ``n_t`` tickets and ``n_s`` search states at random, keep the ticket with the most successes on those
    states, and read off its search estimate and its (already measured) held-out score. No new rollouts: this
    shows how much a small search overstates its pick, which one real 50-episode session cannot show.
    """
    rng = np.random.default_rng(0)
    out = []
    for n_t, n_s in BUDGETS:
        n_t, n_s = min(n_t, len(search_mat)), min(n_s, search_mat.shape[1])
        est, held = np.empty(reps), np.empty(reps)
        for i in range(reps):
            t = rng.choice(len(search_mat), n_t, replace=False)
            k = search_mat[np.ix_(t, rng.choice(search_mat.shape[1], n_s, replace=False))].sum(1)
            est[i], held[i] = k.max() / n_s, eval_mat[t[np.argmax(k)]].mean()  # ties: first drawn
        out.append({"tickets": n_t, "states": n_s, "episodes": n_t * n_s, "search_est": float(est.mean()),
                    "eval": float(held.mean()), "eval_q10_q90": np.quantile(held, [0.1, 0.9]).tolist()})
    return out


# ---------------------------------------------------------------------------- the experiment, step by step


def baselines(cfg, ev, new_ledger) -> dict:
    """The unsteered base (8 salts on eval, one on search) and the zero ticket, all before any search."""
    salts = [ev(GAUSS, "eval", n=cfg.eval_n, salt=s) for s in range(cfg.pass_k)]  # salt 0 = the paired base
    B = {"salts": salts, "base": salts[0], "base_search": ev(GAUSS, "search", n=cfg.search_n)}
    with new_ledger(method="zero_ticket") as L:
        B["zero_search"] = ev(fixed(np.zeros(DIM)), "search", n=cfg.search_n, ledger=L, category="search")
        B["zero"] = ev(fixed(np.zeros(DIM)), "eval", n=cfg.eval_n, ledger=L, category="eval")
        L.robot_steps["eval"] += B["base"].env_steps
        L.set_base(id="base_v1_gaussian", successes=B["base"])
        L.set_final(successes=B["zero"])
        L.extra.update(search=rate(B["zero_search"].k, cfg.search_n), vs_base=paired(B["base"], B["zero"]))
    print(f"Gaussian base: eval {B['base'].summary} | search {B['base_search'].summary}\n"
          f"zero ticket: eval {B['zero'].summary} | search {B['zero_search'].summary}")
    return B


def run_searches(cfg, ev, new_ledger, B) -> tuple[list, list]:
    """Every method x seed: search, then one held-out eval of its pick. One results JSON per run."""
    base, runs, pools = B["base"], [], []
    for method in METHODS:
        for seed in cfg.seeds:
            rng = np.random.default_rng([seed, METHODS.index(method)])  # different tickets per method
            with new_ledger(method=f"{method}_s{seed}") as L:
                print(f"  {method} seed {seed}")
                r = SEARCH[method](rng, cfg, L)
                final = ev(fixed(r["ticket"]), "eval", n=cfg.eval_n, ledger=L, category="eval")
                L.robot_steps["eval"] += base.env_steps  # the shared base evaluation
                L.set_base(id="base_v1_gaussian", successes=base)
                L.set_final(successes=final)
                if method == "random":  # extra held-out rollouts of the whole pool, for the scatter plot
                    pool_eval = score(r["pool"], "eval", range(cfg.eval_n), cfg, L, "eval")
                    assert np.array_equal(pool_eval[int(np.argmax(r["pool_k"]))], final.successes), "parity"
                    pools.append((seed, r, pool_eval))
                    L.extra.update(pool=r["pool"].tolist(), pool_search=bits(r["pool_search"]),
                                   pool_eval=bits(pool_eval))
                r.update(method=method, seed=seed, final=final, search_minutes=L.robot_minutes["search"],
                         search_steps=L.robot_steps["search"], vs_base=paired(base, final),
                         vs_zero=paired(B["zero"], final))
                skip = ("final", "pool", "pool_k", "pool_search")
                L.extra.update({k: v for k, v in r.items() if k not in skip})
                L.extra.update(ticket=r["ticket"].tolist(), eval_successes=bits(final.successes)[0])
                if method in PEEKED:
                    L.extra["note"] = "added after seeing w = 0's search score and quick-mode eval numbers"
            print(f"  {method} seed {seed}: search {format_rate(r['search_k'], cfg.search_n)} -> "
                  f"eval {final.summary} (McNemar vs base p = {r['vs_base']['p_mcnemar']:.3g}) | "
                  f"{r['search_minutes']:.0f} search robot-min")
            runs.append(r)
    return runs, pools


def random_ticket_baselines(cfg, ev, new_ledger, B, pools) -> list[dict]:
    """Honesty rule 4: the first ticket of each pool, used with no selection at all."""
    out = []
    for seed, r, _ in pools:
        with new_ledger(method=f"random_ticket_s{seed}") as L:
            first = ev(fixed(r["pool"][0]), "eval", n=cfg.eval_n, ledger=L, category="eval")
            L.robot_steps["eval"] += B["base"].env_steps
            L.set_base(id="base_v1_gaussian", successes=B["base"])
            L.set_final(successes=first)
            L.extra.update(ticket=r["pool"][0].tolist(), note="first ticket of the random pool, no search")
        out.append(rate(first.k, first.n))
    return out


def build_scatter(runs, pools, B) -> list[dict]:
    """Every ticket with both a full search score and a held-out score, tagged for the plot."""
    def point(id_, kind, w, search_k, eval_s, tags=()):
        return {"id": id_, "kind": kind, "tags": list(tags), "search_k": int(search_k),
                "eval_k": int(np.sum(eval_s)), "norm": float(np.linalg.norm(w)), "w": w, "eval_s": eval_s}

    S = []
    for seed, r, pool_eval in pools:
        chosen = int(np.argmax(r["pool_k"]))
        S += [point(f"random_s{seed}_t{j}", "random pool", w, r["pool_k"][j], pool_eval[j],
                    ["chosen"] if j == chosen else []) for j, w in enumerate(r["pool"])]
    S += [point(f"{r['method']}_s{r['seed']}", r["method"], r["ticket"], r["search_k"],
                r["final"].successes, ["chosen"]) for r in runs if r["method"] != "random"]
    S.append(point("zero", "zero", np.zeros(DIM), B["zero_search"].k, B["zero"].successes, ["zero"]))
    min((t for t in S if t["kind"] == "random pool"), key=lambda t: t["eval_k"])["tags"].append("worst")
    max(S, key=lambda t: t["eval_k"])["tags"].append("eval-best (optimistic)")  # picked ON EVAL
    return S


def support_split(cfg, B, chosen: EvalResult, S: list[dict], L: Ledger) -> tuple[list, dict]:
    """Group eval states by Gaussian successes in pass_k tries; rerun the 0-success states with more tries.

    States selected for "0 of 8" are partly states where the base was unlucky, so pass@8 undercounts what
    the base can do there. More tries on exactly those states show how much (charged to eval: no selection).
    """
    from lastmile.common.eval import policy_seed

    c = np.sum([s.successes for s in B["salts"]], axis=0)
    union = np.any([t["eval_s"] for t in S], axis=0)  # solved by at least one ticket we scored (an oracle)
    support = [{"c": v, "n": int(np.sum(c == v)), "base_k": int(np.sum(B["base"].successes[c == v])),
                "zero_k": int(np.sum(B["zero"].successes[c == v])),
                "chosen_k": int(np.sum(chosen.successes[c == v])),
                "any_ticket_k": int(np.sum(union[c == v]))} for v in range(cfg.pass_k + 1)]
    hard, tries = np.flatnonzero(c == 0), range(cfg.pass_k, cfg.pass_more)
    more = np.zeros((len(hard), len(tries)), bool)
    if len(hard) and len(tries):
        more = rollout_seeds(GAUSS, [INIT_SETS["eval"][i] for i in hard for _ in tries],
                             [policy_seed("eval", int(i), s) for i in hard for s in tries], robot=cfg.robot,
                             n_workers=cfg.n_workers, ledger=L, category="eval",
                             init_set="eval").successes.reshape(len(hard), len(tries))
    extra_k = more.sum(1)
    hard_info = {"idx": hard.tolist(), "extra_tries": len(tries), "extra_k": extra_k.tolist(),
                 "pass_more": cfg.pass_more, "solved_with_more": int(np.sum(extra_k > 0)),
                 "pass_at_more": int(np.sum(c > 0) + np.sum(extra_k > 0)),
                 "chosen_solves": hard[chosen.successes[hard]].tolist(),
                 "zero_solves": hard[B["zero"].successes[hard]].tolist(),
                 "still_unsolved": [{"idx": int(i), "seed": INIT_SETS["eval"][i],
                                     "tickets": [t["id"] for t in S if t["eval_s"][i]]}
                                    for i in hard[extra_k == 0]]}
    return support, hard_info


def run_summary(r: dict, cfg: Config) -> dict:
    return {"run": f"{r['method']}_s{r['seed']}", "method": r["method"], "seed": r["seed"],
            "search": rate(r["search_k"], cfg.search_n), "eval": rate(r["final"].k, r["final"].n),
            "curve": r["curve"], "search_minutes": r["search_minutes"], "vs_base": r["vs_base"],
            "vs_zero": r["vs_zero"]}


def headline(cfg, ev, new_ledger, B, runs, pools, random_ticket) -> tuple[dict, Path]:
    """The best of all runs by search score, charged for every search, plus all the analysis."""
    base, zero, salts = B["base"], B["zero"], B["salts"]
    best = max(runs, key=lambda r: r["search_k"])  # ties: the first in (method, seed) order
    clean = max((r for r in runs if r["method"] not in PEEKED), key=lambda r: r["search_k"])
    with new_ledger(method="golden_ticket") as L:
        L.robot_steps["search"] += sum(r["search_steps"] for r in runs) + B["base_search"].env_steps
        L.robot_steps["eval"] += sum(s.env_steps for s in salts) + best["final"].env_steps
        L.set_base(id="base_v1_gaussian", successes=base)
        L.set_final(successes=best["final"])
        if best["method"] in PEEKED:
            L.extra["note"] = (
                f"The pick comes from {best['method']}, a method added after seeing w = 0's search "
                f"score and quick-mode eval numbers on the first 64 eval seeds (part of this eval set). "
                f"Best run by search among the methods that did not depend on that peek: "
                f"{clean['method']}_s{clean['seed']}, eval {clean['final'].k}/{cfg.eval_n} "
                "(analysis.peek_free).")
        L.extra["eval_successes"] = {"gaussian_salts": bits([s.successes for s in salts]),
                                     "zero": bits(zero.successes)[0],
                                     "chosen": bits(best["final"].successes)[0]}
        S = build_scatter(runs, pools, B)
        pool_pts = [t for t in S if t["kind"] == "random pool"]
        worst = next(t for t in S if "worst" in t["tags"])
        opt = next(t for t in S if "eval-best (optimistic)" in t["tags"])
        xs = np.array([t["search_k"] / cfg.search_n for t in pool_pts])
        ys = np.array([t["eval_k"] / cfg.eval_n for t in pool_pts])
        slope, intercept = np.polyfit(xs, ys, 1)
        support, hard = support_split(cfg, B, best["final"], S, L)
        ext = {}
        if cfg.ext:  # an independent 1024-state check of the two picks and of the base
            for name, w in [("gaussian", None), ("chosen", best["ticket"]), ("optimistic", opt["w"])]:
                res = ev(GAUSS if w is None else fixed(w), "eval_ext", ledger=L, category="eval")
                ext[name] = rate(res.k, res.n)
                print(f"  eval_ext {name}: {res.summary}")
        # GIF (illustration only): the first eval state where the base fails and the chosen ticket succeeds.
        fixed_states = np.flatnonzero(~base.successes & best["final"].successes)
        r0 = pools[0][1]
        gif = {"seed_index": int(fixed_states[0]) if len(fixed_states) else 0,
               "clips": [["Gaussian base", None], ["zero ticket", [0.0] * DIM],
                         [f"chosen ({best['method']}_s{best['seed']})", best["ticket"].tolist()],
                         ["worst ticket", worst["w"].tolist()], ["random ticket A", r0["pool"][0].tolist()],
                         ["random ticket B", r0["pool"][1].tolist()]]}
        A = {"search_n": cfg.search_n, "eval_n": cfg.eval_n, "pass_k": cfg.pass_k,
             "gaussian": {"search": rate(B["base_search"].k, cfg.search_n), "eval": rate(base.k, base.n),
                          "salts_k": [s.k for s in salts],
                          "pass_at_k": (np.cumsum([s.successes for s in salts], axis=0) > 0).sum(1).tolist()},
             "zero": {"search": rate(B["zero_search"].k, cfg.search_n), "eval": rate(zero.k, zero.n)},
             "random_ticket": random_ticket,
             "runs": [run_summary(r, cfg) for r in runs],
             "headline": dict(run_summary(best, cfg), ticket=best["ticket"].tolist()),
             "peek_free": dict(run_summary(clean, cfg), ticket=clean["ticket"].tolist()),
             "optimistic": {"id": opt["id"], "search": rate(opt["search_k"], cfg.search_n),
                            "eval": rate(opt["eval_k"], cfg.eval_n)},
             "regression": {"slope": float(slope), "intercept": float(intercept),
                            "r": float(np.corrcoef(xs, ys)[0, 1]), "n_tickets": len(pool_pts),
                            "mean_pool_eval": float(ys.mean()), "mean_pool_search": float(xs.mean()),
                            "r_norm_eval": float(np.corrcoef([t["norm"] for t in pool_pts], ys)[0, 1])},
             "worst": {"id": worst["id"], "search": rate(worst["search_k"], cfg.search_n),
                       "eval": rate(worst["eval_k"], cfg.eval_n)},
             "support": support, "hard_states": hard, "eval_ext": ext, "gif": gif,
             "any_ticket_eval": rate(int(np.any([t["eval_s"] for t in S], axis=0).sum()), cfg.eval_n),
             # replayed from the three random pools, so this describes N(0, I) tickets only
             "small_budgets": small_budgets(np.concatenate([r["pool_search"] for _, r, _ in pools]),
                                            np.concatenate([pe for _, _, pe in pools])),
             "scatter": [{k: v for k, v in t.items() if k not in ("w", "eval_s")} for t in S]}
        L.extra["analysis"] = A
    return A, L.path


def report(A: dict, runs: list, cfg: Config, wall: float, path: Path) -> None:
    n_e, n_s = cfg.eval_n, cfg.search_n
    print(f"\nsummary (held-out eval, n={n_e}; Gaussian base, salt 0: {fmt(A['gaussian']['eval'])})")
    for m in METHODS:
        rs = [r for r in runs if r["method"] == m]
        print(f"  {m:9s} per seed [{', '.join(str(r['final'].k) for r in rs)}] of {n_e}, pooled "
              f"{format_rate(sum(r['final'].k for r in rs), n_e * len(rs))}; search pooled "
              f"{format_rate(sum(r['search_k'] for r in rs), n_s * len(rs))}")
    print(f"  zero ticket {fmt(A['zero']['eval'])} | random ticket (no selection) "
          f"{[t['k'] for t in A['random_ticket']]} of {n_e}, pool mean "
          f"{A['regression']['mean_pool_eval']:.1%}")
    for name in ("headline", "peek_free"):
        h = A[name]
        print(f"  {name} ({h['run']}): search {fmt(h['search'])} -> eval {fmt(h['eval'])}; "
              f"McNemar vs Gaussian p = {h['vs_base']['p_mcnemar']:.3g}, "
              f"vs zero p = {h['vs_zero']['p_mcnemar']:.3g}")
    o = A["optimistic"]
    print(f"  optimistic (best of {len(A['scatter'])} tickets picked ON EVAL, {o['id']}): "
          f"{fmt(o['eval'])}; slope eval~search {A['regression']['slope']:.2f}")
    H = A["hard_states"]
    print(f"  pass@{cfg.pass_k} {A['gaussian']['pass_at_k'][-1]}/{n_e}; the {len(H['idx'])} states at "
          f"0/{cfg.pass_k} succeed {H['extra_k']} times in {H['extra_tries']} more Gaussian tries -> "
          f"pass@{cfg.pass_more} {format_rate(H['pass_at_more'], n_e)}; still unsolved (seed: tickets that "
          f"solve it): {[(u['seed'], u['tickets']) for u in H['still_unsolved']]}")
    for b in A["small_budgets"]:
        print(f"  replayed N(0, I) random search, {b['tickets']} tickets x {b['states']} states: pick's "
              f"search score {b['search_est']:.1%} -> eval {b['eval']:.1%} (10-90%: "
              f"{b['eval_q10_q90'][0]:.1%}-{b['eval_q10_q90'][1]:.1%})")
    print(f"  search robot-minutes, all {len(runs)} runs: {sum(r['search_minutes'] for r in runs):.0f} | "
          f"wall {wall / 60:.1f} min | headline JSON {path}")


def fmt(r: dict) -> str:
    return format_rate(r["k"], r["n"])


# ---------------------------------------------------------------------------- media


def make_plots(R: dict, out: Path = MEDIA) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from lastmile.common import plotting

    plotting.setup()
    P, n_s, n_e = plotting.PALETTE, R["search_n"], R["eval_n"]
    gs, ge = R["gaussian"]["search"], R["gaussian"]["eval"]
    colors = dict(zip(METHODS, P))

    # 1. search curves (best-so-far on search) + held-out eval of each chosen ticket
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1.4]})
    for m in METHODS:
        runs = [r for r in R["runs"] if r["method"] == m]
        if m == "racing":  # no best-so-far before the last round: plot each finalist only
            for r in runs:
                x, k = r["curve"][-1]
                ax.plot(x, k / n_s, "s", color=colors[m], ms=7, alpha=0.8)
            continue
        x = [c[0] for c in runs[0]["curve"]]
        ks = np.array([[c[1] for c in r["curve"]] for r in runs]) / n_s
        ax.plot(x, ks.T, color=colors[m], lw=1, alpha=0.3)
        ax.plot(x, ks.mean(0), color=colors[m], label=f"{m} (mean of {len(runs)} seeds)")
    ax.plot([], [], "s", color=colors["racing"], label="racing finalist (per seed)")
    refs = [(gs, "0.3", ":", "Gaussian base"), (R["zero"]["search"], P[4], "--", "zero ticket")]
    for r, col, ls, name in refs:
        ax.axhline(r["sr"], color=col, ls=ls, lw=1.2, label=f"{name} on search (band: 95% Wilson)")
        ax.axhspan(*r["ci"], color=col, alpha=0.08, lw=0)
    ax.set(xlabel="search episodes (per run)", ylim=(0, 1.02),
           ylabel=f"best ticket so far, success on search (n={n_s})",
           title="Search-time estimates (a max over candidates: optimistic, no interval)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="lower right")
    labels = []
    for j, m in enumerate(METHODS):
        for i, r in enumerate([r for r in R["runs"] if r["method"] == m]):
            e = r["eval"]
            plotting.heldout_point(ax2, j + 0.15 * (i - 1), e["k"], e["n"], color=colors[m], label=None)
        labels.append(m)
    for j, (name, rs, c) in enumerate([("zero", [R["zero"]["eval"]], P[4]), ("Gaussian", [ge], "0.3"),
                                       ("random ticket", R["random_ticket"], P[5])], start=len(METHODS)):
        for i, t in enumerate(rs):
            dx = 0.15 * (i - (len(rs) - 1) / 2)
            plotting.heldout_point(ax2, j + dx, t["k"], t["n"], color=c, label=None)
        labels.append(name)
    ax2.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, 1.02),
            title=f"Held-out eval (n={n_e}), 95% Wilson")
    ax2.tick_params(axis="x", rotation=30)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    plotting.save_fig(fig, out / "search_curves.png")

    # 2. search score vs held-out score for every ticket scored on both
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    S = R["scatter"]
    pool = [t for t in S if t["kind"] == "random pool"]
    xs, ys = np.array([t["search_k"] / n_s for t in pool]), np.array([t["eval_k"] / n_e for t in pool])
    jit = np.random.default_rng(0).uniform(-0.004, 0.004, len(xs))  # search scores are on a 1/64 grid
    ax.scatter(xs + jit, ys, s=18, color=P[0], alpha=0.6, label="random tickets (pools; bars not drawn)")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="search = eval")
    fit = R["regression"]
    xx = np.linspace(xs.min(), xs.max(), 10)
    ax.plot(xx, fit["intercept"] + fit["slope"] * xx, color=P[0], lw=1.2,
            label=f"least squares, slope {fit['slope']:.2f} (r = {fit['r']:.2f})")
    marks = [("chosen", "*", P[1], 12), ("zero", "P", P[4], 9), ("worst", "v", P[5], 7),
             ("eval-best (optimistic)", "D", P[3], 7)]
    for kind, mk, c, size in marks:
        sel = [t for t in S if kind in t["tags"]]
        px, ex = plotting.wilson_err([t["search_k"] for t in sel], n_s)
        py, ey = plotting.wilson_err([t["eval_k"] for t in sel], n_e)
        ax.errorbar(px, py, xerr=ex, yerr=ey, fmt=mk, ms=size, color=c, mec="k", mew=0.5, elinewidth=0.6,
                    alpha=0.9, zorder=5, label=f"{kind} ticket" + ("s" * (len(sel) > 1)) + " (95% Wilson)")
    px, ex = plotting.wilson_err([gs["k"]], n_s)
    py, ey = plotting.wilson_err([ge["k"]], n_e)
    ax.errorbar(px, py, xerr=ex, yerr=ey, fmt="o", color="k", ms=7, capsize=3, zorder=6,
                label="Gaussian base (95% Wilson)")
    ax.set(xlabel=f"success on search (n={n_s}) = what the search sees",
           ylabel=f"success on held-out eval (n={n_e})", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
           title="Tickets that look best on search regress on eval")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=7.5, loc="upper left")
    plotting.save_fig(fig, out / "search_vs_heldout.png")

    # 3. support: success by how often the Gaussian base solves each eval state in pass_k tries
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    B = R["support"]
    c = np.array([b["c"] for b in B])
    series = [("Gaussian base, salt 0", "base_k", "0.3"), ("zero ticket", "zero_k", P[4]),
              ("chosen ticket", "chosen_k", P[1]),
              ("any scored ticket\n(oracle, picked on eval)", "any_ticket_k", P[2])]
    for j, (name, key, col) in enumerate(series):
        p, err = plotting.wilson_err([b[key] for b in B], [b["n"] for b in B])
        ax.errorbar(c + 0.16 * (j - 1.5), p, yerr=err, fmt="o", color=col, capsize=3, label=name)
    for b in B:
        ax.text(b["c"], -0.1, f"n={b['n']}", ha="center", fontsize=7.5)
    ax.set(xlabel=f"Gaussian successes out of {R['pass_k']} tries from that start state (eval)",
           ylabel="success on those states (95% Wilson)", ylim=(-0.14, 1.04), xticks=c,
           title="Success by how often the Gaussian base solves each start state")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # outside: the plot is full
    plotting.save_fig(fig, out / "support.png")


def tickets_gif(R: dict, cfg: Config, out: Path) -> None:
    """Replay six policies on one eval start state and tile them 3 x 2, border green on success.

    Each episode is one already counted in an evaluation above (same env seed, same policy seed), rerun in
    process with its actions recorded and then re-rendered, which reproduces it exactly.
    """
    from lastmile.common.eval import policy_seed
    from lastmile.common.plotting import replay_frames, save_gif, side_by_side, tile
    from lastmile.envs.cupdrop import CupDropEnv

    env = CupDropEnv(robot=cfg.robot, render_size=(160, 160))
    seed_index = R["gif"]["seed_index"]
    seed, clips = INIT_SETS["eval"][seed_index], []
    for label, w in R["gif"]["clips"]:
        make = GAUSS if w is None else fixed(np.array(w))  # None: the Gaussian base
        res = rollout_seeds(make, [seed], [policy_seed("eval", seed_index)], robot=cfg.robot, n_workers=1,
                            record_trajectories=True)
        traj = res.trajectories[0]
        frames = [np.array(f) for f in replay_frames(env, traj, label=label, every=5)]  # writable copies
        color = np.array([40, 170, 60] if traj["success"] else [200, 50, 50], np.uint8)
        for f in frames:
            f[:4], f[-4:], f[:, :4], f[:, -4:] = color, color, color, color
        clips.append(frames + [frames[-1]] * 4)
    path = save_gif(tile(clips, cols=3), out / "tickets_grid.gif", fps=4, max_size=488)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, eval seed {seed})")
    path = save_gif(side_by_side(clips[0], clips[2]), out / "before_after.gif", fps=4, max_size=324)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------- main


def main(cfg: Config) -> None:
    if cfg.replot:
        R = json.loads(Path(cfg.replot).read_text())["extra"]["analysis"]
        make_plots(R)
        tickets_gif(R, cfg, MEDIA)
        return
    t0 = time.time()
    root = cfg.results_root or str(ROOT / "runs" / "ch03_quick" if cfg.quick else DEFAULT_RESULTS_ROOT)
    ev = partial(evaluate, robot=cfg.robot, n_workers=cfg.n_workers)
    new_ledger = partial(Ledger, chapter="ch03", robot=cfg.robot, config=cfg, results_root=root)
    B = baselines(cfg, ev, new_ledger)
    runs, pools = run_searches(cfg, ev, new_ledger, B)
    random_ticket = random_ticket_baselines(cfg, ev, new_ledger, B, pools)
    analysis, path = headline(cfg, ev, new_ledger, B, runs, pools, random_ticket)
    report(analysis, runs, cfg, time.time() - t0, path)
    if cfg.media:  # quick runs draw into runs/ch03_quick/media so the committed media stay untouched
        out = ROOT / "runs" / "ch03_quick" / "media" if cfg.quick else MEDIA
        make_plots(analysis, out)
        tickets_gif(analysis, cfg, out)


if __name__ == "__main__":
    main(parse(Config))
