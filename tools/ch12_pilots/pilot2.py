"""Chapter 12, pilot 2 (run once before algos/12_best_of_n.py had its final form; kept for the record).

4096 labeled base episodes (seeds 70 000+), classifier and MC-value verifiers trained on 256, 1024 and
4096 of them, and best-of-N for N = 2..32 on 256 dev seeds (79 000-79 255), never search or eval. This is
where train_episodes = 4096 and the N grid came from. Its robot steps are in prior_costs.json.
Run: uv run python tools/ch12_pilots/pilot2.py   (writes to runs/ch12_pilots/)
"""

import importlib
import json
import sys
from functools import partial

from lastmile.common.eval import policy_seed
from lastmile.common.ledger import REPO_ROOT, Ledger
from lastmile.common.rollout import rollout_seeds

sys.path.insert(0, str(REPO_ROOT / "algos"))  # spawned workers inherit sys.path and re-import the module
B = importlib.import_module("12_best_of_n")
W = REPO_ROOT / "runs/ch12_pilots"


def dev(make, ledger, n=256):
    seeds = [79_000 + i for i in range(n)]
    return rollout_seeds(make, seeds, [policy_seed("ch12_dev", i) for i in range(n)], n_workers=6,
                         ledger=ledger, category="search")


if __name__ == "__main__":
    W.mkdir(parents=True, exist_ok=True)
    cfg, L = B.Config(), Ledger("pilot", "pilot")
    data = B.collect(cfg, 0, 4096, L)
    print("base", dev(B.base_policy, L).summary)
    for n in [8, 32]:
        print("random", n, dev(partial(B.bon_policy, n, "random"), L).summary)
    for ne in [256, 1024, 4096]:
        for kind in ["clf", "q"]:
            info = B.train_verifier(cfg, data, kind, ne, 0, W / f"p2_{kind}_{ne}.pt")
            row = [dev(partial(B.bon_policy, n, "verifier", str(W / f"p2_{kind}_{ne}.pt")), L).k
                   for n in [2, 4, 8, 16, 32]]
            print(ne, kind, f"auc {info['val_auc']:.3f}", "k/256 at N = 2..32:", row)
    print("robot steps", L.robot_steps)
    (W / "pilot2_steps.json").write_text(json.dumps(L.robot_steps))
