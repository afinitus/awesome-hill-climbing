"""Chapter 12, pilot 1 (run once before algos/12_best_of_n.py had its final form; kept for the record).

1024 labeled base episodes on the verifier seed-0 range (70 000+), three verifiers, and best-of-N trials
on 128 dev seeds (79 000-79 127), never search or eval. Its robot steps are in prior_costs.json.
Run: uv run python tools/ch12_pilots/pilot1.py   (writes to runs/ch12_pilots/)
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


def dev(make, ledger, n=128):
    seeds = [79_000 + i for i in range(n)]
    return rollout_seeds(make, seeds, [policy_seed("ch12_dev", i) for i in range(n)], n_workers=6,
                         ledger=ledger, category="search")


if __name__ == "__main__":
    W.mkdir(parents=True, exist_ok=True)
    cfg, L = B.Config(), Ledger("pilot", "pilot")
    data = B.collect(cfg, 0, 1024, L)
    for kind in ["clf", "q", "state"]:
        print(kind, B.train_verifier(cfg, data, kind, 1024, 0, W / f"p1_{kind}.pt"))
    print("base", dev(B.base_policy, L).summary)
    print("random 8", dev(partial(B.bon_policy, 8, "random"), L).summary)
    for kind in ["clf", "q", "state"]:
        for n in [4, 16]:
            print(kind, n, dev(partial(B.bon_policy, n, "verifier", str(W / f"p1_{kind}.pt")), L).summary)
    print("robot steps", L.robot_steps)
    (W / "pilot1_steps.json").write_text(json.dumps(L.robot_steps))
