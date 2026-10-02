"""Replay the saved corrected Chapter 5 seed-0 critic from its committed critic.

Diagnostic only: this neither selects a policy nor changes published results.
"""

import hashlib
import importlib
import json
import pathlib
import sys
from functools import partial

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "algos"))
from lastmile.common.rollout import evaluate
from lastmile.envs.cupdrop import obs_layout
from tools.review_harness import AuditedEnv

qc = importlib.import_module("05_rlpd_qchunk")
if __name__ == "__main__":
    checkpoint = ROOT / "checkpoints/ch05_qc_s0_review.npz"
    r = evaluate(
        partial(qc.QCPolicy, 32, "critic", str(checkpoint)),
        "eval",
        n_workers=2,
        record_trajectories=True,
        env_factory=AuditedEnv,
    )
    sl = obs_layout("so101")[1]
    success = [t for t in r.trajectories if t["success"]]
    shifted = [
        int(t["seed"])
        for t in success
        if np.linalg.norm(t["final_obs"][sl["cup_pos"]] - t["obs"][0][sl["cup_pos"]]) > 0.01
    ]
    below = [int(t["seed"]) for t in success if t["infos"]["min_cube_center_above_floor_when_inside"][-1] < 0]
    row = {
        "method": "corrected_qc_s0",
        "k": r.k,
        "n": r.n,
        "ci": r.ci,
        "env_steps": r.env_steps,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "success_with_cup_shift_gt_1cm": shifted,
        "successful_episode_with_local_cube_center_below_floor": below,
        "max_cube_penetration_m": max(
            float(t["infos"]["max_cube_contact_penetration"][-1]) for t in r.trajectories
        ),
        "episode_successes": "".join(str(int(x)) for x in r.successes),
        "scope": "Diagnostic replay of one exported corrected critic; no tuning or exhaustive exploit search.",
    }
    print(json.dumps(row))
    (ROOT / "runs/prelaunch_ch05").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs/prelaunch_ch05/qc-replay-probe.json").write_text(json.dumps(row, indent=2) + "\n")
    assert r.k == 238
