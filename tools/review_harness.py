"""Reproduce prelaunch physical probes without changing published experiment results.

Run from the repository root with ``uv run python tools/review_harness.py``.
Writes ``runs/prelaunch/harness-probes.json``. Contact penetration is sampled at
control-step boundaries, not every physics substep; these probes are not an
exhaustive proof against tunnelling. A shifted cup can still satisfy the v1 metric.
"""

import json
from functools import partial
from pathlib import Path

import numpy as np

from lastmile.common.policy import RandomPolicy
from lastmile.common.policy_flow import FlowChunkPolicy
from lastmile.common.rollout import evaluate
from lastmile.envs.cupdrop import CupDropEnv, obs_layout
from lastmile.envs.knob_controller import TUNED_KNOBS, KnobController


class AuditedEnv(CupDropEnv):
    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.max_penetration = 0.0
        self.min_floor_clearance = float("inf")
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        c = self.data.contact
        depths = -c.dist[(c.geom1 == self._cube_geom) | (c.geom2 == self._cube_geom)]
        if len(depths):
            self.max_penetration = max(self.max_penetration, float(depths.max()))
        if self._cube_cup_dist() < self.cup.inner_radius - 0.005:
            rotation = self.data.xmat[self._cup_body].reshape(3, 3)
            local = rotation.T @ (self.data.xpos[self._cube_body] - self.data.xpos[self._cup_body])
            self.min_floor_clearance = min(self.min_floor_clearance, float(local[2] - self.cup.floor))
        info["max_cube_contact_penetration"] = self.max_penetration
        info["min_cube_center_above_floor_when_inside"] = self.min_floor_clearance
        return obs, reward, term, trunc, info


def main():
    ars = json.loads(Path("results/ch01/ars_s4_so101_20260930-143150.json").read_text())
    low = dict(TUNED_KNOBS["so101"], lift_h=0.04, drop_h=0.05, speed=1.0)
    cases = [
        ("base_v1", partial(FlowChunkPolicy.load, "checkpoints/base_v1_so101.pt"), "eval"),
        ("ars_s4", partial(KnobController, ars["extra"]["best_knobs"], "so101"), "eval"),
        ("low_carry_adversary", partial(KnobController, low, "so101"), "search"),
        ("random_action_adversary", RandomPolicy, "search"),
    ]
    report = []
    for name, factory, init_set in cases:
        result = evaluate(factory, init_set, n_workers=6, record_trajectories=True, env_factory=AuditedEnv)
        sl = obs_layout("so101")[1]
        successful = [t for t in result.trajectories if t["success"]]
        shoved = [
            t["seed"]
            for t in successful
            if np.linalg.norm(t["final_obs"][sl["cup_pos"]] - t["obs"][0][sl["cup_pos"]]) > 0.01
        ]
        below = [
            t["seed"] for t in successful if t["infos"]["min_cube_center_above_floor_when_inside"][-1] < 0
        ]
        row = {
            "name": name,
            "init_set": init_set,
            "k": result.k,
            "n": result.n,
            "ci": result.ci,
            "success_with_cup_shift_gt_1cm": shoved,
            "successful_episode_with_local_cube_center_below_floor": below,
            "max_cube_penetration_m": max(
                float(t["infos"]["max_cube_contact_penetration"][-1]) for t in result.trajectories
            ),
            "episode_successes": "".join(str(int(s)) for s in result.successes),
        }
        report.append(row)
        print(json.dumps(row), flush=True)
    path = Path("runs/prelaunch/harness-probes.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
