"""Damped-least-squares inverse kinematics for a gripper that points straight down.

The env gives us an end-effector position target every control step. We want joint targets that put the
IK site there with the tool pointing down and a fixed yaw. Each step starts from the previous solution
(warm start), so a few iterations are enough: the target only moves by <= 2 cm per step.

Position matters more than orientation (a tilted gripper can still carry a cube; a gripper 2 cm off target
misses it), so each iteration uses a two-level task-priority update:

    dq1 = Jp^T (Jp Jp^T + lambda^2 I)^-1 ep                    position task, damped least squares
    N   = I - pinv(Jp) Jp                                       directions that do not move the site
    dq  = dq1 + N (Jr N)^T ((Jr N)(Jr N)^T + lambda^2 I)^-1 (er - Jr dq1)   orientation, inside N

Jp and Jr come from `mj_jacSite`. The damping keeps steps bounded near singularities and for unreachable
targets: the arm stops at the closest pose it can reach instead of flailing. Joint limits are enforced
by clipping after every iteration.

Why this matters on the SO-101: its wrist pitch is limited to +/-95 deg, so "straight down" is only
reachable up to ~9 cm above the table. Higher up, the priority scheme keeps the position exact and lets the
gripper tilt a little. With 5 joints the SO-101 has exactly two spare degrees of freedom after position,
which is what "pitch down" and "hold yaw" (via wrist roll) need. The PiPER has three and gets full orientation.

One trap remains near the edge of reach: holding the gripper vertical keeps the arm stretched out, which is
exactly the pose from which it cannot reach further, so the solve can stall centimeters short of a target
that a tilted gripper would reach. When the position error is still above `pos_tol` after the normal
iterations, a few extra position-only iterations give up the orientation and close the gap.
"""

from __future__ import annotations

import mujoco
import numpy as np


class DownIK:
    """Warm-started DLS IK on a scratch `MjData`, so solving never disturbs the simulation state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        joints: tuple[str, ...],
        target_rot: np.ndarray,
        damping: float = 0.01,
        iterations: int = 4,
        max_step: float = 0.25,
        pos_tol: float = 1e-3,
        fallback_iterations: int = 8,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(site).id
        jids = [model.joint(j).id for j in joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dofadr = np.array([model.jnt_dofadr[j] for j in jids])
        margin = 0.02  # rad; stay clear of the hard stops so tracking error does not slam into them
        self.lo = model.jnt_range[jids, 0] + margin
        self.hi = model.jnt_range[jids, 1] - margin
        self.target_rot = np.asarray(target_rot, dtype=float)
        self.damping2 = damping**2
        self.iterations = iterations
        self.max_step = max_step
        self.pos_tol = pos_tol
        self.fallback_iterations = fallback_iterations
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Site position and rotation matrix for arm joints `q` (forward kinematics)."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def orientation_error(self, rot: np.ndarray) -> np.ndarray:
        """Rotation vector (rad) that turns `rot` into the target, valid for small errors."""
        # If target = (I + [w]x) rot, then sum_i rot[:, i] x target[:, i] = 2 w.
        return 0.5 * np.cross(rot.T, self.target_rot.T).sum(axis=0)

    def solve(self, target_pos: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(q_init, dtype=float).copy(), self.lo, self.hi)
        for _ in range(self.iterations):
            q = self._iterate(q, target_pos, with_orientation=True)
        for _ in range(self.fallback_iterations):  # stalled short of the target: trade orientation for reach
            if np.linalg.norm(target_pos - self.forward(q)[0]) < self.pos_tol:
                break
            q = self._iterate(q, target_pos, with_orientation=False)
        return q

    def _iterate(self, q: np.ndarray, target_pos: np.ndarray, with_orientation: bool) -> np.ndarray:
        """One IK iteration from `q`, respecting joint limits."""
        pos, rot = self.forward(q)
        ep = target_pos - pos
        er = self.orientation_error(rot) if with_orientation else None
        mujoco.mj_comPos(self.model, self.data)  # mj_jacSite needs cdof and subtree_com
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site)
        Jp = self._jacp[:, self.dofadr]
        Jr = self._jacr[:, self.dofadr]
        # A joint that would cross its limit is parked at the limit and removed from the solve, so the
        # remaining joints take over (otherwise clipping would silently break the position task).
        free = np.ones(len(q), dtype=bool)
        for _ in range(len(q)):
            dq = self._priority_step(Jp * free, Jr * free, ep, er)
            below, above = q + dq < self.lo, q + dq > self.hi
            blocked = free & (below | above)
            if not blocked.any():
                break
            q = np.where(blocked & below, self.lo, np.where(blocked & above, self.hi, q))
            free &= ~blocked
        return np.clip(q + dq, self.lo, self.hi)

    def _priority_step(self, Jp: np.ndarray, Jr: np.ndarray, ep: np.ndarray, er: np.ndarray | None) -> np.ndarray:
        """One task-priority DLS step: position first, orientation (if given) in the position task's null space."""
        Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + self.damping2 * np.eye(3))
        dq = Jp_pinv @ ep
        if er is None:
            return np.clip(dq, -self.max_step, self.max_step)
        # The null-space projector must be exact (undamped pinv): a damped one leaks, and the orientation
        # task would then quietly undo the position step whenever "straight down" is out of reach.
        N = np.eye(Jp.shape[1]) - np.linalg.pinv(Jp, rcond=1e-3) @ Jp
        JrN = Jr @ N
        dq += N @ JrN.T @ np.linalg.solve(JrN @ JrN.T + self.damping2 * np.eye(3), er - Jr @ dq)
        return np.clip(dq, -self.max_step, self.max_step)
