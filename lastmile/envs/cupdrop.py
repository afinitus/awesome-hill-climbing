"""CupDropEnv: pick up a 2.5 cm cube and drop it into a cup (docs/DESIGN.md section 2).

The scene is built in code with `mujoco.MjSpec`: load the vendored Menagerie arm, then add a table, a
free cube, a free open cup, lights and cameras. Nothing is welded or teleported: the cube only moves
because the fingers push and squeeze it, and the cup is a light free body, so a rim hit shoves or tips it
instead of acting as an unbreakable guide rail. Every failure (missed grasp, slip, rim hit) is physical.

Control is at 10 Hz. The policy sends an end-effector delta plus a gripper command; the env integrates an
EE target, clips it to a workspace box, runs damped-least-squares IK, and hands joint targets to the arm's
position servos (with robot-specific gripper gain and force-limit adjustments in RobotSpec).
"""

from __future__ import annotations

import mujoco
import numpy as np

from lastmile.envs.ik import DownIK
from lastmile.envs.robots import CupSpec, RobotSpec, down_rotation, get_robot

CONTROL_DT = 0.1
MAX_DELTA = 0.02  # meters of EE target motion per step at |action| = 1
MAX_LEAD = 0.03  # meters the EE target may lead the measured EE position (per axis)
CUBE_HALF = 0.0125
CUBE_MASS = 0.015
MAX_CUBE_YAW = np.deg2rad(15.0)
SUCCESS_MARGIN = 0.005
SUCCESS_STREAK = 3
MAX_CUP_TILT = np.deg2rad(20.0)  # a cup tipped further than this does not count as holding the cube

# Stiff contacts for every geom. MuJoCo's soft defaults let the squeezed 15 g cube sink millimeters into
# the table, the cup and the fingers, and a controller can then "succeed" by pushing through things.
# solimp near 1 makes the constraint almost rigid; solref = 10 ms stays >= 2 physics steps, as MuJoCo
# recommends for stability.
CONTACT_SOLREF = (0.01, 1.0)
CONTACT_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)

VARIANTS = ("v1", "hard", "bias")
HARD_CUP = CupSpec(inner_radius=0.03)
HARD_REGION_SCALE = (1.5, 1.5)  # the hard cube region is 18 x 24 cm around the same center

OBS_KEYS = ("joint_pos", "ee_pos", "gripper_open", "cube_pos", "cube_to_ee", "cube_yaw", "cup_pos", "time_frac")


def obs_layout(robot: str | RobotSpec) -> tuple[int, dict[str, slice]]:
    """(obs_dim, obs_slices) for a robot, without building the simulation."""
    spec = get_robot(robot) if isinstance(robot, str) else robot
    sizes = {"joint_pos": len(spec.arm_joints) + 1, "ee_pos": 3, "gripper_open": 1, "cube_pos": 3,
             "cube_to_ee": 3, "cube_yaw": 2, "cup_pos": 2, "time_frac": 1}
    slices, i = {}, 0
    for key in OBS_KEYS:
        slices[key] = slice(i, i + sizes[key])
        i += sizes[key]
    return i, slices


def _look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion for a camera at `pos` looking at `target` (MuJoCo cameras look along their -z)."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).ravel())
    return quat


def build_model(spec: RobotSpec, cup: CupSpec, render_size: tuple[int, int] = (256, 256)) -> mujoco.MjModel:
    """Arm + table + cube + cup + cameras, compiled into one MjModel."""
    s = mujoco.MjSpec.from_file(str(spec.mjcf))
    s.option.timestep = spec.timestep
    # Multi-point contacts for convex pairs: without it the cube rests on a single point of the thin cup
    # floor (cylinder-box) and sinks through it.
    s.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    # Stiff contacts between two free bodies (cube in cup) need a converged solver; Menagerie's SO-101 caps
    # it at 10 iterations for speed, which lets a dropped cube launch the cup. The solver stops early once
    # converged, so the higher cap costs little.
    s.option.iterations = max(s.option.iterations, 100)
    s.option.ls_iterations = max(s.option.ls_iterations, 50)
    s.visual.global_.offwidth = max(640, render_size[1])
    s.visual.global_.offheight = max(480, render_size[0])
    s.visual.headlight.ambient = [0.3, 0.3, 0.3]
    s.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    s.visual.headlight.specular = [0.0, 0.0, 0.0]

    _tune_gripper(s, spec)
    for light in s.lights:  # lights shipped with the robot model; ours are added below
        light.diffuse = 0.5 * np.asarray(light.diffuse)
    for name, (body, pos, quat) in spec.extra_sites.items():
        s.body(body).add_site(name=name, pos=pos, quat=quat, size=[0.004, 0, 0], group=3)
    for name, (body, pos, target, up, fovy) in spec.extra_cameras.items():
        s.body(body).add_camera(name=name, pos=pos, quat=_look_at_quat(pos, target, up), fovy=fovy)

    s.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                  rgb1=[0.82, 0.78, 0.72], rgb2=[0.76, 0.72, 0.66], width=300, height=300)
    s.add_material(name="table", textures=["", "grid"], texrepeat=[8, 8], texuniform=True)
    s.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                  rgb1=[0.55, 0.65, 0.75], rgb2=[0.15, 0.18, 0.22], width=256, height=256)

    world = s.worldbody
    world.add_light(pos=[0.3, 0.3, 1.2], dir=[-0.2, -0.2, -1], diffuse=[0.45, 0.45, 0.45], castshadow=True)
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.01], material="table",
                   friction=[1.0, 0.005, 0.0001])

    cx, cy = spec.cube_center
    cube = world.add_body(name="cube", pos=[cx, cy, CUBE_HALF])
    cube.add_freejoint(name="cube")
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, mass=CUBE_MASS,
                  rgba=[0.85, 0.2, 0.15, 1], condim=4, friction=[1.0, 0.005, 0.0001])

    # The cup: a floor disc plus a ring of overlapping wall boxes (an open hollow cylinder) on a free joint.
    cup_body = world.add_body(name="cup", pos=[*spec.cup_pos, 0.0])
    cup_body.add_freejoint(name="cup")
    cup_rgba = [0.2, 0.45, 0.85, 1]
    cup_body.add_geom(name="cup_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, density=cup.density,
                      size=[cup.inner_radius + cup.wall, cup.floor / 2, 0], pos=[0, 0, cup.floor / 2], rgba=cup_rgba)
    r_mid = cup.inner_radius + cup.wall / 2
    half_len = 1.1 * r_mid * np.tan(np.pi / cup.segments)
    gap = 0.001  # the walls start 1 mm above the table, so an upright cup rests on its floor disc alone
    for k in range(cup.segments):  # (20 boxes on the table would add ~80 contacts to every physics step)
        a = 2 * np.pi * k / cup.segments
        cup_body.add_geom(name=f"cup_wall{k}", type=mujoco.mjtGeom.mjGEOM_BOX, density=cup.density,
                          size=[cup.wall / 2, half_len, (cup.height - gap) / 2],
                          pos=[r_mid * np.cos(a), r_mid * np.sin(a), (cup.height + gap) / 2],
                          quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)], rgba=cup_rgba, friction=[0.8, 0.005, 0.0001])

    for geom in s.geoms:
        geom.solref = CONTACT_SOLREF
        geom.solimp = CONTACT_SOLIMP

    # Scene cameras, placed relative to the workspace so both arm scales get the same framing.
    k = spec.view_scale
    look = np.array([0.6 * cx, (cy + spec.cup_pos[1]) / 2, 0.06 * k])  # between the base and the task area
    front = look + k * np.array([0.40, 0.34, 0.26])
    world.add_camera(name="front", pos=front, quat=_look_at_quat(front, look), fovy=55)
    mid = np.array([(cx + spec.cup_pos[0]) / 2, (cy + spec.cup_pos[1]) / 2, 0.0])
    top = mid + np.array([0.0, 0.0, 0.55 * k])
    world.add_camera(name="top", pos=top, quat=_look_at_quat(top, mid, up=(1.0, 0.0, 0.0)), fovy=55)
    return s.compile()


def _tune_gripper(s: mujoco.MjSpec, spec: RobotSpec) -> None:
    """Apply the robot's gripper gain and force limit (see RobotSpec) to the Menagerie position servo."""
    act = s.actuator(spec.gripper_actuator)
    if spec.gripper_kp is not None:  # a position servo is force = kp * (ctrl - q) - kv * qdot
        act.gainprm[0] = spec.gripper_kp
        act.biasprm[1] = -spec.gripper_kp
    if spec.gripper_force is not None:
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-spec.gripper_force, spec.gripper_force]



class RenderUnavailable(RuntimeError):
    """Raised by CupDropEnv.render when this machine cannot render offscreen (no OpenGL context)."""

class CupDropEnv:
    """The CupDrop task. See docs/DESIGN.md section 2 for the full contract."""

    def __init__(self, robot: str = "so101", variant: str = "v1", max_steps: int = 150,
                 render_size: tuple[int, int] = (256, 256), dr: bool = False, obs_mode: str = "state"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if obs_mode != "state":
            raise NotImplementedError("only obs_mode='state' exists in v0.1")
        self.robot = get_robot(robot)
        self.variant = variant
        self.max_steps = max_steps
        self.render_size = render_size
        self.dr = dr

        self.cup = HARD_CUP if variant == "hard" else CupSpec()
        region = np.array(self.robot.cube_size)
        self.cube_region = region * HARD_REGION_SCALE if variant == "hard" else region
        n_arm = len(self.robot.arm_joints)
        self.bias = np.array(self.robot.bias_q) if variant == "bias" else np.zeros(n_arm)

        self.model = build_model(self.robot, self.cup, render_size)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = round(CONTROL_DT / self.model.opt.timestep)
        self.obs_dim, self.obs_slices = obs_layout(self.robot)

        m = self.model
        self._arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in self.robot.arm_joints])
        self._arm_act = np.array([self._actuator_for(j) for j in self.robot.arm_joints])
        self._grip_qadr = m.jnt_qposadr[m.joint(self.robot.gripper_joint).id]
        self._grip_act = m.actuator(self.robot.gripper_actuator).id
        self._cube_body = m.body("cube").id
        self._cube_geom = m.geom("cube").id
        self._cube_qadr = m.jnt_qposadr[m.joint("cube").id]
        self._cup_body = m.body("cup").id
        self._cup_qadr = m.jnt_qposadr[m.joint("cup").id]
        self._site = m.site(self.robot.ik_site).id
        self._jaws = np.array([m.body(b).id for b in self.robot.jaw_bodies])
        self._friction0 = m.geom_friction.copy()
        self._ws_lo = np.array(self.robot.workspace_lo)
        self._ws_hi = np.array(self.robot.workspace_hi)
        self._cup_home = np.array(self.robot.cup_pos)

        self.ik = DownIK(m, self.robot.ik_site, self.robot.arm_joints, down_rotation(self.robot.ik_yaw))
        self._state_size = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
        self._renderer: mujoco.Renderer | None = None
        self.reset(seed=0)

    # ------------------------------------------------------------------ helpers
    def _actuator_for(self, joint: str) -> int:
        jid = self.model.joint(joint).id
        for a in range(self.model.nu):
            if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and self.model.actuator_trnid[a, 0] == jid:
                return a
        raise ValueError(f"no actuator drives joint {joint}")

    def _gripper_ctrl(self, cmd: float) -> float:
        t = (np.clip(cmd, -1.0, 1.0) + 1.0) / 2.0  # 0 = open, 1 = closed
        return (1 - t) * self.robot.gripper_open + t * self.robot.gripper_closed

    def _apply_ctrl(self, q_arm: np.ndarray) -> None:
        # Bias variant: the servos reach q + bias while believing they are at q (zero errors).
        self.data.ctrl[self._arm_act] = q_arm + self.bias
        self.data.ctrl[self._grip_act] = self._gripper_ctrl(self._grip_cmd)

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, options: dict | None = None):
        """Start an episode. `options` is accepted for Gymnasium-style callers and ignored."""
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)

        # Resample starts that would overlap the cup (only possible in the wider "hard" region).
        half = self.cube_region / 2
        keep_out = self.cup.inner_radius + self.cup.wall + np.sqrt(2) * CUBE_HALF + 0.01
        while True:
            xy = np.array(self.robot.cube_center) + rng.uniform(-half, half)
            yaw = rng.uniform(-MAX_CUBE_YAW, MAX_CUBE_YAW)
            if np.linalg.norm(xy - self._cup_home) > keep_out:
                break
        d.qpos[self._cube_qadr:self._cube_qadr + 3] = [xy[0], xy[1], CUBE_HALF]
        d.qpos[self._cube_qadr + 3:self._cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self._cup_qadr:self._cup_qadr + 7] = [*self._cup_home, 0.0, 1, 0, 0, 0]

        # Domain randomization (off in the v0.1 chapters): one sliding-friction scale for every geom.
        # MuJoCo combines two geoms' friction with max() (or takes the higher-priority geom's), so scaling
        # the cube alone would be masked by the table and fingers; scaling all of them scales every contact.
        m.geom_friction[:] = self._friction0
        if self.dr:
            m.geom_friction[:, 0] *= rng.uniform(0.7, 1.3)

        self._q_cmd = np.array(self.robot.home_q, dtype=float)
        self._grip_cmd = -1.0
        d.qpos[self._arm_qadr] = self._q_cmd + self.bias
        d.qpos[self._grip_qadr] = self.robot.gripper_open
        self._apply_ctrl(self._q_cmd)
        # Let the arm settle under gravity before the first observation (0.2 s, same for every seed).
        mujoco.mj_step(m, d, nstep=2 * self.n_substeps)
        d.time = 0.0
        mujoco.mj_forward(m, d)  # positions and contacts for the current qpos (mj_step leaves them one substep old)

        self._ee_target = self.ik.forward(self._q_cmd)[0]
        self._steps = 0
        self._streak = 0
        self._success = False
        self._near_miss = self._cube_cup_dist()
        return self._obs(), self._info()

    def step(self, action):
        # Actions are quantized to float32 here, the precision rollouts record them in, so replaying a
        # recorded trajectory (or stepping in-process with float64 actions) gives bit-identical episodes.
        a = np.asarray(action, dtype=np.float32).reshape(4).astype(float)
        a = np.clip(np.nan_to_num(a), -1.0, 1.0)
        # Integrate the EE target, but keep it within MAX_LEAD of where the arm actually is: otherwise an
        # unreachable target (or a gripper pressed on the table) winds up and the arm lurches later.
        ee_now = self._believed_ee()
        target = self._ee_target + MAX_DELTA * a[:3]
        target = ee_now + np.clip(target - ee_now, -MAX_LEAD, MAX_LEAD)
        self._ee_target = np.clip(target, self._ws_lo, self._ws_hi)
        q_prev = self._q_cmd
        self._q_cmd = self.ik.solve(self._ee_target, q_prev)
        self._grip_cmd = float(a[3])
        # Ramp the joint targets over the control period, like a servo's motion profile, instead of
        # feeding a step: a step makes the force-limited servos overshoot and ring.
        for k in range(1, self.n_substeps + 1):
            self._apply_ctrl(q_prev + (k / self.n_substeps) * (self._q_cmd - q_prev))
            mujoco.mj_step(self.model, self.data)
        # mj_step leaves positions and contacts from the start of its last substep. Recompute them for the
        # current qpos, so observations, success and the next step's MAX_LEAD clamp read the same values
        # here as after set_state() (which is what makes save/restore resume exactly).
        mujoco.mj_forward(self.model, self.data)
        self._steps += 1

        dist = self._cube_cup_dist()
        self._near_miss = min(self._near_miss, dist)
        self._streak = self._streak + 1 if self._cube_in_cup(dist) else 0
        newly = not self._success and self._streak >= SUCCESS_STREAK
        self._success = self._success or newly
        terminated = self._success
        truncated = not terminated and self._steps >= self.max_steps
        return self._obs(), float(newly), terminated, truncated, self._info()

    def _cube_cup_dist(self) -> float:
        """Horizontal distance from the cube center to the cup's axis (wherever the cup is now)."""
        return float(np.linalg.norm(self.data.xpos[self._cube_body, :2] - self.data.xpos[self._cup_body, :2]))

    def _cube_in_cup(self, dist: float) -> bool:
        """Released cube inside an upright cup: within the inner cylinder, below the rim, touching no finger.

        Requiring the release means carrying the cube into the cup and holding it there does not count;
        the task is to drop it in.
        """
        d = self.data
        upright = d.xmat[self._cup_body, 8] > np.cos(MAX_CUP_TILT)  # cup z-axis vs world z
        below_rim = d.xpos[self._cube_body, 2] - d.xpos[self._cup_body, 2] < self.cup.height
        inside = dist < self.cup.inner_radius - SUCCESS_MARGIN
        return bool(upright and below_rim and inside and not self._jaws_touching().any())

    def _jaws_touching(self) -> np.ndarray:
        """bool[2]: which of the two finger bodies touch the cube right now."""
        g1, g2 = self.data.contact.geom1, self.data.contact.geom2
        other = np.concatenate([g2[g1 == self._cube_geom], g1[g2 == self._cube_geom]])
        return np.isin(self._jaws, self.model.geom_bodyid[other])

    def _grasped(self) -> bool:
        """Cube touches both finger bodies and is lifted off the table."""
        if self.data.xpos[self._cube_body, 2] < CUBE_HALF + 0.005:
            return False
        return bool(self._jaws_touching().all())

    def _believed_ee(self) -> np.ndarray:
        """EE position computed from the encoder readings (differs from the truth only with bias)."""
        if self.bias.any():
            return self.ik.forward(self.data.qpos[self._arm_qadr] - self.bias)[0]
        return self.data.site_xpos[self._site].copy()

    def _obs(self) -> np.ndarray:
        d = self.data
        q_arm = d.qpos[self._arm_qadr] - self.bias  # what the (mis-zeroed) encoders report
        ee = self._believed_ee()
        g = d.qpos[self._grip_qadr]
        g_open = (g - self.robot.gripper_closed) / (self.robot.gripper_open - self.robot.gripper_closed)
        cube = d.xpos[self._cube_body]
        R = d.xmat[self._cube_body].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return np.concatenate([
            q_arm, [g], ee, [np.clip(g_open, 0.0, 1.0)], cube, ee - cube, [np.sin(yaw), np.cos(yaw)],
            d.xpos[self._cup_body, :2], [self._steps / self.max_steps],
        ]).astype(np.float32)

    def _info(self) -> dict:
        return {"success": self._success, "grasped": self._grasped(), "near_miss": self._near_miss,
                "steps": self._steps, "ee_target": self._ee_target.copy()}

    # ------------------------------------------------------------------ save / restore
    def get_state(self) -> dict:
        physics = np.empty(self._state_size)
        mujoco.mj_getState(self.model, self.data, physics, mujoco.mjtState.mjSTATE_INTEGRATION)
        return {"physics": physics, "q_cmd": self._q_cmd.copy(), "grip_cmd": self._grip_cmd,
                "ee_target": self._ee_target.copy(), "steps": self._steps, "streak": self._streak,
                "success": self._success, "near_miss": self._near_miss,
                "friction": self.model.geom_friction.copy()}

    def set_state(self, state: dict) -> None:
        mujoco.mj_setState(self.model, self.data, state["physics"], mujoco.mjtState.mjSTATE_INTEGRATION)
        self.model.geom_friction[:] = state["friction"]
        self._q_cmd = state["q_cmd"].copy()
        self._grip_cmd = state["grip_cmd"]
        self._ee_target = state["ee_target"].copy()
        self._steps = state["steps"]
        self._streak = state["streak"]
        self._success = state["success"]
        self._near_miss = state["near_miss"]
        mujoco.mj_forward(self.model, self.data)

    def observe(self) -> np.ndarray:
        """Current observation without stepping (handy after set_state)."""
        return self._obs()

    # ------------------------------------------------------------------ rendering
    def render(self, camera: str = "front") -> np.ndarray:
        if self._renderer is None:
            h, w = self.render_size
            try:
                self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            except Exception as err:  # no OpenGL context: headless Linux without EGL, macOS CI runners
                raise RenderUnavailable(
                    f"MuJoCo cannot create an offscreen renderer here ({err}). On headless Linux set "
                    "MUJOCO_GL=egl (or osmesa); GitHub's macOS runners have no GPU display."
                ) from err
        name = self.robot.wrist_camera if camera == "wrist" else camera
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


FAILURE_MODES = ("missed_grasp", "slip", "rim_hit", "off_target", "still_held")


class OutcomeTracker:
    """Label an episode from its observations and infos alone (so it works on logged trajectories too).

    The label depends on where the cube was the last time it was held (measured from the cup axis):

    success       the env's success flag
    missed_grasp  the cube was never lifted between both fingers
    off_target    let go above the cup's opening, or near the cup above the rim, but did not end up inside
                  (released off-center, bounced out, cup knocked over)
    rim_hit       lost below rim height right next to the cup: the carried cube ran into the wall
    slip          lost anywhere else (dropped in transit, squeezed out)
    still_held    time ran out with the cube still in the gripper
    """

    def __init__(self, env: CupDropEnv):
        self._sl = env.obs_slices
        cup = env.cup
        self._r_inner = cup.inner_radius
        self._rim_zone = cup.inner_radius + cup.wall + CUBE_HALF + 0.02  # cube center scraping the outside
        self._rim_top = cup.height + CUBE_HALF  # cube bottom below the rim
        self._near = cup.inner_radius + cup.wall + 0.03
        self.ever_grasped = False
        self.holding = False
        self.held_at: tuple[float, float] | None = None  # (distance to cup axis, cube height) when last held
        self.success = False

    def update(self, obs: np.ndarray, info: dict) -> None:
        self.holding = bool(info["grasped"])
        if self.holding:
            cube = obs[self._sl["cube_pos"]]
            self.held_at = (float(np.linalg.norm(cube[:2] - obs[self._sl["cup_pos"]])), float(cube[2]))
        self.ever_grasped |= self.holding
        self.success = bool(info["success"])

    @classmethod
    def label_trajectory(cls, env: CupDropEnv, traj: dict) -> str:
        """Label a trajectory recorded by `rollout.evaluate(..., record_trajectories=True)`."""
        tracker = cls(env)
        after = np.vstack([traj["obs"][1:], traj["final_obs"][None]])  # obs after each step
        for t, obs in enumerate(after):
            tracker.update(obs, {key: values[t] for key, values in traj["infos"].items()})
        return tracker.label

    @property
    def label(self) -> str:
        if self.success:
            return "success"
        if not self.ever_grasped:
            return "missed_grasp"
        if self.holding:
            return "still_held"
        dist, z = self.held_at
        if dist < self._r_inner:
            return "off_target"
        if dist < self._rim_zone and z < self._rim_top:
            return "rim_hit"
        return "off_target" if dist < self._near else "slip"
