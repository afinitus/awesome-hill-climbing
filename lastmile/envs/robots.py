"""Per-robot configuration for CupDropEnv.

Everything the task needs to know about an arm lives in one `RobotSpec`: which MJCF to load, which
joints the IK may move, how to map the gripper command, a home pose, and where the cube and cup go.
Positions are in the robot base frame (x forward, y left, z up, table top at z = 0), in meters.

The two arms differ mostly in scale, so the PiPER layout is the SO-101 layout pushed further out. With the
gripper pointing straight down, both arms are limited by their wrist-pitch range, not their full reach:
the SO-101 works best 0.15-0.30 m from its base below ~9 cm height, the PiPER out to ~0.37 m. The cube,
the cup and the knob units stay the same for both.

Bias offsets (variant="bias", rad, base to wrist): SO-101 (0.06, 0.05, 0.04, -0.03, 0.0) and PiPER
(0.04, 0.04, 0.04, 0.0, -0.04, 0.0), i.e. 2-3.5 deg zero errors. They shift the true gripper position by
1-2.5 cm from where the encoders say it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent / "assets"


@dataclass(frozen=True)
class CupSpec:
    """An open cylindrical cup, built from a ring of wall segments on a floor disc."""

    inner_radius: float = 0.04
    height: float = 0.06  # rim height above the table
    wall: float = 0.005
    floor: float = 0.006  # floor thickness; thick enough that a dropped cube cannot punch through it
    segments: int = 20
    density: float = 1000.0  # kg/m^3, a sturdy plastic cup of about 125 g: light enough to push or tip


@dataclass(frozen=True)
class RobotSpec:
    name: str
    mjcf: Path
    arm_joints: tuple[str, ...]  # joints the IK controls, base to wrist
    gripper_joint: str  # joint reported in the observation
    gripper_actuator: str
    gripper_open: float  # actuator target for command -1 (fully open)
    gripper_closed: float  # actuator target for command +1 (fully closed)
    home_q: tuple[float, ...]  # arm joint angles at reset (gripper pointing down, above the table)
    ik_site: str  # site whose position is the end effector ("ee_pos")
    ik_yaw: float  # fixed yaw of the gripper frame in the base frame (rad); wrist roll absorbs base rotation
    workspace_lo: tuple[float, float, float]  # EE target box (the env clips its target to it)
    workspace_hi: tuple[float, float, float]
    cube_center: tuple[float, float]  # center of the cube start region (v1)
    cube_size: tuple[float, float]  # full x, y extent of the cube start region (v1)
    cup_pos: tuple[float, float]
    jaw_bodies: tuple[str, str]  # the two finger bodies, used to detect a grasp
    wrist_camera: str
    timestep: float  # physics step; 0.1 s / timestep substeps per env step
    bias_q: tuple[float, ...]  # joint zero errors for variant="bias" (rad), unknown to the controller
    gripper_kp: float | None = None  # overrides the gripper servo's position gain (None: Menagerie value)
    gripper_force: float | None = None  # caps the gripper servo's force/torque (None: Menagerie value)
    extra_sites: dict = field(default_factory=dict)  # sites added: name -> (body, pos, quat)
    extra_cameras: dict = field(default_factory=dict)  # cameras added: name -> (body, pos, target, up, fovy)
    view_scale: float = 1.0  # distance multiplier for the scene cameras


SO101 = RobotSpec(
    name="so101",
    mjcf=ASSETS / "so101" / "so101.xml",
    arm_joints=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
    gripper_joint="gripper",
    gripper_actuator="gripper",
    gripper_open=0.8,
    gripper_closed=-0.17,
    home_q=(0.0, -0.207, 0.159, 1.619, 0.046),  # site at (0.22, 0, 0.085), pointing down
    ik_site="gripperframe",
    ik_yaw=0.0,
    workspace_lo=(0.08, -0.25, 0.0),
    workspace_hi=(0.34, 0.25, 0.25),
    cube_center=(0.23, 0.0),
    cube_size=(0.12, 0.16),
    cup_pos=(0.19, -0.16),
    jaw_bodies=("gripper", "moving_jaw_so101_v1"),
    wrist_camera="wrist_cam",
    timestep=0.005,
    bias_q=(0.06, 0.05, 0.04, -0.03, 0.0),
    # LeRobot's SO-101 follower limits the gripper servo to 50% of its torque so it cannot burn out
    # squeezing an object; we do the same (Menagerie's 2.94 Nm is the full stall torque).
    gripper_force=1.47,
)

PIPER = RobotSpec(
    name="piper",
    mjcf=ASSETS / "piper" / "piper.xml",
    arm_joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    gripper_joint="joint7",
    gripper_actuator="gripper",
    gripper_open=0.035,
    gripper_closed=0.0,
    home_q=(0.0, 1.682, -1.202, 0.0, 1.178, 0.0),  # site at (0.30, 0, 0.10), pointing down
    ik_site="tcp",
    ik_yaw=0.0,
    workspace_lo=(0.12, -0.4, 0.0),
    workspace_hi=(0.55, 0.4, 0.4),
    cube_center=(0.32, 0.0),
    cube_size=(0.12, 0.16),
    cup_pos=(0.26, -0.18),
    jaw_bodies=("link7", "link8"),
    wrist_camera="wrist_cam",
    timestep=0.002,
    bias_q=(0.04, 0.04, 0.04, 0.0, -0.04, 0.0),
    # Menagerie's finger servo (kp=40) presses a 2.5 cm cube with only ~0.2 N per finger, so the grasp
    # would survive only on generous friction. A stiffer gain gives a few N, still inside its 10 N limit.
    gripper_kp=400.0,
    extra_sites={"tcp": ("link6", (0.0, 0.0, 0.135), (0.7071068, 0.0, -0.7071068, 0.0))},
    extra_cameras={"wrist_cam": ("link6", (-0.07, 0.0, 0.03), (0.0, 0.0, 0.16), (-1.0, 0.0, 0.0), 75.0)},
    view_scale=1.3,
)

ROBOTS: dict[str, RobotSpec] = {"so101": SO101, "piper": PIPER}


def get_robot(name: str) -> RobotSpec:
    if name not in ROBOTS:
        raise ValueError(f"unknown robot {name!r}; choose from {sorted(ROBOTS)}")
    return ROBOTS[name]


def down_rotation(yaw: float) -> np.ndarray:
    """Rotation matrix of a tool frame whose x-axis points straight down, turned by `yaw` about z.

    Both IK sites use x as the approach axis (the PiPER "tcp" site is added with that convention), so one
    target works for both arms.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # +90 deg about y: x -> -z
    return rz @ ry
