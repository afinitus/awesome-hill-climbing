"""The CupDrop task, the robots it runs on, and the scripted knob expert."""

from lastmile.envs.cupdrop import FAILURE_MODES, CupDropEnv, OutcomeTracker, obs_layout
from lastmile.envs.knob_controller import DEFAULT_KNOBS, KNOB_SPACE, TUNED_KNOBS, KnobController
from lastmile.envs.robots import ROBOTS, RobotSpec, get_robot

__all__ = [
    "DEFAULT_KNOBS",
    "FAILURE_MODES",
    "KNOB_SPACE",
    "ROBOTS",
    "TUNED_KNOBS",
    "CupDropEnv",
    "KnobController",
    "OutcomeTracker",
    "RobotSpec",
    "get_robot",
    "obs_layout",
]
