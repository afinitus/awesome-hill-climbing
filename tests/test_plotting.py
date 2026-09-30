"""Media helpers: replaying a recorded episode and joining clips side by side (no OpenGL needed)."""

import numpy as np
import pytest

from lastmile.common.plotting import replay_frames, side_by_side


class FakeEnv:
    """Counts steps; 'succeeds' after 3 steps; renders a frame whose value is the step count."""

    def reset(self, seed=None):
        self.t = 0

    def step(self, action):
        self.t += 1
        return None, 0.0, False, False, {"success": self.t >= 3}

    def render(self, camera="front"):
        return np.full((8, 6, 3), self.t, np.uint8)


def test_replay_frames_keeps_every_other_frame_and_the_last():
    traj = {"seed": 0, "actions": np.zeros((5, 4)), "success": True}
    frames = replay_frames(FakeEnv(), traj, every=2)
    assert [int(f[0, 0, 0]) for f in frames] == [0, 2, 4, 5]


def test_replay_frames_rejects_a_diverged_replay():
    with pytest.raises(AssertionError):
        replay_frames(FakeEnv(), {"seed": 0, "actions": np.zeros((2, 4)), "success": True})


def test_side_by_side_pads_the_shorter_clip():
    a = [np.zeros((8, 6, 3), np.uint8)] * 3
    b = [np.full((8, 5, 3), 7, np.uint8)]
    out = side_by_side(a, b)
    assert len(out) == 3 and out[0].shape == (8, 6 + 4 + 5, 3)
    assert (out[2][:, -5:] == 7).all() and (out[2][:, 6:10] == 255).all()


def test_wilson_err_matches_wilson_and_skips_empty_groups():
    from lastmile.common.eval import wilson
    from lastmile.common.plotting import wilson_err

    p, err = wilson_err([27, 0, 0], [30, 12, 0])
    lo, hi = wilson(27, 30)
    assert p[0] == 0.9 and np.isclose(p[0] - err[0, 0], lo) and np.isclose(p[0] + err[1, 0], hi)
    assert err[0, 1] == 0 and err[1, 1] > 0  # 0/12: nothing below, a real interval above
    assert np.isnan(p[2]) and err.shape == (2, 3)


def test_tile_fills_a_grid_and_pads_the_last_row():
    from lastmile.common.plotting import tile

    a = [np.zeros((8, 6, 3), np.uint8)] * 2
    b = [np.full((8, 6, 3), 7, np.uint8)] * 3
    out = tile([a, b, a], cols=2)
    assert len(out) == 3 and out[0].shape == (8 + 4 + 8, 6 + 4 + 6, 3)
    assert (out[0][12:, 10:] == 255).all()  # the empty fourth cell is white
    assert (out[2][:8, :6] == 0).all() and (out[2][:8, 10:] == 7).all()


def test_optional_media_skips_only_render_failures(capsys):
    import pytest

    from lastmile.common.plotting import optional_media
    from lastmile.envs.cupdrop import RenderUnavailable

    with optional_media("a gif"):
        raise RenderUnavailable("no display")
    assert "skipped a gif" in capsys.readouterr().out
    with pytest.raises(ValueError):
        with optional_media("a gif"):
            raise ValueError("a real bug is not swallowed")
