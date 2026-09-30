"""Shared plot style and media helpers so every chapter's figures look like one project.

Success rates are always drawn with their Wilson 95% interval shaded: with n = 64 search
episodes the interval is roughly +-12 points, and a curve without it invites over-reading.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from lastmile.common.eval import wilson

PALETTE = ["#2f6690", "#d1495b", "#3a7d44", "#edae49", "#6c5b7b", "#4a4a4a"]
GIF_MAX_BYTES = 3 * 1024 * 1024


def setup() -> None:
    """Apply the project style: readable fonts, light grid, a small fixed palette."""
    mpl.rcParams.update(
        {
            "figure.figsize": (6.4, 4.0),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
        }
    )


def _wilson_band(k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.array([wilson(int(ki), int(ni)) for ki, ni in zip(k, n)])
    return bounds[:, 0], bounds[:, 1]


def success_curve(
    ax: plt.Axes,
    x: Sequence[float],
    k: Sequence[int],
    n: int | Sequence[int],
    label: str | None = None,
    color: str | None = None,
    marker: str = "o",
) -> plt.Line2D:
    """Plot success rate ``k / n`` against ``x`` with the Wilson interval shaded."""
    x = np.asarray(x, dtype=float)
    k = np.asarray(k)
    n = np.broadcast_to(np.asarray(n), k.shape)
    (line,) = ax.plot(x, k / n, marker=marker, label=label, color=color)
    lo, hi = _wilson_band(k, n)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.18, linewidth=0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    return line


def heldout_point(
    ax: plt.Axes, x: float, k: int, n: int, label: str = "held-out eval", color: str = PALETTE[5]
) -> None:
    """Mark a held-out evaluation distinctly from search-time estimates (diamond + error bar)."""
    lo, hi = wilson(k, n)
    p = k / n
    ax.errorbar([x], [p], yerr=[[p - lo], [hi - p]], fmt="D", color=color, capsize=4, label=label)


def save_fig(fig: plt.Figure, path: str | Path) -> Path:
    """Save a figure (creating parent folders) and close it to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_gif(frames: Sequence[np.ndarray], path: str | Path, fps: int = 10, max_size: int = 256) -> Path:
    """Write ``frames`` (``[H, W, 3]`` uint8) as a looping GIF no larger than ``max_size`` px.

    Warns if the file exceeds 3 MB, the budget for media committed to the repo.
    """
    import imageio.v3 as iio
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    small = []
    for frame in frames:
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        scale = max_size / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        small.append(np.asarray(img))
    iio.imwrite(path, np.stack(small), extension=".gif", duration=1000 / fps, loop=0)
    size = path.stat().st_size
    if size > GIF_MAX_BYTES:
        warnings.warn(f"{path} is {size / 1e6:.1f} MB (> 3 MB); use fewer frames or a lower fps")
    return path


def replay_frames(env, traj: dict, label: str = "", every: int = 2, camera: str = "front") -> list:
    """Re-run a recorded episode's actions in a rendering ``env`` and caption every ``every``-th frame.

    Rollouts are bit-for-bit reproducible (DESIGN.md §2), so recording ``{"seed", "actions", "success"}``
    in a fast worker and rendering later gives the exact same episode; the assert checks that it did.
    """
    from PIL import Image, ImageDraw

    env.reset(seed=int(traj["seed"]))
    frames, info = [env.render(camera)], {"success": False}
    for t, action in enumerate(traj["actions"]):
        info = env.step(action)[4]
        if (t + 1) % every == 0 or t + 1 == len(traj["actions"]):
            frames.append(env.render(camera))
    assert bool(info["success"]) == bool(traj["success"]), "replay diverged from the recorded episode"
    out = []
    for frame in frames:
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        if label:
            ImageDraw.Draw(img).text((6, 6), label, fill=(20, 20, 20))
        out.append(np.asarray(img))
    return out


def side_by_side(*clips: Sequence[np.ndarray], gap: int = 4) -> list[np.ndarray]:
    """Join clips left to right frame by frame (a before/after GIF); shorter clips hold their last frame."""
    length = max(len(c) for c in clips)
    padded = [list(c) + [c[-1]] * (length - len(c)) for c in clips]
    white = np.full((padded[0][0].shape[0], gap, 3), 255, np.uint8)
    frames = []
    for t in range(length):
        parts = [padded[0][t]]
        for clip in padded[1:]:
            parts += [white, clip[t]]
        frames.append(np.concatenate(parts, axis=1))
    return frames


def wilson_err(k: Sequence[int], n: int | Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Rates ``k / n`` and their Wilson 95% half-widths as ``(p, err)``, ready for ``errorbar``.

    ``err`` has shape ``[2, len(k)]`` (below, above), as ``ax.errorbar(x, p, yerr=err)`` expects. Groups
    with ``n = 0`` give NaN, so they are simply not drawn.
    """
    k = np.atleast_1d(np.asarray(k, dtype=int))
    n = np.broadcast_to(np.asarray(n, dtype=int), k.shape)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    lo, hi = np.array([wilson(int(a), int(b)) if b else (np.nan, np.nan) for a, b in zip(k, n)]).T
    return p, np.clip([p - lo, hi - p], 0, None)  # Wilson's bound at k = 0 can sit 1e-17 above 0


def tile(clips: Sequence[Sequence[np.ndarray]], cols: int, gap: int = 4) -> list[np.ndarray]:
    """Arrange equal-sized clips in a grid, ``cols`` per row, frame by frame (short clips hold their end).

    Missing cells in the last row are left white.
    """
    clips = [list(c) for c in clips]
    blank = [np.full_like(clips[0][0], 255)]
    clips += [blank] * (-len(clips) % cols)
    rows = [side_by_side(*clips[i : i + cols], gap=gap) for i in range(0, len(clips), cols)]
    length = max(len(r) for r in rows)
    rows = [r + [r[-1]] * (length - len(r)) for r in rows]
    white = np.full((gap, rows[0][0].shape[1], 3), 255, np.uint8)
    frames = []
    for t in range(length):
        parts = [rows[0][t]]
        for row in rows[1:]:
            parts += [white, row[t]]
        frames.append(np.concatenate(parts, axis=0))
    return frames
