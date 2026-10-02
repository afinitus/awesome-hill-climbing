"""Cost accounting: every run reports what it cost, not just how well it did (DESIGN.md §7).

A method that reaches 97% after 10,000 robot-minutes is a very different result from one that
gets there in 10. The ledger keeps three budgets side by side:

* **robot-minutes** by category (``search``, ``train``, ``eval``). Sim steps are converted at
  10 Hz, ``steps / 600``, so sim and real hardware share one unit.
* **human-minutes** (demos, labels, takeovers, resets, annotation).
* **compute** (CPU hours including rollout workers, GPU hours, wall-clock minutes).

On a clean exit the ``with`` block writes ``results/<chapter>/<method>_<robot>_<timestamp>.json``.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import numpy as np

from lastmile.common.eval import INIT_SET_VERSION, EvalResult, wilson

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
STEPS_PER_MINUTE = 600  # 10 Hz control
ROBOT_CATEGORIES = ("search", "train", "eval")


def _portable(value: Any) -> Any:
    """Config values with this checkout's absolute path replaced by a repo-relative one, so results
    files are the same on every machine (and do not publish a local home directory)."""
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v) for v in value]
    if isinstance(value, (str, Path)):
        text = str(value)
        root = str(REPO_ROOT)
        if text == root:
            return "."
        if text.startswith(root + "/"):
            return text[len(root) + 1:]
    return value
HUMAN_CATEGORIES = ("demos", "labels", "takeovers", "resets", "annotation")


def git_sha() -> str:
    """Short commit hash of the repo, or ``"nogit"`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "nogit"


def check_robot_category(category: str) -> None:
    """Reject categories the leaderboard does not know, so a typo cannot hide rollouts from the budget."""
    if category not in ROBOT_CATEGORIES:
        raise ValueError(f"unknown robot-minute category {category!r}; choose from {ROBOT_CATEGORIES}")


@dataclasses.dataclass
class _Score:
    """One scored evaluation: the JSON entry plus the env seeds behind it (kept for the pairing check)."""

    entry: dict[str, Any]
    seeds: tuple[int, ...] | None


def _score(result: EvalResult | Sequence[bool], init_set: str | None) -> _Score:
    """Summarize an evaluation. An ``EvalResult`` brings its own init set and seeds."""
    seeds = None
    if isinstance(result, EvalResult):
        if init_set is not None and result.init_set is not None and init_set != result.init_set:
            raise ValueError(f"init_set={init_set!r} contradicts the result, which was scored on {result.init_set!r}")
        init_set = init_set or result.init_set
        env_seeds = np.asarray(result.seeds)
        if env_seeds.ndim != 1 or not np.issubdtype(env_seeds.dtype, np.integer):
            raise ValueError("evaluation seeds must be a 1-D array of integers")
        seeds = tuple(int(s) for s in env_seeds)
        result = result.successes
    outcomes = np.asarray(result)
    if outcomes.ndim != 1 or outcomes.size == 0 or not np.isin(outcomes, [False, True]).all():
        raise ValueError("successes must be a non-empty 1-D array of binary outcomes")
    if seeds is not None and len(seeds) != len(outcomes):
        raise ValueError("evaluation seeds must have one entry per outcome")
    successes = outcomes.astype(bool)
    k, n = int(successes.sum()), len(successes)
    lo, hi = wilson(k, n)
    entry = {"sr": k / n if n else 0.0, "ci": [lo, hi], "k": k, "n": n, "init_set": init_set or "eval"}
    return _Score(entry, seeds)


def _to_json(obj: Any) -> Any:
    """Make numpy scalars/arrays, paths and dataclasses serializable."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__} to JSON")


class Ledger:
    """Context manager that accumulates costs and writes one results JSON per run."""

    def __init__(
        self,
        chapter: str,
        method: str,
        robot: str = "so101",
        variant: str = "v1",
        setting: str = "sim",
        config: Any = None,
        results_root: str | Path = DEFAULT_RESULTS_ROOT,
    ):
        self.chapter = chapter
        self.method = method
        self.robot = robot
        self.variant = variant
        self.setting = setting
        self.config: dict[str, Any] = (
            dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config or {})
        )
        self.results_root = Path(results_root)
        # A plain dict with fixed keys: `robot_steps["serach"] += n` raises instead of inventing a
        # category the leaderboard would not count.
        self.robot_steps: dict[str, int] = {c: 0 for c in ROBOT_CATEGORIES}
        self.human_minutes: defaultdict[str, float] = defaultdict(float, {c: 0.0 for c in HUMAN_CATEGORIES})
        self.worker_cpu_seconds = 0.0  # added by rollout_seeds for work done in worker processes
        self.gpu_hours = 0.0
        self.extra: dict[str, Any] = {}
        self.base: dict[str, Any] | None = None
        self.final: dict[str, Any] | None = None
        self._seeds: dict[str, tuple[int, ...] | None] = {}  # env seeds behind base / final, when known
        self.path: Path | None = None  # set once the JSON is written
        self._cpu_start = self._wall_start = 0.0

    # -- recording -------------------------------------------------------------------

    def set_base(self, id: str, successes: EvalResult | Sequence[bool], init_set: str | None = None) -> None:
        """Score of the starting policy, evaluated on the same episodes as the final policy.

        Pass the ``EvalResult`` itself (preferred: it carries its init set and seeds) or a bare success
        array, in which case ``init_set`` defaults to ``"eval"``.
        """
        score = _score(successes, init_set)
        self.base = {"id": id, **score.entry}
        self._seeds["base"] = score.seeds
        self._check_paired()

    def set_final(self, successes: EvalResult | Sequence[bool], init_set: str | None = None) -> None:
        """Held-out score of the improved policy (its rollouts are charged to ``eval``)."""
        score = _score(successes, init_set)
        self.final = {**score.entry, "init_set_version": INIT_SET_VERSION}
        self._seeds["final"] = score.seeds
        self._check_paired()

    def _check_paired(self) -> None:
        """Base and final must be the same episodes, or the reported difference (and McNemar) mean nothing."""
        if self.base is None or self.final is None:
            return
        for key in ("init_set", "n"):
            if self.base[key] != self.final[key]:
                raise ValueError(f"base and final differ in {key}: {self.base[key]!r} vs {self.final[key]!r}; "
                                 "score both on the same initial states")
        base_seeds, final_seeds = self._seeds.get("base"), self._seeds.get("final")
        if base_seeds is not None and final_seeds is not None and base_seeds != final_seeds:
            raise ValueError("base and final were scored on different env seeds")

    @property
    def robot_minutes(self) -> dict[str, float]:
        return {c: steps / STEPS_PER_MINUTE for c, steps in self.robot_steps.items()}

    @property
    def cpu_hours(self) -> float:
        return (time.process_time() - self._cpu_start + self.worker_cpu_seconds) / 3600

    @property
    def wall_minutes(self) -> float:
        return (time.perf_counter() - self._wall_start) / 60

    # -- context manager -------------------------------------------------------------

    def __enter__(self) -> Self:
        self._cpu_start = time.process_time()
        self._wall_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:  # a crashed run should not appear on the leaderboard
            self.write()
        return False

    def to_dict(self) -> dict[str, Any]:
        for category in self.robot_steps:
            check_robot_category(category)
        for name, costs in (("robot steps", self.robot_steps), ("human minutes", self.human_minutes),
                            ("compute", {"worker CPU seconds": self.worker_cpu_seconds, "GPU hours": self.gpu_hours})):
            for category, value in costs.items():
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{name} {category!r} must be finite and non-negative, got {value}")
        return {
            "schema": SCHEMA_VERSION,
            "chapter": self.chapter,
            "method": self.method,
            "robot": self.robot,
            "variant": self.variant,
            "setting": self.setting,
            "base": self.base,
            "final": self.final,
            "budget": {
                "robot_minutes": self.robot_minutes,
                "human_minutes": dict(self.human_minutes),
                "compute": {
                    "cpu_hours": self.cpu_hours,
                    "gpu_hours": self.gpu_hours,
                    "wall_minutes": self.wall_minutes,
                },
            },
            "extra": _portable(self.extra),
            "git_sha": git_sha(),
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "config": _portable(self.config),
        }

    def write(self) -> Path:
        """Write the results JSON; a numeric suffix avoids clobbering same-second runs."""
        out_dir = self.results_root / self.chapter
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.method}_{self.robot}_{datetime.now().astimezone():%Y%m%d-%H%M%S}"
        path, i = out_dir / f"{stem}.json", 1
        while path.exists():
            path, i = out_dir / f"{stem}-{i}.json", i + 1
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_to_json) + "\n")
        self.path = path
        return path


def load_results(root: str | Path = DEFAULT_RESULTS_ROOT) -> list[dict[str, Any]]:
    """Every results JSON under ``root`` (schema 1), each with its ``_path`` added."""
    results = []
    for path in sorted(Path(root).rglob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("schema") == SCHEMA_VERSION:
            data["_path"] = str(path)
            results.append(data)
    return results
