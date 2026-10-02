import json
from dataclasses import dataclass

import numpy as np
import pytest

from lastmile.common.eval import EvalResult, wilson
from lastmile.common.ledger import Ledger, load_results


@dataclass
class Config:
    lr: float = 0.1
    iters: int = 5


def test_results_json_round_trip(tmp_path):
    with Ledger("ch01", "ars", robot="so101", variant="v1", config=Config(), results_root=tmp_path) as L:
        L.robot_steps["search"] += 1200
        L.robot_steps["eval"] += 600
        L.human_minutes["demos"] += 2.5
        L.set_base(id="knobs_default", successes=[True] * 10 + [False] * 10)
        L.set_final(successes=[True] * 19 + [False], init_set="eval")
        L.extra["best_knobs"] = {"grip_height": 0.02}

    assert L.path.parent == tmp_path / "ch01" and L.path.name.startswith("ars_so101_")
    data = json.loads(L.path.read_text())
    assert set(data) == {
        "schema",
        "chapter",
        "method",
        "robot",
        "variant",
        "setting",
        "base",
        "final",
        "budget",
        "extra",
        "git_sha",
        "timestamp",
        "config",
    }
    assert data["schema"] == 1 and data["setting"] == "sim"
    assert data["base"] == {
        "id": "knobs_default",
        "sr": 0.5,
        "ci": pytest.approx(list(wilson(10, 20))),
        "k": 10,
        "n": 20,
        "init_set": "eval",
    }
    assert data["final"]["k"] == 19 and data["final"]["init_set_version"] == "v1"
    budget = data["budget"]
    assert budget["robot_minutes"] == {"search": 2.0, "train": 0.0, "eval": 1.0}
    assert budget["human_minutes"]["demos"] == 2.5 and len(budget["human_minutes"]) == 5
    assert set(budget["compute"]) == {"cpu_hours", "gpu_hours", "wall_minutes"}
    assert data["config"] == {"lr": 0.1, "iters": 5} and data["extra"]["best_knobs"]["grip_height"] == 0.02
    assert isinstance(data["git_sha"], str) and data["git_sha"]

    loaded = load_results(tmp_path)
    assert len(loaded) == 1 and loaded[0]["final"] == data["final"]


def test_nothing_written_on_crash(tmp_path):
    with pytest.raises(ValueError), Ledger("ch01", "broken", results_root=tmp_path):
        raise ValueError("boom")
    assert load_results(tmp_path) == []


def _result(successes, init_set, first_seed=0):
    n = len(successes)
    return EvalResult(successes=np.array(successes), steps_to_success=np.full(n, -1), near_miss=np.zeros(n),
                      env_steps=0, seeds=np.arange(first_seed, first_seed + n), policy_seeds=np.arange(n),
                      init_set=init_set)


def test_unknown_robot_category_is_rejected(tmp_path):
    ledger = Ledger("ch09", "typo", results_root=tmp_path)
    with pytest.raises(KeyError):
        ledger.robot_steps["serach"] += 100  # a typo must not become a category the leaderboard ignores
    ledger.robot_steps["demos"] = 100  # plain assignment slips past the dict, so write() checks again
    with pytest.raises(ValueError, match="unknown robot-minute category"):
        ledger.write()


def test_scores_take_init_set_from_the_eval_result(tmp_path):
    ledger = Ledger("ch01", "ars", results_root=tmp_path)
    ledger.set_base("knobs_default", _result([True, False] * 32, "search"))
    assert ledger.base["init_set"] == "search" and ledger.base["n"] == 64
    with pytest.raises(ValueError, match="contradicts"):
        ledger.set_base("knobs_default", _result([True] * 4, "search"), init_set="eval")


def test_base_and_final_must_be_the_same_episodes(tmp_path):
    ledger = Ledger("ch01", "ars", results_root=tmp_path)
    ledger.set_base("knobs_default", _result([True, False] * 32, "search"))
    with pytest.raises(ValueError, match="differ in init_set"):
        ledger.set_final(_result([True] * 64, "eval"))

    ledger = Ledger("ch01", "ars", results_root=tmp_path)
    ledger.set_base("knobs_default", _result([True] * 8, "eval"))
    with pytest.raises(ValueError, match="differ in n"):
        ledger.set_final(_result([True] * 16, "eval"))
    with pytest.raises(ValueError, match="different env seeds"):
        ledger.set_final(_result([True] * 8, "eval", first_seed=100))
    ledger.set_final(_result([True] * 8, "eval"))
    assert ledger.final["k"] == 8


def test_results_config_uses_repo_relative_paths():
    from lastmile.common.ledger import REPO_ROOT, _portable

    cfg = {"results_root": str(REPO_ROOT / "results"), "runs": [str(REPO_ROOT / "runs" / "x")], "k": 3, "other": "/tmp/y"}
    assert _portable(cfg) == {"results_root": "results", "runs": ["runs/x"], "k": 3, "other": "/tmp/y"}


@pytest.mark.parametrize("outcomes", [[[True, False], [False, False]], [], [float("nan")], [0.2, 1.0]])
def test_malformed_outcomes_cannot_publish_a_plausible_score(tmp_path, outcomes):
    ledger = Ledger("test", "malformed", results_root=tmp_path)
    with pytest.raises(ValueError, match="non-empty 1-D array of binary outcomes"):
        ledger.set_final(outcomes)
    assert ledger.final is None


def test_eval_result_seed_count_must_match_outcomes(tmp_path):
    result = _result([True, False], "eval")
    result.seeds = np.array([20_000])
    with pytest.raises(ValueError, match="one entry per outcome"):
        Ledger("test", "malformed", results_root=tmp_path).set_final(result)


@pytest.mark.parametrize("seeds", [[20_000.5, 20_001.5], [float("nan"), 20_001],
                                   [[20_000], [20_001]], [True, False]])
def test_eval_result_seeds_cannot_be_silently_coerced_for_pairing(tmp_path, seeds):
    result = _result([True, False], "eval")
    result.seeds = np.asarray(seeds)
    with pytest.raises(ValueError, match="1-D array of integers"):
        Ledger("test", "malformed", results_root=tmp_path).set_final(result)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("category", ["robot", "human", "cpu", "gpu"])
def test_invalid_costs_cannot_be_written(tmp_path, category, value):
    ledger = Ledger("test", "malformed", results_root=tmp_path)
    if category == "robot":
        ledger.robot_steps["train"] = value
    elif category == "human":
        ledger.human_minutes["demos"] = value
    elif category == "cpu":
        ledger.worker_cpu_seconds = value
    else:
        ledger.gpu_hours = value
    with pytest.raises(ValueError, match="finite and non-negative"):
        ledger.write()
    assert not list(tmp_path.rglob("*.json"))
