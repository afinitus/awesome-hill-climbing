"""Tests for tools/leaderboard.py, using synthetic results files (no simulation needed)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

LEADERBOARD_PY = Path(__file__).resolve().parents[1] / "tools" / "leaderboard.py"


@pytest.fixture(scope="module")
def lb():
    """Import tools/leaderboard.py by path (tools/ is a folder of scripts, not a package)."""
    spec = importlib.util.spec_from_file_location("leaderboard", LEADERBOARD_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def write_result(
    root: Path,
    chapter: str,
    method: str,
    k: int,
    n: int = 256,
    base_k: int | None = None,
    search: float = 0.0,
    train: float = 0.0,
    eval_: float = 0.0,
    human: float = 0.0,
    setting: str = "sim",
    **extra,
) -> Path:
    """Write a schema-1 results file the way lastmile.common.ledger does."""
    data = {
        "schema": 1,
        "chapter": chapter,
        "method": method,
        "robot": "so101",
        "variant": "v1",
        "setting": setting,
        "final": {"sr": k / n, "k": k, "n": n, "init_set": "eval", "init_set_version": "v1"},
        "budget": {
            "robot_minutes": {"search": search, "train": train, "eval": eval_},
            "human_minutes": {
                "demos": human,
                "labels": 0.0,
                "takeovers": 0.0,
                "resets": 0.0,
                "annotation": 0.0,
            },
            "compute": {"cpu_hours": 0.0, "gpu_hours": 0.0, "wall_minutes": 1.0},
        },
        "extra": {},
        "git_sha": "test",
        "timestamp": "2026-09-30T12:00:00",
        "config": {},
    }
    if base_k is not None:
        data["base"] = {"id": "base", "sr": base_k / n, "k": base_k, "n": n, "init_set": "eval"}
    data.update(extra)
    path = root / "results" / chapter / f"{method}_so101_{len(list(root.rglob('*.json')))}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def data_rows(table: str) -> list[list[str]]:
    """Split a markdown table into cells, skipping the header and alignment rows."""
    return [[c.strip() for c in line.strip("|").split("|")] for line in table.splitlines()[2:]]


def block(readme: Path, lb) -> str:
    text = readme.read_text()
    return text[text.index(lb.START) : text.index(lb.END)]


def test_wilson_matches_reference_numbers(lb):
    # The reference intervals printed in the curriculum (docs/research/curriculum.md §3).
    for (k, n), (lo, hi) in {
        (15, 30): (33.2, 66.8),
        (30, 30): (88.6, 100.0),
        (243, 256): (91.5, 97.0),
    }.items():
        got = lb.wilson(k, n)
        assert round(100 * got[0], 1) == lo and round(100 * got[1], 1) == hi


def test_table_rows_sorted_and_formatted(tmp_path, lb):
    write_result(tmp_path, "ch03", "random_search", k=200, base_k=128, search=40.0)
    write_result(tmp_path, "ch10", "hg_dagger", k=240, base_k=128, train=5.0, human=12.5)
    write_result(tmp_path, "ch01", "ars", k=248, base_k=105, search=12.0, train=0.5, eval_=99.0)
    write_result(tmp_path, "ch03", "cem", k=230, base_k=128, search=40.0)

    rows = data_rows(lb.build_table(lb.load_runs(tmp_path / "results")))

    # chapter order is natural (ch3 before ch10); inside a chapter the best final SR comes first
    assert [(r[0], r[1]) for r in rows] == [
        ("ch01", "ars"),
        ("ch03", "cem"),
        ("ch03", "random_search"),
        ("ch10", "hg_dagger"),
    ]
    ars = dict(zip(lb.COLUMNS, rows[0]))
    assert ars["Final SR [CI]"].startswith("96.9% [")  # 248/256
    assert ars["Base SR [CI]"].startswith("41.0% [")  # 105/256
    assert ars["Δ"] == "+55.9 pp"
    assert ars["Improvement robot-min"] == "12.5"  # search + train; the 99 eval minutes are excluded
    assert ars["Human-min"] == "0"
    assert ars["n eval"] == "256"
    assert dict(zip(lb.COLUMNS, rows[3]))["Human-min"] == "12.5"


def test_missing_optional_fields_are_tolerated(tmp_path, lb):
    path = tmp_path / "results" / "ch12" / "minimal.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"chapter": "ch12", "method": "best_of_8", "final": {"k": 30, "n": 30}}))

    (row,) = data_rows(lb.build_table(lb.load_runs(tmp_path / "results")))
    cells = dict(zip(lb.COLUMNS, row))
    assert cells["Final SR [CI]"] == "100.0% [88.6, 100.0]"  # CI computed from k and n
    assert cells["Base SR [CI]"] == "—" and cells["Δ"] == "—"
    assert cells["Improvement robot-min"] == "n/a"  # unknown cost is not the same as zero cost


def test_hardware_rows_never_print_100_percent(tmp_path, lb):
    write_result(tmp_path, "ch03", "golden_ticket", k=30, n=30, base_k=15, search=18.0, setting="real")
    (row,) = data_rows(lb.build_table(lb.load_runs(tmp_path / "results")))
    cells = dict(zip(lb.COLUMNS, row))
    assert cells["Final SR [CI]"] == "30/30 [88.6, 100.0]"
    assert "100.0%" not in " ".join(row)


def test_hardware_rows_without_counts_are_skipped(tmp_path, lb, capsys):
    path = tmp_path / "results" / "ch03" / "real.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"chapter": "ch03", "method": "t", "setting": "real", "final": {"sr": 1.0}}))
    assert lb.load_runs(tmp_path / "results") == []
    assert "must report k and n" in capsys.readouterr().err


def test_malformed_and_quick_files_are_skipped(tmp_path, lb, capsys):
    write_result(tmp_path, "ch01", "ars", k=200)
    write_result(tmp_path, "ch01", "smoke", k=10, n=16, config={"quick": True})
    bad = tmp_path / "results" / "ch01"
    (bad / "broken.json").write_text("{not json")
    (bad / "list.json").write_text("[1, 2, 3]")
    (bad / "no_final.json").write_text(json.dumps({"chapter": "ch01", "method": "x"}))
    (bad / "bad_sr.json").write_text(json.dumps({"chapter": "ch01", "method": "y", "final": {"sr": 3.0}}))

    runs = lb.load_runs(tmp_path / "results")

    assert [r.method for r in runs] == ["ars"]
    err = capsys.readouterr().err
    assert err.count("warning: skipping") == 4
    assert "--quick" in err


def test_zero_results_write_an_explanatory_row(tmp_path, lb):
    readme = tmp_path / "README.md"
    lb.main(["--root", str(tmp_path), "--no-plots"])
    text = block(readme, lb)
    assert "No results yet" in text
    assert len(data_rows(text.split("\n", 1)[1])) >= 1


def test_marker_replacement_is_idempotent(tmp_path, lb):
    write_result(tmp_path, "ch01", "ars", k=248, base_k=105, search=12.0)
    readme = tmp_path / "README.md"
    readme.write_text(f"# lastmile\n\nIntro.\n\n{lb.START}\nstale table\n{lb.END}\n\nFooter.\n")

    assert lb.main(["--root", str(tmp_path), "--no-plots"]) == 0
    first = readme.read_text()
    lb.main(["--root", str(tmp_path), "--no-plots"])

    assert readme.read_text() == first
    assert "stale table" not in first
    assert first.startswith("# lastmile\n\nIntro.\n\n") and first.endswith("\n\nFooter.\n")
    assert first.count(lb.START) == 1 and first.count(lb.END) == 1


def test_broken_markers_are_reported_not_mangled(tmp_path, lb):
    readme = tmp_path / "README.md"
    readme.write_text(f"# lastmile\n{lb.END}\n{lb.START}\n")
    assert lb.main(["--root", str(tmp_path), "--no-plots"]) == 1
    assert readme.read_text() == f"# lastmile\n{lb.END}\n{lb.START}\n"


def test_markers_are_added_when_missing(tmp_path, lb):
    readme = tmp_path / "README.md"
    readme.write_text("# lastmile\n")
    lb.main(["--root", str(tmp_path), "--no-plots"])
    text = readme.read_text()
    assert text.startswith("# lastmile\n") and lb.START in text and lb.END in text


def test_plots_are_created_and_linked(tmp_path, lb):
    write_result(tmp_path, "ch01", "ars", k=248, base_k=105, search=12.0)
    write_result(tmp_path, "ch03", "golden_ticket", k=200, base_k=128)  # zero cost -> drawn at the floor
    write_result(tmp_path, "ch03", "golden_ticket", k=27, n=30, search=18.0, human=6.0, setting="real")

    lb.main(["--root", str(tmp_path)])

    for name in lb.PLOT_FILES.values():
        png = tmp_path / "media" / "leaderboard" / name
        assert png.exists() and png.stat().st_size > 1000
        assert f"](media/leaderboard/{name})" in (tmp_path / "README.md").read_text()


def test_many_runs_switch_to_one_panel_per_chapter(tmp_path, lb):
    """Past MAX_LABELLED runs the per-point names would overprint, so the plot facets by chapter."""
    for i in range(lb.MAX_LABELLED + 1):
        write_result(tmp_path, f"ch0{i % 4}", f"m{i}", k=100 + i, base_k=100, search=float(i))
    runs = lb.load_runs(tmp_path / "results")
    assert len(runs) > lb.MAX_LABELLED
    for png in lb.save_plots(runs, tmp_path / "plots"):
        assert png.exists() and png.stat().st_size > 1000


def test_plots_with_zero_results(tmp_path, lb):
    assert len(lb.save_plots([], tmp_path / "plots")) == 2


def test_variants_are_labelled_and_sorted_after_v1(tmp_path, lb):
    """A run on the hard variant is a different task: it must not look like a v1 row."""
    write_result(tmp_path, "ch01", "ars", k=120, base_k=100, variant="hard")
    write_result(tmp_path, "ch01", "ars", k=200, base_k=100)
    write_result(tmp_path, "ch01", "cem", k=90, base_k=100)

    rows = [dict(zip(lb.COLUMNS, r)) for r in data_rows(lb.build_table(lb.load_runs(tmp_path / "results")))]

    assert [(r["Method"], r["Robot"]) for r in rows] == [("ars", "so101"), ("cem", "so101"), ("ars", "so101 (hard)")]


def test_old_init_set_version_is_tagged(tmp_path, lb, capsys):
    from lastmile.common.eval import INIT_SET_VERSION

    assert lb.INIT_SET_VERSION == INIT_SET_VERSION  # the tool keeps its own copy to stay dependency-free
    path = write_result(tmp_path, "ch01", "ars", k=200)
    data = json.loads(path.read_text())
    data["final"]["init_set_version"] = "v0"
    path.write_text(json.dumps(data))

    (row,) = data_rows(lb.build_table(lb.load_runs(tmp_path / "results")))

    assert dict(zip(lb.COLUMNS, row))["n eval"] == "256 (init sets v0)"
    assert "init sets 'v0'" in capsys.readouterr().err


def test_unpaired_base_gets_no_delta(tmp_path, lb):
    path = write_result(tmp_path, "ch01", "ars", k=200, base_k=100)
    data = json.loads(path.read_text())
    data["base"].update(n=64, k=25, sr=25 / 64, init_set="search")
    path.write_text(json.dumps(data))

    (row,) = data_rows(lb.build_table(lb.load_runs(tmp_path / "results")))

    assert dict(zip(lb.COLUMNS, row))["Δ"] == "n/a (unpaired)"


def test_every_non_eval_category_counts_as_improvement_budget(tmp_path, lb):
    path = write_result(tmp_path, "ch09", "typo", k=200, search=1.0, eval_=50.0)
    data = json.loads(path.read_text())
    data["budget"]["robot_minutes"]["serach"] = 12.0  # hand-edited or pre-validation file
    path.write_text(json.dumps(data))

    (run,) = lb.load_runs(tmp_path / "results")

    assert run.robot_minutes == 13.0
