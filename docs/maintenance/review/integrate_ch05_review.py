"""Publish the corrected full Chapter 5 run and preserve its predecessor.

The run started before the prior-run accounting decision. This one-time, checked
migration adds only the named development budget; outcome counts, CIs, paired
statistics and learning curves remain exactly as emitted by the training run.
Run without --apply to inspect the plan; --apply archives, replaces and copies the newly emitted plots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIOR_NAME = "prelaunch_invalid_full_run"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    old_dir = ROOT / "results/ch05"
    new_root = ROOT / "runs/prelaunch_ch05"
    archive = ROOT / "docs/maintenance/review/archive/ch05_target_policy_bug"
    old_files, new_files = sorted(old_dir.glob("*.json")), sorted((new_root / "results/ch05").glob("*.json"))
    if archive.exists():
        raise RuntimeError("Archive already exists; refusing a second migration")
    assert len(old_files) == len(new_files) == 13, "Both complete runs must have 13 rows"
    assert {p.name for p in old_files}.isdisjoint(p.name for p in new_files), "Replacement filenames must differ"
    old = [json.loads(p.read_text()) for p in old_files]
    fresh = [json.loads(p.read_text()) for p in new_files]
    assert {r["method"] for r in old} == {r["method"] for r in fresh}
    assert all(r["config"]["quick"] is False and r["final"]["n"] == 256 for r in fresh)
    assert all(PRIOR_NAME not in r["extra"]["prior_steps"] for r in fresh), "Run already charges this prior"
    assert all(r["extra"] == old[0]["extra"] for r in old), "Old rows must share one analysis"
    assert all(r["extra"] == fresh[0]["extra"] for r in fresh), "Fresh rows must share one analysis"
    assert all(r["config"] == fresh[0]["config"] for r in fresh), "Fresh rows must share one configuration"
    for name in ("base_vs_qc.gif", "dip.png", "eval_per_seed.png", "learning_curves.png"):
        assert (new_root / "media/ch05" / name).is_file(), f"Missing fresh media: {name}"
    analysis = old[0]["extra"]
    old_prior = sum(v["train"] for v in analysis["prior_steps"].values())
    demo_steps = sum(v["demos"] for v in analysis["shared_steps"].values())
    warm_steps = sum(v["warmup"] for v in analysis["shared_steps"].values())
    total_row_train = sum(v["train"] for k, v in analysis.items() if k.startswith("cost_steps_"))
    unique_train = total_row_train - 12 * old_prior - 3 * demo_steps - 2 * warm_steps
    assert unique_train == 572_459 and demo_steps == 4_999
    assert all(sum(v["search"] for v in analysis["prior_steps"].values()) == v["search"]
               for k, v in analysis.items() if k.startswith("cost_steps_"))
    manifest = {
        "reason": "The previous QC target selected by target-pair minimum; acting selected by online-ensemble mean.",
        "replacement": "Same online-mean candidate argmax for acting and bootstrap; target pair values that candidate.",
        "old_files": {p.name: digest(p) for p in old_files},
        "fresh_training_files_before_accounting": {p.name: digest(p) for p in new_files},
        "source_sha256_at_integration": digest(ROOT / "algos/05_rlpd_qchunk.py"),
        "base_checkpoint_sha256": digest(ROOT / "checkpoints/base_v1_so101.pt"),
        "corrected_qc_seed0_checkpoint_sha256": digest(ROOT / "checkpoints/ch05_qc_s0_review.npz"),
        "source_provenance": {
            "raw_git_sha": "Ledger records HEAD at write time, excluding uncommitted edits. Raw git_sha fields are preserved and do not alone reproduce this rerun.",
            "selector_fix": "The process started with the uncommitted online-mean selector fix already applied.",
            "post_launch_source_changes": "Chapter 5 changes after launch were historical-cost constants/metadata and their comments, the fresh-cost plot label, and documentation comments; no later training or policy-selection semantics changed.",
            "reproduction_source": "Use the final reviewed algos/05_rlpd_qchunk.py matching source_sha256_at_integration, not the raw git_sha alone. Its historical charge is added explicitly to the raw run by this migration; plot labels are refreshed from the emitted curves.",
        },
        "command": "uv run python algos/05_rlpd_qchunk.py --out runs/prelaunch_ch05",
        "accounting": {
            "sum_of_old_12_row_train_steps": total_row_train,
            "subtract_12_duplicated_pilot_charges": 12 * old_prior,
            "subtract_3_extra_copies_of_shared_demos": 3 * demo_steps,
            "subtract_2_extra_copies_of_shared_warmup": 2 * warm_steps,
            "unique_prior_full_run_train_steps": unique_train,
            "new_prior_search_steps": 0,
            "prior_demo_human_minutes": demo_steps / 600,
            "eval": "Old eval remains diagnostic cost in the archived rows, excluded from improvement cost; no eval-driven selection was introduced.",
        },
        "outcomes": {r["method"]: {"old": o["final"]["k"], "new": r["final"]["k"]}
                     for r in fresh for o in old if o["method"] == r["method"]},
    }
    print(json.dumps({"rows": len(fresh), "prior_train_steps": unique_train,
                      "prior_train_minutes": unique_train / 600, "outcomes": manifest["outcomes"]}, indent=2))
    if not args.apply:
        return
    archive.mkdir(parents=True)
    for p in old_files:
        shutil.copy2(p, archive / p.name)
    raw_dir = archive / "corrected_before_accounting"
    raw_dir.mkdir()
    for p in new_files:
        shutil.copy2(p, raw_dir / p.name)
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (archive / "README.md").write_text(
        "# Superseded Chapter 5 run\n\n"
        "These original files preserve the run with the QC target-policy mismatch. "
        "They are excluded from the public results tree and leaderboard. "
        "The corrected_before_accounting/ subdirectory preserves the corrected rerun's raw emitted records, "
        "before the explicitly documented historical-development charge was added. "
        "The manifest records their hashes, the corrected rerun and deduplicated development cost. "
        "The historical measurements are unchanged; they do not describe the corrected algorithm.\n")
    for path, row in zip(new_files, fresh):
        before_final = json.dumps(row["final"], sort_keys=True)
        s = row["extra"]
        row["budget"]["robot_minutes"]["train"] += unique_train / 600
        row["budget"]["human_minutes"]["demos"] += demo_steps / 600
        for key, value in s.items():
            if key.startswith("cost_steps_"):
                value["train"] += unique_train
        s["prior_steps"][PRIOR_NAME] = {"search": 0, "train": unique_train}
        s["prior_human_minutes"] = {PRIOR_NAME: {"demos": demo_steps / 600}}
        s["prelaunch_review"] = {
            "qc_target": "online ensemble mean selects; target-pair minimum evaluates same candidate",
            "accounting_migration": "Added archived invalid full-run development cost after training; no outcomes or learned-policy values altered.",
            "archive": "docs/maintenance/review/archive/ch05_target_policy_bug/manifest.json",
        }
        assert json.dumps(row["final"], sort_keys=True) == before_final
        (old_dir / path.name).write_text(json.dumps(row, indent=2) + "\n")
    for p in old_files:
        p.unlink()
    for p in (new_root / "media/ch05").iterdir():
        if p.is_file():
            shutil.copy2(p, ROOT / "media/ch05" / p.name)
    print("Archived 13 original rows; published 13 corrected rows and fresh media.")


if __name__ == "__main__":
    main()
