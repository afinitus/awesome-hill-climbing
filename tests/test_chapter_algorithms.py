"""Regression checks for policy selection in the executable teaching algorithms."""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def load_chapter(filename: str):
    name = "chapter_test_" + filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "algos" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_qc_backup_values_the_same_candidate_as_the_acting_policy():
    chapter = load_chapter("05_rlpd_qchunk.py")
    # Online mean chooses candidate 0 in row 0, candidate 1 in row 1. The target-pair minimum
    # prefers the opposite candidate in both rows: maximizing it would change the backed-up policy.
    online = torch.tensor([[[9.0, 2.0], [1.0, 7.0]], [[5.0, 4.0], [3.0, 5.0]]])
    target = torch.tensor([[[0.2, 0.9], [0.8, 0.3]], [[0.1, 0.7], [0.6, 0.4]]])
    torch.testing.assert_close(chapter.qc_bootstrap_value(online, target), torch.tensor([0.1, 0.3]))


def test_qc_backup_ties_choose_first_candidate_as_acting_argmax_does():
    chapter = load_chapter("05_rlpd_qchunk.py")
    online = torch.tensor([[[4.0, 1.0]], [[0.0, 3.0]]])
    target = torch.tensor([[[0.2, 0.9]], [[0.3, 0.8]]])
    torch.testing.assert_close(chapter.qc_bootstrap_value(online, target), torch.tensor([0.2]))


@pytest.mark.parametrize("filename", ["03_golden_ticket.py", "06_residual_td3.py", "07_dsrl.py"])
def test_replot_preserves_saved_quick_run_provenance_and_media_root(filename, tmp_path, monkeypatch):
    chapter = load_chapter(filename)
    config = asdict(chapter.Config(quick=True))
    if filename == "06_residual_td3.py":
        config["eval_set"] = "search"
    saved = {"config": config, "final": {"init_set": "search"},
             "extra": {"analysis": {"marker": "saved"}, "config": config}}
    path = tmp_path / "quick.json"
    path.write_text(json.dumps(saved))
    plots = Mock()
    monkeypatch.setattr(chapter, "make_plots", plots)
    if filename == "03_golden_ticket.py":
        gifs = Mock()
        monkeypatch.setattr(chapter, "tickets_gif", gifs)
    # The command's default quick=False must not switch saved search outcomes to eval or public media.
    chapter.main(chapter.Config(replot=str(path)))
    plots.assert_called_once()
    out = plots.call_args.args[1]
    assert "runs" in out.parts and any(part.endswith("_quick") for part in out.parts)
    if filename == "03_golden_ticket.py":
        assert plots.call_args.kwargs["final_set"] == "search"
        assert gifs.call_args.kwargs["final_set"] == "search"
        assert gifs.call_args.args[1].quick is True
        assert gifs.call_args.args[2] == out


def test_residual_replot_respects_saved_custom_output_root(tmp_path, monkeypatch):
    chapter = load_chapter("06_residual_td3.py")
    out = tmp_path / "pilot"
    saved = {"extra": {"config": asdict(chapter.Config(out=str(out)))}}
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(saved))
    plots = Mock()
    monkeypatch.setattr(chapter, "make_plots", plots)
    chapter.main(chapter.Config(replot=str(path)))
    assert plots.call_args.args[1] == out / "media/ch06"


def test_bc_accounts_for_each_training_worker_cpu_time(tmp_path):
    from lastmile.common.ledger import Ledger

    chapter = load_chapter("00_bc_flow.py")
    futures = [Mock(), Mock()]
    futures[0].result.return_value = {"loss": 0.7, "cpu_seconds": 3.25}
    futures[1].result.return_value = {"loss": 0.5, "cpu_seconds": 5.5}
    pool = Mock()
    pool.submit.side_effect = futures
    ledger = Ledger("test", "bc", results_root=tmp_path)
    ledger.worker_cpu_seconds = 2.0  # previously charged rollout CPU must be retained
    datasets = {seed: SimpleNamespace(obs=None, chunks=None) for seed in (0, 1)}
    paths = chapter.train_all(chapter.Config(), pool, datasets, 4000, "fixture", tmp_path, ledger)
    assert ledger.worker_cpu_seconds == 10.75
    assert set(paths) == {0, 1} and pool.submit.call_count == 2


def test_bc_update_budget_rounds_up_to_whole_epochs():
    chapter = load_chapter("00_bc_flow.py")
    # The saved base checkpoint has 1449 samples, six batches per epoch: 4000 requests become 4002 updates.
    assert chapter.epochs_for(1449, 4000, 256) == 667
