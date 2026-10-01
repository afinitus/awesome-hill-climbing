"""Regression checks for policy selection in the executable teaching algorithms."""

import importlib.util
import sys
from pathlib import Path

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
