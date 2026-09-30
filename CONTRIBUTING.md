# Contributing

This repo is two things: the **Awesome Hill Climbing list** (README, `papers/`, `data/`) and the **`lastmile` course code** (`lastmile/`, `algos/`). Both have rules.

## Adding to the list

The README and the pages under `papers/` are generated. Do not edit them by hand.

1. Add a row to [`data/papers.csv`](data/papers.csv) (papers, blog posts, talks) or [`data/resources.csv`](data/resources.csv) (code, simulators, benchmarks, courses, books, surveys).
2. Run `make awesome` and commit the data file together with the regenerated pages. CI fails if they are out of sync.

Columns of `data/papers.csv`:

| Column | Meaning |
|---|---|
| `id` | Any unique id. Use the next free `N###`. |
| `date` | First public version, `YYYY-MM-DD` (arXiv v1 date). `make links` fills and checks it for arXiv entries. |
| `title`, `short` | Full title, and the short name shown in the list (usually the method's name). |
| `url`, `code` | Canonical link (`https://arxiv.org/abs/<id>` for papers) and the code link if there is one. |
| `org` | Short affiliation list. Leave blank to get "First-author et al." from arXiv. |
| `family` | One key from [`data/families.json`](data/families.json): `blackbox`, `onpolicy`, `offpolicy`, `residual`, `advbc`, `hitl`, `testtime`, `reward`, `worldmodel`, `simreal`, `industry`, `generalrl`. |
| `kind` | `paper`, `blog`, `talk` or `thread`. |
| `key` | `1` for must-know work (shown with ⭐ on the front page). Use sparingly and say why in the PR. |
| `verified` | `read-once` for a new entry. Maintainers change it to `checked` or `corrected` after a second pass against the source. |
| `one_line` | 40 words at most: the mechanism first, then the single most telling result **with its baseline and setting**, copied exactly from the paper. No hype. Say "chart reading" if the number comes from a figure. |

What belongs: work about improving a robot policy that already exists (RL fine-tuning, residuals, steering, weighted BC, corrections, test-time search, reward and world models used for improvement or evaluation, sim-to-real fine-tuning, black-box search), the classic ideas underneath, and general RL results that robot work builds on. What does not: new pretraining recipes or architectures with no improvement loop, and pure benchmarks of base models.

# The `lastmile` code

lastmile is a course as much as a codebase: one robot, one task, one base policy that works about half the time, and a sequence of methods that climb it toward 95%+. People will read the code to learn from it and quote the numbers, so every contribution has to be readable and every number has to be one you would defend. [docs/DESIGN.md](docs/DESIGN.md) is the contract the code follows; this file covers the rules for contributions and the workflow.

## Setup

You need [uv](https://docs.astral.sh/uv/) and macOS or Linux. No GPU is needed.

```bash
make setup      # uv sync with the dev and bo extras
```

The virtualenv lives in `.venv.nosync`, and `.venv` is a symlink to it. The repo often sits on a Mac Desktop that iCloud syncs, and iCloud skips folders whose names end in `.nosync`, so thousands of venv files never get uploaded. Always run Python through uv (`uv run python ...`, `uv run pytest`).

## Running things

| Command | What it does |
|---|---|
| `make test` | The fast test suite (under a minute). |
| `make validate` | Checks the environment against the design contract (`tools/validate_env.py`). |
| `make smoke` | Runs every `algos/[0-9]*.py` with `--quick`. Each must finish in under 2 minutes. CI runs the same target. |
| `make leaderboard` | Rebuilds the README table and `media/leaderboard/` plots from `results/**/*.json`. |
| `make awesome` | Rebuilds `README.md` and `papers/*.md` from `data/`. |
| `make links` | Checks every arXiv link, title and date in `data/papers.csv` against the arXiv API. |
| `uv run python algos/03_golden_ticket.py --help` | Shows one chapter's options. |

CI (`.github/workflows/ci.yml`) runs `make test` and `make smoke` on Ubuntu and macOS for every pull request.

## The honesty rules

Most last-mile results in the literature are hard to compare because they select on the test set, hide their search budget, or report 30/30 as "100%". These rules exist so our numbers do not have those problems. They apply to code, printed summaries, plots, chapter docs and posts about the repo.

1. **Search and evaluation use disjoint initial states.** Anything you choose (knobs, a noise ticket, a checkpoint, a hyperparameter, N in best-of-N) is chosen on the `search` set only (64 seeds, 10 000–10 063). `eval` (256 seeds) is held out and never used for selection; `eval_ext` (1024) is for final claims. If a chapter also shows a number selected on `eval`, label it *optimistic* and print it next to the held-out one, as Chapter 3 does to show the gap.
2. **Every success rate has a Wilson 95% interval,** in tables, plots, printed summaries and docs. Use `lastmile.common.eval.wilson` (or `format_rate`), not a normal approximation, which breaks at 0/n and n/n. Reference values:

   | k/n | Wilson 95% |
   |---|---|
   | 15/30 | [33.2, 66.8] |
   | 27/30 | [74.4, 96.5] |
   | 30/30 | [88.6, 100] |
   | 57/60 | [86.3, 98.3] |
   | 243/256 | [91.5, 97.0] |

3. **Report single-attempt success next to any pass@k.** pass@k uses simulator resets, so it is the ceiling for selection methods, not a success rate you could deploy.
4. **Include the unsteered and the random-selection baselines.** A search or steering method is compared with the base sampled normally *and* with a randomly chosen candidate (a random ticket, a random pick of N) on the same init set. Without both you cannot tell whether the method did the work or the selection did.
5. **Count search episodes in the budget.** Every rollout except the final held-out evaluation is improvement budget, charged to `search` or `train` (pass `ledger=L, category="search"` to `evaluate`). Evaluations of the base and the final policy are charged to `eval`. The leaderboard's cost column is search + train, so hidden search shows up as a suspiciously cheap row.
6. **Show at least one failure per chapter.** A regression, a seed that diverges, a setting where the method does not help: put it in `docs/chapters/chNN.md` with its numbers. A chapter where everything works teaches less and is harder to believe.
7. **Never write "100%" for real hardware.** Report k/n and the interval: 30/30 means "at least 88.6% with 95% confidence". A lower bound of 95% needs 73 successes in a row, 99% needs 381 (`min_successes_for_lower_bound`). In sim, "95%+" means a point estimate of at least 95% on `eval` (n = 256).
8. **Say explicitly that search and steering can only select behavior the base already has.** Tickets (Ch 3), noise-space RL (Ch 7) and best-of-N (Ch 12) reweight what the base can already produce; the base's support is their ceiling (pass@k as k grows without bound, not any finite pass@k). Methods that select again at every step (best-of-N per chunk, a noise actor) can beat any finite pass@k, because they combine good chunks from different samples into one episode, but they cannot solve a start state from which no sequence of base chunks succeeds. Chapters built on selection state this in their docstring and chapter doc.

Two related habits: comparisons between methods use the paired tests in `lastmile.common.eval` (`mcnemar_exact`, `bootstrap_diff`), because common random numbers make runs on the same set paired; and sim results use at least 3 training seeds, reported per seed and pooled.

## Algorithm files

Each method is one runnable file, `algos/NN_name.py`, in the style of CleanRL (DESIGN.md §0 and §8):

- **One file, about 150–450 lines,** readable top to bottom. It imports only from `lastmile/` (plus numpy, torch and the like). **No imports between algorithm files**: if two chapters need the same helper, it belongs in `lastmile/`, or it is short enough to copy.
- **Top docstring:** the learning goal, the base RL idea, the 2025–26 papers it mirrors (with ids from [data/papers.csv](data/papers.csv)), and how to run it.
- **A `@dataclass Config` at the top,** parsed with `lastmile.common.cli.parse(Config)` so `--help` works. Include `robot: str = "so101"` and `quick: bool = False`, plus a `QUICK: ClassVar[dict]` of overrides.
- **`--quick` finishes in under 2 minutes** on a laptop (CI runs it). A full run finishes in under 30 minutes unless the docstring says otherwise.
- **Scores come from `evaluate(...)` on named init sets, and costs go through a `Ledger`.** Never write a results JSON by hand. The ledger writes one JSON per method run to `results/chNN/`.
- **Outputs:** the results JSON, plots in `media/chNN/` (use `lastmile.common.plotting` so success plots show Wilson bands), and a short printed summary with intervals.
- **Laptop first:** everything runs on a Mac CPU (torch CPU or MPS). Anything that needs NVIDIA is an optional extra.

A minimal skeleton (`make_base` and `make_final` are picklable policy factories, see DESIGN.md §4):

```python
"""Chapter NN: <title>.

Learning goal: ...  Base idea: ...  Papers mirrored: ...
Run: uv run python algos/NN_name.py [--quick]
"""
from dataclasses import dataclass
from typing import ClassVar

from lastmile.common.cli import parse
from lastmile.common.ledger import Ledger
from lastmile.common.rollout import evaluate


@dataclass
class Config:
    robot: str = "so101"
    iterations: int = 50
    quick: bool = False
    QUICK: ClassVar[dict] = {"iterations": 3}


def main(cfg: Config) -> None:
    with Ledger(chapter="chNN", method="name", robot=cfg.robot, config=cfg) as L:
        base = evaluate(make_base, init_set="eval", robot=cfg.robot, ledger=L, category="eval")
        L.set_base(id="base_v1", successes=base.successes)
        ...  # search or train here, charging every rollout to "search" or "train"
        final = evaluate(make_final, init_set="eval", robot=cfg.robot, ledger=L, category="eval")
        L.set_final(successes=final.successes, init_set="eval")
    print(f"base {base.sr:.1%} {base.ci} -> final {final.sr:.1%} {final.ci}")


if __name__ == "__main__":
    main(parse(Config))
```

Runs made with `--quick` still write a results file, with `"quick": true` in its config. The leaderboard leaves those out.

## Adding a chapter

1. Pick the number and title from [docs/chapters/README.md](docs/chapters/README.md), or open an issue to propose a new method.
2. Write `algos/NN_name.py` following the rules above, including the baselines from honesty rule 4.
3. Run it in full. Commit the results JSON files from full runs only.
4. Run `make leaderboard` and commit the updated README table and plots.
5. Write `docs/chapters/chNN.md`: the idea in plain words, the math, the papers it mirrors (links and `C###` ids), our actual results with intervals and plots, what went wrong (honesty rule 6), and the hardware step.
6. Update the chapter's status in `docs/chapters/README.md`.
7. If you added reusable code to `lastmile/`, add a fast test for it in `tests/`, then run `make test smoke`.

## Adding a robot

The SO-101 is the default and the real-hardware target. The AgileX PiPER is the worked example of a second arm: compare `SO101` and `PIPER` in `lastmile/envs/robots.py`. To add another arm:

1. **Vendor the model** under `lastmile/envs/assets/<robot>/` with its upstream LICENSE and the source commit, as `so101/` and `piper/` were vendored from MuJoCo Menagerie. Do not edit the meshes or the upstream XML. Task-specific sites and cameras go in the spec's `extra_sites` and `extra_cameras`.
2. **Add a `RobotSpec`** to `lastmile/envs/robots.py` and register it in `ROBOTS`. It names the arm joints the IK controls, the gripper joint and actuator with their open and closed targets, a home pose with the gripper pointing down, the IK site (approach axis along its x axis), the workspace box, the cube region and cup position scaled to the arm's reach, the two jaw bodies used to detect a grasp, the wrist camera, the physics timestep and the joint offsets for `variant="bias"`.
3. **Check it.** Run `make validate` and the test suite with the new robot. The scripted `KnobController` must reach a high success rate after tuning (the ceiling check in the curriculum) before any chapter number on that robot means anything, and a 150-step episode must take under 0.25 s of CPU.
4. **Report it.** Results carry a `robot` field, and the leaderboard shows it in its own column. Base checkpoints are per robot (`checkpoints/base_v1_<robot>.pt`).

## Code style

- The code is written for learners. Short docstrings explain *why*, not just what. Use type hints. No dead code, and no abstractions a reader has to chase across files.
- `uv run ruff check .` should pass (line length 110).
- Do not add core dependencies without discussing it first. Heavy or platform-specific packages go in an optional extra in `pyproject.toml`.
- Keep `tests/` fast: the whole suite should stay under 60 seconds.

## Pull request checklist

- [ ] `make test` and `make smoke` pass.
- [ ] Every new success rate has a Wilson interval and names its init set.
- [ ] Nothing was selected on `eval` or `eval_ext` (or the optimistic number is labeled and shown next to the held-out one).
- [ ] Search and training rollouts are charged to the budget; the unsteered and random-selection baselines are included.
- [ ] Results JSON files come from full runs, and `make leaderboard` was rerun.
- [ ] The chapter doc shows at least one failure.
- [ ] Hardware numbers are k/n with an interval, never "100%".
