# Pre-launch review brief: Awesome Hill Climbing

> Historical assignment, preserved for review provenance. Its opening status and counts describe the repository before this review. See [REPORT.md](REPORT.md) for the current findings, completed coverage, checks and launch recommendation.

You are a fresh pair of eyes. The repo is about to be announced on X (Twitter). Your job is to find anything that is **wrong, unfair or controversial, overclaimed, legally risky, or broken**, and to confirm that the experiments in the course were done correctly. Fix what you can, report the rest, and end with a go/no-go.

- Repo: https://github.com/afinitus/awesome-hill-climbing (work on branch `review/prelaunch`)
- Live explainer site: https://afinitus.github.io/awesome-hill-climbing/
- This brief: `docs/maintenance/review/PRELAUNCH-REVIEW.md`

## Status (updated 2026-10-01)

**P0 is done** on `main` (commit "Pre-launch P0 fixes", merged into this branch). An automated pass re-checked every P0 surface against primary sources and applied 395 changes. Each change was confirmed by an independent skeptic, and fresh reviewers then re-read the README, the site and every company mention three times. It covered the must-know, industry and What's-new entries, the family intros, the whole site narrative, the README claims and Start-here blurbs, the foundations table, and the course headlines. Chapter 7 is now pooled PSS (81.0%) everywhere. It also added safety notes for the real-arm steps and removed a dead link.

**Your job:**
1. Spot-check P0. Read the README top and the site as an X reader would. Sample about 20 changed rows from `git show f076f75 -- data/papers.csv` and check them against their sources. Read every company mention once.
2. Then do **P1** (chapter experiments, harness) and **P2** (remaining entries, resources) below. These were never audited.

## What the repo is

1. **An "Awesome" list** of papers, posts and resources on *hill climbing* robot policies: taking a pretrained or imitation-learned policy from ~50% to 95–99%+ success with RL fine-tuning, residual and steering policies, advantage-weighted retraining, human corrections, test-time verifiers, reward and world models, sim-to-real, and black-box search. It has 912 papers/posts in 12 families and 213 resources. It is **generated from data**:
   - `data/papers.csv`, `data/resources.csv`: one row per entry (`one_line` is the description readers see)
   - `data/families.json`: the intro text of each family
   - `data/foundations.csv`: the "Foundations" table
   - `data/explainer.json`: the narrative shown on the site (thesis, loop, decision guide, timeline, lessons, industry recipes, open problems)
   - `tools/build_awesome.py` → `README.md` and `papers/<family>.md`; `tools/build_site.py` → `site/index.html` (deployed to GitHub Pages)
2. **A hands-on course, `lastmile`**: a MuJoCo cube-in-cup task on a simulated SO-101 arm (AgileX PiPER also supported), a frozen base policy at 50.4%, and chapters that try to climb it (`algos/NN_*.py`, `docs/chapters/chNN.md`, results in `results/chNN/*.json`). Chapters 0, 1, 2, 3, 5, 6, 7, 8 and 12 exist. The contract is `docs/DESIGN.md`, and the honesty rules are in `CONTRIBUTING.md`.

Much of it was researched and written with AI assistance (disclosed in the README). That is exactly why it needs an independent human-quality check.

## Ground rules

- **Verify against primary sources** (the paper, the official post, the results JSON). Never fix a number from memory. If a number cannot be verified, rewrite the line using only what the source supports.
- **Neutral and fair.** Describe what papers and companies report, not what we think of them. "The post does not disclose X" is fine. "They hid X" is not. Unconfirmed links (e.g. "company A's method is probably based on paper B") must not be stated as fact.
- **No silent result changes.** If you find a bug in an experiment, fix it, re-run the affected experiment, and update results JSON, plots and doc together. Never edit a number in a doc without the results to back it.
- **Never hand-edit generated files** (`README.md`, `papers/*.md`, `site/index.html`). Edit `data/` or the generator, then regenerate.
- Keep descriptions concrete and short: mechanism first, then the single most telling result *with its baseline and setting*, ≤ 40 words.

## Setup (macOS or Linux, no GPU needed)

```bash
git clone https://github.com/afinitus/awesome-hill-climbing && cd awesome-hill-climbing
git checkout review/prelaunch
make setup          # needs uv: https://docs.astral.sh/uv/  (creates .venv.nosync + .venv symlink)
make test           # ~30 s
make smoke          # every chapter script with --quick, ~5 min
```

After any data or text change:

```bash
python3 tools/build_awesome.py && uv run python tools/build_site.py
python3 tools/build_awesome.py --check && uv run python tools/build_site.py --check
uv run python tools/arxiv_meta.py --dry-run    # every arXiv id, title and date vs the arXiv API
uv run python tools/check_links.py             # every other link
```

## Review checklist, by priority

### P0: what an X reader sees first (do all of this)

1. **README top**: title, banner, the stats line, the "What's new", "Start here", "The hill-climbing loop" and "Foundations" sections. Check every count against `data/` (recompute it), every process claim ("every arXiv link validated", "re-checked against the source", "updated weekly") against `tools/` and `docs/maintenance/`, and every claim about a paper against the paper.
2. **The site** (https://afinitus.github.io/awesome-hill-climbing/): every section. The narrative comes from `data/explainer.json` and the family text from `data/families.json`: verify each factual claim, number and attribution.
3. **Everything said about companies and labs**: the "Industry" family (`papers/industry.md`, rows with `family=industry`), the "How companies climb" section of the site (`industry_recipes` in `data/explainer.json`), and any sentence naming Sunday Robotics, Physical Intelligence, Generalist AI, Dyna Robotics, Figure, Skild AI, Siemens, AgiBot, 1X, Google DeepMind, NVIDIA, Amazon or Stanford. Each claim must match the company's own post/paper, be neutrally worded, and separate "reported" from "independently verified".
4. **Course headline numbers** in the README table, the site's course table and `docs/chapters/README.md`: each must equal what the chapter doc leads with, computed from `results/`. They must be pooled over seeds or chosen on the search set, and **never the best seed picked by its held-out score**. (Fixed in P0: Chapter 7 now shows pooled PSS over 3 seeds everywhere.)
5. **Legal and safety**: LICENSE and the vendored MuJoCo Menagerie model licenses (`lastmile/envs/assets/*/LICENSE`); no long verbatim copies of abstracts or blog text in `one_line` (flag more than about 12 consecutive copied words); no logos or implied endorsements; no secrets, tokens or local paths; the AI-assistance disclosure is present and accurate; real-arm steps in the chapter docs carry a basic safety caution (e-stop, force/speed limits, clear workspace).

### P1: substance

6. **All must-know entries (⭐, `key=1`, 122 rows) and all industry rows**: title, date (arXiv v1), affiliation, family, every number in `one_line` (baseline and sim/real setting right), and the code link.
7. **Each chapter's experiment** (`algos/NN_*.py`, `docs/chapters/chNN.md`, `results/chNN/`):
   - Correctness against the cited papers and textbook RL: TD targets, done/truncation, γ^h for chunks, target networks, the same argmax for acting and for the TD target, the critic on the executed action, zero-init residuals, noise bounds, the CMA-ES/ARS/PI² update formulas, sequential halving, verifier training without label leakage, seeds that really differ.
   - **Eval isolation**: the held-out eval set (seeds 20000–20255) and eval_ext (30000–31023) must never be used for any choice. Trace the code. Every search and training rollout must be charged in the ledger.
   - **Re-derive** every k/n, Wilson interval, pooled number and McNemar p in the doc from `results/chNN/*.json` (use `lastmile.common.eval`).
   - **Reproduce**: run `--quick` (under 2 min; writes nothing into `results/` or `media/`), and re-evaluate at least one headline from saved artifacts (knobs and tickets are stored in the results JSON `extra`; `checkpoints/base_v1_so101.pt` is committed) with the documented seeds, and confirm the k/n matches exactly.
   - Every sentence stating a result is supported; "What went wrong" is honest.
8. **Shared harness** (`lastmile/`): no overlap between the search/eval/eval_ext/stress init sets and the demo/training seed ranges any chapter uses (grep every seed range in `algos/`); common-random-number pairing; episodes not dropped or reordered; success counted once; `get_state`/`set_state` exact; the Wilson/McNemar/bootstrap formulas. Also try to break the task: a trained policy must not succeed by pushing the cube up the cup wall, shoving the cup, or tunnelling.

### P2: breadth

9. The remaining paper rows. At minimum, check every row added in the last two weeks (dates ≥ 2026-09-16, mostly `verified=read-once`) and a random 1-in-5 sample of the rest. Report the error rate you see, so the owner knows how much to trust the untouched rows.
10. Resources (`data/resources.csv`): the right official page/repo, the link resolves, and the description is accurate.

## Leads from an interrupted automated pass (unverified)

An automated audit checked every entry and the narrative before it was stopped. It never reached its verification step, and its tone, chapter and harness audits never ran. Use its output as **leads to confirm or reject**, not as truth:

- `docs/maintenance/review/pass1-other-candidates.md`: (resolved in P0; kept as a note.) It held 87 candidate issues in `data/explainer.json`, `data/families.json`, `data/foundations.csv` and the site generator, majors first (21 major). Several look real, for example: family "What changed" text cut off after item (1) of a numbered list; an internal catalog id ("C155") visible to readers; claims about specific papers (ReinFlow/FPO training VLAs, VIP being "per-task", which world models were used) that the sources contradict; the RECAP espresso gain quoted against the wrong baseline; statements about LWD and 1X disclosures that appear wrong.
- `docs/maintenance/review/pass1-entry-candidates.csv`: the remaining candidate fixes to individual rows that P0 did not cover (rows dated before 2026-09-16 that are not must-know or industry, plus resources) (GitHub renders it as a searchable table), each with the problem, evidence and proposed fields. Expect many to be nitpicks (truncated affiliations, nuance in mechanism wording) and some to be wrong. Confirm each one you apply against the source.

Delete both candidate files before merging, once they are processed.

## Deliverables

1. **`docs/maintenance/review/REPORT.md`** on this branch: every issue you found or confirmed, with severity (critical / major / minor), location, problem, evidence (quote or numbers plus source link) and what you did. Also include the sampled error rate for P2, and a short **go/no-go** with any residual risks.
2. **Fixes as commits on `review/prelaunch`**, with regenerated files and these passing: `make test`, `make smoke`, both `--check` commands, `arxiv_meta.py --dry-run` and `check_links.py`. Commit messages say what changed and why.
3. Open a pull request into `main`. The owner reviews and merges, and the Pages site redeploys automatically on merge. A scheduled job adds new papers to `main` every Monday, so merge `main` into your branch (and regenerate) before opening the PR.

Severity guide: **critical**: a false or unfair statement about a named company or person, a legal problem, or a headline result that is wrong. **major**: a wrong number or attribution, an overclaim, or a broken page section. **minor**: wording, truncation or style.

Rough time: P0 2–3 h, P1 4–6 h, P2 depends on the sample size.

---

### If the reviewer is an AI coding agent

Paste this as the task:

> Clone https://github.com/afinitus/awesome-hill-climbing, check out the branch `review/prelaunch`, read `docs/maintenance/review/PRELAUNCH-REVIEW.md`, and carry it out completely, in priority order (P0, then P1, then P2). Verify every claim against primary sources before changing anything, follow the ground rules, run every check listed, commit fixes on `review/prelaunch`, write `docs/maintenance/review/REPORT.md` with a go/no-go, and open a pull request into `main`.
