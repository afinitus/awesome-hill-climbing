# Weekly paper sweep: the playbook

A scheduled agent runs this once a week to add the newest relevant papers, posts and code to the Awesome
Hill Climbing list. Follow the steps in order. Where a step says **stop**, do step 12 (stopping) and end the run.

The run works in its own git worktree, a clean checkout of `origin/main`. It never touches the main
checkout, where people and other agents may have uncommitted work.

**You may edit only** these files in the worktree: `data/papers.csv`, `data/resources.csv`, `data/meta.json`,
`docs/maintenance/sweep-log.md` and `docs/maintenance/seen-ids.txt`, plus the files the build tools regenerate
(`README.md`, `papers/`, `site/`). Never edit code, tests, chapters, `CONTRIBUTING.md` or this playbook, even to
fix a broken tool: log the problem instead. Never force-push, never rewrite pushed history, never set `key` to 1.

Honesty over volume: a week with zero additions is a fine result. Never invent, round or "improve" a number,
an affiliation or a code link. If you are not sure an item belongs, skip it and say so in the log.

## 0. Setup

These names are used below:

```bash
REPO=/path/to/your/checkout   # the main checkout; the scheduled task prompt gives the exact path
STATE=$HOME/.ahc-sweep        # kept between runs: worktrees, candidates, pending entries, unpushed log
TODAY=$(date +%F)             # e.g. 2026-10-07
WT=$STATE/wt-$TODAY           # this run's worktree
WORK=$STATE/runs/$TODAY       # this run's scratch files
```

Your shell may not keep variables or the working directory between commands, so the `$REPO`, `$STATE`,
`$TODAY`, `$WT`, `$WORK`, `$SINCE` and `$UNTIL` in the code blocks below are placeholders: write each command
with the literal paths and dates (also inside quoted strings such as the `python -c` in step 8).

Where commands run: the three commands at the top of step 1 run anywhere (they use absolute paths or
`git -C "$REPO"`). **Every other command in this playbook runs in the worktree**: start it with `cd <WT> && `,
so that `tools/...`, `data/...` and `git ...` refer to the worktree, never to `$REPO`. Paths such as
`data/meta.json` in the text also mean the worktree's copy. Run Python only as `uv run --no-sync python ...`
and tests as `uv run --no-sync pytest ...`: step 1 links the repo's existing virtualenv into the worktree, and
`--no-sync` keeps uv from changing that shared virtualenv.

## 1. Preflight (stop on any failure)

```bash
mkdir -p "$WORK" "$STATE/pending" "$STATE/applied"
git -C "$REPO" fetch origin                                  # must succeed
git -C "$REPO" worktree add --detach "$WT" origin/main       # must succeed
ln -s "$REPO/.venv.nosync" "$WT/.venv"                       # .venv is git-ignored
```

- `worktree add` fails because `$WT` exists: a run already happened today. End the run (no log entry).
- If `$WT/tools/arxiv_candidates.py` or `$WT/tools/add_entries.py` does not exist, `origin/main` does not have
  the sweep tools yet: **stop** with the reason "sweep tools not on origin/main".
- Read `$WT/data/meta.json`. If `"updated"` equals `$TODAY`, the list was already updated today: run
  `git -C "$REPO" worktree remove "$WT"` and end the run (no log entry).
- List `$STATE/pending/`. Files there come from earlier runs that could not push. `entries-<date>.json` holds
  entries that were triaged and accepted; they go into this week's `new.json` in step 6. For each
  `seen-<date>.jsonl`, run `uv run --no-sync python tools/arxiv_candidates.py --mark-seen <file>` now, so the candidates it
  holds are not shown again.

## 2. The window

`SINCE` = the `"updated"` value in `data/meta.json` (the list covers everything up to it). `UNTIL` = `$TODAY`.

If `SINCE` is more than 21 days ago (earlier runs failed), catch up in slices: set `UNTIL` = `SINCE` + 13 days,
pass `--until $UNTIL` in step 3, search the web only up to `UNTIL`, and in step 8 set `"updated"` to `UNTIL`,
not today. The next run continues from there.

## 3. arXiv candidates

```bash
uv run --no-sync python tools/arxiv_candidates.py --out "$WORK/candidates.jsonl"   # add --until "$UNTIL" when catching up
uv run --no-sync python tools/arxiv_candidates.py --brief "$WORK/candidates.jsonl" > "$WORK/brief.txt"
```

The first command downloads every cs.RO submission from 4 days before `SINCE` to `UNTIL`, plus the robot and RL
papers in cs.LG/cs.AI/cs.CV/stat.ML/cs.NE, keeps those whose title or abstract uses the list's topic terms, and
drops ids already in the list or in `docs/maintenance/seen-ids.txt` (triaged in an earlier week). The 4 extra days
catch papers arXiv announced late; whatever of them is still in the file is new to you and gets triaged like
the rest. It fetches each pool twice (day by day and whole-window, keeping the union, because the API sometimes
returns incomplete sets) and takes 2-10 minutes; it backs off by itself when arXiv is busy.

Expect about 500 candidates a week, a third of them matching 2+ topic groups (test week 2026-09-23..30: 585
topic matches, of which 122 were already listed). The filter keeps 114 of the 120 arXiv papers the curators
listed from that week, and 713 of the 733 listed arXiv papers since 2025.
The first run after this playbook was written has no `seen-ids.txt` yet and also shows the 4 overlap days
(about 250 more).

The tool exits non-zero, and writes no candidates file, when arXiv answered with missing pages, an empty or
implausibly small cs.RO pool, or an unfiltered result set. Then wait 30 minutes and run it once more. If it
fails again, **stop** with the reason "arXiv API: <the message it printed>". Never continue with a partial or empty
candidates file: the window would be marked as checked and its papers lost for good.

## 4. Web candidates

Search the web for items first published between `SINCE` and `UNTIL`. Use your web search tool, one search per
name, for example `"Physical Intelligence" robot blog October 2026`, and open the company's own blog or news
page when it has one:

- **Companies:** Sunday Robotics, Physical Intelligence, Generalist AI, Dyna Robotics, Figure, Tesla Optimus,
  1X, Skild AI, Google DeepMind (robotics), NVIDIA (GEAR, Isaac), Amazon FAR, Toyota Research Institute,
  AgiBot, Galaxea, AgileX, Hugging Face LeRobot.
- **Researchers** (posts, talks and X/Twitter threads that announce or explain results): Sergey Levine, Chelsea
  Finn, Pieter Abbeel, Karol Hausman, Jianlan Luo, Abhishek Gupta, Dorsa Sadigh, Shuran Song, Russ Tedrake,
  Lerrel Pinto, Yuke Zhu, Jim Fan, Tony Zhao, Cheng Chi.
- **Code releases:** `gh search repos "<topic>" --created ">=$SINCE" --sort stars --limit 20` for the topics
  `VLA reinforcement learning`, `robot policy fine-tuning`, `residual policy`, `diffusion policy RL`,
  `robot reward model`, `LeRobot`. Also check whether any paper in `data/papers.csv` dated in the last 90 days
  with an empty `code` column now has official code (search its title; accept only a link from the authors'
  project page, abs page or paper).

Keep an item only if it is new (search `data/papers.csv` and `data/resources.csv` for its URL and title), dated
in the window, and from a primary source (the company's or author's own page, not a news article about it).

## 5. Triage

Never open `candidates.jsonl` itself (about 2 KB per candidate). Work from `$WORK/brief.txt`: one line per
candidate (`id date score [topic groups] title`), most topic groups first.

1. Read every line of `brief.txt`. Skip candidates whose title alone shows they are out of scope (most
   score-1 lines). Note the rest.
2. Read the abstracts of the noted ones, a few dozen ids at a time:
   `uv run --no-sync python tools/arxiv_candidates.py --brief "$WORK/candidates.jsonl" --abstracts <id>,<id>,...`
3. Open the abs page (`https://arxiv.org/abs/<id>`) or the paper only when the abstract does not settle it, or
   to read affiliations, code links and numbers for an included paper.

Decide include or skip for every candidate, including the 4 overlap days.

**Include** work about improving a robot policy that already exists, or about measuring or diagnosing one:
RL fine-tuning or post-training of pretrained policies and VLAs, residual and edit policies, noise or latent
steering, advantage-weighted or filtered BC and data curation driven by outcomes, human corrections and DAgger,
test-time search, verifiers and best-of-N, reward, progress, success and value models used for improvement or
evaluation, world models and twins used to improve or evaluate policies, sim-to-real fine-tuning, black-box,
evolutionary and Bayesian optimization of policies or controllers, and industry reports of deploy-and-improve
recipes. Also general RL results that robot work builds on (the `generalrl` family): value and RL scaling laws,
critic and actor-critic design, RL for flow and diffusion policies, policy-gradient estimator design (group
baselines, importance ratios, clipping, entropy), off-policy or stale-data training, evolution strategies at
scale, process rewards and test-time compute, last-mile RL for driving policies.

**Skip** (and count the reason for the log):
- `pretraining`: new pretraining recipes, architectures, datasets or tokenizers with no improvement loop.
- `benchmark-only`: benchmarks or evaluations of base models that propose no improvement or evaluation method.
- `from-scratch`: RL from scratch (locomotion, games, control) with no pretrained policy, sim-to-real
  fine-tuning or last-mile angle.
- `llm-application`: RL for language models that is an application (math, code, agents, tool use, a domain
  dataset) rather than a general RL method result. The bar for `generalrl` is high: expect at most a handful
  a week.
- `off-topic`: everything else outside the scope (perception-only, planning without learning, hardware).
- `unclear`: you could not decide after opening the abs page. List these arXiv ids in the log.
- `duplicate`: already listed, or a new version of a listed entry.

Surveys, codebases, simulators, benchmarks with an improvement angle, datasets, courses and talks that are
squarely in scope go to `data/resources.csv`, not papers.

## 6. Write the entries

Put every included item in `$WORK/new.json`, a JSON list, together with the entries of any
`$STATE/pending/entries-*.json` file (step 1). Papers, blog posts, talks and threads:

```json
{"type": "paper", "title": "Exact title", "url": "https://arxiv.org/abs/2610.01234", "date": "2026-10-02",
 "family": "residual", "kind": "paper", "short": "MethodName", "org": "Stanford; NVIDIA",
 "code": "https://github.com/owner/repo", "one_line": "..."}
```

Resources:

```json
{"type": "resource", "name": "Name", "url": "https://github.com/owner/repo", "category": "code-robot",
 "org": "Lab or company", "one_line": "..."}
```

Field rules:
- `title`: exactly as on the abs page or post. `short`: the method's name (omit it to use the text before
  the title's colon).
- `url`: `https://arxiv.org/abs/<id>` for arXiv papers; otherwise the canonical post or page URL.
- `date`: `YYYY-MM-DD`: the arXiv v1 date from the candidates file, or the post's publication date.
- `kind`: `paper`, `blog`, `talk` or `thread`.
- `org`: short affiliations from the abs page or the paper's first page, `;`-separated, at most 4. Leave it
  `""` if you did not see them; a byline is filled in later. Company posts: the company.
- `code`: only a link the authors give (abs page comments, project page, paper). Otherwise `""`.
- `family`, one of:

  | family | the item improves or measures a policy by... |
  |---|---|
  | `blackbox` | search with only episode scores: random search, ES, CEM, CMA-ES, BO, noise tickets |
  | `onpolicy` | on-policy policy gradients (PPO, GRPO, REINFORCE) on the policy itself |
  | `offpolicy` | off-policy critics, Q-learning, offline-to-online RL, real-world sample-efficient RL |
  | `residual` | a small add-on to a frozen policy: residual or edit actions, noise or latent steering |
  | `advbc` | supervised updates weighted, filtered or conditioned on outcome or advantage; outcome-driven curation |
  | `hitl` | human (or machine) interventions and corrections, DAgger |
  | `testtime` | choosing among samples at deploy time: verifiers, best-of-N, lookahead, runtime monitors |
  | `reward` | building the score: reward, progress, success and value models |
  | `worldmodel` | learned world models or twins used to evaluate or improve policies |
  | `simreal` | simulation as the practice ground with transfer to real: sim-to-real and real-to-sim fine-tuning |
  | `industry` | a company's deploy-and-improve recipe or deployment report |
  | `generalrl` | a general (non-robot-manipulation) RL result robot work builds on |

  When two fit, pick the family of the mechanism that does the improving.
- `category` (resources): `code-robot`, `code-rl`, `sim`, `benchmark`, `dataset`, `course`, `book`,
  `tutorial`, `survey`, `talk` or `awesome`.
- `one_line`, at most 40 words, in this order: the mechanism (what is trained, searched or edited, and on which
  signal), then the single most telling result **with its baseline and setting**. Preserve the source’s numerical values and paraphrase its
  explanation, e.g. `... raises success from 52% to 81% over the frozen base on four real-robot tasks.` No hype
  words ("novel", "state-of-the-art", "significantly"), no "we" or "this paper". If the number comes from a
  figure, add "(chart reading)". If the abstract gives no number and you did not read one in the paper, write
  the mechanism and setting only. Read `head -30 data/papers.csv` once to match the house style.

## 7. Add them

```bash
uv run --no-sync python tools/add_entries.py "$WORK/new.json" --dry-run
uv run --no-sync python tools/add_entries.py "$WORK/new.json"
uv run --no-sync python tools/arxiv_candidates.py --mark-seen "$WORK/candidates.jsonl"
```

Exit code 2 means some entries are invalid and nothing was written: fix the entries it names and rerun.
Duplicates are skipped and reported, which is fine. For a code link to an existing row (step 4), edit that
row's `code` cell in `data/papers.csv` directly. The last command records every candidate as triaged.

## 8. Validate and build

```bash
uv run --no-sync python tools/arxiv_meta.py          # fixes arXiv v1 dates, fills bylines where org is blank
uv run --no-sync python tools/check_links.py         # non-arXiv links
```

- `arxiv_meta.py` prints `MISSING <id>` or `TITLE? <id>` lines. For an id you added this week: fix the row in
  `data/papers.csv` (wrong id or title) or delete it. For older ids: do not touch them; note them in the log.
- `check_links.py` prints `FAIL` lines. For a URL you added this week: fix it, or blank the `code` cell, or
  delete the row. Older failures and `blocked` lines: note them in the log only.

Then set the date (`UNTIL` instead of `$TODAY` when catching up), rebuild, and count what was added:

```bash
uv run --no-sync python -c "import json,pathlib; p=pathlib.Path('data/meta.json'); m=json.loads(p.read_text()); m['updated']='$TODAY'; p.write_text(json.dumps(m, indent=1))"
uv run --no-sync python tools/build_awesome.py
[ -f tools/build_site.py ] && uv run --no-sync python tools/build_site.py
uv run --no-sync python tools/add_entries.py --count-vs HEAD          # prints "N M": papers/posts and resources added
```

Use these N and M (not step 7's output, which counts rows step 8 may have deleted) in the commit message and
the log. Set `"updated"` even when nothing was added: it records that the window was checked. This is safe only
because step 3 succeeded with complete pools.

## 9. Checks (any failure: step 12)

```bash
uv run --no-sync python tools/build_awesome.py --check
[ -f tools/build_site.py ] && uv run --no-sync python tools/build_site.py --check
uv run --no-sync pytest -q tests/test_sweep_tools.py $(ls tests/test_build_site.py 2>/dev/null)
git status --porcelain
```

Only the list's tests run here; CI runs the full suite after the push. The last command must list nothing
but `data/papers.csv`, `data/resources.csv`, `data/meta.json`, `docs/maintenance/seen-ids.txt`, `README.md`,
`papers/*.md` and `site/` files. Anything else means a tool wrote where it should not: step 12.

## 10. Log entry

Append one entry to the end of `docs/maintenance/sweep-log.md` (create it with the heading
`# Weekly sweep log` if it does not exist). If `$STATE/unpushed-log.md` exists, first append its contents
(entries of earlier runs that could not push). Keep it short and factual:

```markdown
## 2026-10-07: ok
- Window: 2026-09-30 → 2026-10-07 (arXiv searched from 2026-09-26)
- Candidates: 512 arXiv (3 already listed, 240 seen last week), 9 web, 3 repos
- Added: 7 papers/posts (N448-N454), 1 resource, 1 code link; 2 entries carried over from 2026-09-30
- Skipped: 516 (pretraining 131, benchmark-only 39, from-scratch 72, llm-application 95, off-topic 173,
  duplicate 4, unclear 2: 2610.01234, 2610.04321)
- Checks: arxiv_meta ok; check_links 2 old failures (https://..., https://...); build --check ok; pytest ok
```

Status is `ok` (pushed), `blocked` (preflight, arXiv or push problem) or `failed` (a check failed). For
`blocked` and `failed`, add a last line `- Not pushed: <reason, failing command and its last output lines>`.

## 11. Commit and push

```bash
git add -- data/papers.csv data/resources.csv data/meta.json README.md papers docs/maintenance/sweep-log.md docs/maintenance/seen-ids.txt
git commit -m "List: weekly update $TODAY (+N papers, +M resources)" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
  -- data/papers.csv data/resources.csv data/meta.json README.md papers docs/maintenance/sweep-log.md docs/maintenance/seen-ids.txt
git rev-list --count origin/main..HEAD                    # must print: 1
git log -1 --format=%s                                    # must start with: List: weekly update
git push origin HEAD:main
```

If `git ls-files site` prints anything (the repo tracks `site/`), add ` site` to both path lists.
Replace N and M with step 8's numbers (e.g. `(+7 papers, +1 resources)`). If either check before the push
fails, do not push: status `blocked`, step 12.

If the push is rejected because `origin/main` moved, run `git fetch origin` and
`git log --oneline HEAD..origin/main -- data README.md papers docs/maintenance`. If that prints anything,
someone else changed the list meanwhile: do not merge by hand; go to step 12 with status `blocked` (the
entries carry over to next week). Otherwise run `git rebase origin/main`, rerun step 9, repeat the two checks,
and push again (at most twice in total). If the rebase stops with a conflict, run `git rebase --abort` and go
to step 12 with status `blocked`. Never use `--force`.

After a successful push:

```bash
mv "$STATE"/pending/* "$STATE/applied/" 2>/dev/null; rm -f "$STATE/unpushed-log.md"
git -C "$REPO" fetch origin
git -C "$REPO" worktree remove "$WT"
```

The main checkout's `main` is not moved; a human pulls when convenient. End the run.

## 12. Stopping without a push

Nothing from this run reaches GitHub, and the next run starts from a fresh worktree, so a failed week never
blocks the next one. Before ending:

1. Write this run's log entry (step 10's format; if step 10 already ran, take that entry and change its
   status) with status `blocked` or `failed`, and append it to `$STATE/unpushed-log.md` (the next successful
   run publishes it).
2. If step 6 produced entries, copy `$WORK/new.json` to `$STATE/pending/entries-$TODAY.json`.
3. If triage got through step 5, copy `$WORK/candidates.jsonl` to `$STATE/pending/seen-$TODAY.jsonl`.
4. Leave `$WT` as it is for a human to inspect, and say in your final message that the run stopped, why, and
   where the worktree and log are. (A human removes old worktrees with `git -C <REPO> worktree remove --force
   <path>`.)
