"""Build the Awesome Hill Climbing list from the files in data/.

The list is data-driven so it stays consistent at 600+ entries:

    data/papers.csv      one row per paper / post (edit this to add work)
    data/resources.csv   code, simulators, benchmarks, courses, books, surveys, talks
    data/families.json   intro text for each method family
    data/foundations.csv the textbook RL ideas and their classic references
    data/meta.json       {"updated": "YYYY-MM-DD", "new_window_days": 14}

It writes:

    README.md            the front page: what's new, start here, must-know work per family, resources
    papers/<family>.md   the full list for each family, newest first
    papers/README.md     index of the family pages

Usage:
    uv run python tools/build_awesome.py           # rewrite the generated files
    uv run python tools/build_awesome.py --check   # exit 1 if any generated file is stale (CI)

The block between <!-- LEADERBOARD:START --> and <!-- LEADERBOARD:END --> in README.md is
owned by tools/leaderboard.py and is preserved verbatim.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
README = ROOT / "README.md"
PAPERS_DIR = ROOT / "papers"
REPO_URL = "https://github.com/afinitus/awesome-hill-climbing"
RECENT_FROM = "2025-09"  # fixed start of the recent-work timeline and counts

# Hand-picked entry points (ids in data/papers.csv) and why each is worth reading first.
START_HERE = [
    ("C357", "Sunday's framing of hill climbing: scale pretraining until in-house gains carry over to unseen homes, then climb reliability in-house. Proposes the \"Solve\" reporting standard."),
    ("C179", "Advantage-conditioned retraining of a VLA: a value model tags chunks positive or negative, the model is retrained with its supervised loss on that tag, and at test time it is conditioned on positive (optionally with classifier-free guidance)."),
    ("C276", "Off-policy RL around a VLA with no RL gradient through it: propose chunks, apply a small bounded edit, execute the best under a Q-ensemble. The VLA itself keeps training on its own supervised loss."),
    ("C236", "Black-box search (random search, CEM, zeroth-order) for one fixed input-noise vector for a frozen diffusion or flow policy, training no new networks. Simple enough to run before RL: it beat Gaussian noise sampling on 46 of 51 sim and real tasks."),
    ("C324", "Noise-space steering constrained to what the base policy can already do, trained only in a scanned digital twin, transferred with zero real RL."),
    ("C259", "Train a verifier on your own evaluation rollouts and rerank N samples at deploy time. No policy weight updates."),
    ("C435", "A controlled study of advantage-guided (RECAP-style) post-training that separates how the advantage is built, calibrated and used, screens each choice with offline diagnostics, then tests the recipe on four real bimanual tasks."),
]

RESOURCE_SECTIONS = [
    ("Code: robot learning and last-mile RL", ["code-robot"]),
    ("Code: general RL and black-box optimization", ["code-rl"]),
    ("Simulators and robot models", ["sim"]),
    ("Benchmarks and datasets", ["benchmark", "dataset"]),
    ("Courses, books and tutorials", ["course", "book", "tutorial"]),
    ("Surveys", ["survey"]),
    ("Talks and podcasts", ["talk"]),
    ("Related awesome lists", ["awesome"]),
]

CHAPTERS = [  # (number, method, base idea, papers mirrored, planned release)
    ("0", "The base policy", "Behavior-clone a flow policy from mixed-quality demos", "Rectified flow, ACT, Diffusion Policy", "v0.1"),
    ("1", "Hill climbing & ARS", "Finite differences on a smoothed objective", "Kohl & Stone 2004, ARS, TD-ES", "v0.1"),
    ("2", "CEM, CMA-ES, PI², BO", "Reward-weighted averaging; BO: a surrogate model", "Zero-order primer, BO review", "v0.1"),
    ("3", "The golden ticket", "Black-box search in noise space", "Golden Ticket", "v0.1"),
    ("4", "PPO/GRPO for a flow policy", "Likelihood-ratio PG + anchors", "SimpleVLA-RL, πRL, PAC-ACT", "v0.3"),
    ("5", "SAC → RLPD → Q-chunking", "Bellman backups on replay", "Q-chunking, Three Regimes, IPE", "v0.2"),
    ("6", "Residual RL", "Frozen base + bounded add-on", "ResFiT, DAWN, Res-HIL", "v0.2"),
    ("7", "DSRL noise steering", "RL in latent-noise space", "DSRL, SCORE, PSS, RFS", "v0.2"),
    ("8", "Propose, edit, select", "Edit policy + argmax-Q", "EXPO-FT, Real-Time EXPO-FT", "v0.2"),
    ("9", "Filtered BC, AWR, RECAP-lite", "Improvement as supervised learning", "π*0.6/RECAP, CFGRL", "v0.3"),
    ("10", "DAgger, HG-DAgger, RaC", "On-policy labels", "RaC, SOP, FlowDAgger", "v0.3"),
    ("11", "Rewards & progress models", "Shaping and learned judges", "Robometer, TOPReward", "v0.3"),
    ("12", "Test-time best-of-N", "Greedy selection with a verifier", "UF-OPS, Q-Planning, SeeQ", "v0.1"),
]


# ---------------------------------------------------------------------------- helpers
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh)]


def anchor(text: str) -> str:
    """GitHub-style heading anchor."""
    a = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return a.replace(" ", "-")


def esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def tracked(folder: str) -> list[str]:
    """Files under `folder` that are committed to git (so a chapter counts as available only once
    it is published, not while it is being written locally)."""
    try:
        out = subprocess.run(["git", "ls-files", folder], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return [str(p.relative_to(ROOT)) for p in (ROOT / folder).glob("*.py")]


def join_and(items: list[str]) -> str:
    """'0', '0 and 1', '0, 1 and 2'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def month_label(d: str) -> str:
    try:
        return dt.date.fromisoformat(d[:7] + "-01").strftime("%B %Y")
    except ValueError:
        return d


def name_of(p: dict) -> str:
    return esc(p.get("short") or p["title"])


def org_of(p: dict) -> str:
    """The org field, with a name that an earlier length limit cut off ("University of…") shown as "et al."."""
    org = p.get("org", "")
    if not org.endswith("…"):
        return org
    head = re.sub(r"[;,][^;,]*…$", "", org)
    if head == org:  # no separator before the cut: drop the partial last word
        head = re.sub(r"\s+\S*…$", "", org)
    return head.rstrip(" ;,") + " et al."


def entry(p: dict) -> str:
    """Full list entry: star, linked name, (org, date), blurb, code link."""
    star = "⭐ " if p.get("key") == "1" else ""
    meta = ", ".join(x for x in [org_of(p), p.get("date", "")] if x)
    line = f"- {star}**[{name_of(p)}]({p['url']})**" + (f" ({esc(meta)})" if meta else "")
    if p.get("one_line"):
        line += f" — {esc(p['one_line'])}"
    extras = [f"[code]({p['code']})"] if p.get("code") else []
    if p.get("verified") == "corrected":
        extras.append("entry corrected after source review")
    return line + (" · " + " · ".join(extras) if extras else "")


def compact(p: dict, families: dict) -> str:
    """One-line changelog entry for What's new."""
    fam = families.get(p.get("family", ""))
    where = f" · [{fam['name']}](papers/{fam['key']}.md)" if fam else ""
    who = f" ({esc(org_of(p))})" if p.get("org") else ""
    star = "⭐ " if p.get("key") == "1" else ""
    name = name_of(p)
    full = esc(re.sub(r"\s*\([^()]*\)\s*$", "", p["title"]) or p["title"])  # drop a trailing "(ACRONYM)" alias
    if " (" in name and not full.lower().startswith(name.lower()):
        head, alias = name.split(" (", 1)
        if alias.rstrip(")").lower() in full.lower() or (full.lower().startswith(head.lower()) and len(full) > len(head)):
            name = head  # "PSS (Principal Steering Subspaces)": the expansion is already in the title
    if name.endswith("…") and full.startswith(name[:-1].rstrip()):
        name = full  # the short name is only the title cut off: show the full title once
    if full.lower().startswith(name.lower()):
        tail = full[len(name):]
        if tail.lstrip()[:1] in (":", "—", "-"):
            rest, sep = tail.lstrip(" :—-"), ": "
        else:  # "Skill-Space Shooting for ..." reads on; "Compose Your Policies! ..." keeps its punctuation
            rest, sep = tail.strip(), ("" if tail[:1] in "!?,." else " ")
    else:
        rest, sep = full, ": "
    title = f"{sep}{rest}" if rest and len(rest) < 130 and rest.lower() not in name.lower() else ""
    return f"- `{p['date']}` {star}**[{name}]({p['url']})**{title}{who}{where}"


# ---------------------------------------------------------------------------- README sections
def sec_new(papers: list[dict], families: dict, today: dt.date, window: int) -> list[str]:
    cutoff = (today - dt.timedelta(days=window)).isoformat()
    new = [p for p in papers if len(p["date"]) == 10 and p["date"] >= cutoff]
    out = ["## What's new", "",
           (f"**{len(new)} papers and posts from the last {window} days** ({cutoff} to {today.isoformat()}), newest first. "
           "Descriptions are on each family page."), ""]
    out += [compact(p, families) for p in new] or ["*Nothing in the window.*"]
    return out + [""]


def sec_start_here(by_id: dict) -> list[str]:
    out = ["## Start here", "", "Seven reads from the last year to start with:", ""]
    for pid, why in START_HERE:
        p = by_id.get(pid)
        if p:
            out.append(f"1. **[{name_of(p)}]({p['url']})** ({esc(org_of(p))}, {p.get('date', '')[:7]}) — {why}")
    return out + [""]


def sec_loop() -> list[str]:
    return [
        "## The hill-climbing loop", "",
        ("Every method here runs some version of one loop. The families differ in which step they spend effort on "
        "and which part of the system they change."), "",
        "```mermaid",
        "flowchart LR",
        "  S([Start]) --> D[Deploy] --> M[Measure] --> F[Find failures] --> I[Improve] --> R[Re-measure]",
        "  R --> D",
        "```", "",
        "| Step | What happens | Families that live here |",
        "| :--- | :--- | :--- |",
        "| **Start** | A pretrained policy that already succeeds sometimes: BC, ACT, a diffusion or flow policy, a VLA. | |",
        "| **Deploy** | Collect episodes on the robot, a fleet, a simulator, a digital twin or a learned world model. | [World models](papers/worldmodel.md), [sim-to-real](papers/simreal.md) |",
        "| **Measure** | Turn each episode into a number: a success check, a progress model, a VLM judge. | [Reward & progress models](papers/reward.md), [evaluation](papers/worldmodel.md) |",
        "| **Find failures** | Locate where and why it fails: progress dips, runtime monitors, human takeovers. | [Reward models](papers/reward.md), [human corrections](papers/hitl.md) |",
        "| **Improve** | Apply an update. Each family is a different update operator. | [Black-box search](papers/blackbox.md), [on-policy PG](papers/onpolicy.md), [off-policy critics](papers/offpolicy.md), [residual & steering](papers/residual.md), [weighted BC](papers/advbc.md), [DAgger](papers/hitl.md), [test-time search](papers/testtime.md) |",
        "| **Re-measure** | Held-out start states, trial counts and confidence intervals. Then repeat. | [Evaluation](papers/worldmodel.md), [industry loops](papers/industry.md) |",
        "",
        ("**Measuring the nines.** Near the top, the hard part is knowing whether you climbed. "
        "30/30 successes gives a **[88.6%, 100%]** Wilson 95% interval, not proof of perfect reliability. "
        "At a fixed evaluation size, **73/73** gives a lower bound of 95%, and **381/381** gives 99%, "
        "assuming independent trials with a constant success probability. "
        "Report the interval, keep search episodes separate from evaluation episodes, and compare methods on the "
        "same start states."), "",
        ("Three patterns run through the list: these methods mostly **amplify behavior the base policy already has** "
        "(at 0% success there is usually little to climb); many recipes that work **keep the big model frozen or anchored and "
        "train something small** next to it, while others retrain the whole model, sometimes with no anchor when the base "
        "already succeeds often (SimpleVLA-RL); and the bottleneck has moved from the optimizer to **the success signal "
        "and the evaluation**."), "",
    ]


def sec_foundations(found: list[dict]) -> list[str]:
    out = ["## Foundations", "",
           ("Most 2025–26 methods recombine about a dozen textbook ideas. The standard text is "
           "[Sutton & Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (free online)."), "",
           "| Idea | Core equation | Classic references | Used most by |",
           "| :--- | :--- | :--- | :--- |"]
    for f in found:
        refs = " · ".join(f"[{n}]({u})" for n, u in (r.split("|", 1) for r in f["refs"].split(";") if "|" in r))
        out.append(f"| **{esc(f['idea'])}** | `{esc(f['equation'])}` | {refs} | {esc(f['used_by'])} |")
    return out + [""]


def sec_families(by_fam: dict, fam_meta: list[dict]) -> list[str]:
    out = ["## Papers and posts by family", "",
           ("Each family answers one question. Below are the **must-know** works (⭐) per family; the link under each "
           "heading opens the full list, newest first."), ""]
    for f in fam_meta:
        items = by_fam.get(f["key"], [])
        recent = sum(1 for p in items if p["date"] >= RECENT_FROM)
        keys = [p for p in items if p.get("key") == "1"]
        out += [f"### {f['name']}", "",
                f"*{esc(f['question'])}*", "",
                f"{esc(f['tagline'])}", "",
                f"**Loop step:** {f['loop_step']} · **Built on:** {'; '.join(f['built_on'][:4])}", "",
                f"**[All {len(items)} entries →](papers/{f['key']}.md)** ({recent} from {month_label(RECENT_FROM)} onward)", ""]
        out += [entry(p) for p in keys] or ["*No ⭐ picks in this family yet; the full list is linked above.*"]
        out.append("")
    return out


def sec_resources(res: list[dict]) -> list[str]:
    out = ["## Code, tools and learning resources", ""]
    for title, cats in RESOURCE_SECTIONS:
        items = sorted((r for r in res if r.get("category") in cats), key=lambda r: r["name"].lower())
        if not items:
            continue
        out += [f"### {title}", ""]
        for r in items:
            org = f" ({esc(org_of(r))})" if r.get("org") else ""
            out.append(f"- **[{esc(r['name'])}]({r['url']})**{org} — {esc(r['one_line'])}")
        out.append("")
    return out


def sec_handson(code_released: bool) -> list[str]:
    have = {Path(f).name[:2] for f in tracked("algos")} if code_released else set()
    rows = [f"| {n} | {m} | {b} | {p} | {'available' if f'{int(n):02d}' in have else 'planned for ' + s} |"
            for n, m, b, p, s in CHAPTERS]
    intro = [
        "## Hands-on: lastmile", "",
        ("This repo also hosts **`lastmile`**, an open-source course that implements these families on one task: "
        "a low-cost **SO-101** arm (the AgileX **PiPER** is supported in simulation too) picks up a cube and drops it "
        "in a cup. One base policy works about half the time; each chapter tries to push it toward 95%+ and reports "
        "what that cost in robot-minutes and human-minutes. Everything runs in MuJoCo on a laptop (no NVIDIA GPU "
        "needed). Each chapter also describes a real-arm step for the SO-101; none has been run on hardware yet, and each "
        "carries a safety note to read first (clear workspace, power switch or e-stop in reach, low speed limits, supervise every run). "
        "Each hardware step in the chapter notes opens with a safety checklist; read it before you power the arm."), "",
        ("**Benchmark limitation:** CupDrop-v1 scores a drop into the cup at its current position; it can count successes "
        "after the cup was moved or tipped and recovered. These scores do not establish a no-shoving benchmark. "
        "See the [prelaunch review](docs/maintenance/review/REPORT.md)."), "",
    ]
    if code_released:
        intro += [
            ('<p><img src="media/env/so101_tuned.gif" width="330" alt="SO-101 picking up the cube and dropping it in the cup">'
            ' <img src="media/env/piper_tuned.gif" width="330" alt="AgileX PiPER doing the same task"></p>'), "",
            "**What is in the repo today:** the simulated task on both arms, the evaluation harness (fixed search and "
            "held-out start states, Wilson intervals, a cost ledger) and a scripted controller with ten tunable knobs. "
            "Tuned, it scores 255/256 = 99.6% [97.8%, 99.9%] on the held-out set; with the deliberately mis-tuned knobs that chapters 1 and 2 "
            "start from, it scores 123/256 = 48.0% [42.0%, 54.2%]. "
            + (f"Chapters {join_and([str(int(n)) for n in sorted(have)])} are available, with notes and results in "
               "[docs/chapters/](docs/chapters/)." if have else "The chapters themselves are next."), "",
        ]
        intro += ["```bash",
                  "make setup      # uv venv + dependencies",
                  "make test       # fast test suite",
                  "make validate   # sanity-check the task with the scripted controller, on both arms",
                  "```", ""]
        links = ("Design contract: [docs/DESIGN.md](docs/DESIGN.md) · course plan: "
                 "[docs/research/curriculum.md](docs/research/curriculum.md) · chapter notes: [docs/chapters/](docs/chapters/)")
    else:
        intro += [("**Status:** the simulator, the evaluation harness and the first chapters are being reviewed and land "
                  "in this repo next. The plan:"), ""]
        links = "Course plan: [docs/research/curriculum.md](docs/research/curriculum.md)"
    out = intro + ["| Ch | Method | Base idea | Mirrors | Status |", "| ---: | :--- | :--- | :--- | :--- |", *rows, "", links, ""]
    if code_released:
        out += ["### Leaderboard", "", "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->", ""]
    return out


def sec_footer(n_papers: int, n_deep: int, n_checked: int, n_res: int, today: dt.date) -> list[str]:
    return [
        "## How this list is made", "",
        (f"The first version (September 2026) came from an agent-assisted literature sweep: many search angles plus rounds "
        f"of gap-finding (arXiv month by month, citation mining, company blogs and talks). Of the {n_papers} papers "
        f"and posts, {n_checked} are marked `checked` or `corrected` in the catalog, and "
        f"{n_papers - n_deep} are marked `read-once`; these are recorded review statuses, not independent replication. "
        f"The {n_res} resources are also listed from source pages. `tools/arxiv_meta.py` checks every "
        f"arXiv link in the paper list and its first-version date against the arXiv API and flags titles that do not "
        f"match, and `tools/check_links.py` checks the non-arXiv links. The workflow for weekly updates is documented "
        f"in [docs/maintenance/weekly-sweep.md](docs/maintenance/weekly-sweep.md)."), "",
        ("Numbers are as reported by the authors. Some are bar-chart readings or estimates from small trial "
        "counts, and industry numbers are self-reported. Check the paper before citing a number. Research and "
        "drafting were assisted by Claude (Anthropic), which also helped write the lastmile code and chapter notes. "
        "The pre-launch review and fixes also used Codex (OpenAI)."), "",
        "## Contributing", "",
        ("Add a row to [`data/papers.csv`](data/papers.csv) or [`data/resources.csv`](data/resources.csv), run "
        "`uv run python tools/build_awesome.py`, and open a pull request. A good entry has a concrete one-line "
        "description: the mechanism, then the headline number with its baseline. See [CONTRIBUTING.md](CONTRIBUTING.md)."), "",
        "## Citation", "",
        "```bibtex",
        "@misc{awesome-hill-climbing,",
        "  title = {Awesome Hill Climbing: last-mile improvement of robot policies},",
        f"  year  = {{{today.year}}},",
        f"  url   = {{{REPO_URL}}}",
        "}",
        "```", "",
        "## License", "",
        ("The list and the code are MIT-licensed ([LICENSE](LICENSE)). Robot models under `lastmile/envs/assets/` keep "
        "their MuJoCo Menagerie licenses (SO-101: Apache-2.0, PiPER: MIT)."), "",
    ]


# ---------------------------------------------------------------------------- family pages
def family_page(f: dict, items: list[dict]) -> str:
    recent = [p for p in items if p["date"] >= RECENT_FROM]
    older = [p for p in items if p["date"] and p["date"] < RECENT_FROM]
    undated = [p for p in items if not p["date"]]
    out = [f"# {f['name']}", "",
           "[← Awesome Hill Climbing](../README.md) · [all families](README.md)", "",
           f"*{esc(f['question'])}*", "",
           esc(f["tagline"]), "",
           f"- **Loop step:** {f['loop_step']}",
           f"- **Built on:** {'; '.join(f['built_on'])}",
           f"- **Use it when:** {esc(f['use_when'])}",
           f"- **What changed in 2025–26:** {esc(f['trend'])}", "",
           f"**{len(items)} entries**, newest first. ⭐ marks must-know work.", ""]
    month = None
    for p in recent:
        if p["date"][:7] != month:
            month = p["date"][:7]
            out += ["", f"## {month_label(month)}", ""]
        out.append(entry(p))
    if older:
        out += ["", f"## Before {month_label(RECENT_FROM)}", ""]
        out += [entry(p) for p in older]
    if undated:
        out += ["", "## Publication date not stated", ""]
        out += [entry(p) for p in undated]
    return "\n".join(out).rstrip() + "\n"


def papers_index(fam_meta: list[dict], by_fam: dict) -> str:
    out = ["# All papers and posts, by family", "", "[← Awesome Hill Climbing](../README.md)", "",
           "| Family | Question it answers | Entries | Must-know |", "| :--- | :--- | ---: | ---: |"]
    for f in fam_meta:
        items = by_fam.get(f["key"], [])
        out.append(f"| [{f['name']}]({f['key']}.md) | {esc(f['question'])} | {len(items)} | {sum(p.get('key') == '1' for p in items)} |")
    out += ["", "The same data as one searchable table: [`data/papers.csv`](../data/papers.csv).", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------- build
def build() -> dict[Path, str]:
    papers = read_csv(DATA / "papers.csv")
    papers.sort(key=lambda p: (p.get("date", ""), p.get("id", "")), reverse=True)
    by_id = {p["id"]: p for p in papers}
    fam_meta = json.loads((DATA / "families.json").read_text(encoding="utf-8"))
    families = {f["key"]: f for f in fam_meta}
    found, res = read_csv(DATA / "foundations.csv"), read_csv(DATA / "resources.csv")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    today = dt.date.fromisoformat(meta["updated"])
    by_fam = defaultdict(list)
    for p in papers:
        by_fam[p.get("family", "")].append(p)
    unknown = sorted(set(by_fam) - set(families))
    if unknown:
        sys.exit(f"data/papers.csv uses families not in data/families.json: {unknown}")

    n_recent = sum(1 for p in papers if p["date"] >= RECENT_FROM)
    n_key = sum(1 for p in papers if p.get("key") == "1")
    toc = [("What's new", []), ("Start here", []), ("The hill-climbing loop", []), ("Foundations", []),
           ("Papers and posts by family", [f["name"] for f in fam_meta]),
           ("Code, tools and learning resources", [t for t, c in RESOURCE_SECTIONS if any(r.get("category") in c for r in res)]),
           ("Hands-on: lastmile", []), ("How this list is made", []), ("Contributing", [])]
    lines = [
        "# Awesome Hill Climbing [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)", "",
        "<picture>",
        '  <source media="(prefers-color-scheme: dark)" srcset="media/banner-dark.svg">',
        ('  <img src="media/banner-light.svg" width="100%" alt="Climbing the Nines: a success-rate landscape drawn as '
        'contour lines, with a dashed hill-climbing path from a pretrained policy at 50% up past the 99% contour.">'),
        "</picture>", "",
        ("> Papers, posts, code and courses on **hill-climbing robot policies**: improving task reliability after "
        "pretraining or imitation learning. RL fine-tuning, residual and steering "
        "policies, advantage-weighted retraining, human corrections, test-time verifiers, reward and world models, "
        "sim-to-real, and the classic black-box search underneath it all."), "",
        (f"**{len(papers)} papers and posts** · **{n_recent} since {month_label(RECENT_FROM)}** · **{n_key} must-know (⭐)** · "
        f"**{len(res)} tools, courses and benchmarks** · updated **{today.isoformat()}**"), "",
        *([(f"**Interactive explainer: [Climbing the Nines]({meta['site_url']})**: the hill-climbing loop, the RL "
           "ideas underneath, a timeline from September 2025 onward, the course results, and a searchable index of every "
           "paper."), ""] if meta.get("site_url") else []),
        ("A pretrained robot policy that works half the time needs more reliable behavior before deployment. "
        "Even 99% success needs a declared task scope, quality bar, speed and evaluation protocol. "
        "\"Hill climbing\" is what labs and companies call the loop in between: deploy, measure, find failures, improve, "
        "re-measure. This list maps the ways people run that loop, the textbook RL ideas each one comes from, and what "
        "changed since September 2025, through the update date above."), "",
        "## Contents", "",
    ]
    for title, subs in toc:
        lines.append(f"- [{title}](#{anchor(title)})")
        lines += [f"  - [{s}](#{anchor(s)})" for s in subs]
    lines.append("")
    lines += sec_new(papers, families, today, int(meta.get("new_window_days", 14)))
    lines += sec_start_here(by_id)
    lines += sec_loop()
    lines += sec_foundations(found)
    lines += sec_families(by_fam, fam_meta)
    lines += sec_resources(res)
    lines += sec_handson(bool(meta.get("code_released", True)))
    lines += sec_footer(len(papers), sum(1 for p in papers if p.get("verified") != "read-once"),
                        sum(1 for p in papers if p.get("verified") in ("checked", "corrected")), len(res), today)

    files = {README: "\n".join(lines).rstrip() + "\n", PAPERS_DIR / "README.md": papers_index(fam_meta, by_fam)}
    for f in fam_meta:
        files[PAPERS_DIR / f"{f['key']}.md"] = family_page(f, by_fam.get(f["key"], []))
    return files


LB = re.compile(r"<!-- LEADERBOARD:START -->.*?<!-- LEADERBOARD:END -->", re.DOTALL)


def keep_leaderboard(new: str, old: str) -> str:
    m = LB.search(old)
    return LB.sub(lambda _: m.group(0), new) if m else new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a generated file is out of date")
    args = ap.parse_args()
    stale = []
    for path, text in build().items():
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if path == README:
            text = keep_leaderboard(text, old)
        if text != old:
            stale.append(path)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    if args.check and stale:
        print("Stale: " + ", ".join(str(p.relative_to(ROOT)) for p in stale)
              + "\nRun `uv run python tools/build_awesome.py`.", file=sys.stderr)
        sys.exit(1)
    print("Up to date." if not stale else f"Wrote {len(stale)} file(s): "
          + ", ".join(str(p.relative_to(ROOT)) for p in stale))


if __name__ == "__main__":
    main()
