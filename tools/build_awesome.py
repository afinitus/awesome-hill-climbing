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
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
README = ROOT / "README.md"
PAPERS_DIR = ROOT / "papers"
REPO_URL = "https://github.com/afinitus/awesome-hill-climbing"
RECENT_FROM = "2025-09"  # "the last year" for counts and for the month-by-month layout

# Hand-picked entry points (ids in data/papers.csv) and why each is worth reading first.
START_HERE = [
    ("C357", "The industry framing of hill climbing: scale pretraining until in-house gains carry over to unseen homes, then climb reliability in-house. Proposes the \"Solve\" reporting standard."),
    ("C179", "Advantage-conditioned retraining of a VLA: a value model tags chunks positive or negative, the model is retrained with its supervised loss, and guidance pushes it toward positive."),
    ("C276", "Off-policy RL around a VLA without backprop through it: propose chunks, apply a small bounded edit, execute the best under a Q-ensemble."),
    ("C236", "Literal hill climbing: random-search one fixed noise vector for a frozen diffusion or flow policy. The baseline every RL method should beat."),
    ("C324", "Noise-space steering constrained to what the base policy can already do, trained only in a scanned digital twin, transferred with zero real RL."),
    ("C259", "Train a verifier on your own evaluation rollouts and rerank N samples at deploy time. No weight updates."),
    ("C435", "A controlled real-robot ablation of the advantage-guided post-training loop: how to build, calibrate and use the advantage."),
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
    ("1", "Hill climbing & ARS", "Finite differences on a smoothed objective", "Kohl & Stone 2004, ARS, TD-ES", "v0.1"),
    ("2", "CEM, CMA-ES, PI², BO", "Reward-weighted averaging", "Zero-order primer, BO review", "v0.1"),
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


def month_label(d: str) -> str:
    try:
        return dt.date.fromisoformat((d[:7] + "-01")).strftime("%B %Y")
    except ValueError:
        return d


def name_of(p: dict) -> str:
    return esc(p.get("short") or p["title"])


def entry(p: dict) -> str:
    """Full list entry: star, linked name, (org, date), blurb, code link."""
    star = "⭐ " if p.get("key") == "1" else ""
    meta = ", ".join(x for x in [p.get("org", ""), p.get("date", "")] if x)
    line = f"- {star}**[{name_of(p)}]({p['url']})**" + (f" ({esc(meta)})" if meta else "")
    if p.get("one_line"):
        line += f" — {esc(p['one_line'])}"
    extras = [f"[code]({p['code']})"] if p.get("code") else []
    if p.get("verified") == "corrected":
        extras.append("numbers corrected after check")
    return line + (" · " + " · ".join(extras) if extras else "")


def compact(p: dict, families: dict) -> str:
    """One-line changelog entry for What's new."""
    fam = families.get(p.get("family", ""))
    where = f" · [{fam['name']}](papers/{fam['key']}.md)" if fam else ""
    who = f" ({esc(p['org'])})" if p.get("org") else ""
    star = "⭐ " if p.get("key") == "1" else ""
    name, full = name_of(p), esc(p["title"])
    rest = full[len(name):].lstrip(" :—-") if full.lower().startswith(name.lower()) else full
    title = f": {rest}" if rest and len(rest) < 130 else ""
    return f"- `{p['date']}` {star}**[{name}]({p['url']})**{title}{who}{where}"


# ---------------------------------------------------------------------------- README sections
def sec_new(papers: list[dict], families: dict, today: dt.date, window: int) -> list[str]:
    cutoff = (today - dt.timedelta(days=window)).isoformat()
    new = [p for p in papers if len(p["date"]) == 10 and p["date"] >= cutoff]
    out = ["## What's new", "",
           f"**{len(new)} papers and posts from the last {window} days** ({cutoff} to {today.isoformat()}), newest first. "
           "Descriptions are on each family page.", ""]
    out += [compact(p, families) for p in new] or ["*Nothing in the window.*"]
    return out + [""]


def sec_start_here(by_id: dict) -> list[str]:
    out = ["## Start here", "", "Seven reads from the last year that together cover most of the field:", ""]
    for pid, why in START_HERE:
        p = by_id.get(pid)
        if p:
            out.append(f"1. **[{name_of(p)}]({p['url']})** ({esc(p.get('org', ''))}, {p.get('date', '')[:7]}) — {why}")
    return out + [""]


def sec_loop() -> list[str]:
    return [
        "## The hill-climbing loop", "",
        "Every method here runs some version of one loop. The families differ in which step they spend effort on "
        "and which part of the system they change.", "",
        "```mermaid",
        "flowchart LR",
        "  S[\"0 · Start<br/>pretrained policy<br/>that sometimes works\"] --> D[\"1 · Deploy<br/>robot · fleet · sim · twin · world model\"]",
        "  D --> M[\"2 · Measure<br/>success check · progress model · VLM judge\"]",
        "  M --> F[\"3 · Find failures<br/>progress dips · monitors · takeovers\"]",
        "  F --> I[\"4 · Improve<br/>search · RL · residual · weighted BC · DAgger · test-time\"]",
        "  I --> R[\"5 · Re-measure<br/>held-out states + confidence intervals\"]",
        "  R --> D",
        "```", "",
        "**Measuring the nines.** Near the top, the hard part is knowing whether you climbed. "
        "30/30 successes only shows the true rate is at least **88.6%** (95% Wilson interval). "
        "You need **73** straight successes for a lower bound of 95%, and **381** for 99%. "
        "Report the interval, keep search episodes separate from evaluation episodes, and compare methods on the "
        "same start states.", "",
        "Three things hold across the whole list: these methods mostly **amplify behavior the base policy already has** "
        "(at 0% success there is little to climb); the recipes that work **keep the big model frozen or anchored and "
        "train something small** next to it; and the bottleneck has moved from the optimizer to **the success signal "
        "and the evaluation**.", "",
    ]


def sec_foundations(found: list[dict]) -> list[str]:
    out = ["## Foundations", "",
           "Every 2025–26 method recombines about a dozen textbook ideas. The standard text is "
           "[Sutton & Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (free online).", "",
           "| Idea | Core equation | Classic references | Used most by |",
           "| :--- | :--- | :--- | :--- |"]
    for f in found:
        refs = " · ".join(f"[{n}]({u})" for n, u in (r.split("|", 1) for r in f["refs"].split(";") if "|" in r))
        out.append(f"| **{esc(f['idea'])}** | `{esc(f['equation'])}` | {refs} | {esc(f['used_by'])} |")
    return out + [""]


def sec_families(by_fam: dict, fam_meta: list[dict]) -> list[str]:
    out = ["## Papers and posts by family", "",
           "Each family answers one question. Below are the **must-know** works (⭐) per family; the link under each "
           "heading opens the full list, newest first.", ""]
    for f in fam_meta:
        items = by_fam.get(f["key"], [])
        recent = sum(1 for p in items if p["date"] >= RECENT_FROM)
        keys = [p for p in items if p.get("key") == "1"]
        out += [f"### {f['name']}", "",
                f"*{esc(f['question'])}*", "",
                f"{esc(f['tagline'])}", "",
                f"**Loop step:** {f['loop_step']} · **Built on:** {'; '.join(f['built_on'][:4])}", "",
                f"**[All {len(items)} entries →](papers/{f['key']}.md)** ({recent} from {month_label(RECENT_FROM)} onward)", ""]
        out += [entry(p) for p in keys]
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
            org = f" ({esc(r['org'])})" if r.get("org") else ""
            out.append(f"- **[{esc(r['name'])}]({r['url']})**{org} — {esc(r['one_line'])}")
        out.append("")
    return out


def sec_handson(code_released: bool) -> list[str]:
    have = {p.name[:2] for p in (ROOT / "algos").glob("[0-9]*.py")} if code_released else set()
    rows = [f"| {n} | {m} | {b} | {p} | {'available' if f'{int(n):02d}' in have else 'planned for ' + s} |"
            for n, m, b, p, s in CHAPTERS]
    intro = [
        "## Hands-on: lastmile", "",
        "This repo also hosts **`lastmile`**, an open-source course that implements these families on one task: "
        "a low-cost **SO-101** arm (the AgileX **PiPER** is supported in simulation too) picks up a cube and drops it "
        "in a cup. One base policy works about half the time; each chapter tries to push it toward 95%+ and reports "
        "what that cost in robot-minutes and human-minutes. Everything runs in MuJoCo on a laptop (no NVIDIA GPU "
        "needed), and each chapter then gets a real-arm step.", "",
    ]
    if code_released:
        intro += ["```bash",
                  "make setup      # uv venv + dependencies",
                  "make test       # fast test suite",
                  "make validate   # sanity-check the task with the scripted controller, on both arms",
                  "```", ""]
        links = ("Design contract: [docs/DESIGN.md](docs/DESIGN.md) · course plan: "
                 "[docs/research/curriculum.md](docs/research/curriculum.md) · chapter notes: [docs/chapters/](docs/chapters/)")
    else:
        intro += ["**Status:** the simulator, the evaluation harness and the first chapters are being reviewed and land "
                  "in this repo next. The plan:", ""]
        links = "Course plan: [docs/research/curriculum.md](docs/research/curriculum.md)"
    out = intro + ["| Ch | Method | Base idea | Mirrors | Status |", "| ---: | :--- | :--- | :--- | :--- |", *rows, "", links, ""]
    if code_released:
        out += ["### Leaderboard", "", "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->", ""]
    return out


def sec_footer(n_papers: int, n_deep: int, n_res: int, today: dt.date) -> list[str]:
    return [
        "## How this list is made", "",
        f"The first version (September 2026) came from an agent-assisted literature sweep: 18 search angles plus rounds "
        f"of gap-finding (arXiv month by month, citation mining, company blogs and talks). Of the {n_papers} papers "
        f"and posts, {n_deep} were read in full into structured notes, and each of those from September 2025 onward "
        f"was re-checked against its source by a second pass, with corrections applied and flagged. The other "
        f"{n_papers - n_deep}, mostly from the newest weeks, were read once from the source (`verified = read-once` in "
        f"the data) and get the full check in later updates. The {n_res} resources were each opened and confirmed. "
        f"Every arXiv link, title and date is validated against the arXiv API by `tools/arxiv_meta.py`.", "",
        "Numbers are as reported by the authors. Many are bar-chart readings, most real-robot results use 20–60 "
        "trials, and industry numbers are self-reported. Check the paper before citing a number. Research and "
        "drafting were assisted by Claude (Anthropic).", "",
        "## Contributing", "",
        "Add a row to [`data/papers.csv`](data/papers.csv) or [`data/resources.csv`](data/resources.csv), run "
        "`uv run python tools/build_awesome.py`, and open a pull request. A good entry has a concrete one-line "
        "description: the mechanism, then the headline number with its baseline. See [CONTRIBUTING.md](CONTRIBUTING.md).", "",
        "## Citation", "",
        "```bibtex",
        "@misc{awesome-hill-climbing,",
        "  title = {Awesome Hill Climbing: last-mile improvement of robot policies},",
        f"  year  = {{{today.year}}},",
        f"  url   = {{{REPO_URL}}}",
        "}",
        "```", "",
        "## License", "",
        "The list and the code are MIT-licensed ([LICENSE](LICENSE)). Robot models under `lastmile/envs/assets/` keep "
        "their MuJoCo Menagerie licenses (SO-101: Apache-2.0, PiPER: MIT).", "",
    ]


# ---------------------------------------------------------------------------- family pages
def family_page(f: dict, items: list[dict]) -> str:
    recent = [p for p in items if p["date"] >= RECENT_FROM]
    older = [p for p in items if p["date"] < RECENT_FROM]
    out = [f"# {f['name']}", "",
           f"[← Awesome Hill Climbing](../README.md) · [all families](README.md)", "",
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
        "> Papers, posts, code and courses on **hill-climbing robot policies**: taking a pretrained or "
        "imitation-learned policy from ~50% to 95–99%+ success on a real task. RL fine-tuning, residual and steering "
        "policies, advantage-weighted retraining, human corrections, test-time verifiers, reward and world models, "
        "sim-to-real, and the classic black-box search underneath it all.", "",
        f"**{len(papers)} papers and posts** · **{n_recent} from the last year** · **{n_key} must-know (⭐)** · "
        f"**{len(res)} tools, courses and benchmarks** · updated **{today.isoformat()}**", "",
        "A pretrained robot policy that works half the time is a demo. One that works 99% of the time is a product. "
        "\"Hill climbing\" is what labs and companies call the loop in between: deploy, measure, find failures, improve, "
        "re-measure. This list maps every way people run that loop, the textbook RL ideas each one comes from, and what "
        "changed in the last twelve months, down to this week.", "",
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
    lines += sec_footer(len(papers), sum(1 for p in papers if p.get("verified") != "read-once"), len(res), today)

    files = {README: "\n".join(lines).rstrip() + "\n", PAPERS_DIR / "README.md": papers_index(fam_meta, by_fam)}
    for f in fam_meta:
        files[PAPERS_DIR / f"{f['key']}.md"] = family_page(f, by_fam.get(f["key"], []))
    return files


LB = re.compile(r"<!-- LEADERBOARD:START -->.*?<!-- LEADERBOARD:END -->", re.S)


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
