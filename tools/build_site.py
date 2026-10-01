"""Build the explainer page "Climbing the Nines" (site/index.html) from the files in data/.

The page is served by GitHub Pages at https://afinitus.github.io/awesome-hill-climbing/ and is regenerated
from the same data as the Awesome list, so the two never drift apart:

    data/papers.csv       every paper and post (index, family cards, timeline, counts)
    data/families.json    the method families (one section each)
    data/foundations.csv  the textbook RL ideas (the ideas x families matrix)
    data/resources.csv    code, simulators, benchmarks, courses and related lists ("Code, tools and reading")
    data/meta.json        the "updated" date (the only date written into the page)
    data/explainer.json   the cross-cutting narrative: thesis, the loop, decision guide, milestones, lessons,
                          industry recipes, classic search inside modern methods, family tree, open problems
                          and the ideas x families matrix. Written from the September 2026 deep-read of the
                          core papers; edit it by hand.
    tools/build_awesome.py START_HERE picks, the course CHAPTERS table, RESOURCE_SECTIONS and the repo URL
    algos/, results/, docs/chapters/ (git-tracked files only) chapter status and headline results

Usage:
    uv run python tools/build_site.py            # write site/index.html
    uv run python tools/build_site.py --check    # exit 1 if site/index.html is stale
    uv run python tools/build_site.py --out DIR  # write DIR/index.html instead

Only the standard library is needed (plus the two sibling tools), so it runs without the course dependencies.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import html
import json
import re
import subprocess
import sys
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_awesome as ba
import leaderboard as lb

ROOT = ba.ROOT
DATA = ba.DATA
SITE = ROOT / "site"
REPO_URL = ba.REPO_URL
PAGES_URL = "https://afinitus.github.io/awesome-hill-climbing/"
RECENT_FROM = ba.RECENT_FROM

MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September",
         "October", "November", "December"]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen"]

# Short labels for chips, lanes and filters (falls back to the family's full name).
SHORT = {
    "blackbox": "Black-box search", "onpolicy": "On-policy PG", "offpolicy": "Off-policy critics",
    "residual": "Residual & steering", "advbc": "Advantage-weighted BC", "hitl": "Human corrections",
    "testtime": "Test-time search", "reward": "Rewards & progress", "worldmodel": "World models & eval",
    "simreal": "Sim-to-real", "industry": "Industry loops", "generalrl": "General RL",
}
# Which families live at each step of the loop (step number -> family keys).
STEP_FAMILIES = {
    0: [], 1: ["worldmodel", "simreal"], 2: ["reward", "worldmodel"], 3: ["reward", "hitl"],
    4: ["blackbox", "onpolicy", "offpolicy", "residual", "advbc", "hitl", "testtime"], 5: ["worldmodel", "industry"],
}
BADGES = {
    "checked": ("ok", "✓ checked", "Numbers re-checked against the source by an independent pass"),
    "corrected": ("fix", "✓ corrected", "Numbers corrected after an independent check against the source"),
    "read-once": ("once", "read once", "Read once from the source; the full check comes in a later update"),
    "classic": ("classic", "◦ classic", "Older work, read from the source"),
}
ID_RE = re.compile(r"\b[CN]\d{3}\b")


# ---------------------------------------------------------------------------- helpers
def e(text: object) -> str:
    return html.escape(str(text), quote=True)


def git_files(folder: str) -> list[str]:
    """Files under `folder` committed to git (falls back to everything on disk outside a git checkout)."""
    try:
        out = subprocess.run(["git", "ls-files", folder], cwd=ROOT, capture_output=True, text=True, check=True)
        return sorted(out.stdout.split())
    except (OSError, subprocess.CalledProcessError):
        return sorted(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*") if p.is_file())


def mon_year(d: str) -> str:
    """'2026-03-14' or '2026-03' -> 'Mar 2026'; a bare year stays as is."""
    if len(d) >= 7 and d[4] == "-":
        return f"{MON[int(d[5:7])]} {d[:4]}"
    return d


def short_month(ym: str) -> str:
    return f"{MON[int(ym[5:7])]} ’{ym[2:4]}"


def num_word(n: int) -> str:
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def ext(url: str, text: str, cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<a{c} href="{e(url)}" target="_blank" rel="noopener">{text}</a>'


class Ctx:
    """Everything the sections need, loaded once."""

    def __init__(self) -> None:
        self.papers = ba.read_csv(DATA / "papers.csv")
        self.papers.sort(key=lambda p: (p.get("date", ""), p.get("id", "")), reverse=True)
        self.by_id = {p["id"]: p for p in self.papers}
        self.fams = json.loads((DATA / "families.json").read_text(encoding="utf-8"))
        self.fam = {f["key"]: f for f in self.fams}
        self.found = ba.read_csv(DATA / "foundations.csv")
        self.res = ba.read_csv(DATA / "resources.csv")
        self.meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
        self.ex = json.loads((DATA / "explainer.json").read_text(encoding="utf-8"))
        self.updated = dt.date.fromisoformat(self.meta["updated"])
        self.window = int(self.meta.get("new_window_days", 14))
        self.new_from = (self.updated - dt.timedelta(days=self.window)).isoformat()
        self.by_fam = defaultdict(list)
        for p in self.papers:
            self.by_fam[p.get("family", "")].append(p)
        unknown = sorted(set(self.by_fam) - set(self.fam))
        if unknown:
            sys.exit(f"data/papers.csv uses families not in data/families.json: {unknown}")
        self.recent = [p for p in self.papers if p["date"] >= RECENT_FROM]
        self.key_re = re.compile(r"\((" + "|".join(map(re.escape, self.fam)) + r")\)")

    # text ------------------------------------------------------------------
    def clean(self, s: str) -> str:
        """Internal paper ids and parenthesised family keys -> display names; drop review markers."""
        s = re.sub(r"\s*\(ADJ\)", "", s)
        s = self.key_re.sub(lambda m: f"({self.short(m.group(1))})", s)
        return ID_RE.sub(lambda m: self.by_id[m.group()]["short"] if m.group() in self.by_id else m.group(), s)

    def t(self, s: str) -> str:
        """Clean, escape and typeset a narrative string."""
        s = e(self.clean(s))
        return s.replace(" -&gt; ", " → ").replace("-&gt;", "→").replace("+/-", "±")

    def short(self, key: str) -> str:
        return SHORT.get(key) or self.fam.get(key, {}).get("name", key)

    def chip(self, key: str) -> str:
        return f'<a class="chip" href="#fam-{e(key)}">{e(self.short(key))}</a>' if key in self.fam else ""

    def fam_url(self, key: str) -> str:
        return f"{REPO_URL}/blob/main/papers/{key}.md"


def narrative_note() -> str:
    return '<p class="src">Narrative from the Sep 2026 deep-read of the core papers.</p>'


# ---------------------------------------------------------------------------- sections
def sec_hero(c: Ctx) -> str:
    n_checked = sum(1 for p in c.papers if p.get("verified") in ("checked", "corrected"))
    end_ym = c.meta["updated"][:7]
    upd = f"{c.updated.day} {MON[c.updated.month]} {c.updated.year}"
    return f"""<header class="hero"><div class="wrap">
<div class="eyebrow">A field guide to last-mile robot learning · {e(mon_year(RECENT_FROM))} – {e(mon_year(end_ym))}</div>
<h1>Climbing the <span class="nines">Nines</span></h1>
<p class="dek">A pretrained robot policy that works half the time is a demo. One that works 99% of the time is a product. This guide explains how people close that gap: the hill-climbing methods, the textbook RL ideas under each one, and what changed in the last twelve months.</p>
<div class="meta"><span><b>{len(c.papers)}</b> papers &amp; posts</span><span><b>{len(c.recent)}</b> from the last 12 months</span><span><b>{n_checked}</b> re-checked against the source</span><span><b>{len(c.fams)}</b> method families</span><span>Updated <b>{e(upd)}</b></span></div>
<p class="hero-links"><a href="{e(REPO_URL)}">The Awesome list on GitHub</a> · <a href="#course">The hands-on course</a> · <a href="#papers">Search all papers</a></p>
</div><figure class="topo" aria-label="Illustration: a success-rate landscape drawn as contour lines, with a dashed hill-climbing path from a pretrained policy at about 45% up to the 99% contour."><canvas id="topo"></canvas><figcaption>success-rate landscape · illustrative</figcaption></figure></header>"""


def sec_start(c: Ctx) -> str:
    cards = []
    for pid, why in ba.START_HERE:
        p = c.by_id.get(pid)
        if not p:
            continue
        cards.append(
            f'<article class="pick"><div class="when">{e(mon_year(p["date"]))} · {e(p.get("org", ""))}</div>'
            f'<h3>{ext(p["url"], e(p["short"]))}</h3><p>{c.t(why)}</p><p class="num">{c.t(p["one_line"])}</p>'
            f'<div class="chips">{c.chip(p["family"])}</div></article>')
    return (f'<section class="sec" id="start"><div class="sec-head"><h2>Start here</h2><p class="lede">If you read '
            f'{num_word(len(cards))} things from the past year, read these. Each one is a different way to climb, and '
            f'together they cover most of the field.</p></div><div class="picks">\n' + "\n".join(cards) + "\n</div></section>")


def sec_new(c: Ctx) -> str:
    new = [p for p in c.papers if len(p["date"]) == 10 and p["date"] >= c.new_from]
    items = []
    for p in new[:10]:
        star = '<span class="star" title="must-know">★</span> ' if p.get("key") == "1" else ""
        items.append(f'<li><span class="d">{e(p["date"])}</span><span class="w">{star}{ext(p["url"], e(p["short"]))}'
                     f'<small>{e(p.get("org", ""))}</small></span>{c.chip(p["family"])}</li>')
    more = (f'<p><a class="btn to-idx" href="#papers" data-period="new">Show all {len(new)} in the paper index</a></p>'
            if len(new) > 10 else "")
    body = f'<ul class="newlist">{"".join(items)}</ul>{more}' if new else "<p>Nothing in the window.</p>"
    return (f'<section class="sec" id="new"><div class="sec-head"><h2>What’s new</h2><p class="lede"><b>{len(new)}</b> '
            f'papers and posts dated in the last {c.window} days ({e(c.new_from)} to {e(c.meta["updated"])}). The list '
            f'is swept for new work every week and this page is rebuilt from it.</p></div>{body}</section>')


def sec_loop(c: Ctx) -> str:
    paras = [p.strip() for p in c.ex["the_loop"].split("\n\n") if p.strip()]
    steps, tail = [], []
    for para in paras:
        m = re.match(r"Step (\d+), ([^.]+)\.\s*(.*)", para, re.DOTALL)
        if m:
            steps.append((int(m.group(1)), m.group(2), m.group(3)))
        elif para.startswith("How the other"):
            tail = [ln[2:].strip().rstrip(";.") for ln in para.splitlines() if ln.startswith("- ")]
    cards, ops = [], []
    for n, name, body in steps:
        lines = body.splitlines()
        text = lines[0] if any(ln.startswith("- ") for ln in lines) else body
        if n == 4:
            ops = [ln[2:].strip().rstrip(";.") for ln in lines if ln.startswith("- ")]
        chips = "".join(c.chip(k) for k in STEP_FAMILIES.get(n, []))
        cls = "step improve" if n == 4 else "step"
        cards.append(f'<div class="{cls}" role="listitem"><span class="n">step {n}</span><h4>{e(name.capitalize())}</h4>'
                     f'<p>{c.t(text)}</p><div class="chips">{chips}</div></div>')
    ops_html = "".join(f"<li>{c.t(o[:1].upper() + o[1:])}.</li>" for o in ops)
    tail_html = " ".join(f"{c.t(x[:1].upper() + x[1:])}." for x in tail)
    return (f'<section class="sec" id="loop"><div class="sec-head"><h2>The hill-climbing loop</h2>'
            f'<p class="lede">{c.t(c.ex["thesis"])}</p>{narrative_note()}</div>\n'
            f'<div class="loop" role="list">{"".join(cards)}</div>\n'
            f'<h4 class="mt">The update operators (step 4)</h4><ul class="ops">{ops_html}</ul>\n'
            f'<div class="prose mt"><p class="small">{tail_html}</p></div></section>')


def sec_nines(c: Ctx) -> str:
    lo30 = lb.wilson(30, 30)[0]

    def need(bound: float) -> int:
        n = 1
        while lb.wilson(n, n)[0] < bound:
            n += 1
        return n

    n95, n99 = need(0.95), need(0.99)
    slo, shi = lb.wilson(778, 785)
    return f"""<section class="sec" id="nines"><div class="sec-head"><h2>Measuring the nines</h2><p class="lede">Near the top, the hard part is knowing whether you actually climbed. A small evaluation cannot tell 95% from 99%, and many headline results are 20–50 trials.</p></div>
<div class="nines-grid">
<div class="stat"><div class="big">30/30</div><p>only shows the true success rate is at least <b>{100 * lo30:.1f}%</b> (95% Wilson interval). A perfect score on 30 trials is not 100%.</p></div>
<div class="stat"><div class="big">{n95}</div><p>consecutive successes with no failures are needed before the interval’s lower bound reaches 95%.</p></div>
<div class="stat"><div class="big">{n99}</div><p>consecutive successes are needed for a lower bound of 99%. This is why Sunday reports 785 attempts and Dyna reports 24-hour runs.</p></div>
<div class="stat"><div class="big">778/785</div><p>Sunday’s laundry result gives a 95% interval of <b>[{100 * slo:.1f}%, {100 * shi:.1f}%]</b>. Tight enough to call it a “Solve”.</p></div>
</div>
<div class="calc" aria-labelledby="calc-h"><h4 id="calc-h">Interval calculator</h4><div class="row"><label for="wk">Successes<input id="wk" type="number" min="0" value="27" inputmode="numeric"></label><label for="wn">Trials<input id="wn" type="number" min="1" value="30" inputmode="numeric"></label><div class="out" id="wout" aria-live="polite"></div></div><div class="bar" aria-hidden="true"><div class="ci" id="wci"></div><div class="pt" id="wpt"></div></div><div class="axis" aria-hidden="true"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div><p class="small">Wilson score interval at 95%. Report it next to every success rate, keep search episodes separate from evaluation episodes, and use paired start states when comparing two methods.</p></div>
</section>"""


def sec_ideas(c: Ctx) -> str:
    mat = c.ex.get("idea_matrix", {})
    used = {k for v in mat.values() for k in v.get("core", []) + v.get("some", [])}
    cols = [f["key"] for f in c.fams if f["key"] in used]
    head = "".join(f'<th scope="col"><a href="#fam-{e(k)}">{e(c.short(k))}</a></th>' for k in cols)
    rows = []
    for f in c.found:
        m = mat.get(f["idea"], {})
        refs = " · ".join(ext(u, e(n)) for n, u in (r.split("|", 1) for r in f["refs"].split(";") if "|" in r))
        cells = []
        for k in cols:
            if k in m.get("core", []):
                cells.append('<td class="c"><span class="dotc" role="img" aria-label="core"></span></td>')
            elif k in m.get("some", []):
                cells.append('<td class="c"><span class="dotr" role="img" aria-label="sometimes"></span></td>')
            else:
                cells.append('<td class="c"></td>')
        classic = f'<span class="small">{e(m["classic"])}</span><br>' if m.get("classic") else ""
        rows.append(f'<tr><th scope="row" class="idea"><b>{e(f["idea"])}</b><code>{e(f["equation"])}</code>{classic}'
                    f'<span class="refs">{refs}</span></th>{"".join(cells)}</tr>')
    return (f'<section class="sec" id="ideas"><div class="sec-head"><h2>The base RL ideas</h2><p class="lede">Every 2025–26 '
            f'method is a recombination of about a dozen textbook ideas. This matrix shows which ideas each family is built '
            f'on, with the classic references. The standard text is {ext("http://incompleteideas.net/book/the-book-2nd.html", "Sutton &amp; Barto")} (free online).</p></div>'
            f'<div class="scroll"><table class="matrix"><thead><tr><th scope="col">Idea · core equation · classic methods</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div><div class="legend"><span><span class="dotc"></span> '
            f'the family is built on it</span><span><span class="dotr"></span> some methods in the family use it</span></div></section>')


def sec_guide(c: Ctx) -> str:
    rows = "".join(
        f'<div class="g-row"><div class="sit">{c.t(g["situation"])}</div><div class="use"><span class="lab">Use</span>'
        f'{c.t(g["use"])}</div><div class="why"><span class="lab">Evidence</span>{c.t(g["why"])}</div></div>'
        for g in c.ex["decision_guide"])
    return (f'<section class="sec" id="guide"><div class="sec-head"><h2>Which method when</h2><p class="lede">Pick by what you '
            f'have: how often the base succeeds, whether you have a simulator, how much robot time and human time you can '
            f'spend, and whether you can touch the weights.</p>{narrative_note()}</div><div class="guide">{rows}</div></section>')


def work_card(c: Ctx, p: dict) -> str:
    cls, label, title = BADGES.get(p.get("verified", ""), ("once", e(p.get("verified", "")), ""))
    code = f' · {ext(p["code"], "code")}' if p.get("code") else ""
    return (f'<div class="work"><div class="top"><span class="when">{e(mon_year(p["date"]))}</span>'
            f'<span class="badge {cls}" title="{e(title)}">{label}</span></div><h5>{ext(p["url"], e(p["short"]))}</h5>'
            f'<div class="org">{e(p.get("org", ""))}</div><p class="ex">{c.t(p["one_line"])}</p>'
            f'<div class="foot">{e(p.get("kind", "paper"))}{code}</div></div>')


def sec_families(c: Ctx) -> str:
    nav = "".join(c.chip(f["key"]) for f in c.fams)
    out = [(f'<section class="sec" id="families"><div class="sec-head"><h2>The {num_word(len(c.fams))} families</h2>'
            f'<p class="lede">Each family answers one question. For each: where it sits in the loop, the ideas it is built '
            f'on, when to use it, what changed in the last year, and its must-know works with their headline numbers.</p>'
            f'</div><div class="famnav">{nav}</div>')]
    for f in c.fams:
        items = c.by_fam.get(f["key"], [])
        keys = [p for p in items if p.get("key") == "1"]
        n_recent = sum(1 for p in items if p["date"] >= RECENT_FROM)
        built = "".join(f"<li>{c.t(b)}</li>" for b in f.get("built_on", []))
        works = "".join(work_card(c, p) for p in keys)
        out.append(
            f'<article class="family" id="fam-{e(f["key"])}"><div class="fam-head"><span class="steptag">Loop step · '
            f'{e(f["loop_step"])}</span><h3>{e(f["name"])}</h3><p class="question">{c.t(f["question"])}</p>'
            f'<p class="tagline">{c.t(f["tagline"])}</p><p class="famlinks">{ext(c.fam_url(f["key"]), f"All {len(items)} entries on GitHub →")}'
            f' <span class="small">({n_recent} from {e(mon_year(RECENT_FROM))} on, {len(keys)} must-know)</span> · '
            f'<a class="to-idx" href="#papers" data-fam="{e(f["key"])}">filter the index</a></p></div>'
            f'<div class="fam-body"><div class="notes"><div><h4>Built on</h4><ul class="built">{built}</ul></div>'
            f'<div><h4>Use it when</h4><p>{c.t(f.get("use_when", ""))}</p></div>'
            f'<div><h4>What changed in 2025–26</h4><p>{c.t(f.get("trend", ""))}</p></div></div>'
            + (f'<div><h4>Must-know works</h4><div class="works">{works}</div></div>' if keys else "")
            + "</div></article>")
    out.append("</section>")
    return "\n".join(out)


def swimlane(c: Ctx) -> tuple[str, int]:
    """The SVG and how many papers it plots (papers dated only by year cannot be placed on a month axis)."""
    y0, m0 = int(RECENT_FROM[:4]), int(RECENT_FROM[5:7])
    y1, m1 = c.updated.year, c.updated.month
    months = [(y0 + (m0 - 1 + i) // 12, (m0 - 1 + i) % 12 + 1) for i in range((y1 - y0) * 12 + m1 - m0 + 1)]
    left, right, top, lane = 190.0, 984.0, 28.0, 38.0
    mw = (right - left) / len(months)
    height = top + lane * len(c.fams) + 6
    parts = []
    for i, (y, m) in enumerate(months):
        x = left + i * mw
        label = f"{MON[m]} ’{str(y)[2:]}" if (i == 0 or m == 1 or i == len(months) - 1) else MON[m]
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top:.0f}" x2="{x:.1f}" y2="{height - 6:.0f}"/>'
                     f'<text class="mo" x="{x + mw / 2:.1f}" y="20" text-anchor="middle">{label}</text>')
    idx = {ym: i for i, ym in enumerate(months)}
    dots_p, dots_k, n = [], [], 0
    for li, f in enumerate(c.fams):
        yc = top + lane * li + lane / 2
        items = [p for p in c.by_fam.get(f["key"], []) if p["date"] >= RECENT_FROM and len(p["date"]) >= 7
                 and (int(p["date"][:4]), int(p["date"][5:7])) in idx]
        parts.append(f'<line class="grid" x1="0" y1="{top + lane * (li + 1):.1f}" x2="{right:.0f}" y2="{top + lane * (li + 1):.1f}" style="opacity:.5"/>'
                     f'<text class="lane" x="0" y="{yc + 4:.1f}">{e(c.short(f["key"]))}</text>'
                     f'<text class="cnt" x="180" y="{yc + 4:.1f}" text-anchor="end">{len(items)}</text>')
        for p in items:
            d = p["date"]
            ym = (int(d[:4]), int(d[5:7]))
            ndays = calendar.monthrange(*ym)[1]
            day = int(d[8:10]) if len(d) == 10 else (ndays + 1) // 2
            x = left + (idx[ym] + (day - 0.5) / ndays) * mw
            h = zlib.crc32(p["id"].encode())
            y = yc + ((h % 1000) / 999 - 0.5) * (lane - 14)
            dot = f'<circle class="{"k" if p.get("key") == "1" else "p"}" cx="{x:.1f}" cy="{y:.1f}" r="{4.4 if p.get("key") == "1" else 3.2}" data-id="{e(p["id"])}"/>'
            (dots_k if p.get("key") == "1" else dots_p).append(dot)
            n += 1
    return (f'<div class="scroll"><svg class="swim" viewBox="0 0 1000 {height:.0f}" width="100%" role="img" '
            f'aria-label="Swimlane chart: {n} papers from {e(mon_year(RECENT_FROM))} to {e(mon_year(c.meta["updated"]))}, one row per '
            f'family, one dot per paper; accented dots are must-know works.">' + "".join(parts + dots_p + dots_k) + "</svg></div>", n)


def sec_timeline(c: Ctx) -> str:
    miles = "".join(
        f'<div class="mile"><span class="d">{e(mon_year(m["date"]))}</span><div class="nm">{c.t(m["name"])}'
        f'<small>{c.t(m.get("org", ""))}</small></div><div class="nt">{c.t(m["note"])} {c.chip(m.get("family", ""))}</div></div>'
        for m in c.ex["timeline"])
    svg, n = swimlane(c)
    skipped = len(c.recent) - n
    note = (f' ({num_word(skipped)} more {"is" if skipped == 1 else "are"} dated only by year and not plotted)'
            if skipped else "")
    return (f'<section class="sec" id="timeline"><div class="sec-head"><h2>The last 12 months</h2><p class="lede">'
            f'{n} papers and posts from {e(MONTH[int(RECENT_FROM[5:7])])} {RECENT_FROM[:4]} to '
            f'{e(MONTH[c.updated.month])} {c.updated.year}, one dot each, sorted into family lanes{note}. Accented dots '
            f'are the must-know works. Hover over a dot for its title and click to find it in the index; on a touch '
            f'screen, tap once for the title and again to jump to it.</p></div>{svg}'
            f'<div class="legend"><span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4.5" fill="var(--accent)"/></svg> '
            f'must-know work</span><span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4" fill="var(--dot)"/></svg> '
            f'other paper or post</span><span>Row totals at left.</span></div>'
            f'<h3 class="mt2">Milestones</h3>{narrative_note()}<div class="miles">{miles}</div></section>')


def sec_industry(c: Ctx) -> str:
    cards = "".join(
        f'<article class="orgc"><h3>{e(r["org"])}</h3><div><div class="lab">Recipe</div><p>{c.t(r["recipe"])}</p></div>'
        f'<p class="ev">{c.t(r["evidence"])}</p><div><div class="lab">Disclosed / not disclosed</div>'
        f'<p class="disc">{c.t(r["disclosed"])}</p></div></article>'
        for r in c.ex["industry_recipes"])
    return (f'<section class="sec" id="industry"><div class="sec-head"><h2>How companies climb</h2><p class="lede">What each '
            f'company says its loop is, the evidence it gives, and what it has not disclosed. Most industry numbers bundle '
            f'new pretraining with the post-training loop, so none of them isolates how much hill climbing itself '
            f'contributed.</p>{narrative_note()}</div><div class="orgs">{cards}</div></section>')


def sec_classic(c: Ctx) -> str:
    parts = re.split(r"\n\n\d+\)\s*", c.ex["classic_inside_modern"].strip())
    intro, items = parts[0], parts[1:]
    lis = "".join(f"<li><p>{c.t(x.strip())}</p></li>" for x in items)
    return (f'<section class="sec" id="classic"><div class="sec-head"><h2>Classic search, still inside</h2>'
            f'<p class="lede">{c.t(intro)}</p>{narrative_note()}</div><ol class="classic">{lis}</ol></section>')


def sec_lessons(c: Ctx) -> str:
    cards = "".join(f'<div class="lesson"><h3>{c.t(x["lesson"])}</h3><p>{c.t(x["evidence"])}</p></div>'
                    for x in c.ex["lessons"])
    n = len(c.ex["lessons"])
    return (f'<section class="sec" id="lessons"><div class="sec-head"><h2>{num_word(n).capitalize()} lessons</h2>'
            f'<p class="lede">Patterns that show up across families, with the evidence behind each.</p>{narrative_note()}'
            f'</div><div class="lessons">{cards}</div></section>')


def sec_open(c: Ctx) -> str:
    lis = "".join(f"<li>{c.t(x)}</li>" for x in c.ex["open_problems"])
    return (f'<section class="sec" id="open"><div class="sec-head"><h2>Open problems</h2>{narrative_note()}</div>'
            f'<ul class="open">{lis}</ul></section>')


def sec_tree(c: Ctx) -> str:
    return (f'<section class="sec" id="tree"><div class="sec-head"><h2>Family tree</h2><p class="lede">From textbook ideas to '
            f'classic methods to this year’s papers.</p></div><div class="scroll"><pre class="tree">'
            f'{c.t(c.ex["family_tree_ascii"])}</pre></div></section>')


# The result each chapter leads with in its own doc: pooled over seeds where the chapter has seeds,
# and never a best-of-seeds pick by held-out score (that would select on the held-out set).
CHAPTER_HEADLINES = {
    "00": (r"bc_flow", "base_v1"),
    "01": (r"ars_s\d+", "ARS, pooled over 3 seeds"),
    "02": (r"cmaes_s\d+", "CMA-ES, pooled over 3 seeds"),
    "03": (r"racing_s1", "peek-free pick (racing), chosen on the search set"),
    "05": (r"qc_s\d+", "Q-chunking best-of-32, pooled over 3 seeds"),
    "06": (r"residual-td3_v1_s\d+", "residual TD3, pooled over 3 seeds"),
    "07": (r"noise-steering-best", "best steering run, chosen on the search set"),
    "08": (r"expo-b0\.1_s\d+", "EXPO edit (β = 0.1, chosen on search), pooled over 3 seeds"),
    "12": (r"bon-q-n32_s\d+", "best-of-32 with a learned verifier, pooled over 3 seeds"),
}


def chapter_results() -> dict[str, dict]:
    """Each chapter's headline result from git-tracked results/chNN/*.json (quick runs left out):
    {"k", "n", "lo", "hi", "label", "base_sr"} with k and n summed over the headline's seeds."""
    rows: dict[str, list[lb.Run]] = defaultdict(list)
    for rel in git_files("results"):
        if not rel.endswith(".json"):
            continue
        try:
            data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
            if lb.is_quick_run(data):
                continue
            run = lb.parse_run(ROOT / rel, data)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        nn = run.chapter.removeprefix("ch")
        if nn not in CHAPTER_HEADLINES or run.variant != "v1":
            continue
        if run.final.init_set not in (None, "eval") or run.final.init_set_version not in (None, lb.INIT_SET_VERSION):
            continue
        if re.fullmatch(CHAPTER_HEADLINES[nn][0], run.method) and run.final.k is not None and run.final.n:
            rows[nn].append(run)
    out = {}
    for nn, runs in rows.items():
        k, n = sum(r.final.k for r in runs), sum(r.final.n for r in runs)
        lo, hi = lb.wilson(k, n)
        base = next((r.base.sr for r in runs if r.base is not None), None)
        out[nn] = {"k": k, "n": n, "lo": lo, "hi": hi, "label": CHAPTER_HEADLINES[nn][1], "base_sr": base}
    return out


def sec_course(c: Ctx) -> str:
    # Same rule as build_awesome.sec_handson: nothing counts as available until meta.json says the code is out.
    released = bool(c.meta.get("code_released", True))
    algos = {Path(f).name[:2] for f in git_files("algos") if f.endswith(".py")} if released else set()
    docs = set(git_files("docs/chapters"))
    best = chapter_results()
    rows = []
    for n, method, idea, mirrors, release in ba.CHAPTERS:
        nn = f"{int(n):02d}"
        avail = nn in algos
        doc = f"docs/chapters/ch{nn}.md"
        name = ext(f"{REPO_URL}/blob/main/{doc}", e(method)) if doc in docs else e(method)
        status = ('<span class="pill ok">available</span>' if avail
                  else f'<span class="pill">planned · {e(release)}</span>')
        h = best.get(nn) if avail else None
        if h:
            base = f' · base {lb._pct(h["base_sr"])}%' if h["base_sr"] is not None and nn != "00" else ""
            res = (f'<b>{100 * h["k"] / h["n"]:.1f}% [{100 * h["lo"]:.1f}, {100 * h["hi"]:.1f}]</b>'
                   f'<span class="small">{e(h["label"])} · {h["k"]}/{h["n"]}{e(base)}</span>')
        else:
            res = '<span class="small">—</span>'
        rows.append(f'<tr><td class="n">{e(n)}</td><td class="m">{name}</td><td class="b">{e(idea)}</td>'
                    f'<td>{e(mirrors)}</td><td>{status}</td><td class="r">{res}</td></tr>')
    n_avail = sum(1 for ch in ba.CHAPTERS if f"{int(ch[0]):02d}" in algos)
    table = "\n".join(rows)
    return f"""<section class="sec" id="course"><div class="sec-head"><h2>Build it yourself</h2><p class="lede">This guide is the map for <b>lastmile</b>, an open-source course in the same repo: one robot, one task, one base policy that works about half the time, and {num_word(len(ba.CHAPTERS))} ways to push it toward 95%+, each scored by success and by what it cost in robot-minutes and human-minutes.</p></div>
<div class="prose"><p>A low-cost <b>SO-101</b> arm (the AgileX PiPER is supported in simulation too) picks up a cube and drops it in a cup. Everything runs in MuJoCo on a laptop without an NVIDIA GPU, and each chapter then gets a real-arm step. Every run reports a Wilson interval on fixed held-out start states, robot-minutes (search, training and evaluation kept separate) and human-minutes.</p>
<p><b>{n_avail}</b> of {len(ba.CHAPTERS)} chapters are available. The result column is the headline each chapter leads with, on the held-out set with its 95% Wilson interval: pooled over seeds where there are seeds, and never the best seed picked by its held-out score. The chapter notes have every run and what went wrong.</p></div>
<div class="scroll mt"><table class="chapters"><thead><tr><th>Ch</th><th>Method</th><th>Base idea it teaches</th><th>Papers it mirrors</th><th>Status</th><th>Headline result [95% CI]</th></tr></thead><tbody>
{table}
</tbody></table></div>
<p class="mt"><a class="btn" href="{e(REPO_URL)}#hands-on-lastmile">Get the code</a> <a class="btn ghost" href="{e(REPO_URL)}/blob/main/docs/DESIGN.md">Design contract</a> <a class="btn ghost" href="{e(REPO_URL)}/tree/main/docs/chapters">Chapter notes</a></p>
</section>"""


def sec_reading(c: Ctx) -> str:
    groups = []
    for title, cats in ba.RESOURCE_SECTIONS:
        items = sorted((r for r in c.res if r.get("category") in cats), key=lambda r: r["name"].lower())
        if not items:
            continue
        lis = "".join(
            f'<li>{ext(r["url"], e(r["name"]))}'
            + (f' <span class="small">({e(r["org"])})</span>' if r.get("org") else "")
            + f'<span class="ol">{c.t(r.get("one_line", ""))}</span></li>' for r in items)
        groups.append(f'<details class="res"><summary>{e(title)} <span class="small">({len(items)})</span></summary>'
                      f'<ul>{lis}</ul></details>')
    n = sum(1 for r in c.res if any(r.get("category") in cats for _, cats in ba.RESOURCE_SECTIONS))
    return (f'<section class="sec" id="reading"><div class="sec-head"><h2>Code, tools and reading</h2><p class="lede">'
            f'{n} resources from the list: code to start from, simulators, benchmarks, courses, surveys, talks and related '
            f'Awesome lists. Open a group to see them.</p></div><div class="resg">{"".join(groups)}</div></section>')


def sec_papers(c: Ctx) -> str:
    opts = "".join(f'<option value="{e(f["key"])}">{e(c.short(f["key"]))}</option>' for f in c.fams)
    rows = []
    for p in c.papers:
        cls, label, title = BADGES.get(p.get("verified", ""), ("once", e(p.get("verified", "")), ""))
        star = '<span class="star" title="must-know">★</span> ' if p.get("key") == "1" else ""
        name = e(p["title"])
        code = f' · {ext(p["code"], "code")}' if p.get("code") else ""
        rows.append(
            f'<tr id="p-{e(p["id"])}" data-fam="{e(p["family"])}" data-date="{e(p["date"])}" data-key="{e(p.get("key", "0"))}">'
            f'<td class="d">{e(p["date"])}</td><td class="t">{star}{ext(p["url"], name)}<span class="ol">{c.t(p["one_line"])}{code}</span></td>'
            f'<td class="o">{e(p.get("org", ""))}</td><td class="f">{e(c.short(p["family"]))}</td>'
            f'<td class="s"><span class="badge {cls}" title="{e(title)}">{label}</span></td></tr>')
    return f"""<section class="sec" id="papers"><div class="sec-head"><h2>All {len(c.papers)} papers</h2><p class="lede">Every paper and post in the list, newest first. Filter by family or period, or search titles, labs and descriptions. “Checked” means an independent pass re-read the source and confirmed or corrected the numbers.</p></div>
<div class="filters"><label for="q">Search<input id="q" type="search" placeholder="e.g. EXPO, residual, Physical Intelligence"></label><label for="fam">Family<select id="fam"><option value="">All families</option>{opts}</select></label><label for="per">Period<select id="per"><option value="">All time</option><option value="year" data-from="{e(RECENT_FROM)}">Since {e(mon_year(RECENT_FROM))}</option><option value="new" data-from="{e(c.new_from)}">Last {c.window} days</option></select></label><label class="cb" for="key"><input id="key" type="checkbox"> Must-know only</label><span class="count" id="cnt" aria-live="polite"></span></div>
<div class="idx-box"><table class="idx" id="idx"><thead><tr><th>Date</th><th>Paper or post</th><th>Lab</th><th>Family</th><th>Status</th></tr></thead><tbody>
{chr(10).join(rows)}
</tbody></table></div></section>"""


def sec_footer(c: Ctx) -> str:
    n_deep = sum(1 for p in c.papers if p.get("verified") != "read-once")
    n_checked = sum(1 for p in c.papers if p.get("verified") in ("checked", "corrected"))
    upd = f"{c.updated.day} {MONTH[c.updated.month]} {c.updated.year}"
    return f"""<footer class="method" id="method"><div class="wrap"><div class="prose"><h2>How this was made</h2>
<p>The list started from an agent-assisted literature sweep in September 2026: many search angles plus rounds of gap-finding (arXiv month by month, citation mining, company blogs and talks), and it is swept for new work every week. Of the {len(c.papers)} papers and posts, {n_deep} were read in full into structured notes and {n_checked} were re-checked against their source by a second pass, with corrections applied and flagged. The rest, mostly from the newest weeks, were read once from the source and get the full check in later updates.</p>
<p>The narrative sections (the loop, the decision guide, milestones, industry recipes, classic search, lessons, open problems and the family tree) were written from the September 2026 deep-read of the core papers and fact-checked against them. Everything else on this page, from the counts and family sections to the timeline, the course table and the paper index, is regenerated from the repository’s data files by <code>tools/build_site.py</code> whenever the list changes.</p>
<p>Numbers are as reported by the authors. Many are read from bar charts, most real-robot results use 20–60 trials, and industry numbers are self-reported with no ablations. Treat this as a map, and check the paper before you cite a number. Research and drafting were assisted by Claude (Anthropic).</p>
<p class="small">Updated {e(upd)} · <a href="{e(REPO_URL)}">source on GitHub</a> · <a href="{e(REPO_URL)}/blob/main/CONTRIBUTING.md">corrections and additions welcome</a> · MIT license</p></div></div></footer>"""


TOC = [("start", "Start here"), ("new", "What’s new"), ("loop", "The hill-climbing loop"), ("nines", "Measuring the nines"),
       ("ideas", "The base RL ideas"), ("guide", "Which method when"), ("families", "The families"),
       ("timeline", "The last 12 months"), ("industry", "How companies climb"), ("classic", "Classic search, still inside"),
       ("lessons", "Lessons"), ("open", "Open problems"), ("tree", "Family tree"), ("course", "Build it yourself"), ("reading", "Code, tools and reading"),
       ("papers", "All papers"), ("method", "How this was made")]


def render() -> str:
    c = Ctx()
    toc = []
    for sid, label in TOC:
        if sid == "families":
            label = f"The {num_word(len(c.fams))} families"
        elif sid == "lessons":
            label = f"{num_word(len(c.ex['lessons'])).capitalize()} lessons"
        elif sid == "papers":
            label = f"All {len(c.papers)} papers"
        toc.append((sid, label))
    side = []
    for sid, label in toc:
        side.append(f'<a href="#{sid}">{e(label)}</a>')
        if sid == "families":
            side.append('<div class="sub">' + "".join(f'<a href="#fam-{e(f["key"])}">{e(c.short(f["key"]))}</a>'
                                                      for f in c.fams) + "</div>")
    mobile = "".join(f'<li><a href="#{sid}">{e(label)}</a></li>' for sid, label in toc)
    desc = (f"How robot policies hill-climb from ~50% to 99%: the last-mile methods, the textbook RL ideas they come from, "
            f"and {len(c.papers)} papers and posts, updated {c.meta['updated']}.")
    body = "\n".join([
        sec_hero(c),
        f'<div class="wrap layout"><aside class="toc"><nav aria-label="Contents">{"".join(side)}</nav></aside><main>',
        f'<details class="toc-mobile"><summary>Contents</summary><ol>{mobile}</ol></details>',
        sec_start(c), sec_new(c), sec_loop(c), sec_nines(c), sec_ideas(c), sec_guide(c), sec_families(c),
        sec_timeline(c), sec_industry(c), sec_classic(c), sec_lessons(c), sec_open(c), sec_tree(c), sec_course(c),
        sec_reading(c), sec_papers(c), "</main></div>", sec_footer(c), '<div id="tip" hidden></div>',
    ])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Climbing the Nines</title>
<meta name="description" content="{e(desc)}">
<meta property="og:title" content="Climbing the Nines">
<meta property="og:description" content="{e(desc)}">
<meta property="og:url" content="{PAGES_URL}">
<meta property="og:type" content="article">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cpath d='M2 28 L11 14 L16 20 L23 6 L30 28 Z' fill='%231C58C0'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&amp;family=JetBrains+Mono:wght@400;600&amp;family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,400&amp;display=swap">
<style>
{CSS.strip()}
</style>
</head>
<body>
<!-- Generated by tools/build_site.py from data/; do not edit by hand. -->
{body}
<script>
{JS.strip()}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------- assets
CSS = r"""
:root{
  /* A field survey: sticky contents rail, a ~68ch reading column, wide figures break out. Topographic motif. */
  --paper:#F1F4EE; --panel:#E6EBE2; --panel-2:#DCE3D7; --ink:#17211C; --ink-2:#48564E; --muted:#6F7D75;
  --rule:#CAD3C7; --contour:#A7B4A2; --accent:#1C58C0; --accent-ink:#1646A0; --accent-soft:#DAE4F6;
  --ok:#1E7236; --fix:#8F5A00; --dot:#9AA79D;
  --f-display:"Bricolage Grotesque","Avenir Next","Segoe UI",system-ui,sans-serif;
  --f-body:"Newsreader","Iowan Old Style",Georgia,serif;
  --f-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --s-1:.8125rem; --s0:1.0625rem; --s1:1.25rem; --s2:1.6rem; --s3:2.2rem; --s4:clamp(2.6rem,6vw,4.6rem);
  --radius:6px; color-scheme:light;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#0E1411; --panel:#161F1A; --panel-2:#1D2822; --ink:#E2E9E4; --ink-2:#A5B2AA; --muted:#7F8D85;
  --rule:#2A3630; --contour:#33433A; --accent:#86A8FF; --accent-ink:#A9C1FF; --accent-soft:#1A2743;
  --ok:#62C27E; --fix:#E3A83A; --dot:#4E5E55; color-scheme:dark;
}}
:root[data-theme="dark"]{
  --paper:#0E1411; --panel:#161F1A; --panel-2:#1D2822; --ink:#E2E9E4; --ink-2:#A5B2AA; --muted:#7F8D85;
  --rule:#2A3630; --contour:#33433A; --accent:#86A8FF; --accent-ink:#A9C1FF; --accent-soft:#1A2743;
  --ok:#62C27E; --fix:#E3A83A; --dot:#4E5E55; color-scheme:dark;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;-webkit-text-size-adjust:100%}
@media (prefers-reduced-motion: reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--f-body);font-size:var(--s0);line-height:1.6;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;overflow-wrap:break-word}
img{max-width:100%} [hidden]{display:none!important}
a{color:var(--accent-ink);text-decoration-thickness:1px;text-underline-offset:2px}
a:hover{text-decoration-thickness:2px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px}
h1,h2,h3,h4,h5{font-family:var(--f-display);line-height:1.12;text-wrap:balance;margin:0;letter-spacing:-.01em}
h2{font-size:var(--s3);font-weight:750}
h3{font-size:var(--s2);font-weight:700}
h4{font-size:var(--s-1);font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-2)}
p{margin:0}
code,.mono{font-family:var(--f-mono);font-size:.86em}
.wrap{max-width:1240px;margin:0 auto;padding-inline:clamp(16px,3vw,40px)}
.mt{margin-top:18px} .mt2{margin-top:34px;font-size:1.3rem}
.src{font-family:var(--f-mono);font-size:11px;letter-spacing:.03em;color:var(--muted)}

/* hero */
.hero{padding-block:clamp(28px,5vw,56px) 0}
.eyebrow{font-family:var(--f-mono);font-size:var(--s-1);letter-spacing:.06em;color:var(--ink-2);text-transform:uppercase}
.hero h1{font-size:var(--s4);font-weight:800;letter-spacing:-.035em;margin-top:10px;line-height:.98}
.hero h1 .nines{color:var(--accent)}
.dek{font-size:clamp(1.15rem,1.8vw,1.4rem);line-height:1.45;max-width:46ch;margin-top:18px;color:var(--ink)}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:18px;font-family:var(--f-mono);font-size:var(--s-1);color:var(--ink-2)}
.meta b{color:var(--ink);font-weight:600}
.hero-links{margin-top:14px;font-family:var(--f-display);font-size:.95rem}
.topo{position:relative;margin:22px 0 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.topo canvas{display:block;width:100%;height:clamp(170px,22vw,270px)}
.topo figcaption{position:absolute;right:8px;bottom:6px;font-family:var(--f-mono);font-size:11px;color:var(--muted);background:var(--paper);padding:0 4px}

/* layout */
.layout{display:grid;grid-template-columns:minmax(0,1fr);gap:40px;padding-block:36px 80px}
@media (min-width:1100px){.layout{grid-template-columns:210px minmax(0,1fr)}}
.toc{display:none}
@media (min-width:1100px){.toc{display:block}}
.toc nav{position:sticky;top:calc(env(safe-area-inset-top,0px) + 20px);max-height:calc(100vh - 40px);overflow:auto;
  font-family:var(--f-display);font-size:.9rem;padding-right:8px}
.toc a{display:block;color:var(--ink-2);text-decoration:none;padding:3px 0 3px 10px;border-left:2px solid transparent}
.toc a:hover{color:var(--ink);border-left-color:var(--rule)}
.toc a.on{color:var(--ink);border-left-color:var(--accent)}
.toc .sub a{font-size:.82rem;padding-left:20px}
.toc-mobile{margin-bottom:8px}
@media (min-width:1100px){.toc-mobile{display:none}}
.toc-mobile summary{font-family:var(--f-display);font-weight:700;cursor:pointer}
.toc-mobile ol{columns:2;column-gap:20px;font-family:var(--f-display);font-size:.95rem;padding-left:18px}
@media (max-width:480px){.toc-mobile ol{columns:1}}
main{min-width:0}
section.sec{padding-block:44px 8px;border-top:1px solid var(--rule)}
section.sec:first-of-type{border-top:0;padding-top:0}
.sec-head{display:grid;gap:10px;margin-bottom:22px;max-width:70ch}
.lede{font-size:var(--s1);line-height:1.5;color:var(--ink)}
.prose{max-width:68ch;display:grid;gap:14px}
.small{font-size:var(--s-1);color:var(--ink-2)}
.btn{display:inline-block;font-family:var(--f-display);font-weight:600;font-size:.92rem;padding:7px 14px;border-radius:99px;
  background:var(--accent);color:var(--paper);text-decoration:none;margin:0 6px 6px 0}
.btn:hover{filter:brightness(1.08)}
.btn.ghost{background:transparent;color:var(--accent-ink);border:1px solid var(--accent)}

/* start-here picks */
.picks{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(270px,100%),1fr));gap:14px}
.pick{border:1px solid var(--rule);border-radius:var(--radius);padding:16px 16px 14px;background:var(--panel);display:grid;gap:8px;align-content:start;min-width:0}
.pick .when{font-family:var(--f-mono);font-size:11.5px;color:var(--ink-2);letter-spacing:.02em}
.pick h3{font-size:1.18rem}
.pick p{font-size:.98rem;line-height:1.5}
.pick .num{font-size:.9rem;color:var(--ink);background:var(--accent-soft);padding:7px 9px;border-radius:4px;line-height:1.45}

/* what's new */
.newlist{list-style:none;padding:0;margin:0 0 14px;display:grid;gap:0;border-top:1px solid var(--rule)}
.newlist li{display:grid;grid-template-columns:96px minmax(0,1fr) auto;gap:12px;align-items:baseline;padding:8px 0;border-bottom:1px solid var(--rule)}
@media (max-width:640px){.newlist li{grid-template-columns:minmax(0,1fr) auto}.newlist .d{grid-column:1/-1}}
.newlist .d{font-family:var(--f-mono);font-size:12px;color:var(--ink-2)}
.newlist .w{font-family:var(--f-display);font-weight:650}
.newlist .w a{color:var(--ink)}
.newlist small{display:block;font-weight:500;color:var(--ink-2);font-size:.8rem}
.star{color:var(--accent)}

/* loop */
.loop{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;border:1px solid var(--rule);border-radius:var(--radius);overflow:hidden;background:var(--rule)}
@media (max-width:860px){.loop{grid-template-columns:minmax(0,1fr)}}
.step{padding:14px 14px 16px;display:grid;gap:8px;align-content:start;background:var(--panel)}
.step .n{font-family:var(--f-mono);font-size:12px;color:var(--accent-ink)}
.step h4{color:var(--ink);font-size:.95rem;letter-spacing:.04em}
.step p{font-size:.93rem;line-height:1.5;color:var(--ink-2)}
.step.improve{background:var(--accent-soft)}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{font-family:var(--f-display);font-size:.74rem;font-weight:600;padding:2px 8px;border-radius:99px;border:1px solid var(--rule);background:var(--paper);color:var(--ink);text-decoration:none;white-space:nowrap}
a.chip:hover{border-color:var(--accent)}
.ops{display:grid;gap:8px;margin-top:10px;max-width:78ch;padding-left:20px}

/* nines */
.nines-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr));gap:12px;margin-top:6px}
.stat{border-left:3px solid var(--accent);padding:6px 12px}
.stat .big{font-family:var(--f-display);font-size:2rem;font-weight:800;letter-spacing:-.02em;line-height:1.05}
.stat p{font-size:.94rem;color:var(--ink-2);margin-top:4px}
.calc{margin-top:22px;border:1px solid var(--rule);border-radius:var(--radius);padding:16px;background:var(--panel);display:grid;gap:12px;max-width:760px}
.calc h4{color:var(--ink)}
.calc .row{display:flex;flex-wrap:wrap;gap:12px;align-items:end}
.calc label{display:grid;gap:4px;font-family:var(--f-display);font-size:.85rem;font-weight:600}
.calc input{font-family:var(--f-mono);font-size:1rem;width:110px;padding:6px 8px;border:1px solid var(--rule);border-radius:4px;background:var(--paper);color:var(--ink)}
.calc .out{font-family:var(--f-mono);font-size:.95rem}
.bar{position:relative;height:18px;border-radius:3px;background:var(--panel-2);overflow:hidden}
.bar .ci{position:absolute;top:0;bottom:0;background:var(--accent);opacity:.35}
.bar .pt{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--ink)}
.axis{display:flex;justify-content:space-between;font-family:var(--f-mono);font-size:11px;color:var(--muted)}

/* matrix */
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%}
table{border-collapse:collapse;width:100%}
.matrix{min-width:900px;font-size:.92rem}
.matrix th,.matrix td{border-bottom:1px solid var(--rule);padding:8px 6px;vertical-align:top}
.matrix thead th{font-family:var(--f-display);font-size:.74rem;font-weight:700;text-align:center;vertical-align:bottom;color:var(--ink-2);line-height:1.2}
.matrix thead th a{color:inherit;text-decoration:none}
.matrix thead th:first-child{text-align:left}
.matrix td.c{text-align:center;vertical-align:middle}
.matrix .idea{text-align:left;font-weight:400;min-width:260px}
.matrix .idea b{font-family:var(--f-display);font-weight:700;display:block}
.matrix .idea code{display:block;color:var(--ink-2);font-size:.8rem}
.matrix .refs{font-size:.8rem}
.dotc{display:inline-block;width:12px;height:12px;border-radius:50%;background:var(--accent)}
.dotr{display:inline-block;width:11px;height:11px;border-radius:50%;border:2px solid var(--accent);opacity:.8}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:var(--s-1);color:var(--ink-2);margin-top:10px;align-items:center}
.legend span{display:inline-flex;gap:6px;align-items:center}

/* guide */
.guide{display:grid;gap:0;border-top:1px solid var(--rule)}
.g-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr) minmax(0,1.3fr);gap:18px;padding:14px 0;border-bottom:1px solid var(--rule)}
@media (max-width:860px){.g-row{grid-template-columns:minmax(0,1fr);gap:6px}}
.g-row .sit{font-family:var(--f-display);font-weight:600;font-size:.98rem;line-height:1.35}
.g-row .use{font-size:.96rem}
.g-row .why{font-size:.9rem;color:var(--ink-2)}
.g-row .lab,.orgc .lab{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:2px}

/* families */
.famnav{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.family{padding-block:34px 10px;border-top:1px dashed var(--rule)}
.famnav + .family{border-top:0}
.fam-head{display:grid;gap:8px;max-width:72ch}
.steptag{font-family:var(--f-mono);font-size:11.5px;color:var(--accent-ink);letter-spacing:.03em}
.question{font-size:var(--s1);line-height:1.4;font-style:italic}
.tagline{color:var(--ink-2)}
.famlinks{font-family:var(--f-display);font-size:.92rem}
.fam-body{display:grid;gap:26px;margin-top:22px}
.notes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}
@media (max-width:1000px){.notes{grid-template-columns:minmax(0,1fr)}}
.notes p{font-size:.93rem;color:var(--ink);margin-top:8px}
.built{margin:8px 0 0;padding-left:18px;display:grid;gap:4px;font-size:.93rem}
.works{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(310px,100%),1fr));gap:12px;margin-top:10px}
.work{border:1px solid var(--rule);border-radius:var(--radius);padding:14px 14px 12px;display:grid;gap:8px;align-content:start;background:var(--paper);min-width:0}
.work .top{display:flex;justify-content:space-between;gap:8px;align-items:baseline;flex-wrap:wrap}
.work .when{font-family:var(--f-mono);font-size:11.5px;color:var(--ink-2)}
.work h5{font-size:1.08rem;font-weight:700;line-height:1.2}
.work h5 a{color:var(--ink);text-decoration:none}
.work h5 a:hover{text-decoration:underline}
.work .org{font-size:.84rem;color:var(--ink-2)}
.work .ex{font-size:.94rem;line-height:1.5}
.work .foot{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;font-size:.8rem;color:var(--ink-2);font-family:var(--f-mono)}
.badge{font-family:var(--f-mono);font-size:10.5px;padding:1px 6px;border-radius:3px;border:1px solid currentColor;white-space:nowrap}
.badge.ok{color:var(--ok)} .badge.fix{color:var(--fix)} .badge.classic,.badge.once{color:var(--muted)}

/* timeline swimlane */
.swim{min-width:900px;display:block}
.swim text{font-family:var(--f-display);fill:var(--ink-2)}
.swim .lane{font-size:12px;font-weight:600;fill:var(--ink)}
.swim .cnt{font-family:var(--f-mono);font-size:10.5px;fill:var(--muted)}
.swim .mo{font-family:var(--f-mono);font-size:10.5px;fill:var(--muted)}
.swim .grid{stroke:var(--rule);stroke-width:1}
.swim circle{stroke:transparent;stroke-width:5px;cursor:pointer}
.swim .p{fill:var(--dot)}
.swim .k{fill:var(--accent)}
.swim circle:hover{fill:var(--ink)}
#tip{position:fixed;z-index:10;pointer-events:none;max-width:320px;background:var(--ink);color:var(--paper);font-family:var(--f-display);font-size:.8rem;line-height:1.35;padding:7px 9px;border-radius:5px;box-shadow:0 4px 18px rgba(0,0,0,.18)}
#tip .d{font-family:var(--f-mono);font-size:10.5px;opacity:.75;display:block;margin-bottom:2px}
.miles{display:grid;gap:0;margin-top:10px;border-top:1px solid var(--rule)}
.mile{display:grid;grid-template-columns:86px minmax(0,220px) minmax(0,1fr);gap:14px;padding:10px 0;border-bottom:1px solid var(--rule);align-items:baseline}
@media (max-width:760px){.mile{grid-template-columns:minmax(0,1fr);gap:3px}}
.mile .d{font-family:var(--f-mono);font-size:12px;color:var(--ink-2)}
.mile .nm{font-family:var(--f-display);font-weight:700;line-height:1.25}
.mile .nm small{display:block;font-weight:500;color:var(--ink-2);font-size:.78rem;margin-top:2px}
.mile .nt{font-size:.93rem}

/* industry */
.orgs{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(330px,100%),1fr));gap:14px}
.orgc{border:1px solid var(--rule);border-radius:var(--radius);padding:16px;display:grid;gap:10px;align-content:start;background:var(--panel);min-width:0}
.orgc h3{font-size:1.25rem}
.orgc p{font-size:.94rem;line-height:1.5}
.orgc .ev{font-family:var(--f-mono);font-size:.8rem;background:var(--accent-soft);padding:7px 9px;border-radius:4px}
.orgc .disc{font-size:.88rem;color:var(--ink-2)}

/* classic, lessons, open, tree */
.classic{display:grid;gap:14px;max-width:80ch;counter-reset:c;padding:0;margin:0}
.classic li{list-style:none;display:grid;grid-template-columns:34px minmax(0,1fr);gap:10px}
.classic li::before{counter-increment:c;content:counter(c);font-family:var(--f-display);font-weight:800;font-size:1.3rem;color:var(--accent)}
.classic p{font-size:.97rem}
.lessons{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(340px,100%),1fr));gap:14px}
.lesson{border-top:2px solid var(--ink);padding-top:10px;display:grid;gap:8px;align-content:start;min-width:0}
.lesson h3{font-size:1.1rem;line-height:1.3}
.lesson p{font-size:.9rem;color:var(--ink-2)}
.open{display:grid;gap:10px;max-width:80ch;padding-left:20px}
.open li{font-size:.97rem}
pre.tree{font-family:var(--f-mono);font-size:12.5px;line-height:1.5;background:var(--panel);border:1px solid var(--rule);border-radius:var(--radius);padding:16px;margin:0;min-width:820px;color:var(--ink)}

/* course */
.chapters{min-width:880px;font-size:.9rem}
.chapters th{font-family:var(--f-display);font-size:.76rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-2);text-align:left;padding:8px 10px 8px 0;border-bottom:2px solid var(--ink)}
.chapters td{padding:9px 10px 9px 0;border-bottom:1px solid var(--rule);vertical-align:top}
.chapters td.n{font-family:var(--f-display);font-weight:800;color:var(--accent);font-size:1.05rem}
.chapters td.m{font-family:var(--f-display);font-weight:700}
.chapters td.b{color:var(--ink-2)}
.chapters td.r{min-width:150px}
.chapters td.r b{font-family:var(--f-mono);font-size:.82rem;font-weight:600;display:block}
.chapters td.r .small{display:block;font-family:var(--f-mono);font-size:11px}
@media (max-width:760px){
  .chapters{min-width:0}
  .chapters thead{display:none}
  .chapters tr{display:grid;grid-template-columns:30px minmax(0,1fr);gap:3px 10px;padding:10px 0;border-bottom:1px solid var(--rule)}
  .chapters td{border:0;padding:0;grid-column:2}
  .chapters td.n{grid-column:1;grid-row:1 / span 5}
}
.pill{font-family:var(--f-mono);font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid var(--rule);color:var(--ink-2);white-space:nowrap}
.pill.ok{border-color:var(--ok);color:var(--ok)}

/* paper index */
.filters{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:end;margin-bottom:12px}
.filters label{display:grid;gap:4px;font-family:var(--f-display);font-size:.82rem;font-weight:600;color:var(--ink-2);min-width:0}
.filters input[type=search],.filters select{font-family:var(--f-body);font-size:1rem;padding:6px 8px;border:1px solid var(--rule);border-radius:4px;background:var(--paper);color:var(--ink);min-width:0;max-width:100%}
.filters input[type=search]{width:min(340px,calc(100vw - 40px))}
.filters .cb{display:flex;gap:6px;align-items:center;font-size:.9rem;color:var(--ink)}
.count{font-family:var(--f-mono);font-size:var(--s-1);color:var(--ink-2)}
.idx-box{max-height:760px;overflow:auto;border:1px solid var(--rule);border-radius:var(--radius);padding:0 12px}
.idx{font-size:.88rem}
.idx th{font-family:var(--f-display);font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-2);text-align:left;padding:8px 10px 8px 0;border-bottom:2px solid var(--ink);position:sticky;top:0;background:var(--paper);z-index:1}
.idx td{padding:7px 10px 7px 0;border-bottom:1px solid var(--rule);vertical-align:top}
.idx td.d{font-family:var(--f-mono);font-size:11.5px;color:var(--ink-2);white-space:nowrap;font-variant-numeric:tabular-nums}
.idx td.t a{color:var(--ink)}
.idx td.t{min-width:320px}
.idx td.o{font-size:.82rem;color:var(--ink-2);min-width:120px}
.idx td.f{white-space:nowrap;font-family:var(--f-display);font-size:.8rem}
.idx .ol{display:block;font-size:.82rem;color:var(--ink-2);line-height:1.45;margin-top:2px}
.idx tr.hl td{background:var(--accent-soft)}
@media (max-width:760px){
  .idx-box{padding:0 10px}
  .idx thead{display:none}
  .idx tr{display:flex;flex-wrap:wrap;gap:2px 10px;padding:9px 0;border-bottom:1px solid var(--rule)}
  .idx tr[hidden]{display:none}
  .idx td{border:0;padding:0;min-width:0!important}
  .idx td.t,.idx td.o{flex-basis:100%;order:2}
}
/* code, tools and reading */
.resg{display:grid;gap:8px}
.res{border:1px solid var(--rule);border-radius:var(--radius);background:var(--panel);min-width:0}
.res summary{cursor:pointer;padding:10px 14px;font-family:var(--f-display);font-weight:700}
.res ul{margin:0;padding:0 14px 12px 32px;display:grid;gap:8px}
.res li{font-size:.92rem;min-width:0;overflow-wrap:anywhere}
.res .ol{display:block;font-size:.84rem;color:var(--ink-2);line-height:1.45}
footer.method{border-top:1px solid var(--rule);padding-block:28px 60px;color:var(--ink-2);font-size:.92rem}
footer.method h2{font-size:1.5rem;color:var(--ink)}
"""

JS = r"""
(function(){
  // ---------- topographic hero
  var cv=document.getElementById('topo');
  function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
  function field(x,y,A){
    var cx=0.79*A, cy=0.42;
    var r2=Math.pow((x-cx)/1.25,2)+Math.pow((y-cy)/0.8,2);
    var main=0.995/(1+r2/3.1);
    var b=0.32*Math.exp(-(Math.pow(x-0.16*A,2)/0.35+Math.pow(y-0.28,2)/0.05));
    var b2=0.22*Math.exp(-(Math.pow(x-0.47*A,2)/0.22+Math.pow(y-0.86,2)/0.04));
    return Math.min(0.999, main+(b+b2)*(1-main));
  }
  function draw(){
    if(!cv) return;
    var dpr=window.devicePixelRatio||1, W=cv.clientWidth, H=cv.clientHeight;
    if(!W||!H) return;
    cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr);
    var g=cv.getContext('2d'); if(!g) return; g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
    var A=W/H, NX=Math.max(60,Math.round(A*34)), NY=34;
    var z=[]; for(var j=0;j<=NY;j++){var row=[];for(var i=0;i<=NX;i++){row.push(field(i/NX*A,j/NY,A));}z.push(row);}
    var levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99];
    var cContour=css('--contour'), cAccent=css('--accent'), cInk=css('--ink'), cMuted=css('--ink-2'), cPaper=css('--paper');
    var sx=W/NX, sy=H/NY, labels={};
    function lerp(a,b,L){return (L-a)/(b-a);}
    levels.forEach(function(L){
      var hi=(L>=0.9); g.beginPath(); g.strokeStyle=hi?cAccent:cContour; g.lineWidth=hi?1.4:1; g.globalAlpha=hi?0.9:1;
      var minPt=null;
      for(var j=0;j<NY;j++){for(var i=0;i<NX;i++){
        var a=z[j][i], b=z[j][i+1], c=z[j+1][i+1], d=z[j+1][i];
        var idx=(a>L?8:0)|(b>L?4:0)|(c>L?2:0)|(d>L?1:0); if(idx===0||idx===15) continue;
        var T=[(i+lerp(a,b,L))*sx, j*sy], R=[(i+1)*sx,(j+lerp(b,c,L))*sy], B=[(i+lerp(d,c,L))*sx,(j+1)*sy], Lf=[i*sx,(j+lerp(a,d,L))*sy];
        var segs={1:[[Lf,B]],2:[[B,R]],3:[[Lf,R]],4:[[T,R]],5:[[Lf,T],[B,R]],6:[[T,B]],7:[[Lf,T]],8:[[Lf,T]],9:[[T,B]],10:[[Lf,B],[T,R]],11:[[T,R]],12:[[Lf,R]],13:[[B,R]],14:[[Lf,B]]}[idx];
        segs.forEach(function(s){g.moveTo(s[0][0],s[0][1]);g.lineTo(s[1][0],s[1][1]);
          [s[0],s[1]].forEach(function(p){ if(p[1]>H*0.2&&p[1]<H*0.8&&(!minPt||p[0]<minPt[0])) minPt=p; });});
      }}
      g.stroke(); g.globalAlpha=1; labels[L]=minPt;
    });
    g.font='600 11px "JetBrains Mono", ui-monospace, monospace'; g.textBaseline='middle';
    [[0.5,'50%'],[0.9,'90%']].forEach(function(p){var pt=labels[p[0]]; if(!pt) return;
      var w=g.measureText(p[1]).width+6; g.fillStyle=cPaper; g.fillRect(pt[0]-w/2,pt[1]-8,w,16);
      g.fillStyle=p[0]>=0.9?cAccent:cMuted; g.textAlign='center'; g.fillText(p[1],pt[0],pt[1]);});
    // stochastic hill climb (deterministic seed)
    var seed=7; function rnd(){seed=(seed*16807)%2147483647;return seed/2147483647;}
    var x=0.30*A, y=0.74, path=[[x,y]], step=0.055;
    for(var k=0;k<400;k++){
      var f0=field(x,y,A); if(f0>0.992) break;
      var best=null,bf=f0;
      for(var t=0;t<6;t++){var ang=rnd()*Math.PI*2, nx=x+Math.cos(ang)*step, ny=y+Math.sin(ang)*step*0.8;
        if(ny<0.04||ny>0.96) continue; var fn=field(nx,ny,A); if(fn>bf){bf=fn;best=[nx,ny];}}
      if(best){x=best[0];y=best[1];path.push([x,y]);}
    }
    var P=path.map(function(p){return [p[0]/A*W,p[1]*H];});
    g.strokeStyle=cInk; g.lineWidth=1.6; g.setLineDash([4,4]); g.beginPath();
    P.forEach(function(p,i){ if(i===0) g.moveTo(p[0],p[1]); else g.lineTo(p[0],p[1]); }); g.stroke(); g.setLineDash([]);
    g.fillStyle=cInk; P.forEach(function(p,i){ if(i%3===0){g.beginPath();g.arc(p[0],p[1],2.1,0,7);g.fill();} });
    var s=P[0], end=P[P.length-1];
    g.beginPath(); g.arc(s[0],s[1],5,0,7); g.fillStyle=cPaper; g.fill(); g.lineWidth=2; g.strokeStyle=cInk; g.stroke();
    g.beginPath(); g.arc(end[0],end[1],5.5,0,7); g.fillStyle=cAccent; g.fill();
    g.font='600 12px "Bricolage Grotesque", system-ui, sans-serif'; g.fillStyle=cInk; g.textAlign='left';
    var sv=Math.round(field(path[0][0],path[0][1],A)*100);
    g.fillText('pretrained policy · '+sv+'%', s[0]+10, s[1]+14);
    g.textAlign='right'; g.fillText('last-mile · 99%+', end[0]-10, end[1]-14);
  }
  if(cv){ draw(); if(window.ResizeObserver){ new ResizeObserver(function(){draw();}).observe(cv); }
    try{ window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);}catch(_){}
    new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
    if(document.fonts&&document.fonts.ready){document.fonts.ready.then(draw);} }

  // ---------- Wilson calculator
  var kIn=document.getElementById('wk'), nIn=document.getElementById('wn'), out=document.getElementById('wout'), ci=document.getElementById('wci'), pt=document.getElementById('wpt');
  function wil(){
    var k=parseInt(kIn.value,10), n=parseInt(nIn.value,10);
    if(!(n>=1)||!(k>=0)||k>n){ out.textContent='Enter successes between 0 and trials (trials ≥ 1).'; ci.style.width='0'; return; }
    var z=1.96, p=k/n, den=1+z*z/n, c=(p+z*z/(2*n))/den, h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;
    var lo=Math.max(0,c-h), hi=Math.min(1,c+h);
    out.textContent=k+'/'+n+' = '+(100*p).toFixed(1)+'%  ·  95% interval ['+(100*lo).toFixed(1)+'%, '+(100*hi).toFixed(1)+'%]';
    ci.style.left=(100*lo)+'%'; ci.style.width=(100*(hi-lo))+'%'; pt.style.left='calc('+(100*p)+'% - 1px)';
  }
  if(kIn&&nIn){ kIn.addEventListener('input',wil); nIn.addEventListener('input',wil); wil(); }

  // ---------- paper index filter
  var q=document.getElementById('q'), fam=document.getElementById('fam'), per=document.getElementById('per'), key=document.getElementById('key'), cnt=document.getElementById('cnt');
  var rows=[].slice.call(document.querySelectorAll('#idx tbody tr'));
  var text=rows.map(function(tr){return tr.textContent.toLowerCase();});
  function filt(){
    if(!q) return;
    var s=(q.value||'').toLowerCase().trim(), f=fam.value, opt=per.options[per.selectedIndex], from=opt?(opt.getAttribute('data-from')||''):'', kk=key.checked, n=0;
    rows.forEach(function(tr,i){ var ok=(!f||tr.getAttribute('data-fam')===f)&&(!from||tr.getAttribute('data-date')>=from)&&(!kk||tr.getAttribute('data-key')==='1')&&(!s||text[i].indexOf(s)>=0);
      tr.hidden=!ok; if(ok) n++; });
    cnt.textContent='Showing '+n+' of '+rows.length;
  }
  if(q){ q.addEventListener('input',filt); fam.addEventListener('change',filt); per.addEventListener('change',filt); key.addEventListener('change',filt); filt(); }
  function showInIndex(opts){
    if(!q) return;
    q.value=opts.q||''; fam.value=opts.fam||''; per.value=opts.period||''; key.checked=false; filt();
    var box=document.querySelector('.idx-box'); if(box) box.scrollTop=0;
  }
  [].forEach.call(document.querySelectorAll('a.to-idx'),function(a){
    a.addEventListener('click',function(){ showInIndex({fam:a.getAttribute('data-fam'),period:a.getAttribute('data-period')}); });
  });

  // ---------- swimlane tooltip
  var tip=document.getElementById('tip');
  function info(id){ var tr=document.getElementById('p-'+id); if(!tr) return null;
    var a=tr.querySelector('td.t a'); return {tr:tr, title:a?a.textContent:'', d:tr.getAttribute('data-date'), f:(tr.querySelector('td.f')||{}).textContent||''}; }
  function move(ev){ var x=ev.clientX+14, y=ev.clientY+14; if(x+330>window.innerWidth) x=Math.max(4,ev.clientX-330); tip.style.left=x+'px'; tip.style.top=y+'px'; }
  // Mouse: hover shows the title, click jumps to the index. Touch: the first tap on a dot shows the title,
  // a second tap on the same dot jumps (a tap fires pointerleave before click, so the click re-shows the tip).
  var armed=null, ptype='mouse';
  function show(el,ev){ var p=info(el.getAttribute('data-id')); if(!p||!tip) return;
    tip.textContent=''; var d=document.createElement('span'); d.className='d'; d.textContent=p.d+' · '+p.f; tip.appendChild(d);
    tip.appendChild(document.createTextNode(p.title)); tip.hidden=false; move(ev); }
  function unarm(){ armed=null; if(tip) tip.hidden=true; }
  document.addEventListener('click',function(ev){ if(armed&&!(ev.target.closest&&ev.target.closest('.swim circle[data-id]'))) unarm(); });
  window.addEventListener('scroll',function(){ if(armed) unarm(); },{passive:true});
  [].forEach.call(document.querySelectorAll('.swim circle[data-id]'),function(el){
    el.addEventListener('pointerenter',function(ev){ ptype=ev.pointerType||'mouse'; if(ptype!=='touch') show(el,ev); });
    el.addEventListener('pointerdown',function(ev){ ptype=ev.pointerType||'mouse'; });
    el.addEventListener('pointermove',function(ev){ if(ptype!=='touch'&&tip&&!tip.hidden) move(ev); });
    el.addEventListener('pointerleave',function(){ if(ptype!=='touch'&&tip) tip.hidden=true; });
    el.addEventListener('click',function(ev){ var p=info(el.getAttribute('data-id')); if(!p) return;
      if(ptype==='touch'&&armed!==el){ armed=el; show(el,ev); return; }
      armed=null; showInIndex({}); rows.forEach(function(r){r.classList.remove('hl');}); p.tr.classList.add('hl');
      document.getElementById('papers').scrollIntoView();
      var box=document.querySelector('.idx-box'); if(box) box.scrollTop=Math.max(0,p.tr.offsetTop-60); if(tip) tip.hidden=true; });
  });

  // ---------- toc highlight
  var links=[].slice.call(document.querySelectorAll('.toc a[href^="#"]'));
  if('IntersectionObserver' in window){
    var map={}; links.forEach(function(a){map[a.getAttribute('href').slice(1)]=a;});
    var io=new IntersectionObserver(function(es){ es.forEach(function(en){ if(en.isIntersecting){ links.forEach(function(a){a.classList.remove('on');}); var a=map[en.target.id]; if(a) a.classList.add('on'); } }); },{rootMargin:'-20% 0px -70% 0px'});
    Object.keys(map).forEach(function(id){var el=document.getElementById(id); if(el) io.observe(el);});
  }
})();
"""


# ---------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the generated page is out of date")
    ap.add_argument("--out", type=Path, default=SITE, help="output directory (default: site/)")
    args = ap.parse_args(argv)
    page = render()
    path = args.out / "index.html"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if page == old:
        print(f"Up to date: {path}")
        return 0
    if args.check:
        print(f"Stale: {path}\nRun `uv run python tools/build_site.py`.", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    print(f"Wrote {path} ({len(page.encode('utf-8')) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
