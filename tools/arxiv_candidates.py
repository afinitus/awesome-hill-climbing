"""List new arXiv submissions that may belong in the Awesome Hill Climbing list (the weekly sweep's first step).

    uv run python tools/arxiv_candidates.py --out cands.jsonl     # window: data/meta.json "updated" - 4 days → today
    uv run python tools/arxiv_candidates.py --since 2026-09-23 --until 2026-09-30 --out /tmp/cands.jsonl
    uv run python tools/arxiv_candidates.py --brief cands.jsonl   # one short line per candidate (triage by title)
    uv run python tools/arxiv_candidates.py --brief cands.jsonl --abstracts 2609.12345,2609.23456   # read these
    uv run python tools/arxiv_candidates.py --mark-seen cands.jsonl   # after triage: never show these ids again
    uv run python tools/arxiv_candidates.py --show-queries        # print the arXiv queries and exit

How it works. Keyword searches on the arXiv API are fuzzy and miss a lot (a test against a hand-curated week
found about 60% of the listed papers), so this tool does not rely on them. It downloads three broad pools of
submissions in the window, in full (about 1,300 papers a week; about 35 requests and 2 minutes, more when the
API is busy):

    robotics            every cs.RO submission (cross-lists included)
    robotics-crosslist  cs.LG / cs.AI / cs.CV submissions that mention robots, VLAs, manipulation, humanoids...
    general-rl          cs.LG / cs.AI / stat.ML / cs.NE submissions about reinforcement learning

and then keeps the papers whose title or abstract contains the list's topic terms (ROBOT_TOPICS for the two
robot pools, GENERAL_RL_TOPICS for the RL pool; whole-word, case-insensitive, hyphen-tolerant matching). Ids
already in data/papers.csv or data/resources.csv are dropped, and so are ids an earlier sweep already triaged
(docs/maintenance/seen-ids.txt, written by --mark-seen). Output is JSONL, newest first, one paper per line:

    {"id", "url", "date", "title", "authors", "categories", "primary_category", "abstract",
     "pools": [...], "matched": [topic names that matched], "score": len(matched),
     "overlap": true if dated before meta.json "updated"}

A printed summary (to stderr when the JSONL goes to stdout) gives counts per pool and new vs already listed.
This is a recall tool, not a judge: most candidates will NOT belong in the list. A person or agent reads each
one and applies the inclusion rules in CONTRIBUTING.md (see docs/maintenance/weekly-sweep.md).

Why the default window starts 4 days before "updated": the API only returns papers once arXiv has announced
them, which can be 1-3 days after the submission date. Overlapping windows catch late announcements; ids that
are already listed or were triaged last week (seen-ids.txt) are dropped, so what "overlap" still marks are
late announcements, to be triaged like any other candidate.

Failing loudly. A sweep that silently misses papers is worse than one that stops: the next window starts where
this one ended, so anything missed is missed for good. The tool exits non-zero (and writes no output) when a
query's pages come back short of the API's totalResults after retries, when a query returns more than MAX_POOL
results (the date filter was ignored), or when the cs.RO pool is implausibly small for the window (an empty or
truncated API answer; see --min-per-day).

Two passes per pool. The API can return an incomplete result set whose totalResults agrees with it, which no
page check can catch: on 2026-09-30 the cs.RO query for 09-23..09-30 returned 629 papers (totalResults 629),
while the eight one-day queries for the same days returned 705, a superset; the missing 76 were the window's
oldest papers. So every pool is fetched once day by day and once in windows of up to 14 days, and the union is
kept; the summary prints both counts, so a disagreement is visible.

API etiquette (https://info.arxiv.org/help/api/tou.html): one request at a time, 3 seconds between requests,
and back off on 429/5xx, as tools/arxiv_meta.py does.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
API = "https://export.arxiv.org/api/query"
PAGE = 200  # results per request; smaller pages fail less often when the API is busy
PAUSE = 3.0  # seconds between requests (arXiv API terms)
OVERLAP_DAYS = 4
SLICE_DAYS = 14  # the windowed pass uses slices of at most this many days, so MAX_POOL holds for any window
MAX_POOL = 5000  # a 14-day slice is ~300-1,300 papers per pool; far more means the date filter was not applied
EMPTY_RETRIES = 4  # an empty page where results are expected is retried this many times (arXiv does this under load)
MIN_RO_PER_DAY = 15  # cs.RO gets ~80 submissions a day (~30 on weekends); fewer than 15 a day means a bad answer
SEEN = ROOT / "docs" / "maintenance" / "seen-ids.txt"
SEEN_KEEP_MONTHS = 6  # --mark-seen forgets ids older than this (they can no longer fall in a window)
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}
ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")

ROBOT_TERMS = ["robot", "robotic", "robotics", "manipulation", "VLA", "vision-language-action", "embodied",
               "humanoid", "legged", "locomotion", "dexterous", "grasping"]
RL_TERMS = ["reinforcement learning", "RL", "policy optimization", "policy gradient", "actor-critic", "Q-learning",
            "offline RL", "GRPO", "PPO", "RLVR", "evolution strategies", "evolutionary strategies"]

# The pools, downloaded in full. (name, category filter, terms the title/abstract must contain or None, topics).
# "ANDNOT cat:cs.RO" keeps the cross-list pools disjoint from "robotics".
POOLS = [
    ("robotics", ["cs.RO"], None, "robot"),
    ("robotics-crosslist", ["cs.LG", "cs.AI", "cs.CV"], ROBOT_TERMS, "robot"),
    ("general-rl", ["cs.LG", "cs.AI", "stat.ML", "cs.NE"], RL_TERMS, "general"),
]

# Topic terms for robot papers, one group per kind of work the list covers. A paper is a candidate if it
# matches any group. Terms in WEAK are common in robotics papers outside the list's scope, so they count only
# when the paper is also about a policy (POLICY_TERMS).
ROBOT_TOPICS: dict[str, list[str]] = {
    "rl": ["reinforcement learning", "RL", "policy gradient", "PPO", "GRPO", "actor-critic", "Q-learning",
           "Q-function", "critic", "off-policy", "on-policy", "offline-to-online"],
    "post-training": ["fine-tuning", "finetuning", "fine-tune", "post-training", "self-improvement",
                      "self-improving", "policy improvement", "improve the policy", "autonomous improvement",
                      "continual learning", "lifelong"],
    "residual-steering": ["residual policy", "residual RL", "residual reinforcement", "residual correction",
                          "residual", "steering", "steer", "noise space", "latent noise", "initial noise",
                          "action editing", "guidance"],
    "outcome-weighted-bc": ["advantage", "weighted behavior cloning", "filtered behavior cloning", "data curation",
                            "mixed-quality", "return-conditioned", "classifier-free guidance"],
    "corrections": ["human-in-the-loop", "intervention", "correction", "DAgger", "interactive imitation",
                    "takeover", "recovery", "failed rollout", "failure data", "shared autonomy"],
    "test-time": ["test-time", "inference-time", "runtime", "best-of-N", "verifier", "verification", "rerank",
                  "reranking", "failure detection", "failure prediction", "monitor"],
    "reward-value": ["reward", "value function", "value model", "progress estimation", "task progress",
                     "success detection", "success detector", "preference"],
    "world-model-eval": ["world model", "world action model", "world-action model", "video prediction",
                         "imagination", "imagined", "policy evaluation", "evaluating policies", "digital twin",
                         "real-to-sim"],
    "sim-real": ["sim-to-real", "sim2real", "real-to-sim", "domain randomization"],
    "search-evolution": ["evolution strategies", "evolutionary", "evolving", "evolve", "Bayesian optimization",
                         "CMA-ES", "black-box", "zeroth-order", "random search", "quality-diversity",
                         "policy search", "harness", "code-as-policy", "code as policies"],
    "action-chunking": ["action chunking", "action chunk"],
}
WEAK = {"RL", "critic", "fine-tune", "fine-tuning", "finetuning", "lifelong", "continual learning", "residual",
        "steer", "guidance", "advantage", "intervention", "correction", "recovery", "runtime", "verification",
        "monitor", "reward", "preference", "imagination", "imagined", "digital twin", "evolving", "evolve",
        "evolutionary", "black-box", "harness"}
POLICY_TERMS = ["policy", "policies", "VLA", "vision-language-action", "controller", "agent"]

# Topic terms for the general-RL pool (data/families.json "generalrl"): results robot work builds on.
GENERAL_RL_TOPICS: dict[str, list[str]] = {
    "grl-critics-scaling": ["scaling law", "compute-optimal", "critic", "value function", "Q-function",
                            "overestimation", "plasticity", "rank collapse", "value-based", "TD learning"],
    "grl-flow-diffusion": ["flow matching", "flow policy", "diffusion policy", "MeanFlow", "flow-based policy",
                           "diffusion-based policy", "denoising policy", "consistency policy"],
    "grl-llm-rl-method": ["off-policy", "stale", "asynchronous", "importance sampling", "group baseline",
                          "group relative", "trust region", "entropy collapse", "on-policy distillation",
                          "reward-weighted", "self-generated"],
    "grl-test-time": ["process reward", "test-time", "best-of-N", "verifier", "inference-time"],
    "grl-offline-finetune": ["offline-to-online", "offline RL", "offline reinforcement learning",
                             "pretrained policy", "behavior prior", "quality-diversity", "fine-tuning policies"],
    "grl-driving": ["autonomous driving", "driving policy", "end-to-end driving"],
    "grl-evolution": ["evolution strategies", "evolutionary strategies", "evolution strategy", "CMA-ES",
                      "black-box optimization", "black-box optimisation", "zeroth-order"],
}
# RL for language models is ~250 papers a week and mostly applications. An LLM-centric paper (LLM_TERMS, and
# none of CONTROL_TERMS) stays a candidate only if it is about one of the RL-method ideas that transfer to
# robots: critics and off-policy/stale data, scaling, process rewards and test-time compute, the policy-gradient
# estimator itself (importance ratios, clipping, entropy, baselines), distillation from the policy's own
# samples, and evolution strategies. "group relative" / "policy optimization" are deliberately absent: they
# name GRPO, which ~100 of the ~250 weekly LLM-RL papers mention (test week 2026-09-23..30), mostly applications.
LLM_TERMS = ["language model", "LLM", "reasoning", "RLVR", "token", "chain-of-thought", "agentic", "VLM",
             "MLLM", "prompt"]
LLM_TRANSFER = ["critic", "value function", "value model", "off-policy", "stale", "staleness", "asynchronous",
                "scaling law", "process reward model", "test-time scaling", "reward-weighted", "self-generated",
                "trust region", "importance sampling", "importance ratio", "importance weight", "clipping",
                "entropy", "group baseline", "advantage estimation", "optimization bias", "update sparsity",
                "pass@k", "reasoning boundary", "on-policy distillation", "self-distillation",
                "evolution strategies", "evolutionary strategies", "evolution strategy"]
CONTROL_TERMS = ["continuous control", "robot", "robotic", "locomotion", "manipulation", "MuJoCo", "D4RL",
                 "OGBench", "imitation learning", "Atari", "DeepMind Control", "autonomous driving"]


def term_pattern(t: str) -> re.Pattern:
    """Whole-word regex for a term: case-insensitive, '-' and ' ' interchangeable or absent ("fine-tuning"
    matches "fine tuning" and "finetuning"), and an optional plural "s"/"es"."""
    words = [re.escape(w) for w in re.split(r"[-\s]+", t)]
    return re.compile(r"(?<![A-Za-z0-9])" + r"[-\s]?".join(words) + r"(?:s|es)?(?![A-Za-z0-9])", re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern] = {}


def has(text: str, terms: list[str] | set[str]) -> bool:
    for t in terms:
        if t not in _PATTERNS:
            _PATTERNS[t] = term_pattern(t)
        if _PATTERNS[t].search(text):
            return True
    return False


def topics(paper: dict, kind: str) -> list[str]:
    """Names of the topic groups the paper's title + abstract match ("robot" or "general" topic set). An
    LLM-centric general paper gets ["grl-llm-transfer"] if it matches LLM_TRANSFER and nothing otherwise."""
    text = paper["title"] + " " + paper["abstract"]
    if kind == "general":
        if has(text, LLM_TERMS) and not has(text, CONTROL_TERMS):
            return ["grl-llm-transfer"] if has(text, LLM_TRANSFER) else []
        found = [name for name, terms in GENERAL_RL_TOPICS.items() if has(text, terms)]
        return found or (topics(paper, "robot") if has(text, CONTROL_TERMS) else [])
    about_policy = has(text, POLICY_TERMS)
    return [name for name, terms in ROBOT_TOPICS.items()
            if has(text, [t for t in terms if about_policy or t not in WEAK])]


def build_query(cats: list[str], terms: list[str] | None, since: dt.date, until: dt.date,
                exclude: str | None = None) -> str:
    """arXiv search_query for one pool. The date range goes first: with it last, "ANDNOT" makes the API
    ignore it (tested September 2026)."""
    q = f"submittedDate:[{since:%Y%m%d}0000 TO {until:%Y%m%d}2359] AND ("
    q += " OR ".join(f"cat:{c}" for c in cats) + ")"
    if terms:
        quote = lambda t: f'"{t}"' if re.search(r"[^A-Za-z0-9]", t) else t
        q += " AND (" + " OR ".join(f"abs:{quote(t)} OR ti:{quote(t)}" for t in terms) + ")"
    if exclude:
        q += f" ANDNOT cat:{exclude}"
    return q


def pool_queries(since: dt.date, until: dt.date) -> list[tuple[str, str, list[str] | None, str]]:
    """(pool name, query, terms to re-check locally, topic kind) for every pool."""
    return [(name, build_query(cats, terms, since, until, None if name == "robotics" else "cs.RO"), terms, kind)
            for name, cats, terms, kind in POOLS]


def parse_feed(xml: str) -> tuple[int, list[dict]]:
    """Parse one Atom page from the arXiv API into (totalResults, [paper dicts])."""
    root = ET.fromstring(xml)
    total = int(root.findtext("os:totalResults", default="0", namespaces=NS))
    papers = []
    for e in root.findall("a:entry", NS):
        m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", e.findtext("a:id", default="", namespaces=NS))
        if not m:  # old-style ids (e.g. cs/0101001) and API error entries
            continue
        clean = lambda s: re.sub(r"\s+", " ", s or "").strip()
        prim = e.find("arxiv:primary_category", NS)
        papers.append({
            "id": m.group(1),
            "url": f"https://arxiv.org/abs/{m.group(1)}",
            "date": e.findtext("a:published", default="", namespaces=NS)[:10],
            "title": clean(e.findtext("a:title", namespaces=NS)),
            "authors": [clean(a.findtext("a:name", namespaces=NS)) for a in e.findall("a:author", NS)],
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
            "primary_category": prim.get("term") if prim is not None else "",
            "abstract": clean(e.findtext("a:summary", namespaces=NS)),
        })
    return total, papers


def get(params: dict) -> str:
    """One API request, retried with exponential backoff while the API is busy (429/5xx/timeouts)."""
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "awesome-hill-climbing weekly sweep"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
        time.sleep(10 * 2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_all(query: str) -> tuple[list[dict], int]:
    """One pass over all pages of a query: (unique papers, the API's totalResults). An empty page is retried
    EMPTY_RETRIES times with growing pauses (the API returns empty pages under load, even for the first page,
    where it then also reports totalResults 0); after that the pass stops, possibly short."""
    out: dict[str, dict] = {}
    start, total = 0, None
    while total is None or start < total:
        params = {"search_query": query, "start": start, "max_results": PAGE,
                  "sortBy": "submittedDate", "sortOrder": "descending"}
        for attempt in range(EMPTY_RETRIES + 1):
            total, page = parse_feed(get(params))
            time.sleep(PAUSE)
            if total > MAX_POOL:
                raise SystemExit(f"query returned {total} results (> {MAX_POOL}); the date filter was probably "
                                 f"ignored. Query: {query}")
            if page:
                break
            if attempt < EMPTY_RETRIES:
                time.sleep(PAUSE * 2 ** attempt)
        if not page:
            break
        for p in page:
            out.setdefault(p["id"], p)
        start += len(page)
    return list(out.values()), total or 0


def search(query: str, passes: int = 2) -> list[dict]:
    """All results of one query. Exits non-zero if they stay short of totalResults after `passes` full passes:
    a truncated pool would silently drop papers from a window that is then marked as checked."""
    for i in range(passes):
        papers, total = fetch_all(query)
        if len(papers) >= total:
            return papers
        print(f"  warning: got {len(papers)} of {total} results; retrying the query", file=sys.stderr, flush=True)
        if i < passes - 1:
            time.sleep(30)
    raise SystemExit(f"arXiv returned only {len(papers)} of {total} results after {passes} passes; the pool is "
                     f"incomplete, so the window must not be marked as checked. Query: {query}")


def slices(since: dt.date, until: dt.date, days: int = SLICE_DAYS) -> list[tuple[dt.date, dt.date]]:
    """[since, until] cut into consecutive inclusive slices of at most `days` days."""
    out, a = [], since
    while a <= until:
        b = min(until, a + dt.timedelta(days=days - 1))
        out.append((a, b))
        a = b + dt.timedelta(days=1)
    return out


def fetch_pools(since: dt.date, until: dt.date, log=sys.stderr
                ) -> tuple[dict[str, tuple[list[dict], list[str] | None, str]], dict[str, tuple[int, int, int]]]:
    """Download every pool for the window twice (one-day slices, then SLICE_DAYS slices) and keep the union.
    Returns (pools as select() takes them, {pool: (papers from the daily pass, from the windowed pass, union)})."""
    got: dict[str, dict[str, dict[str, dict]]] = {}  # pool → pass → id → paper
    meta: dict[str, tuple[list[str] | None, str]] = {}
    for pass_name, days in (("daily", 1), ("windowed", SLICE_DAYS)):
        for a, b in slices(since, until, days):
            for name, q, terms, kind in pool_queries(a, b):
                meta[name] = (terms, kind)
                bucket = got.setdefault(name, {}).setdefault(pass_name, {})
                for p in search(q):
                    bucket.setdefault(p["id"], p)
        print(f"  {pass_name} pass done", file=log, flush=True)
    pools, stats = {}, {}
    for name, passes in got.items():
        union = {**passes["windowed"], **passes["daily"]}
        pools[name] = (sorted(union.values(), key=lambda p: (p["date"], p["id"]), reverse=True), *meta[name])
        stats[name] = (len(passes["daily"]), len(passes["windowed"]), len(union))
    return pools, stats


def robotics_floor(since: dt.date, until: dt.date, today: dt.date, per_day: int = MIN_RO_PER_DAY) -> int:
    """Fewest cs.RO submissions a healthy API answer can hold for the window: per_day for every day that arXiv
    has surely announced by now (submitted 3+ days before today), 0 for the rest."""
    last = min(until, today - dt.timedelta(days=3))
    return per_day * max(0, (last - since).days + 1)


def read_seen(path: Path = SEEN) -> set[str]:
    """arXiv ids an earlier sweep already triaged (one per line; '#' starts a comment)."""
    if not path.exists():
        return set()
    return {ln.split("#")[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()} - {""}


def mark_seen(jsonl: Path, path: Path = SEEN, today: dt.date | None = None) -> tuple[int, int]:
    """Add the ids in a candidates file to the seen list, forgetting ids older than SEEN_KEEP_MONTHS (the id's
    YYMM prefix). Returns (ids added, ids kept in the file)."""
    today = today or dt.datetime.now().astimezone().date()
    cutoff = (today.year % 100) * 12 + today.month - 1 - SEEN_KEEP_MONTHS
    month = lambda i: int(i[:2]) * 12 + int(i[2:4]) - 1
    seen = read_seen(path)
    new = {json.loads(ln)["id"] for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()}
    added = len(new - seen)
    keep = sorted((i for i in seen | new if re.fullmatch(r"\d{4}\.\d{4,5}", i) and month(i) >= cutoff),
                  key=lambda i: (i[:4], int(i[5:])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# arXiv ids already triaged by the weekly sweep (tools/arxiv_candidates.py --mark-seen).\n"
                    "# arxiv_candidates.py drops them, so overlapping windows do not show them twice.\n"
                    + "".join(i + "\n" for i in keep), encoding="utf-8")
    return added, len(keep)


def abstracts(jsonl: Path, ids: list[str]) -> str:
    """Title, categories, authors and abstract of the given candidates, in the order asked."""
    cands = {c["id"]: c for c in map(json.loads, filter(str.strip, jsonl.read_text(encoding="utf-8").splitlines()))}
    out = []
    for i in ids:
        c = cands.get(i)
        out.append(f"{i}: not in {jsonl}\n" if c is None else
                   f"{i} {c['date']} [{', '.join(c['categories'])}] {c['title']}\n  {', '.join(c['authors'][:6])}"
                   f"{' et al.' if len(c['authors']) > 6 else ''}\n  {c['abstract']}\n")
    return "\n".join(out)


def brief(jsonl: Path) -> str:
    """One short line per candidate, most topic groups first: id, date, score, groups, title."""
    cands = [json.loads(ln) for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cands.sort(key=lambda c: (c["date"], c["id"]), reverse=True)
    cands.sort(key=lambda c: -c["score"])  # stable: newest first within a score
    return "".join(f"{c['id']} {c['date']} {c['score']} [{','.join(c['matched'])}] {c['title']}\n" for c in cands)


def listed_ids(data: Path = DATA) -> set[str]:
    """arXiv ids already in the list (papers.csv url/code and resources.csv url)."""
    ids = set()
    for name in ("papers.csv", "resources.csv"):
        path = data / name
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    for col in ("url", "code"):
                        if m := ARXIV_ID.search(row.get(col) or ""):
                            ids.add(m.group(1))
    return ids


def select(pools: dict[str, tuple[list[dict], list[str] | None, str]], known: set[str],
           updated: dt.date | None) -> tuple[list[dict], int]:
    """Tag every pooled paper with its topics, keep the ones with at least one, drop known ids, sort newest
    first. pools maps name → (papers, terms the paper must contain or None, topic kind).
    Returns (new candidates, number of matching candidates that are already listed)."""
    by_id: dict[str, dict] = {}
    for name, (papers, terms, kind) in pools.items():
        for p in papers:
            if terms and not has(p["title"] + " " + p["abstract"], terms):
                continue  # arXiv's fuzzy search matched a term the paper does not actually contain
            found = topics(p, kind)
            if not found:
                continue
            c = by_id.setdefault(p["id"], {**p, "pools": [], "matched": []})
            c["pools"].append(name)
            c["matched"] += [t for t in found if t not in c["matched"]]
    new = [c for i, c in by_id.items() if i not in known]
    for c in new:
        c["score"] = len(c["matched"])
        c["overlap"] = bool(updated and c["date"] < updated.isoformat())
    new.sort(key=lambda c: (c["date"], c["id"]), reverse=True)
    return new, len(by_id) - len(new)


def main() -> None:
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    updated = dt.date.fromisoformat(meta["updated"])
    today = dt.datetime.now().astimezone().date()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", type=dt.date.fromisoformat,
                    help=f'first submission date, inclusive (default: meta.json "updated" minus {OVERLAP_DAYS} days)')
    ap.add_argument("--until", type=dt.date.fromisoformat, default=today,
                    help="last submission date, inclusive (default: today)")
    ap.add_argument("--out", type=Path, help="write JSONL here (default: stdout)")
    ap.add_argument("--min-score", type=int, default=1,
                    help="keep candidates matching at least this many topic groups (default 1: highest recall)")
    ap.add_argument("--min-per-day", type=int, default=MIN_RO_PER_DAY,
                    help=f"exit non-zero if the cs.RO pool has fewer submissions than this per announced day "
                         f"(default {MIN_RO_PER_DAY}; 0 turns the check off)")
    ap.add_argument("--seen", type=Path, default=SEEN, help="ids already triaged, to drop (default: %(default)s)")
    ap.add_argument("--mark-seen", type=Path, metavar="JSONL",
                    help="add the ids of this candidates file to --seen and exit (run after triage)")
    ap.add_argument("--brief", type=Path, metavar="JSONL", help="print one short line per candidate and exit")
    ap.add_argument("--abstracts", metavar="IDS", help="with --brief: print these comma-separated ids in full instead")
    ap.add_argument("--show-queries", action="store_true", help="print the arXiv queries for the window and exit")
    args = ap.parse_args()
    if args.brief:
        sys.stdout.write(abstracts(args.brief, [i.strip() for i in args.abstracts.split(",") if i.strip()])
                         if args.abstracts else brief(args.brief))
        return
    if args.mark_seen:
        added, kept = mark_seen(args.mark_seen, args.seen, today)
        print(f"marked {added} new ids as seen; {args.seen} now holds {kept}")
        return
    since = args.since or updated - dt.timedelta(days=OVERLAP_DAYS)
    if since > args.until:
        raise SystemExit(f"empty window: {since} is after {args.until}")
    if args.show_queries:
        for a, b in slices(since, args.until, SLICE_DAYS):
            for name, q, _, _ in pool_queries(a, b):
                print(f"{name}:\n  {q}")
        print("(the daily pass runs the same queries for each single day of the window)")
        return
    log = sys.stderr if args.out is None else sys.stdout
    print(f"window {since} → {args.until}", file=log)
    pools, stats = fetch_pools(since, args.until, log)
    for name, (n_day, n_win, n_all) in stats.items():
        print(f"  pool {name:20s} {n_all:5d} submissions (daily pass {n_day}, windowed pass {n_win})",
              file=log, flush=True)
    floor = robotics_floor(since, args.until, today, args.min_per_day)
    if len(pools["robotics"][0]) < floor:
        raise SystemExit(f"the cs.RO pool has {len(pools['robotics'][0])} submissions, fewer than the {floor} "
                         f"expected at minimum for {since} → {args.until}: the API answer is empty or truncated. "
                         f"Retry later; do not mark the window as checked.")
    new, n_listed = select(pools, listed_ids(), updated)
    seen = read_seen(args.seen)
    n_seen = sum(c["id"] in seen for c in new)
    new = [c for c in new if c["id"] not in seen and c["score"] >= args.min_score]
    lines = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in new)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(lines, encoding="utf-8")
    else:
        sys.stdout.write(lines)
    by_pool = {name: sum(name in c["pools"] for c in new) for name in pools}
    n_overlap, n_multi = sum(c["overlap"] for c in new), sum(c["score"] >= 2 for c in new)
    print(f"{len(new) + n_seen + n_listed} match the topic terms: {len(new)} new, {n_listed} already listed, "
          f"{n_seen} already triaged (seen). New by pool: " + ", ".join(f"{k} {v}" for k, v in by_pool.items())
          + f". {n_multi} new ones match 2+ topic groups; {n_overlap} are dated before {updated} (the last update)."
          + (f" Wrote {args.out}" if args.out else ""), file=log)


if __name__ == "__main__":
    main()
