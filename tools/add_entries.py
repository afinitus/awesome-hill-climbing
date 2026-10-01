"""Add verified entries to the list: append items from JSON files to data/papers.csv and data/resources.csv.

    uv run python tools/add_entries.py new.json [more.json ...]            # write the CSVs
    uv run python tools/add_entries.py new.json --dry-run                  # only print what would change
    uv run python tools/add_entries.py --count-vs HEAD                     # rows added since a git ref: "N M"

Each JSON file holds a list of objects. A paper, blog post, talk or thread:

    {"type": "paper", "title": "...", "url": "https://arxiv.org/abs/2609.12345", "date": "2026-09-29",
     "family": "residual", "kind": "paper", "one_line": "...",
     "org": "Stanford; NVIDIA", "code": "https://github.com/...", "short": "FP2"}      # last three optional

A resource (code, simulator, benchmark, dataset, course, book, tutorial, survey, talk, awesome list):

    {"type": "resource", "name": "...", "url": "...", "category": "code-robot", "org": "...", "one_line": "..."}

What it does (the rules from CONTRIBUTING.md "Adding to the list"):
  * validates every item first and writes nothing if any item is invalid (exit code 2): family must be a key
    in data/families.json, kind one of paper/blog/talk/thread, resource category one the README renders,
    date YYYY-MM-DD and not in the future, one_line at most 40 words, URLs http(s);
  * normalizes URLs (arXiv pdf/html/versioned links → https://arxiv.org/abs/<id>);
  * skips duplicates of rows already in the list or earlier in the input, matched by arXiv id, normalized URL
    or normalized title/name (a duplicate is reported, not an error);
  * gives papers the next free N### id, key = 0 and verified = "read-once", and a short name from the title
    when none is given; cleans org (blank when unknown, so tools/arxiv_meta.py fills "First-author et al.");
  * keeps papers.csv sorted newest first and resources.csv sorted by category then name.

It does not touch the network. Run tools/arxiv_meta.py afterwards to fix arXiv dates and fill bylines.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PFIELDS = ["id", "date", "title", "short", "url", "code", "org", "family", "kind", "key", "verified", "one_line"]
RFIELDS = ["name", "url", "org", "category", "one_line"]
KINDS = {"paper", "blog", "talk", "thread"}
MAX_WORDS = 40
UNKNOWN_ORG = re.compile(r"not (stated|listed|shown|given)|per candidate|unknown|affiliation|n/a|^none$|^tbd$", re.IGNORECASE)
ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")


def resource_categories() -> set[str]:
    """The resource categories the README renders, read from tools/build_awesome.py (single source of truth)."""
    spec = importlib.util.spec_from_file_location("build_awesome", ROOT / "tools" / "build_awesome.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {c for _, cats in mod.RESOURCE_SECTIONS for c in cats}


def families(data: Path) -> set[str]:
    return {f["key"] for f in json.loads((data / "families.json").read_text(encoding="utf-8"))}


def canonical_url(u: str) -> str:
    """The URL as stored: arXiv links become https://arxiv.org/abs/<id> (no version), whitespace stripped."""
    u = (u or "").strip()
    if m := ARXIV.search(u):
        return f"https://arxiv.org/abs/{m.group(1)}"
    return u


def url_key(u: str) -> str:
    """A key for duplicate detection: arXiv id, GitHub owner/repo, or the URL without scheme, www and slash."""
    u = (u or "").strip().rstrip("/")
    if m := ARXIV.search(u):
        return "arxiv:" + m.group(1)
    if m := re.match(r"https?://(?:www\.)?github\.com/([^/]+/[^/#?]+)", u, re.IGNORECASE):
        return "gh:" + m.group(1).lower().removesuffix(".git")
    return re.sub(r"^https?://(www\.)?", "", u, flags=re.IGNORECASE).split("#")[0].lower()


def title_key(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("->", "→")).strip()


def clean_org(o: str) -> str:
    """Short affiliation list, or blank when unknown (tools/arxiv_meta.py then fills a byline)."""
    o = clean_text(o)
    if not o or UNKNOWN_ORG.search(o):
        return ""
    o = re.sub(r"\s*\(.*?\)", "", o).strip(" ;,")
    return o if len(o) <= 60 else o[:58].rsplit(" ", 1)[0] + "…"


def short_name(title: str) -> str:
    """The method name before a colon or dash, else the (trimmed) title."""
    head = re.split(r":| — | - ", title, maxsplit=1)[0].strip()
    if 2 < len(head) <= 45 and head != title:
        return head
    return title if len(title) <= 90 else title[:88].rsplit(" ", 1)[0] + "…"


def read_csv(path: Path) -> tuple[list[dict], str]:
    """Rows and the file's line terminator (kept on write so diffs stay minimal)."""
    raw = path.read_bytes()
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows, "\r\n" if b"\r\n" in raw else "\n"


def write_csv(path: Path, fields: list[str], rows: list[dict], eol: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator=eol)
        w.writeheader()
        w.writerows(rows)


def id_order(pid: str) -> tuple[str, int]:
    m = re.match(r"([A-Za-z]*)(\d+)$", pid)
    return (m.group(1), int(m.group(2))) if m else (pid, -1)


def validate(it: dict, fams: set[str], rcats: set[str], today: dt.date) -> list[str]:
    """Problems with one input item (empty list = valid)."""
    errs = []
    typ = it.get("type", "paper")
    if typ not in ("paper", "resource"):
        return [f'type must be "paper" or "resource", got {typ!r}']
    if not re.match(r"https?://", (it.get("url") or "").strip()):
        errs.append("url must start with http:// or https://")
    if (it.get("code") or "").strip() and not re.match(r"https?://", it["code"].strip()):
        errs.append("code must be blank or an http(s) URL")
    one = clean_text(it.get("one_line"))
    if not one:
        errs.append("one_line is empty")
    elif len(one.split()) > MAX_WORDS:
        errs.append(f"one_line has {len(one.split())} words (max {MAX_WORDS})")
    if typ == "paper":
        if not clean_text(it.get("title")):
            errs.append("title is empty")
        if it.get("family") not in fams:
            errs.append(f"family {it.get('family')!r} is not one of {sorted(fams)}")
        if it.get("kind", "paper") not in KINDS:
            errs.append(f"kind {it.get('kind')!r} is not one of {sorted(KINDS)}")
        date = str(it.get("date"))
        try:
            # fromisoformat alone also accepts "20260929" and "2026-W40-1", which would be stored as given.
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                raise ValueError(date)
            if dt.date.fromisoformat(date) > today:
                errs.append(f"date {date} is in the future")
        except ValueError:
            errs.append(f"date {it.get('date')!r} is not YYYY-MM-DD")
    else:
        if not clean_text(it.get("name")):
            errs.append("name is empty")
        if it.get("category") not in rcats:
            errs.append(f"category {it.get('category')!r} is not one of {sorted(rcats)}")
    return errs


def add(items: list[dict], data: Path = DATA, dry_run: bool = False, today: dt.date | None = None) -> dict:
    """Validate and merge items into data/papers.csv and data/resources.csv. Returns a report dict:
    {"errors": [(index, title, [problems])], "papers": [rows added], "resources": [rows added],
     "skipped": [(title, reason)]}. Nothing is written when there are errors or dry_run is set."""
    today = today or dt.datetime.now().astimezone().date()
    fams, rcats = families(data), resource_categories()
    report = {"errors": [], "papers": [], "resources": [], "skipped": []}
    for i, it in enumerate(items):
        if errs := validate(it, fams, rcats, today):
            report["errors"].append((i, it.get("title") or it.get("name") or "?", errs))
    if report["errors"]:
        return report

    ppath, rpath = data / "papers.csv", data / "resources.csv"
    papers, peol = read_csv(ppath)
    res, reol = read_csv(rpath)
    seen_urls = {url_key(p["url"]): f"papers.csv {p['id']}" for p in papers}
    seen_urls.update({url_key(r["url"]): f"resources.csv {r['name']!r}" for r in res})
    seen_titles = {title_key(p["title"]): f"papers.csv {p['id']}" for p in papers}
    seen_titles.update({title_key(r["name"]): f"resources.csv {r['name']!r}" for r in res})
    next_id = 1 + max([id_order(p["id"])[1] for p in papers if p["id"].startswith("N")] or [0])

    for it in items:
        typ = it.get("type", "paper")
        title = clean_text(it.get("title") if typ == "paper" else it.get("name"))
        url = canonical_url(it["url"])
        ukey, tkey = url_key(url), title_key(title)
        if ukey in seen_urls or tkey in seen_titles:
            where = seen_urls.get(ukey) or seen_titles.get(tkey)
            report["skipped"].append((title, f"duplicate of {where}"))
            continue
        if typ == "paper":
            row = {"id": f"N{next_id:03d}", "date": str(it["date"]), "title": title,
                   "short": clean_text(it.get("short")) or short_name(title), "url": url,
                   "code": canonical_url(it.get("code") or ""), "org": clean_org(it.get("org", "")),
                   "family": it["family"], "kind": it.get("kind", "paper"), "key": "0", "verified": "read-once",
                   "one_line": clean_text(it["one_line"])}
            next_id += 1
            papers.append(row)
            report["papers"].append(row)
            where = f"papers.csv {row['id']} (this run)"
        else:
            row = {"name": title, "url": url, "org": clean_org(it.get("org", "")), "category": it["category"],
                   "one_line": clean_text(it["one_line"])}
            res.append(row)
            report["resources"].append(row)
            where = f"resources.csv {title!r} (this run)"
        seen_urls[ukey] = seen_titles[tkey] = where

    if not dry_run and (report["papers"] or report["resources"]):
        papers.sort(key=lambda p: (p["date"], id_order(p["id"])), reverse=True)
        res.sort(key=lambda r: (r["category"], r["name"].lower()))
        write_csv(ppath, PFIELDS, papers, peol)
        write_csv(rpath, RFIELDS, res, reol)
    return report


def count_new(old_papers: str, old_resources: str, data: Path = DATA) -> tuple[int, int]:
    """(papers, resources) in data/ now that are not in the old CSV texts: new paper ids, new resource URLs.
    Rows edited in place (a date fix, a code link) do not count; rows removed again do not count."""
    old_ids = {r["id"] for r in csv.DictReader(io.StringIO(old_papers))}
    old_urls = {url_key(r["url"]) for r in csv.DictReader(io.StringIO(old_resources))}
    papers, _ = read_csv(data / "papers.csv")
    res, _ = read_csv(data / "resources.csv")
    return sum(p["id"] not in old_ids for p in papers), sum(url_key(r["url"]) not in old_urls for r in res)


def git_show(ref: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", type=Path, help="JSON files, each a list of entries")
    ap.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    ap.add_argument("--count-vs", metavar="REF",
                    help='print "N M": papers/posts and resources in data/ that are new since git REF, and exit')
    args = ap.parse_args()
    if args.count_vs:
        n, m = count_new(git_show(args.count_vs, "data/papers.csv"), git_show(args.count_vs, "data/resources.csv"))
        print(n, m)
        return
    if not args.files:
        ap.error("give at least one JSON file (or --count-vs REF)")
    items = []
    for f in args.files:
        loaded = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            sys.exit(f"{f}: expected a JSON list of entries")
        items += loaded
    rep = add(items, dry_run=args.dry_run)
    if rep["errors"]:
        print(f"{len(rep['errors'])} invalid entries; nothing written:")
        for i, title, errs in rep["errors"]:
            print(f"  [{i}] {title[:80]}: " + "; ".join(errs))
        sys.exit(2)
    for p in rep["papers"]:
        print(f"+ paper    {p['id']} {p['date']} {p['family']:10s} {p['title'][:80]}")
    for r in rep["resources"]:
        print(f"+ resource {r['category']:10s} {r['name'][:80]}")
    for title, why in rep["skipped"]:
        print(f"= skipped  {title[:70]}: {why}")
    verb = "would add" if args.dry_run else "added"
    print(f"{verb} {len(rep['papers'])} papers/posts and {len(rep['resources'])} resources; "
          f"skipped {len(rep['skipped'])} duplicates")


if __name__ == "__main__":
    main()
