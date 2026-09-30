"""Fill exact first-version dates (and a byline where the org is blank) for arXiv entries in
data/papers.csv, and flag title mismatches.

    uv run python tools/arxiv_meta.py            # update dates in place, print mismatches
    uv run python tools/arxiv_meta.py --dry-run  # only report

Uses the public arXiv API (https://export.arxiv.org/api/query), 100 ids per request,
with a 3-second pause between requests as the API terms ask.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import re
import time
import urllib.request
from pathlib import Path

PAPERS = Path(__file__).resolve().parents[1] / "data" / "papers.csv"
API = "https://export.arxiv.org/api/query?max_results=200&id_list="
ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")


def fetch(ids: list[str]) -> dict[str, tuple[str, str, str]]:
    """Return {id: (published_date, title, "First-author et al.")} for the given arXiv ids."""
    with urllib.request.urlopen(API + ",".join(ids), timeout=60) as r:
        xml = r.read().decode("utf-8")
    out = {}
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5})(v\d+)?</id>", entry)
        if not m:
            continue
        published = re.search(r"<published>(\d{4}-\d{2}-\d{2})", entry).group(1)
        title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", entry, re.S).group(1)).strip()
        authors = re.findall(r"<name>(.*?)</name>", entry)
        lead = authors[0].split()[-1] if authors else ""
        byline = f"{lead} et al." if len(authors) > 1 else lead
        out[m.group(1)] = (published, title, byline)
    return out


def similar(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with PAPERS.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    ids = sorted({m.group(1) for r in rows if (m := ARXIV_ID.search(r["url"]))})
    meta: dict[str, tuple[str, str, str]] = {}
    for i in range(0, len(ids), 100):
        meta.update(fetch(ids[i:i + 100]))
        time.sleep(3)
    missing, mismatched, changed = [], [], 0
    for r in rows:
        m = ARXIV_ID.search(r["url"])
        if not m:
            continue
        aid = m.group(1)
        if aid not in meta:
            missing.append(aid)
            continue
        published, title, byline = meta[aid]
        if not r.get("org"):
            r["org"] = byline  # no affiliation on record: credit the authors instead
        if similar(r["title"], title) < 0.6 and title.lower()[:25] not in r["title"].lower():
            mismatched.append((aid, r["title"][:70], title[:70]))
        if r["date"] != published:
            r["date"] = published
            changed += 1
    print(f"{len(ids)} arXiv ids, {len(meta)} found, {changed} dates updated")
    for aid in missing:
        print(f"MISSING  {aid}")
    for aid, ours, theirs in mismatched:
        print(f"TITLE?   {aid}: ours={ours!r} arxiv={theirs!r}")
    if not args.dry_run:
        with PAPERS.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
