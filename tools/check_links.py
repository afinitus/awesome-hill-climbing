"""Check that the non-arXiv links in data/papers.csv and data/resources.csv still resolve.

    uv run python tools/check_links.py            # report
    uv run python tools/check_links.py --strict   # exit 1 on any hard failure (404/410/DNS)

arXiv links are validated separately (and more strictly) by tools/arxiv_meta.py.
Sites that block scripts (HTTP 401/403/429/999) are reported as "blocked", not as failures.
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
UA = {"User-Agent": "Mozilla/5.0 (awesome-hill-climbing link check)", "Accept": "*/*"}
BLOCKED = {401, 403, 405, 406, 429, 999}


def status(url: str) -> tuple[str, str]:
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, headers=UA, method=method)
            with urllib.request.urlopen(req, timeout=25) as r:
                return url, f"ok {r.status}"
        except urllib.error.HTTPError as e:
            if e.code in BLOCKED:
                if method == "GET":
                    return url, f"blocked {e.code}"
                continue
            if method == "GET":
                return url, f"FAIL {e.code}"
        except Exception as e:  # DNS, TLS, timeout
            if method == "GET":
                return url, f"FAIL {type(e).__name__}"
    return url, "FAIL unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    urls = []
    for name, cols in (("papers.csv", ("url", "code")), ("resources.csv", ("url",))):
        path = DATA / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                urls += [row[c].strip() for c in cols if row.get(c, "").strip()]
    urls = sorted({u for u in urls if "arxiv.org/abs/" not in u})
    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(status, urls))
    bad = [(u, s) for u, s in results if s.startswith("FAIL")]
    blocked = [(u, s) for u, s in results if s.startswith("blocked")]
    print(f"{len(urls)} links: {len(urls) - len(bad) - len(blocked)} ok, {len(blocked)} blocked scripts, {len(bad)} failed")
    for u, s in blocked:
        print(f"  {s:12s} {u}")
    for u, s in bad:
        print(f"  {s:12s} {u}")
    if args.strict and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
