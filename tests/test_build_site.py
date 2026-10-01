"""The explainer page (tools/build_site.py) builds from data/ and contains what it promises."""
from __future__ import annotations

import csv
import html
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_site", ROOT / "tools" / "build_site.py")
build_site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_site)

SECTIONS = ["start", "new", "loop", "nines", "ideas", "guide", "families", "timeline", "industry", "classic",
            "lessons", "open", "tree", "course", "reading", "papers", "method"]
# Family keys that are not ordinary English words; none should show up as text on the page.
FAMILY_KEYS = ["blackbox", "onpolicy", "offpolicy", "advbc", "hitl", "testtime", "worldmodel", "simreal", "generalrl"]


def papers() -> list[dict]:
    with (ROOT / "data" / "papers.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def visible_text(page: str) -> str:
    page = re.sub(r"<(script|style)\b.*?</\1>", " ", page, flags=re.DOTALL)
    page = re.sub(r"<!--.*?-->", " ", page, flags=re.DOTALL)
    return html.unescape(re.sub(r"<[^>]+>", " ", page))


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("site")
    assert build_site.main(["--out", str(out)]) == 0
    return (out / "index.html").read_text(encoding="utf-8")


def test_standalone_document(page):
    head = page[:600].lower()
    assert head.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in head and 'name="viewport"' in head
    assert "<title>climbing the nines</title>" in head
    assert len(page.encode("utf-8")) < 2_000_000
    # nothing external except Google Fonts
    for tag in re.findall(r"<(?:script|link|img)\b[^>]*>", page):
        for url in re.findall(r'(?:src|href)="(https?://[^"]+)"', tag):
            assert url.startswith(("https://fonts.googleapis.com", "https://fonts.gstatic.com")), url


def test_key_sections(page):
    for sid in SECTIONS:
        assert f'id="{sid}"' in page, sid
    for fam in json.loads((ROOT / "data" / "families.json").read_text(encoding="utf-8")):
        assert f'id="fam-{fam["key"]}"' in page, fam["key"]
        assert f'/blob/main/papers/{fam["key"]}.md' in page
    assert 'id="topo"' in page and 'id="wk"' in page and 'class="swim"' in page


def test_every_paper_in_index(page):
    index = page[page.index('id="papers"'):]
    rows = papers()
    assert f"All {len(rows)} papers" in page
    for p in rows:
        assert f'id="p-{p["id"]}"' in index, p["id"]
        assert html.escape(p["url"], quote=True) in index, p["id"]


def test_no_internal_ids_or_markers_in_visible_text(page):
    text = visible_text(page)
    leaks = sorted(set(re.findall(r"\b[CN]\d{3}\b", text)))
    assert not leaks, leaks
    assert "(ADJ)" not in text and "ADJ)" not in text
    keys = sorted({k for k in FAMILY_KEYS if re.search(rf"\b{k}\b", text)})
    assert not keys, keys


def test_timeline_intro_matches_the_plot(page):
    sec = page[page.index('id="timeline"'):page.index('id="industry"')]
    dots = len(re.findall(r"<circle [^>]*data-id=", sec))  # one per plotted paper (legend swatches have no id)
    lede = visible_text(re.search(r'<p class="lede">(.*?)</p>', sec, re.DOTALL).group(1))
    assert lede.startswith(f"{dots} papers and posts"), (dots, lede[:60])
    assert f'aria-label="Swimlane chart: {dots} papers' in sec


def test_reading_lists_every_resource(page):
    sec = page[page.index('id="reading"'):page.index('id="papers"')]
    with (ROOT / "data" / "resources.csv").open(newline="", encoding="utf-8") as fh:
        res = list(csv.DictReader(fh))
    for r in res:
        assert html.escape(r["url"], quote=True) in sec, r["name"]


def test_chapter_results_pool_the_headline_and_skip_other_sets(tmp_path, monkeypatch):
    def run(chapter, fname, method, k, n, **final):
        d = {"schema": 1, "chapter": chapter, "method": method, "variant": "v1",
             "final": {"sr": k / n, "k": k, "n": n, **final}}
        (tmp_path / "results" / chapter).mkdir(parents=True, exist_ok=True)
        (tmp_path / "results" / chapter / f"{fname}.json").write_text(json.dumps(d), encoding="utf-8")
        return f"results/{chapter}/{fname}.json"
    v = build_site.lb.INIT_SET_VERSION
    files = [
        # ch05 headline is Q-chunking pooled over seeds: the best seed alone must not be reported
        run("ch05", "qc_s0", "qc_s0", 230, 256, init_set="eval", init_set_version=v),
        run("ch05", "qc_s1", "qc_s1", 250, 256, init_set="eval", init_set_version=v),
        run("ch05", "rlpd_s0", "rlpd_s0", 255, 256, init_set="eval", init_set_version=v),  # not the headline method
        # ch07 headline is PSS steering pooled over seeds; runs on other sets or old init sets are skipped,
        # and so are the search-selected best run and the other arms
        run("ch07", "pss_s0", "dsrl-pss8_s0", 200, 256, init_set="eval", init_set_version=v),
        run("ch07", "pss_s1", "dsrl-pss8_s1", 210, 256, init_set="eval", init_set_version=v),
        run("ch07", "pss_s2_search", "dsrl-pss8_s2", 64, 64, init_set="search"),
        run("ch07", "pss_s2_v0", "dsrl-pss8_s2", 250, 256, init_set="eval", init_set_version="v0"),
        run("ch07", "best", "noise-steering-best", 225, 256, init_set="eval", init_set_version=v),
        run("ch07", "res_s0", "dsrl-pss8-res_s0", 250, 256, init_set="eval", init_set_version=v),
    ]
    monkeypatch.setattr(build_site, "ROOT", tmp_path)
    monkeypatch.setattr(build_site, "git_files", lambda folder: files if folder == "results" else [])
    res = build_site.chapter_results()
    assert (res["05"]["k"], res["05"]["n"]) == (480, 512)
    assert (res["07"]["k"], res["07"]["n"]) == (410, 512)


def test_fact_check_fixes_applied(page):
    text = visible_text(page)
    for bad in ["collapsing in 0/5", "fails in 0/5", "TACO's argmax pseudo-count", "finite-difference correction",
                "Q-Planning rounds"]:
        assert bad not in text, bad
    assert "0/5 seeds to ADR 50" in text
    assert "Sep 2026 deep-read" in text


def test_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build_site.main(["--out", str(a)])
    build_site.main(["--out", str(b)])
    assert (a / "index.html").read_text(encoding="utf-8") == (b / "index.html").read_text(encoding="utf-8")
    assert build_site.main(["--out", str(a), "--check"]) == 0
