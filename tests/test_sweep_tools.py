"""Offline tests for the weekly paper-sweep tools (tools/arxiv_candidates.py, tools/add_entries.py). No network."""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import itertools
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    """Import tools/<name>.py by path (tools/ is a folder of scripts, not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ac():
    return load("arxiv_candidates")


@pytest.fixture(scope="module")
def ae():
    return load("add_entries")


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """A temp copy of data/ so tests never touch the real list."""
    d = tmp_path / "data"
    shutil.copytree(ROOT / "data", d)
    return d


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- arxiv_candidates

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>arXiv Query</title>
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry>
    <id>http://arxiv.org/abs/2609.11111v2</id>
    <published>2026-09-28T17:59:59Z</published>
    <title>Steer-0: Latent Noise
      Steering for Frozen VLAs</title>
    <summary>  We steer a frozen VLA by
      choosing its noise. Success rises from 52% to 81%.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:primary_category term="cs.RO" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.RO" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2609.22222v1</id>
    <published>2026-09-24T10:00:00Z</published>
    <title>Stale Critics in Off-Policy Actor-Critic</title>
    <summary>We study stale critics in off-policy reinforcement learning on MuJoCo tasks.</summary>
    <author><name>Grace Hopper</name></author>
    <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""


def test_parse_feed(ac):
    total, papers = ac.parse_feed(FEED)
    assert total == 2 and len(papers) == 2
    p = papers[0]
    assert p["id"] == "2609.11111"  # version suffix dropped
    assert p["url"] == "https://arxiv.org/abs/2609.11111"
    assert p["date"] == "2026-09-28"
    assert p["title"] == "Steer-0: Latent Noise Steering for Frozen VLAs"  # whitespace collapsed
    assert p["abstract"] == "We steer a frozen VLA by choosing its noise. Success rises from 52% to 81%."
    assert p["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert p["categories"] == ["cs.RO", "cs.LG"] and p["primary_category"] == "cs.RO"


def test_parse_empty_feed(ac):
    empty = FEED.split("<entry>")[0].replace(">2<", ">0<") + "</feed>"
    assert ac.parse_feed(empty) == (0, [])


def test_build_query(ac):
    q = ac.build_query(["cs.LG", "cs.AI"], ["vision-language-action", "VLA"], dt.date(2026, 9, 23),
                       dt.date(2026, 9, 30), exclude="cs.RO")
    # The date range must come first: after an ANDNOT the API silently ignores it.
    assert q == ("submittedDate:[202609230000 TO 202609302359] AND (cat:cs.LG OR cat:cs.AI)"
                 ' AND (abs:"vision-language-action" OR ti:"vision-language-action" OR abs:VLA OR ti:VLA)'
                 " ANDNOT cat:cs.RO")
    assert ac.build_query(["cs.RO"], None, dt.date(2026, 9, 23), dt.date(2026, 9, 23)) == (
        "submittedDate:[202609230000 TO 202609232359] AND (cat:cs.RO)")


def test_pool_queries(ac):
    qs = ac.pool_queries(dt.date(2026, 1, 1), dt.date(2026, 1, 7))
    assert [name for name, *_ in qs] == ["robotics", "robotics-crosslist", "general-rl"]
    assert "ANDNOT" not in qs[0][1] and all(q.endswith("ANDNOT cat:cs.RO") for _, q, _, _ in qs[1:])
    for name, q, _, _ in qs:
        assert q.startswith("submittedDate:[202601010000 TO 202601072359]"), name
        assert q.count("(") == q.count(")") and q.count('"') % 2 == 0, name


def test_term_matching(ac):
    assert ac.has("we use fine tuning on a VLA", ["fine-tuning"]) and ac.has("Finetuning helps", ["fine-tuning"])
    assert ac.has("two world models", ["world model"])
    assert not ac.has("CURLING and URL tricks", ["RL"])  # whole words only
    assert ac.has("an RL-based method", ["RL"])


def paper_with(title: str, abstract: str = "") -> dict:
    return {"id": "2609.00001", "date": "2026-09-28", "title": title, "abstract": abstract}


def test_topics(ac):
    robot = paper_with("Noise-Space Steering of a Frozen VLA Policy", "RL in the noise space of a diffusion policy.")
    assert {"rl", "residual-steering"} <= set(ac.topics(robot, "robot"))
    # Weak terms count only for papers about a policy: a mechanism paper that mentions "correction" is not in.
    assert ac.topics(paper_with("Kinematic Error Correction for a Cable Robot"), "robot") == []
    assert "corrections" in ac.topics(paper_with("Correction Data for VLA Policies"), "robot")
    # General RL: LLM-only papers stay only if they are about an idea that transfers.
    assert ac.topics(paper_with("GRPO for Math Reasoning in LLMs", "group relative reinforcement learning"),
                     "general") == []
    assert ac.topics(paper_with("Stale Rollouts in Asynchronous RL for LLMs"), "general") == ["grl-llm-transfer"]
    assert "grl-flow-diffusion" in ac.topics(paper_with("Flow Matching Policies for Offline RL"), "general")


def test_topics_keeps_llm_rl_method_and_es_papers(ac):
    # LLM-RL papers about the estimator itself (DAPO / GSPO style) transfer to robots and must be kept,
    # while a GRPO application with no method idea is still dropped (test_topics).
    gspo = paper_with("Sequence-Level Importance Ratios for Stable RL of Large Language Models",
                      "We replace token-level importance ratios with sequence-level clipping in GRPO for LLM "
                      "reasoning and avoid entropy collapse.")
    assert ac.topics(gspo, "general") == ["grl-llm-transfer"]
    opd = paper_with("On-Policy Distillation for LLM Reasoning", "The student learns from its own samples.")
    assert ac.topics(opd, "general") == ["grl-llm-transfer"]
    es_llm = paper_with("Evolution Strategies for Fine-Tuning LLMs", "ES rivals RL for language model reasoning.")
    assert ac.topics(es_llm, "general") == ["grl-llm-transfer"]
    es = paper_with("Noisy Evolution Strategies Under a Fixed Budget", "Rank-based ES updates.")
    assert ac.topics(es, "general") == ["grl-evolution"]
    assert ac.has("cs.NE: noisy evolution strategies", ac.RL_TERMS)  # ES papers enter the general-RL pool
    chunk = paper_with("Adaptive Action Chunking for VLA Execution", "Chunks of actions from a VLA policy.")
    assert "action-chunking" in ac.topics(chunk, "robot")


def make_feed(total: int, ids: list[str]) -> str:
    entries = "".join(f"""<entry><id>http://arxiv.org/abs/{i}v1</id><published>2026-09-28T00:00:00Z</published>
        <title>T {i}</title><summary>A robot policy.</summary><author><name>A</name></author>
        <category term="cs.RO"/></entry>""" for i in ids)
    return (f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            f"<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>")


@pytest.fixture
def fake_api(ac, monkeypatch):
    """Replace the network with a scripted list of answers; returns the list of requests made."""
    calls = []

    def install(answer):
        def get(params):
            calls.append(params)
            return answer(params, len(calls))
        monkeypatch.setattr(ac, "get", get)
        monkeypatch.setattr(ac.time, "sleep", lambda s: None)
        return calls
    return install


def test_search_pages_and_retries_an_empty_page(ac, fake_api):
    ids = [f"2609.{10000 + k}" for k in range(450)]
    flaky = {"done": False}

    def answer(params, n):
        start = params["start"]
        if start == 200 and not flaky["done"]:  # one empty page under load, then the real one
            flaky["done"] = True
            return make_feed(450, [])
        return make_feed(450, ids[start:start + params["max_results"]])
    calls = fake_api(answer)
    got = ac.search("q")
    assert [p["id"] for p in got] == ids and len(calls) == 4


def test_search_fails_loudly_on_truncated_pages(ac, fake_api):
    ids = [f"2609.{10000 + k}" for k in range(200)]
    fake_api(lambda params, n: make_feed(600, ids if params["start"] == 0 else []))
    with pytest.raises(SystemExit, match="only 200 of 600"):
        ac.search("q")


def test_empty_answer_trips_the_robotics_floor(ac, fake_api):
    calls = fake_api(lambda params, n: make_feed(0, []))
    assert ac.search("q") == [] and len(calls) == ac.EMPTY_RETRIES + 1  # retried, then accepted as 0...
    # ...but a week of cs.RO can never be empty: main() exits non-zero below this floor.
    today = dt.date(2026, 9, 30)
    assert ac.robotics_floor(dt.date(2026, 9, 23), today, today) == 5 * ac.MIN_RO_PER_DAY  # 23..27 announced
    assert ac.robotics_floor(today, today, today) == 0  # nothing announced yet: no floor
    assert ac.robotics_floor(dt.date(2026, 9, 23), today, today, per_day=0) == 0


def test_fetch_pools_keeps_the_union_of_daily_and_windowed_passes(ac, fake_api):
    # The windowed query drops the window's oldest day (seen live on 2026-09-30); the daily pass has it.
    per_day = {f"2026092{k}": [f"2609.{k}{j:04d}" for j in range(3)] for k in (3, 4, 5)}

    def answer(params, n):
        q = params["search_query"]
        if "cat:cs.RO)" not in q or params["start"]:
            return make_feed(0, [])  # other pools empty; single page
        a, b = q[len("submittedDate:["):].split("]")[0].split(" TO ")
        days = [d for d in per_day if a[:8] <= d <= b[:8]]
        if len(days) > 1:
            days = days[1:]
        ids = [i for d in days for i in per_day[d]]
        return make_feed(len(ids), ids)
    calls = fake_api(answer)
    pools, stats = ac.fetch_pools(dt.date(2026, 9, 23), dt.date(2026, 9, 25), log=sys.stderr)
    assert stats["robotics"] == (9, 6, 9)
    assert len(pools["robotics"][0]) == 9 and pools["robotics"][2] == "robot"
    assert len({c["search_query"] for c in calls}) == 3 * (3 + 1)  # 3 pools x (3 days + 1 window)


def test_date_filter_guard(ac, fake_api):
    fake_api(lambda params, n: make_feed(36000, ["2609.00001"]))
    with pytest.raises(SystemExit, match="date filter was probably ignored"):
        ac.search("q")


def test_slices_cover_long_windows(ac):
    d = dt.date
    assert ac.slices(d(2026, 9, 23), d(2026, 9, 30)) == [(d(2026, 9, 23), d(2026, 9, 30))]  # a week: one slice
    assert ac.slices(d(2026, 9, 23), d(2026, 9, 30), days=7) == [(d(2026, 9, 23), d(2026, 9, 29)),
                                                                 (d(2026, 9, 30),) * 2]
    parts = ac.slices(d(2026, 7, 1), d(2026, 9, 30))
    assert parts[0][0] == d(2026, 7, 1) and parts[-1][1] == d(2026, 9, 30) and len(parts) == 7  # 92 days
    assert all((b - a).days < ac.SLICE_DAYS for a, b in parts)
    assert all(b + dt.timedelta(days=1) == a2 for (_, b), (a2, _) in itertools.pairwise(parts))  # no gaps


def test_mark_seen_and_brief(ac, tmp_path):
    cands = tmp_path / "c.jsonl"
    more = {"categories": ["cs.RO"], "authors": ["Ada Lovelace"], "abstract": "We fine-tune a VLA."}
    cands.write_text("".join(json.dumps({**c, **more}) + "\n" for c in [
        {"id": "2609.00002", "date": "2026-09-29", "score": 1, "matched": ["rl"], "title": "B"},
        {"id": "2609.00001", "date": "2026-09-28", "score": 2, "matched": ["rl", "test-time"], "title": "A"}]))
    seen = tmp_path / "seen.txt"
    seen.write_text("# comment\n2601.00009\n2603.00001\n")
    assert ac.mark_seen(cands, seen, today=dt.date(2026, 9, 30)) == (2, 3)
    assert ac.read_seen(seen) == {"2603.00001", "2609.00001", "2609.00002"}  # 2601 is > 6 months old
    assert ac.brief(cands).splitlines() == ["2609.00001 2026-09-28 2 [rl,test-time] A",
                                            "2609.00002 2026-09-29 1 [rl] B"]
    text = ac.abstracts(cands, ["2609.00002", "2609.99999"])
    assert "2609.00002 2026-09-29 [cs.RO] B" in text and "We fine-tune a VLA." in text
    assert "2609.99999: not in" in text


def test_select_filters_dedupes_drops_listed_and_sorts(ac):
    _, papers = ac.parse_feed(FEED)
    off_topic = {**paper_with("A Survey of Lidar Odometry"), "id": "2609.33333"}
    fuzzy = {**paper_with("Cooking Recipes With Transformers"), "id": "2609.44444"}  # no RL term
    pools = {"robotics": (papers[:1] + [off_topic], None, "robot"),
             "general-rl": (papers[1:] + [fuzzy], ac.RL_TERMS, "general")}
    new, n_listed = ac.select(pools, known=set(), updated=dt.date(2026, 9, 25))
    assert [c["id"] for c in new] == ["2609.11111", "2609.22222"]  # newest first, off-topic and fuzzy dropped
    assert new[0]["pools"] == ["robotics"] and "residual-steering" in new[0]["matched"]
    assert new[0]["score"] == len(new[0]["matched"])
    assert [c["overlap"] for c in new] == [False, True]  # 2026-09-24 is before the last update
    assert n_listed == 0
    new, n_listed = ac.select(pools, known={"2609.22222"}, updated=None)
    assert [c["id"] for c in new] == ["2609.11111"] and n_listed == 1


def test_listed_ids_reads_the_list(ac, data):
    ids = ac.listed_ids(data)
    first_arxiv = next(r for r in rows(data / "papers.csv") if "arxiv.org/abs/" in r["url"])
    assert first_arxiv["url"].rsplit("/", 1)[1] in ids


# ---------------------------------------------------------------- add_entries

TODAY = dt.date(2026, 9, 30)


def paper(**kw) -> dict:
    base = {"type": "paper", "title": "Zeta-Steer: A Brand New Steering Method for Frozen Robot Policies",
            "url": "https://arxiv.org/pdf/2609.99999v3", "date": "2026-09-30", "family": "residual",
            "kind": "paper", "one_line": "Steers the noise of a frozen flow policy; 52% to 81% on 4 tasks.",
            "org": "Not stated on abs page"}
    return {**base, **kw}


def test_add_assigns_next_id_and_normalizes(ae, data):
    before = rows(data / "papers.csv")
    top = max(int(r["id"][1:]) for r in before if r["id"].startswith("N"))
    rep = ae.add([paper(), paper(title="Another New Method: Second", url="https://arxiv.org/abs/2609.99998",
                                 date="2026-09-29", family="testtime", code="https://github.com/x/y")],
                 data=data, today=TODAY)
    assert not rep["errors"] and len(rep["papers"]) == 2
    after = rows(data / "papers.csv")
    assert len(after) == len(before) + 2
    new = {r["id"]: r for r in after if r["id"] in (f"N{top + 1:03d}", f"N{top + 2:03d}")}
    assert len(new) == 2
    a = new[f"N{top + 1:03d}"]
    assert a["url"] == "https://arxiv.org/abs/2609.99999"  # pdf + version → abs
    assert a["short"] == "Zeta-Steer"
    assert a["org"] == ""  # unknown org left blank for arxiv_meta's byline
    assert (a["key"], a["verified"]) == ("0", "read-once")
    assert after[0]["id"] == a["id"]  # newest first
    dates = [r["date"] for r in after]
    assert dates == sorted(dates, reverse=True)


def test_add_skips_duplicates(ae, data):
    existing = next(r for r in rows(data / "papers.csv") if "arxiv.org/abs/" in r["url"])
    items = [
        paper(url=existing["url"] + "v2", title="Totally different title"),         # same arXiv id
        paper(url="https://example.org/x", title=existing["title"].upper()),         # same title
        paper(),
        paper(title="Zeta-Steer again", url="https://arxiv.org/abs/2609.99999"),     # duplicate within input
    ]
    rep = ae.add(items, data=data, today=TODAY)
    assert len(rep["papers"]) == 1 and len(rep["skipped"]) == 3
    assert "this run" in rep["skipped"][2][1]
    res = rows(data / "resources.csv")[0]
    rep = ae.add([{"type": "resource", "name": "New name", "url": res["url"].rstrip("/") + "/",
                   "category": "code-rl", "org": "", "one_line": "A repo."}], data=data, today=TODAY)
    assert not rep["resources"] and len(rep["skipped"]) == 1


def test_add_validates_and_writes_nothing_on_error(ae, data):
    before = (data / "papers.csv").read_bytes()
    bad = [
        paper(),  # valid, but must not be written because others are invalid
        paper(title="B", url="https://arxiv.org/abs/2609.90001", family="not-a-family"),
        paper(title="C", url="https://arxiv.org/abs/2609.90002", kind="video"),
        paper(title="D", url="https://arxiv.org/abs/2609.90003", date="2026-10-15"),
        paper(title="E", url="https://arxiv.org/abs/2609.90004", one_line="word " * 41),
        paper(title="F", url="ftp://nope"),
        {"type": "resource", "name": "R", "url": "https://r.org", "category": "videos", "one_line": "x"},
        paper(title="G", url="https://example.org/g", date="20260929"),    # fromisoformat accepts these two
        paper(title="H", url="https://example.org/h", date="2026-W40-1"),
    ]
    rep = ae.add(bad, data=data, today=TODAY)
    assert [i for i, _, _ in rep["errors"]] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert all("YYYY-MM-DD" in rep["errors"][k][2][0] for k in (6, 7))
    assert "family" in rep["errors"][0][2][0] and "future" in rep["errors"][2][2][0]
    assert (data / "papers.csv").read_bytes() == before


def test_dry_run_writes_nothing(ae, data):
    before = (data / "papers.csv").read_bytes(), (data / "resources.csv").read_bytes()
    rep = ae.add([paper(), {"type": "resource", "name": "Zeta Gym", "url": "https://github.com/zeta/gym",
                            "category": "sim", "org": "Zeta Lab", "one_line": "A simulator."}],
                 data=data, dry_run=True, today=TODAY)
    assert len(rep["papers"]) == 1 and len(rep["resources"]) == 1
    assert ((data / "papers.csv").read_bytes(), (data / "resources.csv").read_bytes()) == before


def test_csv_round_trip_is_byte_identical(ae, data):
    # Reading and rewriting must not change the files (same columns, order, quoting and line endings),
    # so a weekly diff shows only the new rows.
    for name, fields in (("papers.csv", ae.PFIELDS), ("resources.csv", ae.RFIELDS)):
        path = data / name
        before = path.read_bytes()
        rws, eol = ae.read_csv(path)
        ae.write_csv(path, fields, rws, eol)
        assert path.read_bytes() == before, name


def test_add_only_inserts_rows_and_keeps_resources_sorted(ae, data):
    before = (data / "papers.csv").read_bytes()
    ae.add([paper()], data=data, today=TODAY)
    lines_before, lines_after = before.splitlines(), (data / "papers.csv").read_bytes().splitlines()
    assert len(lines_after) == len(lines_before) + 1
    assert set(lines_before) <= set(lines_after)
    ae.add([{"type": "resource", "name": "Zeta Gym", "url": "https://github.com/zeta/gym", "category": "sim",
             "org": "Zeta Lab (Robotics)", "one_line": "A simulator -> fast."}], data=data, today=TODAY)
    res = rows(data / "resources.csv")
    assert res == sorted(res, key=lambda r: (r["category"], r["name"].lower()))
    z = next(r for r in res if r["name"] == "Zeta Gym")
    assert z["org"] == "Zeta Lab" and z["one_line"] == "A simulator → fast."


def test_clean_org_and_families(ae, data):
    assert ae.clean_org("Unknown") == "" and ae.clean_org("") == ""
    assert ae.clean_org("Stanford (IRIS lab); NVIDIA") == "Stanford; NVIDIA"
    keys = {f["key"] for f in json.loads((data / "families.json").read_text())}
    assert ae.families(data) == keys and "generalrl" in keys
    assert {"code-robot", "sim", "survey", "awesome"} <= ae.resource_categories()


def test_count_new_counts_added_rows_only(ae, data):
    old_p, old_r = (data / "papers.csv").read_text(), (data / "resources.csv").read_text()
    assert ae.count_new(old_p, old_r, data) == (0, 0)
    ae.add([paper(), paper(title="Second One", url="https://arxiv.org/abs/2609.99997"),
            {"type": "resource", "name": "Zeta Gym", "url": "https://github.com/zeta/gym", "category": "sim",
             "org": "", "one_line": "A simulator."}], data=data, today=TODAY)
    assert ae.count_new(old_p, old_r, data) == (2, 1)
    # A row deleted again after validation (playbook step 8) no longer counts.
    rws, eol = ae.read_csv(data / "papers.csv")
    ae.write_csv(data / "papers.csv", ae.PFIELDS, [r for r in rws if r["title"] != "Second One"], eol)
    assert ae.count_new(old_p, old_r, data) == (1, 1)
