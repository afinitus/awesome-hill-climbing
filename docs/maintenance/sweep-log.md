# Weekly sweep log

## 2026-10-05: failed
- Window: 2026-10-02 → 2026-10-05 (arXiv searched from 2026-09-28; first run without seen-ids.txt, so 411 of the 469 new candidates predate 2026-10-02)
- Candidates: 469 arXiv (46 already listed, 0 seen), 0 web items in window (16 companies and 14 researchers checked; pi.website returned 429/403 and could not be confirmed; X/Twitter poorly indexed), 0 relevant repos (6 topic searches, all hobby LeRobot repos); code-link check of 236 recent rows without code: 1 candidate (C442 github.com/midea-ai/ars) but it returns 404, so not added
- Added (staged, not pushed): 40 papers (N448-N487), 0 resources, 0 code links: testtime 10, simreal 7, generalrl 5, residual 4, hitl 3, blackbox 3, onpolicy 2, advbc 2, worldmodel 2, reward 1, offpolicy 1; LocoWM code link blanked in the pending file (some N ids may shift when they are re-added)
- Skipped: 429 (371 on title alone, mostly LLM-RL applications, pretraining/WAM architectures, from-scratch locomotion/control and off-topic perception/planning; 58 after reading the abstract). Judgment calls skipped: 2609.39018 (URAI), 2609.39971 (ECT), 2610.01083 (WBAG), 2609.38641 (AD-Memo), 2609.38862 (EMPlan). No unclear ids
- Checks: arxiv_meta ok (old TITLE? 2608.22364 WAM-OPD: arXiv title now "WAM-OPD: Joint Video-Action Supervision for World Action Model Post-Tr..."; not touched); arxiv_meta rewrote data/papers.csv with CRLF line endings, converted back to LF; check_links 2 new failures blanked (github.com/midea-ai/ars, zhaozijie2022.github.io/LocoWM), 3 old blocked (2 doi.org, 1 openreview); build --check ok; site --check ok; pytest 1 failed
- Not pushed: `uv run --no-sync pytest -q tests/test_sweep_tools.py tests/test_build_site.py` → FAILED tests/test_sweep_tools.py::test_add_assigns_next_id_and_normalizes: `assert after[0]["id"] == a["id"]  # newest first` / `AssertionError: assert 'N485' == 'N488'`. The test adds a fixture entry dated 2026-09-30 to a copy of the live list and expects it to sort first; it fails whenever the list holds rows dated after 2026-09-30 (it passes on origin/main). The test needs a human fix (e.g. a fixture date relative to the data's newest date) before any later sweep can push

## 2026-10-05: ok
- Re-run of the failed entry above after c785b9c fixed the date-dependent test; same window, candidates, triage and entries
- Added: 40 papers/posts (N448-N487), 0 resources, 0 code links
- Checks: build --check ok; site --check ok; pytest tests/test_sweep_tools.py tests/test_build_site.py ok (33 passed)
