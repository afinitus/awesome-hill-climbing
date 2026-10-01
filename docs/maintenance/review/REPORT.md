# Prelaunch review — 2026-10-01

**Decision: NO-GO for an unqualified, fully validated launch.** The catalogue and course are materially improved by this review, but CupDrop-v1 does not meet the brief's no-shoving requirement, historical evaluation peeking cannot be undone, and the older-row sample contained enough substantive errors that untouched rows should not be described as fully verified. The owner can separately choose to announce a curated, source-linked collection with these limitations. This review does not authorize or publish an X post.

The review began at `efd9e71422f638e631790c8693aa6fe127f9cee8` on `review/prelaunch`. Latest `origin/main` was fetched and merged before the PR; it was already the branch's ancestor, `f076f75ad65e39c1826b7a2f7ca6c429b2eca618`. Findings below refer to the pre-fix state unless explicitly labeled residual. Generated README, family pages and site are rebuilt from data/generators.

## Scope and evidence

- **P0:** read the README introduction and site narrative, all 28 industry rows and nine company recipes, and screened 133 named-company/lab rows. Independently checked 20 deterministic changed-row samples against primary sources: **0/20 new numerical/mechanism errors**. This spot check is separate from the full must-know field review below. Details and sources: [P0 review](P0-REVIEW.md).
- **P1 must-know/industry:** checked all **122 starred entries** across title, first-version date, affiliation, family, description and official-code provenance, and closed metadata/provenance checks for all **28 industry entries**. Reused the complete arXiv API audit for arXiv titles/dates; checked non-arXiv dates against primary pages. Targeted paper sections, tables, figures, author/project links and repository contents support the field checks; this is not full-paper reading or independent experimental replication. The two manifests preserve coverage, access limits and **24 additional corrections: one major and 23 minor**: [first 61 and industry](P1-MUST-KNOW-A.json), [remaining 61](P1-MUST-KNOW-B.json).
- **P1:** inspected every implemented chapter's algorithms and experiment narrative, seed ranges, selection and budget paths; re-derived saved statistical records; replayed saved artifacts; ran all quick scripts. Detailed findings, method adaptations and limits: [chapter review](CHAPTER-REVIEW.md), [recomputation script](verify_chapter_results.py), [physical probes](HARNESS-PROBES.json).
- **P2 recent:** checked the descriptions of all **198 entries dated September 16 onward** against primary abstracts and targeted method/results passages. The two disjoint manifests cover 148 and 50 rows: [recent entries](P2-RECENT.csv), [remaining recent entries](P2-RECENT-CHAPTER.csv). This is source checking, not independent experimental replication or a full institutional/code provenance audit.
- **P2 older:** deterministic random sample of **117/584 eligible older, non-starred, non-industry rows**, selected by shuffling original CSV order with Python `random.Random(20261001)` and taking `ceil(584/5)`. Pre-fix errors: **24 major (20.5%), 39 minor (33.3%), 54 without a confirmed issue (46.2%)**. All identified sample issues were corrected. Minor issues include missing setting, terminology, truncation and attribution clarity; these percentages are not all numerical errors. The sample's high substantive-error rate limits assurance about untouched older rows. See the [catalogue review](P2-REVIEW.md) and its reproducible manifests for final coverage and dispositions.
- **Resources:** all **212 current entries** and all **66 resource leads** examined against official repositories, papers, publisher metadata or organizer pages. The brief's 213 was stale. The pass identified **15 major and 48 minor corrections/clarifications**, including eight findings outside the supplied leads. See [resource report](P2-RESOURCE-REVIEW.md) and [per-entry evidence](P2-RESOURCE-REVIEW.json).

Every catalogue/resource issue's location, severity, primary-source evidence and accepted field changes are recorded in the linked P1/P2 manifests and [catalogue decisions](p2-decisions.json). The [root lead review](P2-ROOT-LEADS.json) and [chapter reviewer's lead decisions](P2-CHAPTER-LEADS.json) preserve additional independent dispositions. These are appendices to this report; do not treat the interrupted pass's proposed wording as verified evidence.

All **373 inherited entry leads** have a disposition: **349 confirmed** and **24 rejected or requiring no change**, with none unresolved. Across the complete review, changes affect **338 distinct paper rows and 63 resource rows** (315 paper rows in the broad catalogue pass and 24 in the must-know/industry field review, with one overlap). The two interrupted-pass candidate files were removed after their evidence and dispositions were preserved in the review manifests.

## Confirmed experiment and harness issues

### Major — a dying worker pool could return an incomplete evaluation

**Location:** `lastmile/common/rollout.py::_run_in_pool`.

If a process pool broke while submitting a later shard, the old handler broke out of submission and could collect only already-submitted successful shards. The result retained the requested seed array, so dropped episodes could silently invalidate the denominator or pairing.

**Action:** cancel submitted futures, tear down the broken pool and raise a seed-specific error. Two regression cases exercise failure on the first submit and failure after a completed earlier shard. Normal rollout math and environment dynamics are unchanged. No observed committed experiment was shown to have hit this failure; this is not a reason to invent replacement historical scores.

### Major — Q-chunk target and acting used different action selectors

**Location:** `algos/05_rlpd_qchunk.py`, Q-chunk critic backup.

Acting chose the candidate maximizing the online ensemble mean; the old target instead maximized the minimum of a random target-critic pair. These can choose different chunks. The [Q-chunking paper, Appendix B.4](https://arxiv.org/html/2507.07969) describes evaluating the action selected by the implicit policy; the chapter disclosed a more cautious target policy, but that differs from the requested same-policy backup.

**Action:** choose the chunk with the online ensemble mean, then evaluate that same chunk using the target-pair minimum. Two focused tests check deliberately opposite rankings and an online tie. Reran the complete original three-seed Chapter 5 configuration, including SAC/RLPD controls and the conservative-critic ablation. Corrected Q-chunking scored **238, 245 and 238/256**, pooled **721/768 = 93.9% [92.0, 95.4]**. QC-CQL scored **154, 140 and 141/256**, pooled **435/768 = 56.6% [53.1, 60.1]**, replacing 268/768 (34.9%). The old conservative-pretraining collapse conclusion does not survive this correction. All six SAC/RLPD final counts and the random control reproduced exactly.

Results, chapter numbers and plots are replaced together. Superseded outputs and the corrected raw outputs before accounting are retained outside the active results tree with hashes. The deduplicated previous full run contributes **572,459 train steps (954.10 robot-minutes)** and **8.33 demonstration human-minutes** to development cost, separate from fresh-recipe cost. The final outcome and cost reconciliation are recorded in the chapter appendix; no published outcome is edited by hand.

### Major, residual — the success definition permits displaced or recovered cups

**Location:** `lastmile/envs/cupdrop.py::_cube_in_cup`, `docs/DESIGN.md`.

A successful cube must be released, horizontally inside the cup's current cylinder, below its rim and accompanied by an upright cup for three control steps. The predicate does not reject earlier cup displacement/tipping and does not explicitly require a lower floor bound. This does not establish the brief's no-shoving/no-tunnelling requirement.

The saved base reproduced **129/256**, including **three successful episodes ending with cup displacement >1 cm**: seeds **20017, 20047, 20122**. Seed 20047 moved the cup about **10 cm**, tipped it and completed a drop after it recovered upright. A rendered final frame and replay showed the cube inside the upright cup; this was not evidence of tunnelling. Saved ARS seed 4 reproduced **254/256**, with zero displaced-cup successes in that replay. Low-carry and random-action adversaries each scored **0/64**; two failed adversaries do not prove exploit resistance.

The corrected Chapter 5 Q-chunk seed 0 independently reproduced **238/256** from its exported critic, including **three successful episodes ending with cup displacement >1 cm** (20068, 20151, 20183). Its maximum sampled cube-contact penetration was **5.33 mm**; no successful episode had the cube center below the cup floor while horizontally inside the tested cylinder. This confirms that the movable-cup limitation also affects a learned policy. See the [replay probe](QC-REPLAY-PROBE.json) and [replay helper](replay_corrected_qc.py).

**Action:** add an explicit limitation to the design contract, chapter introduction, README and site course sections, and check in the [probe command](../../../tools/review_harness.py) and [measured outcomes](HARNESS-PROBES.json). Do not silently redefine CupDrop-v1 and thereby invalidate every historical experiment. A stricter success metric needs a versioned benchmark and reruns before a no-shoving claim is warranted.

### Major — contact-penetration guarantee exceeded the evidence

**Location:** `docs/DESIGN.md`, contact contract.

The design said penetration stayed around 1 mm and the cube did not sink into other objects. Probes sampled maximum cube-contact penetration of **4.64 mm for the base** and **5.71 mm for ARS** at control-step boundaries. None of their successful episodes had the cube center below the cup floor in cup-local coordinates while inside the tested cylinder. The sampling does not cover every physics substep.

**Action:** replace the guarantee with measured values and the sampling limit. No collision solver or contact parameters changed.

### Major, residual — quick runs and historical chapter development used held-out outcomes

**Location:** quick modes across implemented chapters; Chapter 3 diagnostic result and narrative.

Quick modes formerly generated small held-out scores during repeated debugging. They now use `search` for their final diagnostics as well. The Chapter 7 aggregate ledger was corrected to use the selected set consistently. Full experiment configurations keep their declared evaluation sets.

Chapter 3's **224/256** exploratory CEM-near-zero ticket followed earlier held-out peeking; it must not be promoted as a clean headline. The public lead remains the search-selected, peek-free racing ticket at **219/256 (85.5%)**. Both stored outcome bitstrings replayed exactly. Relabeled ambiguous internal “headline” references and tagged the exploratory result for the leaderboard. Software changes cannot erase historical adaptive reuse of held-out data; fresh independent states would be needed for new confirmatory claims.

### Major/minor — empirical and algorithmic claims were too strong

**Locations:** chapters 0, 2, 3, 6 and 7; details in [chapter review](CHAPTER-REVIEW.md).

Clarified that 0/16 observed successes does not prove zero support; three tested latent mappings do not establish that any mapping helps; non-significant paired differences do not establish equivalence; and the residual TD3 implementation's multi-step off-policy target is an approximation. The course's SAC/RLPD and DSRL adaptations remain disclosed. These are instructional variants, not uniform faithful reproductions of their source algorithms. Existing counts were preserved except the independently rerun Chapter 5 experiment.

## Confirmed front-page and catalogue issues

[P0-REVIEW.md](P0-REVIEW.md) records five editorial fixes with primary links: the categorical “99% is a product” framing; AgiBot LWD's mixed success/sub-step metric; universal judge-exploitation wording; unsupported historical full-read/weekly-process wording; and the current AI-assistance disclosure. No new critical false company statement was confirmed in the spot sample. Vendored model licenses match their pinned upstream copies, and all hardware walkthroughs include basic safety notes.

The catalogue decisions include wrong algorithm mechanisms, wrong before/after baselines, omitted simulation settings, relative versus percentage-point confusion, composite metrics presented without context, incomplete affiliations, internal audit notes leaking into public descriptions, and project/placeholder pages mislabeled as code. Examples include DIVO's separate behavior model and actor; HIL-SERL's 29% separate HG-DAgger comparator; SPIRE's simulated success percentages; Max-Q's different 99% and 100% endpoints; and Thinking Machines' distillation-cost comparator. All applied rewrites were checked against primary evidence; optional or unsupported leads were rejected. Descriptions remain at most 40 whitespace-delimited words.

The must-know/industry pass additionally replaced PDE's stale aggregate with its current paper's directly tabulated 81.8% versus 0% comparison; specified PSS's final-five-evaluation mean; qualified best-validation/peak results, chart readings, per-task demonstration counts and pre-generated-noise throughput; and removed a benchmark landing page from the implementation-code field. Repeated DAWN/PSS claims in the chapters were synchronized, and the site's PDE transfer explanation is attributed to the authors as a suggestion. Company-post dates now match the linked artifacts. Three entries whose publication dates could not be verified are explicitly undated, including two YC profiles whose cohort dates were not publication dates. The generator displays them separately instead of placing them on a dated timeline.

Source-copying screening found no normalized 13-word overlap in the P0 sample. The resource screen flagged one entry with a 13+-word overlap in *Self-evolving Embodied AI*: five overlapping windows forming a 17-token enumeration of the source's named taxonomy components, retained as category names. No long copied prose passage was confirmed in these accessible-source checks; this is not copyright clearance. No secrets or user-local paths were found in the tracked-text pattern scan.

**Minor — repository lint:** the complete Ruff check initially reported 33 style/typing/timezone issues across ledger, plotting, tests and generator/link tools. Corrected them without changing generated text or experiment calculations. New ledger timestamps include the local UTC offset; filenames retain local wall-clock time. README generation and all four banner SVGs were checked for unchanged output, and repository-wide Ruff now passes.

## Checks and reproducibility

The final machine-readable check record is [VALIDATION.json](VALIDATION.json). It records command outcomes, smoke timings and the no-mutation check rather than relying on this narrative alone.

Baseline statistical reconciliation traversed **145 result JSONs**, **17,669 nested k/n objects**, **232 explicitly written document rates** and **220 accompanying intervals**, with no arithmetic discrepancy. The durable audit subsequently recomputed **181 discordant-count objects and 666 paired-test objects** (847 checks, including duplicated shared analyses); three Chapter 5 Fisher comparisons were checked separately. An exact convolution of the empirical paired-difference distribution also checked the saved bootstrap interval endpoints: all 666 objects (38 distinct count triples) agreed within one episode-count increment of the recorded Monte Carlo estimates. These counts include repeated copies of analyses and are not counts of independent experiments. The document parser does not cover every prose claim or plot label. Chapter reconciliation handles those separately.

After replacement, the [final verifier output](CHAPTER-VERIFICATION.txt) confirms the same 145 active JSONs, 17,669 count objects and paired/bootstrap totals, plus **233 explicit document rates and 224 intervals**, with no discrepancy. The [publication check](CH05-PUBLICATION-CHECK.json) verifies the 26 archived result hashes, source/checkpoint hashes and accounting-only differences from the raw corrected output. The training log's single absolute checkout-path prefix was removed before publication; its updated hash and the redaction are recorded in the manifest.

Saved-artifact replay reproduced the base **129/256**, ARS **254/256**, exploratory Chapter 3 **224/256**, and public Chapter 3 **219/256**. Shared random-number pairing and shard-order assembly were traced; existing tests verify single/multiworker parity and exact save/restore continuations. MuJoCo integration state, controller state, friction, success streak and termination state are saved. The success latch and rollout active mask count a terminated episode once. [MuJoCo state documentation](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate) supports using `mjSTATE_INTEGRATION` rather than positions and velocities alone.

The Wilson and exact McNemar implementations agree with their standard definitions; paired bootstrap resamples paired outcome differences. Wilson intervals over repeated fixed states are not uncertainty over new tasks or independent training seeds. Bootstrap intervals with few discordant pairs and unadjusted families of paired tests require the already documented caution.

The published configurations' demo/training ranges are disjoint from search, eval, eval_ext and stress. Arbitrary custom CLI seed/budget values are not globally range-validated. Several historical pilots and learned checkpoints are absent from the repository; charged constants were traced but their historical collection cannot be independently reconstructed from committed logs.

The corrected Chapter 5 run began with the selector fix uncommitted. Its raw `git_sha` records repository HEAD at write time, not a complete snapshot of the running source. The archive manifest records this limitation, the final source hash and the raw outputs before the explicit accounting migration. Subsequent Chapter 5 edits during the run affected historical-cost metadata, plot labels and comments, not training or action selection.

Browser rendering was attempted through the available computer-use interface. No browser-control surface was available and native Chrome and Safari returned `cgWindowNotFound`; rendered visual QA could not be completed. HTML structure, sections, counts, links and interactive-control markup are checked by the site test suite. This limitation is separate from source verification and generator consistency.

## Required follow-up before stronger promotion

1. Define and validate a versioned no-shoving success metric, then rerun affected benchmark claims; do not quietly relabel CupDrop-v1 results.
2. Use a fresh, untouched evaluation set for confirmatory comparisons after historical peeking and preserve complete pilot/checkpoint provenance.
3. Extend source review to untouched older entries before claiming complete verification. A primary-source catalogue can be useful without that claim; the sampled error rate must remain visible to the owner.
4. Inspect the rendered site in an available browser. Treat blocked publisher/OpenReview sources according to the narrower evidence scopes in the manifests, not as fully read documents.

The owner reviews and merges the PR. The public Pages site continues to reflect `main` until that merge and deployment.
