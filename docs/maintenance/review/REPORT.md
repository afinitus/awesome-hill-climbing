# Prelaunch review — updated 2026-10-02

**Decision: GO for a scoped announcement of the source-linked catalog and educational simulation course after this PR is reviewed, merged and deployed. NO-GO for calling it an exhaustive collection or a fully validated robotics benchmark.** All 912 catalog descriptions now have a recorded source review, and all 472 files tracked at the follow-up baseline have an inspection record. CupDrop-v1 still permits some cup movement, historical evaluation peeking cannot be undone, and external paper results have not been independently reproduced. This review does not publish an X post.

The review began at `efd9e71422f638e631790c8693aa6fe127f9cee8` on `review/prelaunch`. Latest `origin/main` was fetched and merged before the PR; it was already the branch's ancestor, `f076f75ad65e39c1826b7a2f7ca6c429b2eca618`. Findings below refer to the pre-fix state unless explicitly labeled residual. Generated README, family pages and site are rebuilt from data/generators.

## Whole-repository follow-up — 2026-10-02

The owner asked to extend the initial sampled review to the entire repository. This follow-up inventories **all 472 tracked files at `25a9a1d`**, with a file-specific inspection method and content hash in [REPOSITORY-FILE-COVERAGE.json](second-pass/REPOSITORY-FILE-COVERAGE.json). Source and prose were inspected; structured results were reconciled; meshes/checkpoints were loaded or structurally parsed; every raster frame was decoded and selected animation frames were visually inspected. These are different inspection methods, not a claim to execute every code path or visually inspect every animation frame.

**All 912 catalog descriptions now have a recorded primary-source review**, combining 445 previously completed whole-description reviews with 467 newly completed reviews. The conservative selection does not count an old targeted correction as a whole-description review. Metadata closure covers the 143 prior all-field reviews plus 467 newly reviewed entries and 302 additional metadata-only entries. All 212 resources retain the completed first-pass source review. This is verification of the catalog's listed claims against relevant source passages, tables, figures, author blocks and official artifact links—not full-paper reading, proof checking, or independent reproduction of external experiments. The [scope](second-pass/CATALOG-SCOPE.json) and [final per-entry coverage and applied changes](second-pass/CATALOG-FINAL-COVERAGE.json) make that distinction auditable.

### Additional catalog findings

This pass changes content or metadata in **226 distinct paper rows**: **75** receive corrections, protocol clarifications or source-version updates (**15 major, 60 minor** under the recorded review classifications), and **151** receive only optional affiliation/title/link enrichment. These are not 226 false results. In particular, BORA’s revision update does not prove its older source was wrong. Review-status fields also changed in 621 rows; status-only edits are not counted as content corrections. All accepted descriptions remain at most 40 whitespace-delimited words.

Examples include correcting LexiSafe's 44.3→56.0 **success counts out of 60**, distinguishing GRAFT's last ten online training episodes per task from frozen-policy evaluation, replacing ambiguous percentage gains with trial counts or percentage points, specifying per-task training budgets, and updating BORA to its revised v3 results. Placeholder or inaccessible implementation links were removed; valid partial releases were retained with their scope recorded in the audit. An undated source's unsupported exact date was cleared. A formal institution inferred only from an email domain was narrowed to an author-name attribution.

The decision manifests are [A](second-pass/CATALOG-A-REVIEW.json), [A2](second-pass/CATALOG-A2-REVIEW.json), [A3](second-pass/CATALOG-A3-REVIEW.json), [B](second-pass/CATALOG-B-REVIEW.json), [B2](second-pass/CATALOG-B2-REVIEW.json), [metadata 1](second-pass/CATALOG-METADATA-CHAPTER.json), and [metadata 2](second-pass/CATALOG-METADATA-ROOT.json). They preserve the exact scope, source hashes, locators and access limits. A separate [recheck](second-pass/METADATA-PATCH-RECHECK.json) confirmed 15 sampled affiliation enrichments and rejected two proposed code additions whose official pages still defer public release. Neither rejected link was added. The proposed N012 title change was also declined: its v3 manuscript heading differs from the canonical arXiv abstract/API title, which remains the catalog title; both source versions are recorded in the final disposition.

A normalized 13-consecutive-token screen compared **all 912 descriptions** with cached primary text. Three short overlaps were conservatively paraphrased without changing their reported results. The [screen](second-pass/SOURCE-COPY-SCREEN.json) records its extraction and normalization limits; it is not copyright clearance. The initial all-resource screen and license checks remain applicable.

### Additional code, documentation and presentation findings

- **Major:** Chapter 3 replotting could replay a quick result on the default held-out episode set. It now restores the saved configuration and episode set. Chapter 6/7 replots also preserve quick/custom/one-arm output destinations. Details and regressions: [chapter follow-up](second-pass/CHAPTERS.md).
- **Major:** malformed multidimensional outcomes or noninteger/mismatched seed arrays could produce misleading ledger scores or pairing. The ledger now rejects them and rejects negative/nonfinite budget values before serialization. The leaderboard also rejects impossible counts, rates inconsistent with k/n, invalid intervals and nonfinite/negative costs, and respects known initialization-set version mismatches. Existing valid results are unchanged.
- **Major accounting disclosure:** Chapter 0 omitted training-worker CPU from its historical total. Future runs include it; the old 0.69 CPU-hours remains explicitly incomplete because the missing telemetry cannot be recovered. Nominal update counts are now described as rounded to full epochs. No historical total was invented.
- **Major presentation:** the site's Wilson calculator silently truncated fractional trial counts. It now requires nonempty safe integers with `0 ≤ k ≤ n` and `n > 0`; invalid input clears the result and hides the point. Safari checks covered invalid, boundary and normal cases.
- **Major documentation:** the research curriculum mixed planned hardware/chapters with implemented simulation features and used stale environment settings. It now describes the nine implemented chapters, ten scripts, actual simulation contract, planned work and current experiment limitations.
- **Minor:** date-period labels now say “since September 2025”; generic completeness/reliability claims were narrowed; Wilson examples specify fixed evaluation sizes and their assumptions; finite pass@k is no longer called a hard support ceiling. Contributor examples use search-only quick evaluation and actual `EvalResult` objects. The original review brief is labeled historical.
- **Minor:** missing vendored README preview images now use pinned upstream links with a local modification note. Physical-code comments no longer imply guarantees against cup shoving or contact penetration. CI now checks Ruff and both generated outputs, with read-only repository permission for the test workflow.

The independent [review of root changes](second-pass/ROOT-CHANGE-RECHECK.json) found no concrete regression in its stated scope. New regression cases address behavior that could misreport scores, overwrite publication assets, expose held-out states or omit compute charges; they do not reproduce third-party paper results.

### Follow-up validation and remaining limits

The [follow-up validation record](second-pass/VALIDATION.json) records **176 passing tests**, all ten default quick runs in **264.15 seconds**, no changes/additions under published results/media/checkpoints, repository-wide Ruff, lockfile consistency, generator checks and fresh external-link/title/date checks. Saved-statistic reconciliation again passed for all 145 active result files; the 26 archived Chapter 5 result hashes still match their manifest. A built wheel includes all 115 source model-asset files.

All 61 media files were checked: 40 PNGs, 18 GIFs and three SVGs. All 834 raster frames decoded; every PNG, first/middle/final GIF frames and all rendered SVGs received visual inspection. Safari verification covered the desktop dark theme, a 734-pixel-wide responsive layout, search and combined filters, resource accordions and the calculator. It did not cover every browser, a physical mobile device, a 390-pixel viewport or the light browser theme. The preview used this branch locally; the deployed site remains `main` until merge.

The two scientific follow-ups below still apply. The full catalog review closes the earlier untouched-row coverage gap; it does not establish that published research is correct, that the collection contains every technique, or that no future error will be found. N424's OpenReview PDF remained access-blocked, so its qualitative entry rests on accessible primary abstract/author and official workshop material. Other source/implementation access limits are recorded per entry rather than hidden by a “checked” badge.

## First-pass scope and evidence — 2026-10-01

- **P0:** read the README introduction and site narrative, all 28 industry rows and nine company recipes, and screened 133 named-company/lab rows. Independently checked 20 deterministic changed-row samples against primary sources: **0/20 new numerical/mechanism errors**. This spot check is separate from the full must-know field review below. Details and sources: [P0 review](P0-REVIEW.md).
- **P1 must-know/industry:** checked all **122 starred entries** across title, first-version date, affiliation, family, description and official-code provenance, and closed metadata/provenance checks for all **28 industry entries**. Reused the complete arXiv API audit for arXiv titles/dates; checked non-arXiv dates against primary pages. Targeted paper sections, tables, figures, author/project links and repository contents support the field checks; this is not full-paper reading or independent experimental replication. The two manifests preserve coverage, access limits and **24 additional corrections: one major and 23 minor**: [first 61 and industry](P1-MUST-KNOW-A.json), [remaining 61](P1-MUST-KNOW-B.json).
- **P1:** inspected every implemented chapter's algorithms and experiment narrative, seed ranges, selection and budget paths; re-derived saved statistical records; replayed saved artifacts; ran all quick scripts. Detailed findings, method adaptations and limits: [chapter review](CHAPTER-REVIEW.md), [recomputation script](verify_chapter_results.py), [physical probes](HARNESS-PROBES.json).
- **P2 recent:** checked the descriptions of all **198 entries dated September 16 onward** against primary abstracts and targeted method/results passages. The two disjoint manifests cover 148 and 50 rows: [recent entries](P2-RECENT.csv), [remaining recent entries](P2-RECENT-CHAPTER.csv). This is source checking, not independent experimental replication or a full institutional/code provenance audit.
- **P2 older:** deterministic random sample of **117/584 eligible older, non-starred, non-industry rows**, selected by shuffling original CSV order with Python `random.Random(20261001)` and taking `ceil(584/5)`. Pre-fix errors: **24 major (20.5%), 39 minor (33.3%), 54 without a confirmed issue (46.2%)**. All identified sample issues were corrected. Minor issues include missing setting, terminology, truncation and attribution clarity; these percentages are not all numerical errors. The sample’s high substantive-error rate motivated the complete follow-up above; there are now no untouched catalog-description rows. See the [catalogue review](P2-REVIEW.md) and its reproducible manifests for final coverage and dispositions.
- **Resources:** all **212 current entries** and all **66 resource leads** examined against official repositories, papers, publisher metadata or organizer pages. The brief's 213 was stale. The pass identified **15 major and 48 minor corrections/clarifications**, including eight findings outside the supplied leads. See [resource report](P2-RESOURCE-REVIEW.md) and [per-entry evidence](P2-RESOURCE-REVIEW.json).

Every catalogue/resource issue's location, severity, primary-source evidence and accepted field changes are recorded in the linked P1/P2 manifests and [catalogue decisions](p2-decisions.json). The [root lead review](P2-ROOT-LEADS.json) and [chapter reviewer's lead decisions](P2-CHAPTER-LEADS.json) preserve additional independent dispositions. These are appendices to this report; do not treat the interrupted pass's proposed wording as verified evidence.

All **373 inherited entry leads** have a disposition: **349 confirmed** and **24 rejected or requiring no change**, with none unresolved. Across the initial review, changes affected **338 distinct paper rows and 63 resource rows** (315 paper rows in the broad catalogue pass and 24 in the must-know/industry field review, with one overlap). The two interrupted-pass candidate files were removed after their evidence and dispositions were preserved in the review manifests.

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

Quick modes formerly generated small held-out scores during repeated debugging. They now use `search` for their final diagnostics as well. The Chapter 7 aggregate ledger was corrected to use the selected set consistently. Full experiment configurations keep their declared evaluation sets. Quick-run console, figure and GIF-status labels now name the search set too; Chapter 12’s held-out verifier-validation AUC remains labeled as validation. A [22-figure display check](QUICK-LABEL-CHECK.json) confirmed that full-run labels did not change.

Chapter 3's **224/256** exploratory CEM-near-zero ticket followed earlier held-out peeking; it must not be promoted as a clean headline. The public lead remains the search-selected, peek-free racing ticket at **219/256 (85.5%)**. Both stored outcome bitstrings replayed exactly. Relabeled ambiguous internal “headline” references and tagged the exploratory result for the leaderboard. Software changes cannot erase historical adaptive reuse of held-out data; fresh independent states would be needed for new confirmatory claims.

### Major/minor — empirical and algorithmic claims were too strong

**Locations:** chapters 0, 2, 3, 6 and 7; details in [chapter review](CHAPTER-REVIEW.md).

Clarified that 0/16 observed successes does not prove zero support; three tested latent mappings do not establish that any mapping helps; non-significant paired differences do not establish equivalence; and the residual TD3 implementation's multi-step off-policy target is an approximation. The course's SAC/RLPD and DSRL adaptations remain disclosed. These are instructional variants, not uniform faithful reproductions of their source algorithms. Existing counts were preserved except the independently rerun Chapter 5 experiment.

## Confirmed front-page and catalogue issues

[P0-REVIEW.md](P0-REVIEW.md) records five editorial fixes with primary links: the categorical “99% is a product” framing; AgiBot LWD's mixed success/sub-step metric; universal judge-exploitation wording; unsupported historical full-read/weekly-process wording; and the current AI-assistance disclosure. No new critical false company statement was confirmed in the spot sample. Vendored model licenses match their pinned upstream copies, and all hardware walkthroughs include basic safety notes.

The catalogue decisions include wrong algorithm mechanisms, wrong before/after baselines, omitted simulation settings, relative versus percentage-point confusion, composite metrics presented without context, incomplete affiliations, internal audit notes leaking into public descriptions, and project/placeholder pages mislabeled as code. Examples include DIVO's separate behavior model and actor; HIL-SERL's 29% separate HG-DAgger comparator; SPIRE's simulated success percentages; Max-Q's different 99% and 100% endpoints; and Thinking Machines' distillation-cost comparator. All applied rewrites were checked against primary evidence; optional or unsupported leads were rejected. Descriptions remain at most 40 whitespace-delimited words.

The must-know/industry pass additionally replaced PDE's stale aggregate with its current paper's directly tabulated 81.8% versus 0% comparison; specified PSS's final-five-evaluation mean; qualified best-validation/peak results, chart readings, per-task demonstration counts and pre-generated-noise throughput; and removed a benchmark landing page from the implementation-code field. Repeated DAWN/PSS claims in the chapters were synchronized, and the site's PDE transfer explanation is attributed to the authors as a suggestion. Company-post dates now match the linked artifacts. The first pass left three entries explicitly undated, including two YC profiles whose cohort dates were not publication dates; the follow-up also cleared N195’s unsupported exact date. The generator displays them separately instead of placing them on a dated timeline.

Source-copying screening found no normalized 13-word overlap in the P0 sample. The resource screen flagged one entry with a 13+-word overlap in *Self-evolving Embodied AI*: five overlapping windows forming a 17-token enumeration of the source's named taxonomy components, retained as category names. No long copied prose passage was confirmed in these accessible-source checks; this is not copyright clearance. No secrets or user-local paths were found in the tracked-text pattern scan.

**Minor — repository lint:** the complete Ruff check initially reported 33 style/typing/timezone issues across ledger, plotting, tests and generator/link tools. Corrected them without changing generated text or experiment calculations. New ledger timestamps include the local UTC offset; filenames retain local wall-clock time. README generation and the banner assets were checked for unchanged output, and repository-wide Ruff now passes.

## Checks and reproducibility

The initial machine-readable check record is [VALIDATION.json](VALIDATION.json); the follow-up record is [second-pass/VALIDATION.json](second-pass/VALIDATION.json). It records command outcomes, smoke timings and the no-mutation check rather than relying on this narrative alone.

Baseline statistical reconciliation traversed **145 result JSONs**, **17,669 nested k/n objects**, **232 explicitly written document rates** and **220 accompanying intervals**, with no arithmetic discrepancy. The durable audit subsequently recomputed **181 discordant-count objects and 666 paired-test objects** (847 checks, including duplicated shared analyses); three Chapter 5 Fisher comparisons were checked separately. An exact convolution of the empirical paired-difference distribution also checked the saved bootstrap interval endpoints: all 666 objects (38 distinct count triples) agreed within one episode-count increment of the recorded Monte Carlo estimates. These counts include repeated copies of analyses and are not counts of independent experiments. The document parser does not cover every prose claim or plot label. Chapter reconciliation handles those separately.

After replacement, the [final verifier output](CHAPTER-VERIFICATION.txt) confirms the same 145 active JSONs, 17,669 count objects and paired/bootstrap totals, plus **233 explicit document rates and 224 intervals**, with no discrepancy. The [publication check](CH05-PUBLICATION-CHECK.json) verifies the 26 archived result hashes, source/checkpoint hashes and accounting-only differences from the raw corrected output. The training log's single absolute checkout-path prefix was removed before publication; its updated hash and the redaction are recorded in the manifest.

Saved-artifact replay reproduced the base **129/256**, ARS **254/256**, exploratory Chapter 3 **224/256**, and public Chapter 3 **219/256**. Shared random-number pairing and shard-order assembly were traced; existing tests verify single/multiworker parity and exact save/restore continuations. MuJoCo integration state, controller state, friction, success streak and termination state are saved. The success latch and rollout active mask count a terminated episode once. [MuJoCo state documentation](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate) supports using `mjSTATE_INTEGRATION` rather than positions and velocities alone.

The Wilson and exact McNemar implementations agree with their standard definitions; paired bootstrap resamples paired outcome differences. Wilson intervals over repeated fixed states are not uncertainty over new tasks or independent training seeds. Bootstrap intervals with few discordant pairs and unadjusted families of paired tests require the already documented caution.

The published configurations' demo/training ranges are disjoint from search, eval, eval_ext and stress. Arbitrary custom CLI seed/budget values are not globally range-validated. Several historical pilots and learned checkpoints are absent from the repository; charged constants were traced but their historical collection cannot be independently reconstructed from committed logs.

The corrected Chapter 5 run began with the selector fix uncommitted. Its raw `git_sha` records repository HEAD at write time, not a complete snapshot of the running source. The archive manifest records this limitation, the integration source hash and the raw outputs before the explicit accounting migration. A separate publication-source hash identifies later display-only edits without replacing the historical integration digest. Subsequent Chapter 5 edits during the run affected historical-cost metadata, plot labels and comments, not training or action selection.

The first pass could not obtain a browser window through the computer-use interface. The follow-up resolved that access limitation and completed the scoped Safari checks recorded above. HTML structure, sections, counts, links and interactive-control markup also pass the site test suite.

## Required follow-up before stronger promotion

1. Define and validate a versioned no-shoving success metric, then rerun affected benchmark claims; do not quietly relabel CupDrop-v1 results.
2. Use a fresh, untouched evaluation set for confirmatory comparisons after historical peeking and preserve complete pilot/checkpoint provenance.
The follow-up completed the catalog-description coverage and scoped browser inspection requested after the initial review. Treat blocked publisher/OpenReview sources according to the narrower evidence scopes in the manifests, not as fully read documents.

The owner reviews and merges the PR. The public Pages site continues to reflect `main` until that merge and deployment.
