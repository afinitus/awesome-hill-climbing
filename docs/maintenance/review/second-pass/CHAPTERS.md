# Second pass: executable chapters, harness, and saved artifacts

Reviewed against commit `25a9a1d` on 2026-10-02. This follows the complete first-pass algorithm/narrative review in [CHAPTER-REVIEW.md](../CHAPTER-REVIEW.md), the corrected Chapter 5 rerun, and the recorded replay/statistical checks. The second pass concentrates on previously less-exercised configuration, plotting/replay, accounting and artifact paths; it does not claim a second independent training run for every chapter.

## Coverage and method

[CHAPTER-FILE-COVERAGE.json](CHAPTER-FILE-COVERAGE.json) inventories **310 tracked files**: 10 executable algorithms, 129 library/environment/vendor files, 13 test files, 145 result JSONs, 3 checkpoint/manifest files and 10 chapter documents. Every entry records path, current SHA256, size and inspection method. `tests/test_build_site.py` and `tests/test_sweep_tools.py` receive syntax inspection here; root owns their source-level generator/maintenance review. Root also owns public media decode/visual QA, overall reports and generators.

The shared harness sources were read completely, including CLI override precedence, outcome statistics, ledger serialization/pairing, worker failures, per-episode RNGs, trajectory masking, NumPy/torch flow sampling, state restore, IK, controller phases and the physical success predicate. Executable algorithm cores retain the first-pass source audit; every script received a second-pass inspection of its configurations, seed expressions, main/output/accounting paths and supported replot path. Chapter documents received a result-arithmetic/caveat/reproduction cross-check against the prior narrative audit.

Every vendor mesh was structurally parsed: binary STL record lengths and all normal/vertex floats; OBJ vertices/normals/UVs, face arity and vertex indices. Every XML model/scene compiled through MuJoCo, resolving its meshes/includes and checking finite geometry/masses. This verifies local integrity and usability, not physical fidelity of third-party CAD. The vendored README declares Menagerie commit `4d038b3feae26ec82b46a4d586379114012a8ac7`; Apache 2.0 and MIT license files are present. The audit does not claim a byte-for-byte comparison of every mesh against upstream.

## Confirmed issues fixed

1. **Chapter 3 replot used the wrong episode set (major).** Its figures read the saved `final.init_set`, but its GIF replay used the new CLI configuration, whose default is full/eval. Replotting a quick/search result without `--quick` could therefore expose eval episodes and render different trajectories. Replay now uses the saved configuration and explicit saved init set. Quick replots write under `runs/ch03_quick/media`.
2. **Chapter 6/7 replots could overwrite public media (minor).** They reconstructed quick-run labels but always wrote canonical media. They now restore the original quick/custom/one-arm output routing. Full-run replot destinations are preserved.
3. **Malformed ledger outcomes could create a plausible wrong denominator (major for callers).** For example `[[True,False],[False,False]]` was counted as 1/2 instead of being rejected. Scores now require a nonempty one-dimensional binary array, and an `EvalResult` must have one integer environment seed per outcome. Fractional/boolean seeds and two-dimensional seed arrays are rejected instead of being silently coerced before pairing. Nonfinite/negative robot, human and explicitly accumulated compute charges are rejected before any JSON file is written. Existing valid experiment outcomes and budgets are unaffected.
4. **Chapter 0 omitted training-worker CPU (major accounting disclosure).** `train_job` ran in spawned trainer processes and returned only loss; `train_all` never charged that CPU to the ledger. It now returns elapsed process CPU and the parent adds every trainer's contribution while retaining rollout-worker charges. The historical **0.69 CPU-hours remains unchanged and is now explicitly described as incomplete** in ch00. No replacement total can be recovered from the saved artifacts, so none was invented.
5. **Chapter 0 update budgets were described too exactly (minor).** The trainer rounds requested update budgets up to whole epochs. The saved base has 1449 samples, six batches per epoch and 667 epochs: 4002 updates for the nominal 4000-step budget. The chapter now explains this convention. Existing checkpoints, sweep values and historical metadata remain intact. A causal statement attributing a sweep reversal solely to data exposure is also softened because that sweep does not isolate the cause.
6. **Physical comments overclaimed protection (minor).** The knob controller's latched target does not guarantee that a lucky cup shove cannot succeed; floor thickness does not prove penetration impossible. Comments now state the narrower implemented behavior without changing dynamics or success semantics.
7. **Two vendored README preview images were broken (minor).** The preview files were omitted from the vendored subset. Their links now point to the pinned upstream images, both checked as HTTP 200 image/png. A local documentation note records this modification; XML, meshes and license text are unchanged.

Regression checks cover saved-search replay routing despite default full CLI configuration, custom Chapter 6 output roots, malformed outcomes and seed lengths, every negative/nonfinite budget category, trainer CPU accumulation and exact whole-epoch rounding. These tests use mocks/small inputs rather than retraining published policies.

## Saved values and provenance

The read-only verifier again checks all 145 active result files: 17,669 nested count/rate/Wilson objects; 181 discordant-count and 666 nested paired tests; 666 saved bootstrap objects corresponding to 38 distinct empirical paired distributions. Exact convolution agrees with the saved Monte Carlo percentile endpoints within one episode count. It also verifies 233 explicit chapter percentage statements and 224 explicit intervals. Repeated shared JSON analyses are counted repeatedly; these are not independent experiments. Parser coverage is explicitly narrower than every implicit-denominator prose claim.

Both checkpoints load safely (`weights_only=True` for torch, `allow_pickle=False` for NPZ); all numerical parameters are finite, the base's normalizers are positive, and its architecture/config hash matches the companion manifest. The corrected QC checkpoint digest and base checkpoint digest match the archived review manifest. `algos/05_rlpd_qchunk.py` was not edited: its publication SHA256 remains `9a96e432efaf1cabf57ed23f31ddc883f223ffe70615ed98d0896ee20ba70d31`, separate from the historical integration digest. Raw result `git_sha` still does not capture the selector fix that was uncommitted when training started; the archive explains this provenance limitation.

No published result, checkpoint or media file was regenerated or changed in this pass. Corrected QC remains 721/768; QC-CQL 435/768; Chapter 3's declared search-selected peek-free headline remains 219/256. Previous saved-artifact replays and physical probes are reused as evidence; this pass does not charge or claim a new full policy replay.

## Remaining boundaries, not hidden fixes

- The existing v1 metric can count a cube in a relocated/recovered upright cup. Prior base and corrected-QC probes demonstrate such successes. This is disclosed in the chapter index/design/review; enforcing the original no-shoving aspiration would define a different benchmark and require new results. The probe did not establish tunneling, and control-rate observations do not rule out every between-sample contact artifact.
- Historical Chapter 3 leakage and exploratory eval-picked tickets remain labeled. Search-only default quick runs prevent new smoke leakage; they do not erase historical access to eval data or make post hoc analyses confirmatory.
- `parse` deliberately gives explicit CLI options precedence over QUICK defaults. Chapter 5 `--final-set` and Chapter 6 `--eval-set` can explicitly request held-out data even with `--quick`; search-only smoke claims refer to the supplied default quick recipe. Custom seed ranges/budgets are not globally validated for all possible overlap, emptiness or resource size. The audited defaults are disjoint from all held-out sets. Arbitrary configurations are not certified experiments.
- Learned chapters use the SO-101 base checkpoint; accepting a generic `--robot` field does not provide a trained PiPER base or establish PiPER learning results.
- Pooled Wilson intervals over repeated start states do not quantify uncertainty across independent tasks or training runs. The existing per-seed counts and caveats remain necessary. Families of unadjusted paired tests remain exploratory.
- Cost rows often share whole-run compute or reused selection costs and cannot simply be summed as independent bills. Chapter 0's missing historical trainer CPU is newly disclosed; recovering that number would require original process telemetry, not arithmetic on success counts.
- A structural mesh/checkpoint pass and small regression suite are not a proof of physical fidelity, safety on hardware, exhaustive parameter validity or every possible runtime path.

## Reproduction

From the repository root:

```sh
.venv.nosync/bin/python docs/maintenance/review/second-pass/audit_chapter_files.py
.venv.nosync/bin/python docs/maintenance/review/verify_chapter_results.py
.venv.nosync/bin/python -m pytest -q tests --ignore=tests/test_build_site.py --ignore=tests/test_sweep_tools.py
.venv.nosync/bin/ruff check algos lastmile tests/test_chapter_algorithms.py tests/test_ledger.py docs/maintenance/review/second-pass/audit_chapter_files.py
git diff --name-only 25a9a1d -- results media checkpoints
```

The last command returns no paths, confirming that the published artifacts are unchanged. The scoped suite passed **126 tests in 15.96s** before the final four malformed-seed cases were added. The final changed-path suite passed **35 tests in 1.41s**, including those four cases; Ruff, the 310-file structural/provenance audit and whitespace checks passed. Root will run the final complete suite and smoke/no-mutation check after all second-pass changes converge.
