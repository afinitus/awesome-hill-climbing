# P0 independent spot review — 2026-10-01

This is a fresh spot review of the completed P0 pass at `f076f75`, not a claim to have independently reproduced the papers or re-audited every sentence against every cited paper. The README introduction, Start here, loop, foundations, all explainer narrative sections, family descriptions, and named-company/lab mentions were read. Company results remain author-reported. Course-number reconstruction and browser behavior are covered by the other review workstreams.

## Confirmed issues and fixes

- **Major — reliability framing, `tools/build_awesome.py` and `tools/build_site.py` hero.** “One that works 99% of the time is a product” treats a task success percentage as sufficient for deployment. Sunday's [Solve definition](https://www.sunday.ai/blog/act-2-preview) explicitly requires performance, scope, and adaptation cost; Dyna's [deployment report](https://www.dyna.co/research/scaling-customer-deployments) separately measures speed and production quality. Replaced the categorical claim with scope, quality, speed, and evaluation requirements. No paper result changed.
- **Major — unlabeled composite metric, `data/explainer.json`, AgiBot recipe.** LWD's 0.76/0.95 comparison combines binary grocery success and graded long-horizon sub-step completion. Its [evaluation protocol, §V-A](https://arxiv.org/html/2605.00416v4#S5.SS1.SSS1) scores a long-horizon sub-step 1, 0.5, or 0; it is not uniformly a whole-episode success rate. Added that distinction to the site recipe; the catalog row already carried it.
- **Minor — universal claim, `data/explainer.json`, lessons.** “Any learned judge … gets exploited” exceeded what the cited examples establish. [Robometer](https://arxiv.org/html/2603.02115) itself reports a evaluated configuration with no false-positive successes, while other configurations fail. Changed to “can be exploited”; the caution and supporting examples remain.
- **Minor — provenance/process wording, both generators.** The repository exposes verification labels and a [weekly workflow](../weekly-sweep.md), but does not contain the historical full-read notes, installed schedule, or completed weekly run log that would independently establish all historical process claims. Changed the footer, site badges and narrative provenance notes to report recorded labels, linked sources and documented workflow. Removed an uncounted “most real-robot results use 20–60 trials” claim, retaining the small-sample caution. The site course text distinguishes published full runs from quick diagnostics on search states. This does not assert that the earlier work did not occur.
- **Minor — assistance disclosure, both generator footers.** Added Codex/OpenAI's involvement in the current pre-launch review and fixes alongside the existing Claude/Anthropic disclosure.

Edits were made only to data and generators. The integration workstream regenerates all public outputs and runs the full checks after concurrent changes settle.

## Deterministic changed-row sample

Parsed the before/after CSV snapshots of `f076f75`; 225 existing rows differed. Sorted changed rows by ID, then selected 20 with Python `random.Random(20261001).sample(rows, 20)`. This samples changes, not the full corpus. All 20 source pages were retrievable. The headline mechanism, setting, and reported numbers were checked in the source text/tables, with the SOLE-R1 figure rendered separately. No new numerical/mechanism error was confirmed in this sample: **0/20**. This is a small spot sample and is not the P2 error-rate estimate.

- [C348 IPE](https://arxiv.org/html/2607.27203): source reports 1.26× average fine-tuning improvement over naive Q pretraining, with ensemble rollout initialization.
- [C420 FIND](https://arxiv.org/html/2609.32069): eight real tasks, 55%→71.9% human-assessed success; 456-episode run and 30 scene-recovery interventions agree.
- [C440 U-GROW](https://arxiv.org/html/2609.22879): RoboTwin table gives 58.8 vs VLA-MBPO 54.3; branching uses disagreement between sampling budgets.
- [N405 Closed-Loop Collapse](https://arxiv.org/html/2609.23048): simulated student 0/72 vs teacher 40/72; recovery 18/36 vs 17/36 agree.
- [N073 InsertAnything](https://arxiv.org/html/2609.24511): 0.02 mm nominal clearance, 20/20 HITL-protocol benchmark, and 152/160 across eight unseen tasks agree; row retains protocol distinction.
- [N135 DynaForge](https://arxiv.org/html/2609.25631): simulation data-generation 41.30→78.37; real DP3 3/10, 6/10, 5/10 vs 0/10, 0/10, 1/10 agree.
- [N246 HarnessPAI](https://arxiv.org/html/2609.29166): 96.5 vs 34.9 on LIBERO-PRO and 73.7 after distilling harness data agree.
- [N153 Smooth Exponentials](https://evjang.com/2026/09/10/smooth-exponential.html): October 2027 statement is an author's prediction, properly labeled as such.
- [C450 PreferenceFlow](https://arxiv.org/html/2609.36872): real four-task table gives 90.5 vs frozen 69.0 and intervention BC 84.5.
- [N389 RecastVLA](https://arxiv.org/html/2609.32155): +10.68 points is the matched no-TTT comparison on RoboTwin Clean-to-Clean, not a real-robot average.
- [N398 CCPL](https://arxiv.org/html/2609.28427): retrospective exposure reductions ~37%/17% agree; row correctly does not claim an online trial benefit.
- [C226 TOPReward](https://arxiv.org/html/2602.19313): Table 2 gives 0.942/0.316/0.858 VOC on the 497-episode, 113-task progress subset.
- [N145 DYNA-1 pretraining](https://www.dyna.co/research/pre-training): source makes the roughly-100%/one-hour claim; row attributes it to the company and does not infer RL.
- [C251 SOLE-R1](https://arxiv.org/html/2603.28730): Figure 3 visually counted 24/40 tasks at ≥50% for SOLE-R1 and 7/40 for GPT-5; random-policy initialization and reward-only learning agree.
- [C304 BMD](https://arxiv.org/html/2605.11387): Avoid table reports DSRL 2/24 and DPPO 6.33/24 modes with near-unit success; BMD raises mode coverage. Mode counts must not be confused with success rates.
- [N374 FoLD](https://arxiv.org/html/2609.33551): overall simulated completion 0.606→0.772; hardware is open-loop replay, as retained in the row.
- [N254 G2MAF](https://arxiv.org/html/2609.31286): 20/24 settings, +9.2%/+8.9% relative returns, and ~6% latency agree.
- [C184 Prophet/ProphRL](https://arxiv.org/html/2511.20633): real UR30e table gives 52.1→82.1; 100 RL updates and the FA-GRPO/FlowScale mechanism agree.
- [N228 BrickCraft-Duo](https://arxiv.org/html/2609.28281): five-task table supports ≥95% step completion and ≥60% full-task success, with up to nine steps.
- [C236 Golden Ticket](https://arxiv.org/html/2603.15757): hardware paragraph specifies base 40/50, search 6×25 episodes, selected ticket 49/50 additional evaluations. This is distinct from the paper's optimistic held-out-selected simulation result, which the explainer already labels.

## Company surfaces

Read all 28 industry rows and all nine company recipes. Also read rows mentioning the named organizations (133 matches when using company/lab names rather than ambiguous ordinary words such as “figure” or “generalist”). Most additional matches are affiliations. The latter read is a wording/attribution screen, not an independent source audit of all 133 affiliations and numbers.

Fetched all 28 linked industry documents, including the essays and third-party articles; these were checked as accounts of what the named source says, not as independent confirmation of a company's results. Primary evidence additionally checked for Physical Intelligence RECAP/RL Token, AgiBot SOP/LWD, and the 1X technical report. No inaccessible source in this P0 subset was treated as verified.

- Sunday: primary post supports 778/785, one fixed checkpoint, no per-home adaptation, SE labeling, pretraining-scale ablation, and undisclosed reliability-loop update details. It also describes one-shot SFT separately; the narrative does not equate this with the undisclosed full reliability loop.
- Dyna: DYNA-1 blog and press release differ in their “850+” vs “800+” napkin wording; the corresponding entries preserve each source. Quality 75→93 and throughput 35→95 refer to the later deployment comparison, not the 99.4% completion metric. No unsupported RL objective is ascribed to Dyna.
- Generalist: GEN-1 post supports ~500k human-data hours, ~1 hour robot adaptation, five ingredients including RL, and 64→99 whole-system comparison. It does not isolate RL's contribution; the narrative says so.
- Figure: the logistics post supports 70→95 barcode orientation and the 50% decoder increase; the narrative does not infer RL.
- Siemens: primary paper supports 2,535 factory episodes, ~900 lab episodes, 70% round gate, 65% bag-content error share among failures, and the authors' statement that the final outcome fell short. No final overall success rate is invented.
- AgiBot: SOP source text supports 0.94/0.96/0.98 and 0.571→0.800; LWD supports the 16-robot system, 652.5-hour offline buffer, and mixed metric, now labeled in the recipe.
- Skild: official physical-self-play post supports recent-self opponents, stated scoring objective, Isaac Sim, and 140 simulated years. Narrative avoids inferring the soccer policy's initialization or a numerical real-robot success rate.
- 1X: [2025 report](https://www.1x.tech/1x-world-model.pdf), pp. 5, 7–8, supports 63.06→71.17 alignment and the stated 70%/15-point/90% claim. Figures 7–8 were rendered and inspected: real-world markers are selected checkpoint/task-score comparisons without real-trial counts or error bars; model shading is not a real-world confidence interval. The repo appropriately attributes the claim and notes the missing derivation.
- Google DeepMind: Gemini Robotics 2 source image alt text gives the 32–92% endpoints; <200 embodiment examples is in the post. These are selected task bars, not a uniform aggregate.
- NVIDIA, Amazon, Stanford: source-audited examples include IPE, TOPReward, BMD, Robometer, SIMPLER and the company recipe papers. Broader affiliation verification remains part of catalog auditing.

## Counts, legal, safety and source-copying spot checks

Before concurrent P2 edits: 912 papers/posts, 679 dated since September 2025, 122 starred, 198 dated September 16–30, 12 families, and **212** resources. Generated README counts agree; the review brief's historical 213 resources is stale after the prior removed link.

The root MIT license explicitly preserves the model licenses. Both vendored LICENSE files exactly match the pinned MuJoCo Menagerie commit `4d038b3feae26ec82b46a4d586379114012a8ac7`: [SO-101 Apache-2.0](https://github.com/google-deepmind/mujoco_menagerie/blob/4d038b3feae26ec82b46a4d586379114012a8ac7/robotstudio_so101/LICENSE) and [PiPER MIT](https://github.com/google-deepmind/mujoco_menagerie/blob/4d038b3feae26ec82b46a4d586379114012a8ac7/agilex_piper/LICENSE). No company logo or endorsement claim was found in the inspected banner/site material. This is a repository inspection, not a legal opinion.

A normalized 13-consecutive-word comparison of descriptions against the 57 fetched HTML source texts found no matches. This is a source-copying spot check, not a corpus-wide copyright clearance. A tracked-text pattern scan for common token formats, private-key headers and user-local absolute paths returned no matching files.

All nine existing chapter hardware sections contain workspace, reachable power/e-stop, low-speed/per-step-limit, hands-clear and supervision cautions. The README makes clear that course hardware steps have not been run on real arms.

`pass1-other-candidates.md` contains only a resolution note; it has no remaining leads. It is safe to delete after preserving this disposition in the integrated report. Source spot checks confirmed that the previously cited VIP, LWD, 1X and RECAP narrative corrections are present; this reviewer did not reconstruct all 87 historical decisions.

## Additional targeted primary-source checks

After the initial P0 sample, checked [C433's official event-host transcript](https://ai.engineer/talks/Sjfz1TqxzEs-robot-demos-are-easy-reliability-is-hard). Timestamps 11:38–12:18 describe the video progress model and dips; 13:15–13:52 describe targeted recovery-data collection and fine-tuning. The talk supports the 99.4% napkin-folding claim and approximately 80% initial recipe; the source's erroneous illustrative compounded-probability calculation is not reproduced in the catalog.

Two paper-lead figures were independently inspected for the main catalog reviewer: [HandITL Figure 6](https://arxiv.org/html/2605.15157v2) has base task bars 0.71/0.62/0.78 (mean 0.703) and Copilot bars 0.90/0.99/0.96 (mean 0.95). This is normalized partial-credit sub-goal completion, with one-hour additional datasets, not a binary task-success percentage. The apparatus is 56 DoF: two 7-DoF arms and two 21-DoF hands. [Decoupled Q-Chunking Figure 2](https://arxiv.org/pdf/2512.10926) reports DQC 82, n-step 68, SHARSA 44 and QC 25 across six hard offline goal-conditioned OGBench environments and ten seeds. Its shorter-output policy uses a distilled partial-chunk critic; executing a prefix of a full-chunk policy is the separate naive ablation.

## Scope of assurance

This P0 spot review found no new critical false company statement or sampled numerical error. Its concrete editorial fixes should land before promotion. It does not certify the full catalog, every narrative quantitative claim, live deployment state, or the experiments; the final go/no-go must include the P1 reruns, P2 findings and required checks. Rendered browser QA and complete re-generation are integration checks, not claimed as completed here.
