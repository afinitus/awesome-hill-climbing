# Advantage-weighted & filtered BC

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*How do I push a big imitation policy (a diffusion policy or a flow VLA) up on one task using its own rollouts, human corrections and success labels, without running policy-gradient RL through it?*

Keep the supervised imitation loss, but decide which actions (or which demos) the policy copies and how much, using a success label, a value function or a data-influence score.

- **Loop step:** Improve
- **Built on:** Reward- and advantage-weighted regression (RWR/AWR); Filtered BC and rejection sampling; Advantage- or return-conditioned policies; Classifier-free guidance as a policy-improvement knob; Where the advantage comes from; Curation: hill-climbing the dataset instead of the loss
- **Use it when:** Use this family when you have a decent BC or VLA policy (roughly 20-60% success) on a specific task, a way to label episodes success or failure, and ideally a teleop device for takeovers, but you do not want to backprop RL through a large flow or diffusion sampler. Start with curation (RoboDrop, QoQ or CUPID-style audits, or cheap smoothness filters) when your demos are mixed-quality.
- **What changed in 2025–26:** The big shift came with pi*0.6/RECAP in November 2025. Advantage-conditioned BC, where the policy is conditioned on a binarized advantage token and classifier-free guidance is optional, suits flow VLAs whose likelihood is awkward to weight, and it joined advantage-weighted BC as a main option; RECAP names CFGRL (May 2025) as the closest prior formulation. In the next ten months at least a dozen groups adopted or adapted the recipe: GigaBrain RAMP, chi0, DexPIE, DEED, ROVE, STEAM, Robo-ValueRL, PACE, DistAL, Facet-0, CLIFT and a prizewinning LeHome Challenge entry (1st in the simulation round, 2nd in the real-world final).

**63 entries**, newest first. ⭐ marks must-know work.


## October 2026

- **[MobiAgent](https://arxiv.org/abs/2610.03476)** (Liu et al., 2026-10-02) — Outer loop segments and verifies deployment rollouts, clusters them into atomic skills and fine-tunes flow-matching skill experts without annotations; autonomous data recycling lifts success 7.50% to 27.50% on RoboCasa and 32.5% to 57.5% on Astribot S1.

## September 2026

- **[PrefPI](https://arxiv.org/abs/2609.40165)** (Rho et al., 2026-09-30) — Preference-guided policy iteration: models preferred self-generated trajectories as a conditional distribution and amplifies them with classifier-free guidance, pushing diffusion policies and pi0.5 beyond their initial support; real object transport height 10.7 cm to 19.8 cm with 150 preference-labeled trajectories.
- **[RE-0](https://arxiv.org/abs/2609.32416)** (Jilin University; Dalian University of Technology, 2026-09-26) — Teacher gives local corrections on the student's own failure histories; only environment-verified corrections become on-policy distillation supervision (RE-OPD), with a per-round gain lower bound.
- **[Generate, Track, Improve](https://arxiv.org/abs/2609.31577)** (Caltech, 2026-09-25) — Off-policy RL (structured search plus advantage-weighted regression) fine-tunes a depth-conditioned flow-matching motion generator feeding a tracking policy; in simulation vs the pre-fine-tuned generator, up to +25 points terrain traversal and +80 skill selection; G1 hardware shown qualitatively.
- **[PACL](https://arxiv.org/abs/2609.29000)** (NTU; Changan; Cloudbutterfly, 2026-09-24) — IQL chunk critic with future latent supervision conditions a diffusion actor on high-value chunks, then selects best-of-N; real Franka StackCup improves from 18/25 to 25/25 trials.
- ⭐ **[Dissecting advantage-guided post-training](https://arxiv.org/abs/2609.28161)** (Shanghai Jiao Tong University; Xiaomi Robotics, 2026-09-23) — Ablates separately how critic-derived advantages are built, calibrated and used; n-step TD, per-value-bin normalization and soft exponential weights on the pi0.5 loss worked best. Real bimanual, 4-task mean success: SFT 0.11, source-weighted DAgger 0.45, continuous weighting 0.74.
- **[Imperfection for Precision](https://arxiv.org/abs/2609.26672)** (Samsung Robotics eXperience; NUS; UIUC, 2026-09-22) — ε4P co-trains a flow-matching VLA on imperfect data by noise level (low-precision target-task data at high noise, high-precision mismatched-task data at low noise); real robot: up to +31.7 points; replacing equal high-quality data costs 4.2 points on average.
- **[CARF](https://arxiv.org/abs/2609.21982)** (UC Berkeley, 2026-09-18) — Progress-based scorer trained on successes labels imperfect-data segments; flow-matching policy is attracted to progressive segments and repelled from failure-critical ones, beating baselines in sim and real.
- **[DistAL](https://arxiv.org/abs/2609.18392)** (University of Oxford, 2026-09-16) — Dense reward inside RECAP: negative kNN distance to the base VLA's training embeddings. Hardware mean 44.0 base / 60.0 binary reward / 69.5% DistAL.
- **[ReWeight](https://arxiv.org/abs/2609.13851)** (HKU; TeleAI, China Telecom; Northwestern Polytechnical University, 2026-09-12) — Optimal-transport retrieval of egocentric human demos plus sample weighting for pi0.5 post-training; RoboTwin simulation success 39% robot-only to 57%, real-world 40.0% to 68.8%. Weighted BC, without RL.
- ⭐ **[RoboDrop](https://arxiv.org/abs/2609.10021)** (Tsinghua University; Striding AI, 2026-09-09) — In one warm-up epoch, compares each sample's compressed gradient with one from the most similar clean validation frames, and drops conflicting episodes before re-post-training pi0.5. Real, 4 tasks, 20 rollouts each: 35.0% → 67.5% with automatic filtering.
- **[Facet-0](https://arxiv.org/abs/2609.01596)** (NTU PINE Lab, 2026-09-01) — A wrench-predicting VLA uses contact-selective credit tags and bounded TD3+BC adaptation. Five real precision-assembly tasks: 16% alignment-only, 38% value-guided refinement, 82% full system; authors’ matched pi0.5+RECAP-style implementation scores 15%. · [code](https://github.com/PINE-Lab-NTU/FACET)

## August 2026

- **[PAVE](https://arxiv.org/abs/2608.30378)** (Zhao et al., 2026-08-31) — Combines multi-horizon future-latent prediction with value-derived quality conditions for a flow actor. Matched-data comparisons improve success by 5.0 points on LIBERO-Plus and 4.5 points on RoboTwin Random (simulation).
- **[EXIMO](https://arxiv.org/abs/2608.19891)** (Google DeepMind, 2026-08-20) — VLM-guided subgoals collect exploration trajectories, successful episodes fine-tune the VLA, and residual off-policy RL further optimizes it; evaluated on 22 simulated ALOHA manipulation tasks.
- **[Self-demonstrated fine-tuning](https://arxiv.org/abs/2608.19490)** (UIUC, 2026-08-19) — Replays frozen base-VLA rollouts with new expert demonstrations to prevent forgetting. Real ALOHA gear insertion: 30% with expert-only fine-tuning to 90% with self-demonstration replay.
- **[Max-Q selective imitation](https://arxiv.org/abs/2608.15088)** (Wang et al., 2026-08-15) — A Monte Carlo chunk critic selects imitation targets from buffer or actor actions by Q. Real USB insertion: ACT QChunk-MCBC reaches 99% in 30 minutes; HIL-SERL peaks at 100% after 5 hours.
- **[PACE](https://arxiv.org/abs/2608.15026)** (Dalian UT; Jilin University, 2026-08-15) — A phase-aware distributional critic gives cleaner RECAP labels. LIBERO-Long 73.8 (ReCAP) → 83.3 (PACE).

## July 2026

- ⭐ **[CLIFT](https://arxiv.org/abs/2607.29172)** (UC Berkeley, Google DeepMind, NVIDIA Research, 2026-07-31) — Tunes closed-weight Gemini Robotics On-Device via an SFT API only, tagging the top 30% of chunks by dense-reward advantage over similar-start chunks as 'advantage positive'. Real Unitree G1, 2 cycles of 100 rollouts: Plate Handover 53% → 96%.
- ⭐ **[RedFlow](https://arxiv.org/abs/2607.27782)** (HKUST, 2026-07-30) — Clusters action chunks by estimated progress and joint state; within a cluster, positive-scored chunks (progress gain plus episode outcome) form correction targets for negative chunks, which are also suppressed. Real pi0 clothes folding: 36 → 67/100 (AWR 48). · [code](https://github.com/HEISENYAN/RedFlow)
- ⭐ **[DEED](https://arxiv.org/abs/2607.20345)** (HIVE Robots; Technical University of Denmark, 2026-07-22) — Data-efficient GR00T N1.6 post-training plus a RECAP-adapted refinement (text advantage prefix, VLM value function) for chip restocking on a Unitree G1, ~108 min of data. 50 real episodes each: SFT 32%, iteration 1 42% (CIs overlap), iteration 2 22%.
- **[Robo-ValueRL](https://arxiv.org/abs/2607.09866)** (Renmin University; Beijing Innovation Center of Humanoid Robotics, 2026-07-10) — A history-conditioned VLM value guides quality-conditioned pretraining and filters rollouts for residual adaptation. Real chip insertion improves from 46% to 86% over three rounds of 500 rollouts. · [code](https://github.com/Open-X-Humanoid/Robo-ValueRL)

## June 2026

- ⭐ **[STEAM](https://arxiv.org/abs/2606.29834)** (Institute of Automation, CAS; Tsinghua University; UCAS, 2026-06-29) — An ensemble of small VLMs predicts time offsets between expert frame pairs (forward and reversed); the ensemble minimum gives per-frame advantages for CFGRL training of pi0. Real towel folding: 92.3% vs one-iteration RECAP-style baseline 55.6%, BC 33.3% (trials unstated).
- **[MoRE (Behavior Uncloning)](https://arxiv.org/abs/2606.29201)** (Texas A&M University and others, 2026-06-28) — A temporary mode classifier guides policy-weight edits with a retain loss. Simulated Push-T desired-mode success averages 50.0% before versus 98.3% after MoRE; the original policy already completes the task in either mode. · [code](https://github.com/phai-lab/behavior-uncloning)
- **[ReGuide](https://arxiv.org/abs/2606.28939)** (Texas A&M, 2026-06-27) — Phase-gated latent-dynamics guidance at test time, with the successes distilled back through filtered BC. Tool Hang 0.0312 → 0.2404 after two iterations (sim). · [code](https://github.com/tzuhsiangl/reguide)
- **[LeHome Learning to Fold](https://arxiv.org/abs/2606.27163)** (Independent researcher, 2026-06-25) — AWR sampling, RECAP conditioning and CFG improve a simulated bimanual SO-ARM101 policy that also predicts value: 79.63%, first of 62 teams. The second-place real final instead uses plain BC. · [code](https://github.com/IliaLarchenko/lehome_solution)
- **[HABC](https://arxiv.org/abs/2606.17043)** (ACE Robotics; Tsinghua SIGS; CUHK, 2026-06-15) — Viability and efficiency critic heads turn episode success bits into per-transition flow-matching weights. On three real bimanual tasks, initial online fine-tuning raises SFT success 36/44/12% to 60/78/22%; compared online methods share the same interaction budget. · entry corrected after source review
- ⭐ **[ROVE](https://arxiv.org/abs/2606.17011)** (XPENG Robotics; Fudan; CUHK; SJTU, 2026-06-15) — Splits each human intervention into autonomous rollout, human adaptation and human recovery, placing the failure penalty after adaptation; an optimistic expectile critic drives RECAP-style conditioning. Iteration 1, same robot data, Erase/Bread: 60.0%/73.3% vs RECAP 50.0%/66.7%.
- **[DexPIE](https://arxiv.org/abs/2606.09615)** (Hunan University, 2026-06-08) — One RECAP round with a continuous sigmoid optimality indicator on a dexterous diffusion policy. Average 50.0 BC / 75.3 HG-DAgger / 80.0 RECAP / 87.3% (50 trials per task). · [code](https://github.com/siiuuuuuu/tele_UR)
- **[ForesightFlow](https://arxiv.org/abs/2606.04968)** (BIT; Tsinghua; HKUST (Guangzhou), 2026-06-03) — The VLA generates a success-potential channel used both as the AWR baseline and for best-of-K selection. Real average 35.4% vs IDQL 32.6% vs filtered BC 24.6%. · entry corrected after source review

## May 2026

- **[ProgVLA](https://arxiv.org/abs/2605.28231)** (Kim et al., 2026-05-27) — Compact 0.1B VLA with two-stage Perceiver resampling and jointly trained progress heads (offline-RL critics over normalized remaining-horizon targets) that weight flow-matching imitation; beats imported larger-baseline scores on LIBERO and Meta-World; progress weighting adds 2.3 points on LIBERO.
- **[AFIL (Failing Forward)](https://arxiv.org/abs/2605.08434)** (United Imaging Intelligence; UIUC, 2026-05-08) — A failure action head sharing the VLA backbone gives adaptive CFG-style negative guidance during sampling. LIBERO average pi0.5 96.9% → 98.4% (sim only).

## April 2026

- **[RINSE](https://arxiv.org/abs/2604.23000)** (UCLA, 2026-04-24) — Smoothness scores retain top-ranked demonstrations. On simulated RoboMimic Transport mh, spectral arc length selects 50 of 300 demos for 55% success versus 39% with all 300; trajectory-envelope distance scores 46%.

## March 2026

- **[PTR](https://arxiv.org/abs/2603.16542)** (Zhang et al., 2026-03-17) — PTR: reward-free weighted regression for VLA post-training; a transition scorer's identification posterior-to-uniform ratio reweights heterogeneous mixed-quality samples, compatible with diffusion and flow-matching action heads.
- **[QoQ](https://arxiv.org/abs/2603.09056)** (KAIST; UC Berkeley; NYU, 2026-03-10) — Hessian-free max-cosine influence against a small validation set. Robomimic Can-paired 99.2% vs 64.8% with all data; real cabinet opening 25.0% → 43.3%.
- **[How to Peel](https://arxiv.org/abs/2603.03280)** (UC Berkeley, 2026-03-03) — Human quality scores train a per-step reward, and a residual is fitted with exp(beta*r)-weighted BC. Apple 60% → 100%, while IQL-weighted BC scored 0%. · [code](https://github.com/ToruOwO/how-to-peel)

## February 2026

- ⭐ **[IG-RFT](https://arxiv.org/abs/2602.20715)** (Zhejiang University; Torch Kernel, 2026-02-24) — Trains pi0.5 with 60 demos per task, offline advantage-weighted flow matching from a learned critic, then human-in-the-loop correction rounds with the same weighting. Four real long-horizon tasks, 20 trials each: 18.8% → 40.0% → 85.0%.
- **[GigaBrain-0.5M* (RAMP)](https://arxiv.org/abs/2602.12099)** (GigaAI, 2026-02-12) — RECAP-style conditioning plus world-model future latents and value. Box Packing about 60 pretrain / 75 AWR / 60 RECAP / 95% RAMP (bar readings).
- **[RKO (Preference-aligned Diffusion Policy)](https://arxiv.org/abs/2602.09583)** (KTH; INCAR Robotics, 2026-02-10) — Fine-tunes diffusion policies on preferred/dispreferred cloth-folding demonstrations, combining KTO sample labels with RPO similarity weighting. RKO leads 4/9 real garment/preference settings; Trousers Pref 1 normalized folding score is 0.910 versus vanilla DDPM’s 0.701. · entry corrected after source review
- **[chi0 (kai0)](https://arxiv.org/abs/2602.09021)** (Kinetix AI / OpenDriveLab, 2026-02-09) — Open garment-folding pipeline: model soup plus stage-aware advantage (RECAP-style, eps = 0.3) plus train-deploy alignment on pi0.5. Task A success about 27% → 97% (bar readings, 30 trials). · [code](https://github.com/OpenDriveLab/kai0)

## November 2025

- ⭐ **[pi*0.6 / RECAP](https://arxiv.org/abs/2511.14759)** (Physical Intelligence, 2025-11-18) — Fits a distributional value, binarizes n-step advantages into text conditioning, and supervises the whole flow VLA; CFG is optional. Same-data real laundry: about 2x AWR throughput; success ~96% versus AWR ~91%, PPO ~75% (chart reading).
- **[ExpReS-VLA](https://arxiv.org/abs/2511.06202)** (CMU, 2025-11-09) — Frozen-feature success/failure buffers, retrieval-augmented LoRA BC and a contrastive loss on failures. Real ID: naive FT 84.7% → 98.0%; OOD: 32% → 98%. · entry corrected after source review

## October 2025

- ⭐ **[Hi-ORS](https://arxiv.org/abs/2510.26406)** (Tsinghua SIGS; Tencent Robotics X, 2025-10-30) — Keeps episodes above a rising return threshold, including human-corrected successes, and trains pi0 with reward-weighted flow matching. Three real tasks, two robots: 23.3 percentage points over offline BC (chart reading); about 1.5 hours of real training. · [code](https://github.com/hiors-project/hiors)
- **[RM-RL](https://arxiv.org/abs/2510.15189)** (NTU Singapore, 2025-10-16) — Best-of-scene self-imitation plus REINFORCE for a discrete pick-pose correction. Shelf placement 10/10 for pretrained RM-RL vs 5/10 for plain REINFORCE. This is a cross-method comparison, and the pretrained variant uses about 200 extra prior samples. · [code](https://github.com/NTUMARS/RMRL)

## September 2025

- **[DexFlyWheel](https://arxiv.org/abs/2509.23829)** (Harbin Institute of Technology; Peking University; PsiBot; PKU-PsiBot Lab; Harbin Institute of Technology Suzhou Research Institute, 2025-09-28) — Freezes a diffusion imitation policy, trains a SAC residual scaled by 0.1 with shaped rewards, then filters successful rollouts, augments them and retrains imitation policies in a simulation data-generation loop. · entry corrected after source review
- **[SOE](https://arxiv.org/abs/2509.19292)** (SJTU; Noematrix, 2025-09-23) — Information-bottleneck latent exploration for iterated success-filtered BC. Real Toaster Load 0.56 → 0.75 (0.84 with human steering) in one round. · [code](https://github.com/EricJin2002/SOE)
- **[ARFM](https://arxiv.org/abs/2509.04063)** (Westlake; UCLA; XJTU, 2025-09-04) — Critic-free softmax-advantage-weighted flow-matching loss with adaptive temperature on pi0. LIBERO simulation average: 88.1% base to 92.1%; flow RWR 90.8%, flow ReinboT 91.2%.

## Before September 2025

- ⭐ **[CUPID](https://arxiv.org/abs/2506.19121)** (Stanford; Toyota Research Institute; Cambridge; NVIDIA, 2025-06-23) — Scores each demo's effect on closed-loop success by chaining influence functions with a REINFORCE estimate from labeled eval rollouts, then drops negative-influence demos and retrains. Real TuckBox mass shift, 66% filtered: base 0%, CUPID 84%, oracle 88% (bar readings). · [code](https://github.com/agiachris/cupid)
- ⭐ **[CFGRL](https://arxiv.org/abs/2505.23458)** (UC Berkeley, 2025-05-29) — Trains one flow policy by BC with an optimality token; classifier-free guidance samples reference policy times a non-decreasing advantage function. Sim ExORL walker-stand, same IQL critic: 782±8 vs AWR 603±8. OGBench: mixed vs AWR, beats goal-conditioned BC on most tasks. · [code](https://github.com/kvfrans/cfgrl)
- **[SCIZOR](https://arxiv.org/abs/2505.22626)** (UT Austin; NVIDIA, 2025-05-28) — A task-progress predictor removes suboptimal state-action pairs; deduplication removes redundant pairs. On Sirius-Fleet real-robot tasks, the paper text reports a 32.9-percentage-point success gain over training on the full dataset. · [code](https://github.com/UT-Austin-RPL/SCIZOR)
- **[DataMIL](https://arxiv.org/abs/2505.09603)** (UT Austin; MIT; Stanford, 2025-05-14) — Linear datamodels with a target-demo BC-loss proxy select prior-data clusters. LIBERO-10 37.76 vs 27.72 for all-data co-training. · [code](https://github.com/UT-Austin-RobIn/datamil)
- **[ReinboT](https://arxiv.org/abs/2505.07395)** (Zhang et al., 2025-05-12) — End-to-end VLA predicts dense returns to model data quality and condition actions on high return; authors report gains on CALVIN mixed-quality data and few-shot and real-world OOD generalization. · entry corrected after source review
- **[SIME](https://arxiv.org/abs/2505.01396)** (SJTU; Noematrix, 2025-05-02) — Success-filtered self-improvement with noise injected into the observation embedding to diversify modes. Square-20 (state): 0.534 → 0.620 vs 0.556 for plain self-improvement. · [code](https://github.com/EricJin2002/SIME)
- **[Demo-SCORE](https://arxiv.org/abs/2503.03707)** (Stanford, 2025-03-05) — A success classifier trained on policy rollouts flags demonstration strategies the policy cannot execute. Robomimic square peg (simulation, averaged over data mixtures): 79.9% base to 94.4%. · [code](https://github.com/alessing/demo-score)
- **[DemInf](https://arxiv.org/abs/2502.08623)** (Google DeepMind; Stanford, 2025-02-12) — Label-free demo scoring by contribution to I(S;A), using k-NN estimators in VAE latents. The abstract reports a 5-10% improvement on RoboMimic. · [code](https://github.com/jhejna/demonstration-information)
- **[iRe-VLA](https://arxiv.org/abs/2501.16664)** (Tsinghua IIIS; UC Berkeley, 2025-01-28) — Alternates online RL on an action head over a frozen VLM with full-model SFT on expert demos and RL successes. MetaWorld unseen-10 success: 0.51 SFT / 0.39 PPO-Replay / 0.80 iRe-VLA.
- **[RLDG](https://arxiv.org/abs/2412.09858)** (UC Berkeley, 2024-12-13) — Real-world HIL-SERL specialists generate successful rollouts that become SFT data for OpenVLA/Octo. On FMB Insertion, OpenVLA succeeds in 25/30 trials versus 15/30 when fine-tuned on human demonstrations. · [code](https://github.com/generalist-distillation/RLDG) · entry corrected after source review
- **[GRAPE](https://arxiv.org/abs/2411.19309)** (UNC-Chapel Hill; University of Washington; University of Chicago, 2024-11-28) — Trajectory-level preference optimization aligns OpenVLA using its successful and failed rollouts, ranked by task success, self-evaluation and VLM-generated stage/keypoint costs, with a KL anchor to the SFT reference policy. · [code](https://github.com/aiming-lab/grape)
- **[S2I](https://arxiv.org/abs/2409.19917)** (Chen et al., 2024-09-30) — S2I segments mixed-quality demos, selects good segments via contrastive learning and optimizes poor ones' trajectories; 3 expert demos as reference improve downstream policies on six tasks (ICRA 2025). · [code](https://github.com/Junxix/S2I)
- **[SOAR](https://arxiv.org/abs/2407.20635)** (UC Berkeley, 2024-07-30) — VLM-proposed tasks and VLM success filtering for goal-conditioned BC on autonomous data. Average real-world success 28% to 57% across 10 scenes, with no human annotations. · [code](https://github.com/rail-berkeley/soar)
- **[Learning from Imperfect Demonstrations with Self-Supervision for Robotic Manipulation](https://arxiv.org/abs/2401.08957)** (Wu et al., 2024-01-17) — SSDF: self-supervised quality scoring of failed-trajectory segments using expert plus imperfect data; high-scoring segments augment offline BC training, improving success on ManiSkill2 and a real Franka, reward-free.
- **[RoboCat](https://arxiv.org/abs/2306.11706)** (Google DeepMind, 2023-06-20) — Self-improvement loop for a 1.18B visual-goal-conditioned transformer: fine-tune on 100-1000 demonstrations, collect new robot trajectories, hindsight-relabel goals including failed episodes, and retrain the generalist.
- **[CRR](https://arxiv.org/abs/2006.15134)** (DeepMind, 2020-06-26) — Offline critic-filtered BC (binary or exp weights). Humanoid Run: CRR exp 586±6 vs BC 382±2 and D4PG 1±1. · [code](https://github.com/google-deepmind/acme/tree/master/acme/agents/jax/crr)
- **[AWR](https://arxiv.org/abs/1910.00177)** (UC Berkeley, 2019-10-01) — The base recipe: regress V, then do exp(A/beta)-weighted BC on buffer actions. Hopper-v2 3405±121 vs SAC 2769±552 and PPO 1391±304. · [code](https://github.com/xbpeng/awr)
