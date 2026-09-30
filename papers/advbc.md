# Advantage-weighted & filtered BC

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*How do I push a big imitation policy (a diffusion policy or a flow VLA) up on one task using its own rollouts, human corrections and success labels, without running policy-gradient RL through it?*

Keep the supervised imitation loss, but decide which actions (or which demos) the policy copies and how much, using a success label, a value function or a data-influence score.

- **Loop step:** Improve
- **Built on:** Reward- and advantage-weighted regression (RWR/AWR); Filtered BC and rejection sampling; Advantage- or return-conditioned policies; Classifier-free guidance as a policy-improvement knob; Where the advantage comes from; Curation: hill-climbing the dataset instead of the loss
- **Use it when:** Use this family when you have a decent BC or VLA policy (roughly 20-60% success) on a specific task, a way to label episodes success or failure, and ideally a teleop device for takeovers, but you do not want to backprop RL through a large flow or diffusion sampler. Start with curation (RoboDrop, QoQ or CUPID-style audits, or cheap smoothness filters) when your demos are mixed-quality.
- **What changed in 2025–26:** The big shift came with pi*0.6/RECAP in November 2025. Advantage-weighted BC gave way to advantage-conditioned BC with classifier-free guidance, which fits flow VLAs whose likelihood is awkward to weight, and CFGRL (May 2025) supplied the theory. In the next ten months at least a dozen groups copied or modified the recipe: GigaBrain RAMP, chi0, DexPIE, DEED, ROVE, STEAM, Robo-ValueRL, PACE, DistAL, Facet-0, CLIFT and the LeHome winner.

**58 entries**, newest first. ⭐ marks must-know work.


## September 2026

- **[RE-0](https://arxiv.org/abs/2609.32416)** (2026-09-26) — Teacher gives local corrections on the student's own failure histories; only environment-verified corrections become on-policy distillation supervision (RE-OPD), with a per-round gain lower bound.
- **[PACL](https://arxiv.org/abs/2609.29000)** (NTU; Changan, 2026-09-24) — An IQL chunk critic with a latent-prediction auxiliary loss, a 'good'-conditioned diffusion policy and best-of-N selection. Real StackCup 72% → 100%.
- ⭐ **[Dissecting advantage-guided post-training](https://arxiv.org/abs/2609.28161)** (Shanghai Jiao Tong University; Xiaomi Robotics, 2026-09-23) — Ablates separately how RECAP-style advantages are built, calibrated and used; n-step TD, per-value-bin normalization and soft exponential weights on the pi0.5 loss worked best. Real bimanual, 20 trials each: SFT 0.11, DAgger 0.45, continuous weighting 0.74.
- **[CARF](https://arxiv.org/abs/2609.21982)** (UC Berkeley, 2026-09-18) — Progress-based scorer trained on successes labels imperfect-data segments; flow-matching policy is attracted to progressive segments and repelled from failure-critical ones, beating baselines in sim and real.
- **[DistAL](https://arxiv.org/abs/2609.18392)** (University of Oxford, 2026-09-16) — Dense reward inside RECAP: negative kNN distance to the base VLA's training embeddings. Hardware mean 44.0 base / 60.0 binary reward / 69.5% DistAL.
- **[ReWeight](https://arxiv.org/abs/2609.13851)** (Wang et al., 2026-09-12) — Optimal-transport retrieval of egocentric human demos plus per-sample weighting for VLA post-training; sim success 39% to 57%, real 68.8%. Weighted BC, no RL. · [code](https://reweight-vla.github.io/)
- ⭐ **[RoboDrop](https://arxiv.org/abs/2609.10021)** (Tsinghua University; Striding AI, 2026-09-09) — In one warm-up epoch, compares each sample's compressed gradient with one from the most similar clean validation frames, and drops conflicting episodes before re-post-training pi0.5. Real, 4 tasks, 20 rollouts each: 35.0% → 67.5% with automatic filtering.
- **[Facet-0](https://arxiv.org/abs/2609.01596)** (NTU PINE Lab, 2026-09-01) — A wrench-predicting VLA with advantage-conditioned post-training plus a TD3+BC actor. Mean 16% → 38% (+RL) → 82% (full) vs 15% for pi0.5+RECAP. · [code](https://github.com/PINE-Lab-NTU/FACET)

## August 2026

- **[PAVE](https://arxiv.org/abs/2608.30378)** (Zhao et al., 2026-08-31) — Direct VLA with multi-horizon (25/50/75/100%) JEPA-style transition alignment plus a distributional value critic on deployment trajectories whose advantages become text conditions for a flow-matching actor; best on three sim benchmarks.
- **[EXIMO](https://arxiv.org/abs/2608.19891)** (Google DeepMind, 2026-08-20) — A VLM coach proposes subgoals, success-filtered episodes SFT the VLA, then residual RL follows. About 0.13 → 0.38 averaged over 22 sim tasks (plot readings).
- **[Self-demonstrated fine-tuning](https://arxiv.org/abs/2608.19490)** (UIUC, 2026-08-19) — Replays frozen base-VLA rollouts 1:1 with new expert demos to prevent forgetting. Gear insertion success 30% → 90% on trained colors (10 trials).
- **[Max-Q selective imitation](https://arxiv.org/abs/2608.15088)** (unknown, 2026-08-15) — A Monte Carlo chunk critic, and the actor imitates whichever of the buffer action or its own action has higher Q. Real USB insertion reaches 99% in about 30 min vs 5.0 h for HIL-SERL.
- **[PACE](https://arxiv.org/abs/2608.15026)** (Dalian UT; Jilin University, 2026-08-15) — A phase-aware distributional critic gives cleaner RECAP labels. LIBERO-Long 73.8 (ReCAP) → 83.3 (PACE).

## July 2026

- ⭐ **[CLIFT](https://arxiv.org/abs/2607.29172)** (UC Berkeley, Google DeepMind, NVIDIA Research, 2026-07-31) — Tunes closed-weight Gemini Robotics On-Device via an SFT API only, tagging the top 30% of chunks by dense-reward advantage over similar-start chunks as 'advantage positive'. Real Unitree G1, 2 cycles of 100 rollouts: Plate Handover 53% → 96%.
- ⭐ **[RedFlow](https://arxiv.org/abs/2607.27782)** (HKUST, 2026-07-30) — Clusters action chunks by estimated progress and joint state; within a cluster, actions from good episodes become correction targets for bad ones instead of just down-weighting failures. Real clothes folding: 36 → 67 of 100 (AWR 48). · [code](https://github.com/HEISENYAN/RedFlow)
- ⭐ **[DEED (cautionary replication)](https://arxiv.org/abs/2607.20345)** (HIVE Robots; Technical University of Denmark, 2026-07-22) — Independent low-budget RECAP replication on GR00T N1.6 for chip restocking with a Unitree G1, using about 108 minutes of data. 50 real episodes each: data-efficient SFT 32%, RECAP iteration 1 42% (not significant), iteration 2 22%.
- **[Robo-ValueRL](https://arxiv.org/abs/2607.09866)** (Renmin University; Beijing Humanoid Innovation Center, 2026-07-10) — A better VLM progress value labels chunks low/medium/high for quality conditioning. Chip insertion 46% → 86% over 3 rounds of 500 rollouts, while DAgger ends at 20%. · [code](https://github.com/Open-X-Humanoid/Robo-ValueRL)

## June 2026

- ⭐ **[STEAM](https://arxiv.org/abs/2606.29834)** (Tsinghua University; Institute of Automation, CAS / UCAS, 2026-06-29) — Trains a small VLM on expert frame pairs plus reversed clips to predict time offset, uses predicted progress as per-frame advantage, and trains pi0 with CFGRL. Towel folding: 92.3% vs RECAP 55.6%, BC 33.3% (trial counts not stated).
- **[ReGuide](https://arxiv.org/abs/2606.28939)** (Texas A&M, 2026-06-27) — Phase-gated latent-dynamics guidance at test time, with the successes distilled back through filtered BC. Tool Hang 0.0312 → 0.2404 after two iterations (sim). · [code](https://github.com/tzuhsiangl/reguide)
- **[LeHome Learning to Fold](https://arxiv.org/abs/2606.27163)** (Independent researcher, 2026-06-25) — Open, iterated advantage loop on SO-ARM101 (AWR-style sampling, RECAP token and CFG; the policy is its own critic). 1st of 62 teams at 79.63% in the sim round. · [code](https://github.com/IliaLarchenko/lehome_solution)
- **[HABC](https://arxiv.org/abs/2606.17043)** (ACE Robotics; Tsinghua SIGS; CUHK, 2026-06-15) — Viability and efficiency critic heads turn episode success bits into per-transition flow-matching weights. At equal budget, SFT 36/44/12% → 60/78/22%. · [code](https://github.com/ACERobotics-VLA/HABC)
- ⭐ **[ROVE](https://arxiv.org/abs/2606.17011)** (XPENG Robotics; Fudan; CUHK; SJTU, 2026-06-15) — Splits each human intervention into autonomous rollout, human adaptation and human recovery, placing the failure penalty after adaptation; an optimistic expectile critic drives RECAP-style conditioning. Iteration 1, identical data, Erase/Bread: 60.0%/73.3% vs RECAP 50.0%/66.7%.
- **[DexPIE](https://arxiv.org/abs/2606.09615)** (Hunan University, 2026-06-08) — One RECAP round with a continuous sigmoid optimality indicator on a dexterous diffusion policy. Average 50.0 BC / 75.3 HG-DAgger / 80.0 RECAP / 87.3% (50 trials per task). · [code](https://github.com/siiuuuuuu/tele_UR)
- **[ForesightFlow](https://arxiv.org/abs/2606.04968)** (BIT; Tsinghua; HKUST, 2026-06-03) — The VLA generates a success-potential channel used both as the AWR baseline and for best-of-K selection. Real average 35.4% vs IDQL 32.6% vs filtered BC 24.6%.

## May 2026

- **[AFIL (Failing Forward)](https://arxiv.org/abs/2605.08434)** (United Imaging Intelligence; UIUC, 2026-05-08) — A separate failure action generator is used as a CFG-style negative prompt. LIBERO average pi0.5 96.9% → 98.4% (sim only).

## April 2026

- **[RINSE](https://arxiv.org/abs/2604.23000)** (UCLA, 2026-04-24) — Smoothness scores (spectral arc length, trajectory envelope) keep the top-K demos. Transport mh: 55% using 50 of 300 demos vs 39% with all of them.

## March 2026

- **[Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting](https://arxiv.org/abs/2603.16542)** (Zhang et al., 2026-03-17) — PTR: reward-free weighted regression for VLA post-training; a transition scorer's identification posterior-to-uniform ratio reweights heterogeneous mixed-quality samples, compatible with diffusion and flow-matching action heads.
- **[QoQ](https://arxiv.org/abs/2603.09056)** (KAIST; UC Berkeley; NYU, 2026-03-10) — Hessian-free max-cosine influence against a small validation set. Robomimic Can-paired 99.2% vs 64.8% with all data; real cabinet opening 25.0% → 43.3%.
- **[How to Peel](https://arxiv.org/abs/2603.03280)** (UC Berkeley, 2026-03-03) — Human quality scores train a per-step reward, and a residual is fitted with exp(beta*r)-weighted BC. Apple 60% → 100%, while IQL-weighted BC scored 0%. · [code](https://github.com/ToruOwO/how-to-peel)

## February 2026

- ⭐ **[IG-RFT](https://arxiv.org/abs/2602.20715)** (Zhejiang University; Torch Kernel, 2026-02-24) — Trains pi0.5 in three stages: SFT on 60 demos, an offline advantage-weighted flow-matching loss from a learned critic, then human-in-the-loop correction rounds with the same weighting. 4 real long-horizon tasks, 20 trials each: 18.8% → 40.0% → 85.0%.
- **[GigaBrain-0.5M* (RAMP)](https://arxiv.org/abs/2602.12099)** (GigaAI, 2026-02-12) — RECAP-style conditioning plus world-model future latents and value. Box Packing about 60 pretrain / 75 AWR / 60 RECAP / 95% RAMP (bar readings). · [code](https://github.com/open-gigaai/giga-brain-0)
- **[chi0 (kai0)](https://arxiv.org/abs/2602.09021)** (Kinetix AI / OpenDriveLab, 2026-02-09) — Open garment-folding pipeline: model soup plus stage-aware advantage (RECAP-style, eps = 0.3) plus train-deploy alignment on pi0.5. Task A success about 27% → 97% (bar readings, 30 trials). · [code](https://github.com/OpenDriveLab/kai0)

## November 2025

- ⭐ **[pi*0.6 / RECAP](https://arxiv.org/abs/2511.14759)** (Physical Intelligence, 2025-11-18) — Fits a distributional value on labeled episodes, thresholds n-step chunk advantages into an 'Advantage: positive/negative' token, and retrains the whole flow VLA supervised, using CFG at test time. T-shirt folding: about 96% success vs AWR 91%, PPO 75%.
- **[Diffusion Policies with Value-Conditional Optimization for Offline Reinforcement Learning](https://arxiv.org/abs/2511.08922)** (2025-11-12) — DIVO (IROS 2025): binary advantage-weighted conditioning guides diffusion policy training toward high-value dataset actions, then filters high-return samples; improves D4RL locomotion and sparse-reward AntMaze.
- **[ExpReS-VLA](https://arxiv.org/abs/2511.06202)** (CMU, 2025-11-09) — Frozen-feature success/failure buffers, retrieval-augmented LoRA BC and a contrastive loss on failures. Real ID: naive FT 84.7% → 98.0%; OOD: 32% → 98%. · numbers corrected after check

## October 2025

- ⭐ **[Hi-ORS](https://arxiv.org/abs/2510.26406)** (Tsinghua SIGS; Tencent Robotics X, 2025-10-30) — Drops the critic: keeps episodes whose outcome clears a rising return threshold, human-corrected successes included, and trains pi0 on them with its flow-matching loss. Beats offline BC by 23.3 points over three real tasks after about 1.5 h. · [code](https://github.com/hiors-project/hiors)
- **[RM-RL](https://arxiv.org/abs/2510.15189)** (NTU Singapore, 2025-10-16) — Best-of-scene self-imitation plus REINFORCE for a discrete pick-pose correction. Shelf placement 10/10 for pretrained RM-RL vs 5/10 for plain REINFORCE. This is a cross-method comparison, and the pretrained variant uses about 200 extra prior samples. · [code](https://github.com/NTUMARS/RMRL)

## September 2025

- **[DexFlyWheel](https://arxiv.org/abs/2509.23829)** (Harbin Institute of Technology, Peking University, PsiBot,…, 2025-09-28) — This is a clean template for a sim-side last-mile loop: freeze the imitation policy, train a small bounded SAC residual on dense rewards, keep the successful rollouts, augment them, and retrain the imitation policy.
- **[SOE](https://arxiv.org/abs/2509.19292)** (SJTU; Noematrix, 2025-09-23) — Information-bottleneck latent exploration for iterated success-filtered BC. Real Toaster Load 0.56 → 0.75 (0.84 with human steering) in one round. · [code](https://github.com/EricJin2002/SOE)
- **[ARFM](https://arxiv.org/abs/2509.04063)** (Westlake; UCLA; XJTU, 2025-09-04) — Critic-free softmax-advantage-weighted flow-matching loss with a per-batch adaptive temperature on pi0. LIBERO average 88.1 → 92.1 (flow RWR 90.8). The verifier found it published at AAAI-26.

## Before September 2025

- ⭐ **[CUPID](https://arxiv.org/abs/2506.19121)** (Stanford; Toyota Research Institute; Cambridge; NVIDIA, 2025-06-23) — Scores each demo's effect on closed-loop success by chaining influence functions with a REINFORCE estimate from labeled eval rollouts, then drops negative-influence demos and retrains. TuckBox mass shift, 66% filtered: base 0%, CUPID 84%, oracle 88% (bar readings). · [code](https://github.com/agiachris/cupid)
- ⭐ **[CFGRL](https://arxiv.org/abs/2505.23458)** (UC Berkeley, 2025-05-29) — Trains one flow policy by plain BC with an optimality token, then samples reference policy times a non-decreasing advantage function via classifier-free guidance. Sim ExORL walker-stand, same IQL critic: 782±8 vs AWR 603±8; mixed on OGBench. · [code](https://github.com/kvfrans/cfgrl)
- **[SCIZOR](https://arxiv.org/abs/2505.22626)** (UT Austin; NVIDIA, 2025-05-28) — A self-supervised progress model plus deduplication removes suboptimal chunks. +32.9% on Sirius-Fleet real tasks over the full dataset. · [code](https://github.com/UT-Austin-RPL/SCIZOR)
- **[DataMIL](https://arxiv.org/abs/2505.09603)** (UT Austin; MIT; Stanford, 2025-05-14) — Linear datamodels with a target-demo BC-loss proxy select prior-data clusters. LIBERO-10 37.76 vs 27.72 for all-data co-training. · [code](https://github.com/UT-Austin-RobIn/datamil)
- **[ReinboT](https://arxiv.org/abs/2505.07395)** (2025-05-12) — End-to-end VLA that predicts dense returns to capture data-quality distribution and act toward maximized return; SOTA on CALVIN mixed-quality data, strong few-shot and real-world OOD generalization.
- **[SIME](https://arxiv.org/abs/2505.01396)** (SJTU; Noematrix, 2025-05-02) — Success-filtered self-improvement with noise injected into the observation embedding to diversify modes. Square-20 (state): 0.534 → 0.620 vs 0.556 for plain self-improvement. · [code](https://github.com/EricJin2002/SIME)
- **[Sim-and-Real Co-Training](https://arxiv.org/abs/2503.24361)** (UT Austin, NVIDIA, UC Berkeley, New York University, 2025-03-31) — This is the cheapest data lever for the base-policy stage of a hill-climbing pipeline: a roughly aligned sim cousin plus automated demo multiplication took a 20-50-demo real policy from 45% to 83% on average, and that…
- **[Demo-SCORE](https://arxiv.org/abs/2503.03707)** (Stanford, 2025-03-05) — A tiny success classifier trained on eval rollouts flags demo strategies the policy cannot execute. Square peg: 79.9% base → 94.4%. · [code](https://github.com/alessing/demo-score)
- **[DemInf](https://arxiv.org/abs/2502.08623)** (Google DeepMind; Stanford, 2025-02-12) — Label-free demo scoring by contribution to I(S;A), using k-NN estimators in VAE latents. The abstract reports a 5-10% improvement on RoboMimic. · [code](https://github.com/jhejna/demonstration-information)
- **[iRe-VLA](https://arxiv.org/abs/2501.16664)** (Tsinghua IIIS; UC Berkeley, 2025-01-28) — PPO on a small head over a frozen VLM, then SFT on demos plus all successes. MetaWorld unseen-10: 0.51 SFT / 0.39 PPO-Replay / 0.80 iRe-VLA.
- **[RLDG](https://arxiv.org/abs/2412.09858)** (UC Berkeley, 2024-12-13) — Real-world HIL-SERL specialists generate successful rollouts that become SFT data for OpenVLA/Octo. OpenVLA +33% on FMB Insertion vs fine-tuning on human demos. · [code](https://generalist-distillation.github.io/RLDG/)
- **[GRAPE](https://arxiv.org/abs/2411.19309)** (aiming-lab, University of Washington, plus other…, 2024-11-28) — GRAPE is an early template for improving a VLA from its own rollouts, successes and failures included, with a KL anchor to the SFT policy and no online RL infrastructure. · [code](https://github.com/aiming-lab/grape)
- **[Towards Effective Utilization of Mixed-Quality Demonstrations in Robotic Manipulation…](https://arxiv.org/abs/2409.19917)** (2024-09-30) — S2I segments mixed-quality demos, selects good segments via contrastive learning and optimizes poor ones' trajectories; 3 expert demos as reference improve downstream policies on six tasks (ICRA 2025). · [code](https://tonyfang.net/s2i/)
- **[SOAR](https://arxiv.org/abs/2407.20635)** (UC Berkeley, 2024-07-30) — VLM-proposed tasks and VLM success filtering for goal-conditioned BC on autonomous data. Average success 28% to 57% across 10 scenes, with no humans. · [code](https://github.com/rail-berkeley/soar)
- **[Learning from Imperfect Demonstrations with Self-Supervision for Robotic Manipulation](https://arxiv.org/abs/2401.08957)** (2024-01-17) — SSDF: self-supervised quality scoring of failed-trajectory segments using expert plus imperfect data; high-scoring segments augment offline BC training, improving success on ManiSkill2 and a real Franka, reward-free.
- **[RoboCat](https://arxiv.org/abs/2306.11706)** (Google DeepMind, 2023-06-20) — Fleet-scale fine-tune, deploy, hindsight-relabel, success-filter and retrain flywheel on a 1.18B transformer.
- **[CRR](https://arxiv.org/abs/2006.15134)** (DeepMind, 2020-06-26) — Offline critic-filtered BC (binary or exp weights). Humanoid Run: CRR exp 586±6 vs BC 382±2 and D4PG 1±1. · [code](https://github.com/google-deepmind/acme/tree/master/acme/agents/jax/crr)
- **[AWR](https://arxiv.org/abs/1910.00177)** (UC Berkeley, 2019-10-01) — The base recipe: regress V, then do exp(A/beta)-weighted BC on buffer actions. Hopper-v2 3405±121 vs SAC 2769±552 and PPO 1391±304. · [code](https://github.com/xbpeng/awr)
