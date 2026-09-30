# Human corrections & DAgger

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*My imitation policy works most of the time but fails in states the demos never covered. How do I spend a small amount of human time so the next version fixes exactly those failures?*

Run the policy, let a person (or now a machine) step in where it goes wrong, and train on those on-policy fixes so the policy learns to handle the states it actually reaches.

- **Loop step:** Find failures → Improve
- **Built on:** Compounding error (covariate shift); DAgger: label the learner's own states; Human-gated vs robot-gated takeovers; Reweighting correction data; A takeover as a reward, value or preference signal; Human effort is the budget
- **Use it when:** Use this family when you have a decent imitation policy (roughly 30% to 80% success) that fails in specific, repeatable places, and a person who can recognize and fix those failures. It is the default last-mile tool when you have no reward function, when the task is long-horizon or contact-rich, and when you want the policy to stay close to its pretrained behavior.
- **What changed in 2025–26:** Five shifts in the last 12 months. (1) The policy being corrected is now a VLA. pi0/pi0.5, GO-1, Being-H0.5 and Gr-Dexter replace small BC-RNNs, and the loops are built as systems: SOP streams fleet corrections to a cloud learner, and HELP splits operator roles and reports successes per hour.

**46 entries**, newest first. ⭐ marks must-know work.


## September 2026

- **[Rho](https://arxiv.org/abs/2609.38164)** (Rho Team, 2026-09-29) — Open-weights dual-arm VLA family (YAM Box, UR AI Trainer, FR3 Duo) with embodiment-specific midtraining; a latent policy adapts online from as few as 15 corrected episodes.
- ⭐ **[BlenDAgger](https://arxiv.org/abs/2609.37599)** (Carnegie Mellon University, 2026-09-29) — Keeps the policy running during corrections and executes a per-DoF blend of human and policy actions, favoring the human when they push hard or disagree. Final round, real Almond Scooping: 50% vs 15% for HG-DAgger.
- ⭐ **[Skill-Space Shooting](https://arxiv.org/abs/2609.38178)** (Tsinghua University; UC Berkeley; Shanghai Qi Zhi Institute, 2026-09-29) — A learned value model halts a pi0.5-class policy when its score drops below zero, Gemini 2.5 Pro picks and checks a repair skill, and successful repairs become training data. Coffee progress: 71.25±7.50% vs 46.25±15.00% with extra human demos.
- ⭐ **[TimelyDAgger](https://arxiv.org/abs/2609.33157)** (Tsinghua University; SEEN·E Robotic, 2026-09-27) — Robot-gated DAgger that requests expert takeover when a PCA reconstruction residual on the VLA's internal features crosses a calibrated, outcome-adapted threshold. pi0.5 StackCube-Green (ManiSkill3 sim): 81 vs HG-DAgger 29, Offline BC 49; best in 13 of 15 settings.
- **[Kintsugi-VLA](https://arxiv.org/abs/2609.31048)** (Snegirev et al., 2026-09-25) — Restores simulator states along failed rollouts to estimate recoverability and generate targeted recovery data; SmolVLA recovery success 34.6-38.4%, 5.8-6.7 points over uniform sampling.
- **[Policy-Calibrated DAgger](https://arxiv.org/abs/2609.30462)** (Wang et al., 2026-09-24) — Calibrates DAgger-style noise injection offline from a diffusion policy's own action spread along expert trajectories; beats un-noised DAgger and matches hindsight-optimal noise without sweeps.
- **[TANDEM](https://arxiv.org/abs/2609.28314)** (Sahoo et al., 2026-09-23) — TAMP with VLM-added predicates requests human teleoperation only for steps it cannot plan; 2.9x more demos per human time, 20 demos/task lift VLA success 0% to 60%. · [code](https://prpl-group.com/tandem/)
- ⭐ **[PAKT](https://arxiv.org/abs/2609.25630)** (NVIDIA, 2026-09-22) — Operators correct HIL-SERL-style real-world RL by hand-guiding the arm through an admittance controller sharing the policy's action space, limits and impedance tracker. Vs HIL-SERL on four real insertion tasks: 62%-86% fewer interventions, 23%-48% lower cycle time.
- **[MAGMA-GEN](https://arxiv.org/abs/2609.20056)** (LAAS-CNRS, 2026-09-17) — Turns failed long-horizon manipulation rollouts into recovery supervision: a coach proposes corrections, kept only if re-execution from that point improves outcomes; beats distillation and trajectory repair (CoRL 2026).
- **[HIL-UMI](https://arxiv.org/abs/2609.20659)** (Xi'an Jiaotong University; Peking University, 2026-09-17) — Shadow-mode policy on a handheld UMI gates where to collect; Stack Cube 84 to 100 after 3 rounds.
- **[REVOLVE](https://arxiv.org/abs/2609.14633)** (Jiangsu Key Laboratory, 2026-09-13) — VLM failure detector plus scripted recovery replaces human corrections and resets; average success 40.0% to 58.5% over 5 iterations.

## August 2026

- ⭐ **[GAINS](https://arxiv.org/abs/2608.15707)** (Beijing Institute of Technology; Beijing Innovation Center…, 2026-08-16) — Learns the return distribution with a truncated-quantile critic and acts on its lower tail, since takeover timing varies (over 50% of repeated marks differed by more than 10 frames at 30 fps). Abstract: 22% higher success than RLIF. · [code](https://github.com/nuomizai/HIL-RL)

## July 2026

- **[LIFT](https://arxiv.org/abs/2607.14236)** (Shanghai Jiao Tong University; Shanghai Innovation…, 2026-07-15) — Grafts force sensing onto pi0.5 during online DAgger; towel folding peaked at 0.825 (#2.8K frames) vs vision-only 0.725 (#3.1K). · [code](https://github.com/y-wng/lift)
- ⭐ **[FlowDAgger](https://arxiv.org/abs/2607.08877)** (Microsoft Research; Microsoft; ETH Zurich; University of…, 2026-07-09) — Inverts the flow sampler to find the noise that would have produced the operator's correction, then trains a small noise policy on it with the VLA frozen. MetaWorld, pi0.5: 0.78 vs SFT 0.71; held-out skills 0.88 vs 0.02. · [code](https://github.com/microsoft/FlowDAgger)
- ⭐ **[HELP](https://arxiv.org/abs/2607.09776)** (Shanghai AI Laboratory, 2026-07-08) — A progress model (VLAC-CUT) splits human-in-the-loop pi0.5 rollouts into progress, recovery and idle segments, training only on useful ones to avoid learned slowness. Real round 2, Microplate: 42 tasks/hr at 90% vs HITL-only 25 at 60%. · [code](https://github.com/InternRobotics/VLAC-cut)

## June 2026

- ⭐ **[UniIntervene](https://arxiv.org/abs/2606.12372)** (Nanyang Technological University; Beijing University of…, 2026-06-10) — Adds an intervention agent to HIL-SERL: a VLM-based proxy value model and a temporal value-risk critic detect stagnation, then steer the robot back to a remembered high-value state. HIL-SERL 81% at 34.3% intervention rate → 88% at 14.6%. · [code](https://github.com/Denghaoyuan123/UniIntervene)
- ⭐ **[FlowPRO](https://arxiv.org/abs/2606.05468)** (Tencent Robotics X; Futian Laboratory; Tsinghua University, 2026-06-03) — Operators roll back and teleoperate a fix, giving paired bad and corrected segments; a critic-free DPO-style loss (flow-matching loss as -log pi) with proximal regularizer and SFT trains on them. PACK, 3 rounds: 99±1.0% vs DAgger 88%. · [code](https://wuyeyexvnainai.github.io/flowpro/)
- **[Set-Supervised Diffusion Policy](https://arxiv.org/abs/2606.01865)** (TU Delft; University of Stuttgart, 2026-06-01) — Uses the robot's rejected chunk to build a 'desired set' around the correction; robust to noisy corrections (noisy Square 0.946 vs DP 0.683). · [code](https://github.com/ZhaotingLi/Set_Supervised_DP)

## May 2026

- **[VR-DAgger](https://arxiv.org/abs/2605.27114)** (ETH Zurich, 2026-05-26) — MC-dropout uncertainty picks 2 s failure clips for VR correction; DAgger-style gains at about 40% less operator time (sim).
- **[OHP-RL](https://arxiv.org/abs/2605.15971)** (HKUST, 2026-05-15) — Takeovers as pairwise preferences inside SAC with a critic-based gate; Move Sand Bag 68% vs HIL-SERL 16%.
- **[HandITL](https://arxiv.org/abs/2605.15157)** (Shanghai Jiao Tong University, 2026-05-14) — Bumpless hand-arm takeover for a 21-DoF dexterous VLA; one hour of corrections took avg sub-goal score from about 0.70 to about 0.95.

## April 2026

- ⭐ **[Hi-WM](https://arxiv.org/abs/2604.21741)** (Current Robotics; Tsinghua University; Peking University;…, 2026-04-23) — Runs HG-DAgger inside an action-conditioned video world model: a human steers the imagined robot away from failure, and the corrections fine-tune the real policy. Real-world success averages +37.9 points over base and +19.0 over a human-free world-model baseline.
- **[Human-Robot Copilot](https://arxiv.org/abs/2604.03613)** (UC San Diego, 2026-04-04) — Low-cost bidirectional leader arm for HG-DAgger clips; sim Nut Assembly 56 vs 46 total success at 40 trajectories.

## March 2026

- **[VLA-OPD](https://arxiv.org/abs/2603.26666)** (HKUST, 2026-03-27) — DAgger with a neural teacher: reverse-KL on-policy distillation; LIBERO avg 48.9 to 87.4 (distill only), 93.4 with GRPO. Sim only. · [code](https://irpn-lab.github.io/VLA-OPD/)
- **[DexHiL](https://arxiv.org/abs/2603.09121)** (CASIA; Shanghai Jiao Tong University, 2026-03-10) — Arm-plus-hand takeovers for a dexterous VLA with 50% reweighting; Tissue Extraction 19/20 vs DAgger* 16/20 vs offline 15/20. · [code](https://chenzhongxi-sjtu.github.io/dexhil/)
- **[RoboPocket](https://arxiv.org/abs/2603.05504)** (Shanghai Jiao Tong University; Shanghai Innovation…, 2026-03-05) — Robot-free DAgger with an iPhone handheld gripper and AR policy preview; distributed boost 0.42 to 0.82 with 12 corrections.
- **[TER-DAgger](https://arxiv.org/abs/2603.04038)** (Xi'an Jiaotong-Liverpool University, 2026-03-04) — Force-mismatch trigger plus trajectory-edited corrections for insertion; average 77.2 vs 40.0 for the best baseline (Finetune).

## February 2026

- **[FlowCorrect](https://arxiv.org/abs/2602.22056)** (Karlsruhe Institute of Technology, 2026-02-25) — VR 'nudges' train a tiny LoRA patch on a frozen flow policy; hard Pick-and-Place cases 0/40 to 32/40.
- **[AGPS](https://arxiv.org/abs/2602.11978)** (Peking University; PKU-PsiBot Joint Lab, 2026-02-12) — Replaces the HIL-SERL human with a failure-detector-gated VLM agent; USB insertion 100% at 8 min with zero human interventions.

## January 2026

- ⭐ **[SOP](https://arxiv.org/abs/2601.03044)** (AgiBot Research; Shanghai Innovation Institute, 2026-01-06) — Ten real AgiBot G1 robots run pi0.5 with human takeovers; a cloud learner mixes on-policy data with old demos, publishing weights every 25 steps. 3 h on-policy interaction: 0.571 → 0.800; 80 more demo hours: 0.576 → 0.612.

## September 2025

- **[Selective Progress-Aware Querying for Human-in-the-Loop Reinforcement Learning](https://arxiv.org/abs/2509.20541)** (Muraleedharan et al., 2025-09-24) — SPARQ: query policy requests human feedback only when learning stagnates or worsens; near-perfect success on simulated UR5 cube picking (PyBullet) with about half the always-query feedback budget.
- **[Talk-and-Tweak dual-actor](https://arxiv.org/abs/2509.13774)** (Beijing Xiaomi Robot Technology, 2025-09-17) — HIL off-policy BC+Q fine-tuning of a VLA with language-steered noise; 100% avg vs HG-DAgger 25.3%, though test-time human language corrections may be included.
- ⭐ **[RaC](https://arxiv.org/abs/2509.07953)** (Carnegie Mellon University, 2025-09-09) — Changes HG-DAgger takeovers: the human first recovers to a familiar state, then finishes the sub-task, and the episode ends with the takeover. Shirt-hanging: 78.3% with 5 hours of data vs ALOHA Unleashed 75.0% with about 89 hours.

## Before September 2025

- **[CR-DAgger](https://arxiv.org/abs/2506.16685)** (Stanford University, 2025-06-20) — Compliant physical delta corrections train a residual on a frozen diffusion policy; book flipping 40 to 100 with on-policy delta data. · [code](https://github.com/yifan-hou/cr-dagger)
- **[Genie Centurion](https://arxiv.org/abs/2505.18793)** (AgiBot, 2025-05-24) — Rewind-and-refine DAgger on the GO-1 VLA with a learned success detector; e.g. Microwave-Heating task score 0.97±0.01 at 80 trajectories vs 0.55±0.05 teleop.
- **[RoboCopilot](https://arxiv.org/abs/2503.07771)** (Wu et al., 2025-03-10) — Compliant bilateral teleoperation system that lets a human seamlessly take over from and hand back to the policy, collecting interactive correction data for bimanual manipulation in sim and real.
- **[PVP](https://arxiv.org/abs/2502.03369)** (UCLA, 2025-02-05) — Reward-free +1/-1 proxy values on human vs rejected actions; MetaDrive test success 0.857 vs HG-DAgger 0.045 (NeurIPS 2023, arXiv 2025). · [code](https://github.com/metadriverse/PVP)
- **[Sirius-Fleet](https://arxiv.org/abs/2410.22689)** (UT Austin, 2024-10-30) — World-model-based failure monitor decides when humans step in; real autonomous success 0.250 to 0.715 over 3 rounds. · [code](https://github.com/UT-Austin-RPL/sirius-fleet)
- **[IntervenGen](https://arxiv.org/abs/2405.01472)** (NVIDIA; UC Berkeley, 2024-05-02) — Multiplies 10 human interventions into about 1,000 synthetic ones; sim Coffee 2% to 80%.
- **[RLIF](https://arxiv.org/abs/2311.12996)** (UC Berkeley, 2023-11-21) — -1 reward before each takeover plus off-policy RL; Walker2d 103.83±4.48 vs HG-DAgger 36.75±4.7, and it beats weak experts. · [code](https://github.com/pd-perry/RLIF)
- **[Sirius](https://arxiv.org/abs/2211.08416)** (UT Austin, 2022-11-15) — Deployment-time weighted BC; real Gear Insertion 14/32 to 29/32 over three rounds. · [code](https://github.com/UT-Austin-RPL/sirius)
- **[Fleet-DAgger](https://arxiv.org/abs/2206.14349)** (UC Berkeley, 2022-06-29) — Allocates M humans across N robots; defines Return on Human Effort and claims up to 8.8x higher ROHE than baselines. · [code](https://github.com/BerkeleyAutomation/ifl_benchmark)
- **[ThriftyDAgger](https://arxiv.org/abs/2109.08273)** (UC Berkeley AUTOLAB, 2021-09-17) — Robot-gated on novelty plus risk; sim peg insertion 73/100 autonomous success vs HG-DAgger 57/100, with fewer interventions. · [code](https://github.com/ryanhoque/thriftydagger)
- **[IWR](https://arxiv.org/abs/2012.06733)** (Stanford; UT Austin, 2020-12-12) — Balanced sampling of intervention vs robot data; Threading final 87.3±5.0 vs HG-DAgger 75.3±8.1 and Full Demos 76.7±2.3 (sim).
- **[HG-DAgger](https://arxiv.org/abs/1810.02890)** (Stanford University; UIUC, 2018-10-05) — Human-gated takeovers on a car; mean road-departure rate 0 at every checkpoint from 1.2x10^4 labels onward, while BC and DAgger kept departing.
- **[DAgger](https://arxiv.org/abs/1011.0686)** (Carnegie Mellon University, 2010-11-02) — The origin: aggregate expert labels on the learner's states. Super Tux Kart agent almost never fell off after 5 iterations while BC did not improve with more expert data.
