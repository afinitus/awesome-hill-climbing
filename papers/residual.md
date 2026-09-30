# Residual, edit & steering policies

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*How do I push a big imitation, diffusion or VLA policy from 50% to 95%+ when I can't cheaply fine-tune or backprop through it, without breaking what it already does well?*

Freeze the big pretrained policy and train a small add-on with RL: one that corrects its actions (residual/edit) or one that picks the noise it is fed (steering).

- **Loop step:** Improve
- **Built on:** Residual policy: the base is part of the environment; Start at base performance: zero-init, bound and scale the correction; Off-policy actor-critic with prior data is what makes this fit on a real robot; Latent-noise steering: RL picks the noise, not the action; Edit policy + best-of-N selection by Q; Support versus reach
- **Use it when:** Use it when you already have a decent base (roughly 20-80% success) that you can't or won't fine-tune: a VLA too big to backprop through with RL, a closed checkpoint that allows only forward calls, or a chunked diffusion or ACT policy whose likelihood is awkward for policy gradients. It also fits when you want the base's generality and recovery behaviors kept intact, and when your budget is minutes to hours of real robot time with sparse success labels.
- **What changed in 2025–26:** Four shifts in the last 12 months. (1) Bases moved from task-specific diffusion or ACT policies to frozen generalist VLAs (pi0, pi0.5, GR00T, SmolVLA, OpenVLA-OFT). Freezing became the default because full RL fine-tuning of 3B models is costly and erodes recovery behavior (PPS, TRANSIC-style findings).

**56 entries**, newest first. ⭐ marks must-know work.


## September 2026

- **[FailPatch](https://arxiv.org/abs/2609.34175)** (Xi'an Jiaotong; Dexmal; Nanjing U, 2026-09-28) — Offline, supervised hidden-space residual expert bank with a Null route, trained on logged failures of a frozen pi0.5. RoboTwin average 49.50 → 60.50, vs PPO 55.75. · [code](https://github.com/yupeng-2003/FailPatch)
- ⭐ **[PSS (Principal Steering Subspaces)](https://arxiv.org/abs/2609.33765)** (XPENG Robotics; University of Michigan, 2026-09-27) — Finds the few noise directions that most move a frozen diffusion or flow policy's actions (finite-difference probing at 64 states) and lets SAC steer only in that subspace. Diffusion-Square final success: full-latent DSRL 0.716 vs 0.878 (one seed).
- **[VLaRL](https://arxiv.org/abs/2609.30868)** (MSRA Tokyo; KAIST; U Tokyo, 2026-09-25) — Sim-trained TD3 residual that reads the VLA's own latent, aligned sim-to-real with an OT mapper. Real Flower: button 67.5% → 100%, block push 22.5% → 50%. · numbers corrected after check
- **[FIND](https://arxiv.org/abs/2609.32069)** (TU Darmstadt; Hessian.AI; others, 2026-09-25) — Reset-free real RL. A VLM picks feasible, weakest tasks and judges success for a critic-gated residual on a frozen pi0.5. 8-task human-assessed success goes from 55% to 71.9%.
- **[Res-HIL](https://arxiv.org/abs/2609.30023)** (Siemens; TU Munich, 2026-09-24) — Zero-initialized TD3 residual on a 20-demo ACT. Human takeovers become regression targets plus penalties on the steps just before them. After 10 min: 100/64/50/66/92% vs HIL-SERL 90/30/10/0/0 and ResFiT 0/0/0/0/0.
- **[CereVLA](https://arxiv.org/abs/2609.27468)** (Zeng et al., 2026-09-23) — Lightweight residual refinement on an action-chunked VLA, gated by a recurrent consequence predictor that suppresses unfavorable corrections; SO-101 success 57.5% to 90.0%, 19.6% fewer control steps.
- **[BEE](https://arxiv.org/abs/2609.27450)** (AgiBot; SCUT, 2026-09-23) — Residual actor on a frozen pi0.5, constrained by a learned per-dimension Gaussian model of human corrections (Mahalanobis trust region). Average success 91.2 vs RLT 57.5, DSRL 42.1, base 38.8.5 memory claim is unverified; all table numbers were confirmed. · numbers corrected after check
- **[DynaForge](https://arxiv.org/abs/2609.25631)** (Jin et al., 2026-09-22) — Planner plus object-centric IK with a residual policy and implicit curriculum lifts sim demo-generation success 41.30% to 78.37%; trained DP3 policies reach 30-60% real vs 0-10%.
- **[RouteRLT](https://arxiv.org/abs/2609.26467)** (University of Toronto, 2026-09-22) — A learned router decides when a per-phase RL specialist (RL-Token-style readout) takes over from a frozen SmolVLA. LIBERO Object full-task success 85.00 → 92.22.
- **[Imagine-RL](https://arxiv.org/abs/2609.24033)** (KAIST; USTB; SJTU, 2026-09-21) — DSRL on a frozen pi0 whose critic gets 10-step visual and torque look-ahead from a frozen latent world model. It reports relative gains of 23.6% over DSRL and 60% over pi0 after 50 RL episodes; per-task numbers appear only in a chart.
- **[ForceRFT](https://arxiv.org/abs/2609.22840)** (EIT Ningbo; SJTU; HKUST-GZ; others, 2026-09-19) — Force-aware TD3 residual bounded to 1 mm and 0.5 deg on a frozen force-conditioned SmolVLA, plus human corrections. Plug insertion: 12/30 prior → 25/30 within a 30-min budget.
- **[VT-Bridge](https://arxiv.org/abs/2609.22606)** (Wu et al., 2026-09-18) — 0.98M-parameter tactile residual adapter on frozen pi0/pi0.5/SmolVLA, trained from 50 vision-tactile demos per task (supervised, not RL); completion 11.7% to 62.9% on four contact-rich tasks. · [code](https://hoxnocha.github.io/vt-bridge-web/)
- **[PARTS](https://arxiv.org/abs/2609.21788)** (UT Austin; Autel; UC Berkeley, 2026-09-18) — Masked, bounded TD3 residuals only on human-identified bottleneck subtasks of a frozen pi0.5, triggered by agent-written selectors and verifiers. Full task 50% → 95% on Franka and 32% → 61% on YAM, in tens of minutes. · [code](https://destiny000621.github.io/PARTS/)
- **[High-DoF VLA post-training](https://arxiv.org/abs/2609.19666)** (Wuji Technology; ShanghaiTech, 2026-09-17) — SFT, then human-gated DAgger, then a latent residual TD3 on a frozen pi0.5 for two 20-DoF hands. All 5 tasks reach 20/20. Transfer goes 10% → 65% → 70% → 100% with RL, using 3,000 online transitions per task.
- ⭐ **[Real-Time EXPO-FT](https://arxiv.org/abs/2609.18207)** (Stanford University, 2026-09-16) — Slow pi0.5 proposes 32 chunks from a stale observation; a fast policy adds bounded edits from the fresh one; a Q-ensemble picks one. Real, 4 dynamic tasks, 10 min online data: 12.5/30 → 29/30 (EXPO-FT w/ RTC 25/30). · [code](https://github.com/pd-perry/expo-ft)
- **[PPS (Proxy Policy Steering)](https://arxiv.org/abs/2609.09148)** (Cornell, 2026-09-08) — Nudges the base velocity field by gamma*(v_task - v_ref) from two small proxy models, with no RL. Real 8 tasks: pi0.5 22% → 79%, vs LoRA 59%, residual 19%, DSRL 25%. · [code](https://github.com/TritiumR/pps)
- **[Towards Universal Post-Training (blog)](https://pd-perry.github.io/posts/post-training.html)** (Stanford, 2026-09) — Position essay for an EXPO-FT-style standard robot post-training recipe. It cites EXPO-FT reaching 30/30 with an average of about 19 min of online interaction. The baseline averages cover only the 4 tasks every baseline ran. · [code](https://github.com/pd-perry/expo-ft)

## August 2026

- **[ARLI](https://arxiv.org/abs/2608.23831)** (Siemens; UC Berkeley; Microsoft; ETH, 2026-08-24) — DSRL under asynchronous VLA inference. It feeds the noise policy the committed actions plus a fresher mid-inference frame. Real success goes from around 40% to near 100% in 100-125 episodes.
- **[ORPA](https://arxiv.org/abs/2608.17323)** (AIST Japan, 2026-08-18) — Supervised residual head that turns coarse human feedback ('too left') into joint corrections for a frozen ACT, with no RL. Sim induced-failure Cube Transfer 60.0% → 92.3%.
- **[HAF-Steer](https://arxiv.org/abs/2608.16837)** (Peking University, 2026-08-17) — Inverts demos into flow noise, keeps K=8 DCT modes, BC-initializes, then runs SAC on a frozen humanoid VLA. Toy Storage OOD with pi0.5: 2/10 → 8/10, while DSRL training failed.
- **[ViTaR](https://arxiv.org/abs/2608.15816)** (Wang et al., 2026-08-16) — Tactile feedback selects and scales bounded residual corrections atop a frozen VLA via outcome-guided modeling; 61.3% average on seven UniVTAC contact-rich tasks, +30.6 points over base.
- ⭐ **[MiDAS](https://arxiv.org/abs/2608.11363)** (Carnegie Mellon University, 2026-08-11) — LoRA-tunes pi0.5 on one demo, then freezes it and trains a sparse-reward off-policy edit actor-critic that outputs the full executed chunk, so it can leave the base's action support. LIBERO-Long, 1 demo: 91.2 vs DICE-RL 89.3, DSRL 33.5.
- **[SpeedTuning](https://arxiv.org/abs/2608.09138)** (Stanford, 2026-08-10) — Rainbow DQN picks the playback speed of a frozen ACT's chunks, so it hill-climbs cycle time rather than success. Sim Cube Transfer runs 3.81x faster at 97.9% (ACT 100%). The arXiv date is 2026-08, but the venue is ICRA 2025. · [code](https://github.com/DaivdYuan/SpeedTuning)

## July 2026

- **[RLMM-Flow](https://arxiv.org/abs/2607.26460)** (Wang et al., 2026-07-29) — Frozen pretrained flow policy steered by a latent-noise RL network with critic warm-up and coarse-to-fine latent-to-residual expansion; improves mobile-manipulation success and collision avoidance over imitation-only.
- ⭐ **[Foresight Residual RL](https://arxiv.org/abs/2607.16506)** (Rutgers University, 2026-07-17) — Trains a PPO residual on a frozen per-phase pi0 for each subtask, multiplying each phase's reward by a learned predictor of next-phase success. Sim wrench-on-nut full task: 85.6±3.9% vs 54.5±3.6% with plain per-phase residuals.
- **[LAMP](https://arxiv.org/abs/2607.06323)** (Fudan; PsiBot; Tsinghua; PKU, 2026-07-07) — Residual SAC/RLPD in a 2-D learned latent of hand motions for a 6-DoF dexterous hand. Real average 56.25% → 98.75%, while the same residual RL in the raw action space fails. · [code](https://github.com/dex-lamp/LAMP)
- **[OmniTacTune](https://arxiv.org/abs/2607.03723)** (U Maryland; Georgia Tech, 2026-07-04) — GelSight tactile SAC residual with a scheduled scale (0.05 → 0.15) on a frozen visual policy. Real base 40/10/5/5% → 100/100/90/85% after 40-80 min, including a 12-minute warm-start.

## June 2026

- ⭐ **[SCORE](https://arxiv.org/abs/2606.27475)** (University of Washington; Amazon FAR, 2026-06-25) — Freezes a real-demo flow policy and trains, in a scanned digital twin, a PPO steering policy that can only reweight actions the base produces. Real Franka/LEAP hand, 8 tasks, zero real RL: 37.8 → 89.9 vs Residual-RL 59.5.
- **[HiL-ResRL](https://arxiv.org/abs/2606.22860)** (Huawei Cloud; SCUT, 2026-06-22) — SAC residual plus SpaceMouse interventions on frozen Diffusion Policy or pi0.5. Bases at 50-80% rise to over 90% within 40-90 min of real training.
- **[Object-Centric Residual RL](https://arxiv.org/abs/2606.18953)** (KAIST; MSRA Tokyo; U Tokyo, 2026-06-17) — TD3 residual on 6-DoF object poses, trained in a digital twin against a paired 'sim VLA'. Real GR00T-N1.5 goes from 42% to 76% with zero real-world RL.
- **[Steering Generative Reinforcement Learning into Stable Robotic Controller (SteerGenPO)](https://arxiv.org/abs/2606.16572)** (Wang et al., 2026-06-15) — Replaces stochastic latent sampling of a trained generative RL policy with a learned state-dependent latent actor for deterministic deployment; beats baselines on six Isaac Lab tasks plus Unitree G1 locomotion.
- **[FRS (Flow Reversal Steering)](https://arxiv.org/abs/2606.13675)** (Stanford; UC Berkeley, 2026-06-11) — Flow-reverses coarse human or VLM hints into noise for a frozen pi0.5, then trains a noise-space BC/RL policy. Real DROID ~20% → ~87% zero-shot with human steering. Towel hanging 5% → 50% → 80% after RL.
- **[Learning What to Say to Your VLA](https://arxiv.org/abs/2606.12299)** (Carnegie Mellon University, 2026-06-10) — This is a hill-climbing lever that never touches the VLA's weights.

## May 2026

- **[LP-DS](https://arxiv.org/abs/2606.01151)** (Bilkent University, 2026-05-31) — Learns a residual on the prior noise (w = eps + Delta) with a Lagrangian trust region, so steering doesn't send off-distribution noise or collapse modes. Walker2D about 5000 vs about 4000 return for the best baseline (ICML 2026). · [code](https://github.com/HikmetSimsir/lpds)
- **[HOIST](https://arxiv.org/abs/2606.00252)** (University of Florida, 2026-05-29) — DSRL-style noise RL on a fine-tuned GR00T N1.6 humanoid pushing a suspended load. Sim placement error 22.44 cm → 2.54 cm after 20 RL rollouts (6.25 cm at 30). Real error 9.28 → 6.38 cm, but yaw error rose from 14.5 to 28.9 deg.
- **[ZPRL](https://arxiv.org/abs/2605.19919)** (HKU; Shanghai Qizhi; SJTU, 2026-05-19) — SAC perturbs a small variational-bottleneck latent inside a frozen flow policy rather than its actions. Real tasks gain 33.7% on average over the IL base (Insert Bills 20% → 77.5%), with about 28.5 h of real RL in total. · [code](https://github.com/manutdmoon/ZPRL)
- ⭐ **[BMD (Behavioral Mode Discovery)](https://arxiv.org/abs/2605.11387)** (Naver Labs Europe; Stanford, 2026-05-12) — Discovers modes in a frozen diffusion policy via a noise-steering policy conditioned on a per-episode code with a DIAYN-style classifier reward, then fine-tunes keeping modes separate. Avoid, success ~1.00 throughout: DSRL keeps 2.00/24 modes, DPPO 6.33/24, DSRL[BMD] 10.0/24.
- **[TMRL](https://arxiv.org/abs/2605.12236)** (U Washington; Amazon FAR, 2026-05-12) — Context-smoothed pretraining plus a learned noise-level dial lets steering leave the base's narrow support. Real widowx-sausage reaches 1.0 at about 50 episodes while DSRL stays at 0 (RSS 2026). · [code](https://github.com/matthewh6/tmrl)
- **[Retrieve-then-Steer](https://arxiv.org/abs/2605.10094)** (College of Artificial Intelligence, Xi'an Jiaotong…, 2026-05-11) — This is a zero-gradient hill-climbing baseline for a fixed station.
- ⭐ **[UniSteer](https://arxiv.org/abs/2605.10821)** (Microsoft Research; Tsinghua University; University of…, 2026-05-11) — Trains a SAC noise actor for frozen pi0, inverting the flow decoder by fixed-point iteration so human teleop corrections become noise-space targets. Real AgileX Piper: success 20 → 90 in 66 min average, vs DSRL 55, DAgger 60.

## March 2026

- **[ExpertGen](https://arxiv.org/abs/2603.15956)** (RAI Institute; UT Austin; Sony AI, 2026-03-16) — DSRL + FastTD3 noise steering in massively parallel sim with binary reward only. On an 8-task AnyTask mean: ExpertGen 85.3 vs DP prior 40.3, residual RL 53.7, DSRL-SAC 37.7.
- ⭐ **[DICE-RL](https://arxiv.org/abs/2603.10263)** (Stanford University; TRI hardware, 2026-03-10) — Freezes a diffusion/flow BC prior as a noise-to-chunk sampler, trains a small residual on the same noise with a TD3+BC-style loss, and runs the best-Q sample. Real UR5 BeltAssembly (30 trials): 56.67% → 93.33% after about 420 episodes. · [code](https://github.com/zhanyisun/dice-rl)
- **[LPS](https://arxiv.org/abs/2603.05296)** (Yonsei; Microsoft Research, 2026-03-05) — Latent actor picks the noise for a one-step MeanFlow policy, and the action-space Q-gradient is backpropagated through the generator. Real average: BC-FM 25/80, DSRL 28/80, LPS 45/80. · [code](https://github.com/jellyho/LPS)

## February 2026

- **[SPARR](https://arxiv.org/abs/2602.23253)** (NVIDIA; U Washington, 2026-02-26) — Frozen sim PPO assembly policy plus a real-world image and force residual trained RLPD-style, seeded with the prior's own successes. 95-100% on 10 AutoMate tasks in 0.5 h, with no human in the loop.
- ⭐ **[DAWN](https://arxiv.org/abs/2602.10539)** (Nanyang Technological University; Mila / Universite de…, 2026-02-11) — Speeds up residual SAC by fixing the critic: prefill the replay buffer with 20K base-policy transitions and add LayerNorm to the critic. PegInsertionSide: 90% success about 6x faster than Policy Decorator (~0.3M vs ~2-2.3M env steps, plot reading). · [code](https://github.com/Guozheng-Ma/DAWN)
- ⭐ **[RFS](https://arxiv.org/abs/2602.01789)** (University of Washington; Amazon FAR, 2026-02-02) — One PPO policy outputs both the starting noise for a frozen flow-matching base and an additive residual, so it can switch modes and leave the demo manifold. Sim, 6 IsaacLab dexterous tasks: 0.861 vs base 0.250, DSRL-only 0.483.

## December 2025

- **[Equi-DSRL](https://arxiv.org/abs/2512.11345)** (Yonsei; UC Berkeley, 2025-12-12) — Equivariant noise-steering actor and critics on equivariant diffusion policies. Square D2 peak success: base 0.275, DSRL 0.552, Equi-DSRL 0.644. Plain DSRL showed Q divergence in 3/5 Lift seeds.

## October 2025

- ⭐ **[PLD (Probe, Learn, Distill)](https://arxiv.org/abs/2511.00091)** (NVIDIA GEAR; CMU; UC Berkeley; UT Austin, 2025-10-30) — Trains a bounded Gaussian residual with SAC on a frozen VLA, lets the VLA drift before the residual recovers, and distills successes into the VLA. LIBERO: OpenVLA-OFT 91.8 → 99.2; real Franka pick-up: 30/30 vs 10/30 (human data).

## September 2025

- ⭐ **[ResFiT](https://arxiv.org/abs/2509.19301)** (Amazon FAR; Stanford; CMU; UC Berkeley, 2025-09-23) — Keeps an ACT chunked BC policy frozen and trains a small per-step residual with a tuned TD3-style off-policy recipe (demos in each batch, LayerNorm critic, n-step returns). Real 29-DoF humanoid: WoollyBallPnP 14% → 64% after 134 rollouts. · [code](https://github.com/amazon-far/residual-offpolicy-rl)

## Before September 2025

- **[DSRL](https://arxiv.org/abs/2506.15799)** (UC Berkeley; U Washington; Amazon, 2025-06-18) — RL over the input noise of a frozen diffusion or flow policy. Real Franka 2/10 → 9/10 after about 40 episodes. Real pi0 'turn on toaster' 5/20 → 18/20 after 80 episodes. · [code](https://github.com/ajwagen/dsrl)
- **[Policy Decorator](https://arxiv.org/abs/2412.13630)** (UC San Diego; Hillbot, 2024-12-18) — Bounded SAC residual plus a progressive exploration schedule on frozen BeT or Diffusion Policy. Bases at 15-78% reach near-perfect success on 8 sim tasks, with millions of env steps. · [code](https://github.com/tongzhoumu/policy_decorator)
- **[ResiP](https://arxiv.org/abs/2407.16677)** (MIT Improbable AI; Harvard, 2024-07-23) — Per-step PPO residual on a frozen chunked diffusion policy. Sim assembly goes from 5-54% to 88-99% (low randomization) with 50 demos, while diffusion BC plateaus at about 80% even with 100K demos on one_leg. · [code](https://github.com/ankile/robust-rearrangement)
- **[TRANSIC](https://arxiv.org/abs/2405.10315)** (Stanford, 2024-05-16) — Gated residual trained by max likelihood on human SpaceMouse corrections over a frozen sim policy. Real Franka average 81%, and it beats HG-DAgger, IWR and BC fine-tuning, which forget. · [code](https://github.com/transic-robot/transic)
- **[PLAS](https://arxiv.org/abs/2011.07213)** (CMU, 2020-11-14) — Offline RL where the actor outputs a tanh-bounded latent to a frozen CVAE decoder. Actions stay on data support. The conceptual ancestor of noise steering. · [code](https://github.com/Wenxuan-Zhou/PLAS)
- **[Residual Policy Learning](https://arxiv.org/abs/1812.06298)** (MIT, 2018-12-15) — pi = pi_base + f_theta with DDPG+HER on a frozen hand controller or MPC. On sim Push it reaches 0.9 success with about 10x fewer samples than learning from scratch. · [code](https://github.com/k-r-allen/residual-policy-learning)
- **[Residual RL for Robot Control](https://arxiv.org/abs/1812.03201)** (Siemens; UC Berkeley; TU Hamburg, 2018-12-07) — TD3 residual on a hand-engineered controller. Real Sawyer insertion with misaligned blocks: 15/20 vs 2/20 for the controller, in about 8K samples (about 3 h).
