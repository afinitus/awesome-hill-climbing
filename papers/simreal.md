# Sim-to-real & real-to-sim

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*How do I get the trial-and-error gains of RL for a robot policy without paying for thousands of risky, hand-reset real-world episodes?*

Uses a simulator, often a scanned or generated copy of the real scene, as a safe practice gym where the policy can hill-climb with RL before or alongside a short real-robot stage. Real demos or real replay data keep it tied to reality.

- **Loop step:** Deploy → Improve
- **Built on:** Domain randomization; Real-to-sim: digital twins and cousins; BC-anchored sim RL (sim-real co-training); Sim-to-online RL on the real robot; Sim value as a shaping signal; Residual correction on a sim-made base; Gradient-free and label-light last-mile knobs (the 'other' bucket)
- **Use it when:** Use this family when real-world trial and error is slow, unsafe or needs human resets, and you can build a simulator that gets the coarse task structure right: rigid objects, known geometry, a calibratable controller. Pick by situation.
- **What changed in 2025–26:** Four shifts in the last 12 months. (1) The simulator changed jobs. Earlier sim-to-real trained small policies from scratch and hoped for zero-shot transfer.

**36 entries**, newest first. ⭐ marks must-know work.


## September 2026

- ⭐ **[F4R](https://arxiv.org/abs/2609.35575)** (Nanyang Technological University, Xi'an Jiaotong…, 2026-09-28) — A VLM diagnoses real pi0.5 failures, the scene is rebuilt in Isaac Sim, and the policy is co-trained on failure-conditioned sim data, then PPO-refined and redeployed. Real OOD success: 26.25% → 90.0%, vs 71.25% for budget-matched Targeted BC.
- **[ZeroBot](https://arxiv.org/abs/2609.34010)** (Kapelyukh et al., 2026-09-27) — Single-view image-to-3D meshes feed massively parallel sim RL with a contact-state-sampling action space; no demos or pretraining, 87% real-world success after ~119 s training. · [code](https://zerobot-rl.github.io)
- **[Robot-GST](https://arxiv.org/abs/2609.33872)** (Liu et al., 2026-09-27) — Builds 3D Gaussian-splat + SAM3D scenes from RGB-D for simulation-before-execution, filtering infeasible action sequences; validated on rigid, soft and deformable object manipulation. · [code](https://robot-gst.github.io)
- **[Self-Adaptive VLA](https://arxiv.org/abs/2609.30092)** (UMass Amherst; Genesis AI, 2026-09-24) — A context token from failed trials recovers from hardware offsets. Actuation bias average 7.5% → 72.5% (up to 6 summed trials) vs nominal 88.8%.0%.
- **[Morphometric Imitation](https://arxiv.org/abs/2609.28660)** (Sadjadpour et al., 2026-09-23) — Contact-aware morphometric retargeting of human hand motion, RL refinement for dynamic feasibility, then visuomotor distillation; 89.3% zero-shot success over 300 real trials on 30 objects. · [code](https://morphometricimitation.github.io)
- **[InsertAnything](https://arxiv.org/abs/2609.24511)** (Ma et al., 2026-09-21) — Sim-only RL insertion policy using fingertip force feedback and decoupled gated rewards; 95.0% zero-shot on eight unseen real insertions (0.02 mm clearance), 20/20 on ManipulationNet peg-in-hole. · [code](https://mzhsoul.github.io/InsertAnything/)
- **[A Sim-to-Real Integration Pipeline for Training and Deployment of Chunk-Based VLA…](https://arxiv.org/abs/2609.21817)** (Kappel et al., 2026-09-18) — Replays simulated expert trajectories open-loop on a real Franka FR3 to record real observations, unifying data collection, closed-loop evaluation and sim-to-real gap measurement for chunk-based VLAs. · [code](https://gitlab.isir.upmc.fr/kappel/sim2real_public_chunk_control)
- **[DEXTERA](https://arxiv.org/abs/2609.21045)** (UT Austin, University of Florida, BrainCo, 2026-09-17) — Real-to-sim data engine from one photo. A 1:1 sim-real co-training mix raises mean real success from 29.2% (sim-only) to 61.9% across ACT, DP and pi0.5. There is no real-only baseline and no RL fine-tuning of the deployed policy.
- ⭐ **[DLS (Where Success Breaks)](https://arxiv.org/abs/2609.06114)** (Show Lab, National University of Singapore, 2026-09-05) — Finds pi0/pi0.5 failures in parallel twins, turns hand-defined task phases into a dense potential, and applies a critic-free contrastive velocity loss with a real-data anchor, all in sim. pi0.5 real ID/Unseen: 84.4/72.2 vs PPO 81.1/67.8, SFT 56.7/41.1.

## August 2026

- **[DREAM](https://arxiv.org/abs/2608.29078)** (Sato et al., 2026-08-29) — Reconstructs the deployment workspace in sim, translates instructions to symbolic goals via LLM, and uses TAMP to generate verified, randomized demos for VLA fine-tuning without human teleoperation.
- **[SCAPE](https://arxiv.org/abs/2608.19425)** (Zhu et al., 2026-08-19) — Predicts scenario-level real-world policy performance from few paired sim/real samples plus large sim rollouts with conformal calibration; error down 4.9-34.7% (driving), 14.5-27.7% (quadruped, Go2 validated).

## July 2026

- **[RoboSnap](https://arxiv.org/abs/2607.06699)** (Zhang et al., 2026-07-07) — Single RGB image to sim-ready scene (collision-aware foreground assets, 3DGS background); releases DROID-Sim from 564 real scenes for replay, synthetic data generation and sim-real correlated policy evaluation. · [code](https://robosnap.github.io)
- ⭐ **[SILO](https://arxiv.org/abs/2607.04616)** (NVIDIA; UC San Diego, 2026-07-06) — Trains PPO from scratch in sim on a capsule-chain cable for single-clip routing, then deploys through a 10 Hz twin whose sim joint angles become real targets. Real 3-clip routing, zero demos: 18/24 vs hierarchical IL 12/24.

## June 2026

- **[MoRE (Behavior Uncloning)](https://arxiv.org/abs/2606.29201)** (Texas A&M University and others, 2026-06-28) — A short fine-tune backpropagates a frozen mode classifier into policy weights to remove unwanted behavior modes. Push-T 50.0/54.0 → 98.3/100.0 avg/max SR. · [code](https://github.com/phai-lab/behavior-uncloning)
- **[SimFoundry](https://arxiv.org/abs/2606.28276)** (Ranawaka et al., 2026-06-26) — Zero-shot video-to-sim digital twins and affordance-preserving digital cousins; 0.911 mean Pearson sim-real correlation across 7 tasks and 5 policies, 17-40% success gains from variant training. · [code](https://research.nvidia.com/labs/gear/simfoundry/)
- ⭐ **[TacCoRL](https://arxiv.org/abs/2606.11743)** (UCLA, UC San Diego, UESTC, Peking University, University of…, 2026-06-10) — Adds gated tactile tokens to pi0.5, co-trains on real plus tactile-twin sim demos, then runs sim PPO with a real-data BC anchor. Real, 4 bimanual tasks: 72.5% visuo-tactile vs 50.0% vision-only; unanchored sim RL fell below co-training.
- ⭐ **[Video2Sim2Real](https://arxiv.org/abs/2606.08828)** (Georgia Institute of Technology, University of…, 2026-06-07) — Builds a sim twin from one RGB-D human video, re-optimizes three object keyframes, and trains only a PPO finger residual under domain randomization. Real: 95.7% (67/70) across 7 tasks, zero robot demos, vs 15.7% for a FoundationPose-based baseline. · [code](https://github.com/video2sim2real/video2sim2real)
- **[TTT-VLA](https://arxiv.org/abs/2606.03127)** (ByteDance Seed, 2026-06-02) — Reward-free test-time tuning of a latent prompt via a state-grounding proxy loss. SimplerEnv WidowX overall 51.1% → 67.4% (sim only; the architecture is retrained).

## May 2026

- **[COAST](https://arxiv.org/abs/2605.17144)** (University of Pennsylvania, 2026-05-16) — Gradient-free activation steering of a frozen VLA with conceptors fit on about 15 success/failure rollouts. Abstract: over 20 (sim) and 40 (real) absolute points.

## April 2026

- **[Rewind-IL](https://arxiv.org/abs/2604.16683)** (Vanderbilt, Waterloo, Sydney, 2026-04-17) — Training-free drift detector (TIDE, conformal threshold) plus respawn to a safe subtask boundary for a frozen ACT policy. About +13 points unperturbed and about +58 perturbed.

## March 2026

- ⭐ **[Sim-to-Real VLA Generalization Study](https://arxiv.org/abs/2603.22876)** (The Chinese University of Hong Kong, Shenzhen; Shenzhen…, 2026-03-24) — Controlled study of zero-shot sim-to-real for a VLA trained only in RoboTwin 2.0, isolating which randomization factors matter and adding binary-success GRPO in sim. Real average success: SFT 5.6% → SFT+RL 33.4% → SFT+RL+DR 42.8%.
- ⭐ **[Sim-to-Real VLA RL with Generative 3D Worlds](https://arxiv.org/abs/2603.18532)** (Horizon Robotics, 2026-03-19) — Generates 100 physical ManiSkill 3 scenes with a GPT-4o designer and EmbodiedGen, runs PPOFlow on an open pi0 BridgeV2 checkpoint across them, then transfers zero-shot. Real WidowX, 240 trials: 21.7% → 75%; the only baseline is imitation.
- ⭐ **[HandelBot](https://arxiv.org/abs/2603.12243)** (Stanford University; Amazon FAR, 2026-03-12) — Freezes the best sim piano rollout as an open-loop trajectory, nudges each finger's lateral joint from MIDI key errors, then adds bounded residual RL rewarded by MIDI. Real, 5-song mean key-press F1: 77.6 vs open-loop sim 41.8. · [code](https://github.com/amberxie88/handelbot)

## February 2026

- ⭐ **[What Matters for Sim-to-Online RL](https://arxiv.org/abs/2602.20220)** (ETH Zurich; Google DeepMind, 2026-02-23) — Empirical study of continuing sim-pretrained SAC on real robots: keep or warm-start replay, update the actor once per 20 critic steps at LR 1e-5, restore optimizer state. Franka vision grasping reaches near-perfect success in about ten real minutes. · [code](https://github.com/yardenas/panda-rl-kit)
- ⭐ **[RL-Co (RLinf-Co / Beyond Imitation)](https://arxiv.org/abs/2602.12628)** (Tsinghua University, Harbin Institute of Technology, Peking…, 2026-02-13) — Co-trains the VLA on real plus MimicGen sim demos, then runs PPO in a rough ManiSkill3 task copy with a weighted real-demo BC loss; no real-robot RL. Real average success, pi0.5: SFT co-training 45.9 → 66.2. · [code](https://github.com/RLinf/RLinf)
- **[RKO (Preference-aligned Diffusion Policy)](https://arxiv.org/abs/2602.09583)** (KTH; INCAR Robotics, 2026-02-10) — Offline DPO/KTO-style preference fine-tuning of a real cloth-folding diffusion policy with winning and losing demos. Trousers P1 score 0.701 (DDPM) → 0.910. Better filed under preference / weighted-BC methods.
- ⭐ **[TwinRL](https://arxiv.org/abs/2602.09023)** (Peking University, Tsinghua University, HKUST, Simplexity…, 2026-02-09) — SFTs Octo-Small on 30 real demos plus twin trajectories, runs RL in a phone-scanned twin to seed the real buffer, then real HIL-SERL-style RL. 4 real Franka tasks: near-100% success, at least 30% less real interaction than HIL-SERL. · [code](https://github.com/zhourui9813/TwinRL)
- **[RLinf-USER](https://arxiv.org/abs/2602.07837)** (Tsinghua University and others, 2026-02-08) — Infrastructure, not an algorithm. Async actor-learner system for real-robot online RL. Peg-insertion convergence drops from 8000+ s to about 1500 s; pi0 Pick-and-Place goes 39/60 → 58/60 with HG-DAgger. Better filed under real-world RL infrastructure. · [code](https://github.com/RLinf/RLinf)

## October 2025

- ⭐ **[VT-Refine](https://arxiv.org/abs/2510.14930)** (Columbia University, NVIDIA, UC San Diego, 2025-10-16) — BC-trains a visuo-tactile diffusion policy on about 30 real demos, fine-tunes it with DPPO and sparse success reward in an Isaac Gym twin simulating the same tactile pad, and redeploys. Real mean success: 0.50 → 0.85 (visuo-tactile). · [code](https://github.com/NVlabs/vt-refine)
- **[Online Imitation-Pretrained World Models](https://arxiv.org/abs/2510.02538)** (UC San Diego, UC Berkeley, UNC Chapel Hill, Sudo AI, 2025-10-02) — Reward-free online imitation RL pretrains a TD-MPC2-style world model in sim for coverage, then a 15-30 demo real fine-tune. Real 32/40 vs 15/40 for the best baseline (v2 numbers; v1 differed).

## Before September 2025

- **[SGFT](https://arxiv.org/abs/2502.02705)** (University of Washington, Microsoft Research, 2025-02-04) — Precursor. The frozen sim critic serves as a potential-based shaping reward and H-step terminal value for real-world SAC/TD-MPC2. 100% on hammering and pushing within about an hour, 70% on insertion within two hours. · [code](https://github.com/WEIRDLabUW/sgft)
- **[ACDC (Digital Cousins)](https://arxiv.org/abs/2410.07408)** (Stanford University, 2024-10-09) — Precursor. Sim 'cousin' scenes built from one image. Zero-shot real door opening: 25% for the exact twin vs 90% cousin-only and 95% twin+cousin. · [code](https://github.com/cremebrule/digital-cousins)
- **[RL-GSBridge](https://arxiv.org/abs/2409.20291)** (Shanghai Jiao Tong University and others, 2024-09-30) — Precursor. A 3DGS-rendered twin lets image-based SAC transfer zero-shot. small_cube grasping 96.88% sim → 96.88% real, vs 12.50% real for mesh renders. · [code](https://github.com/IRMV-Manipulation-Group/RL-GSBridge)
- **[SplatSim](https://arxiv.org/abs/2409.10161)** (Carnegie Mellon University, 2024-09-16) — Precursor. Gaussian-splat rendering over PyBullet physics. A zero-shot RGB diffusion policy averages 86.25% sim-to-real vs 97.5% real-to-real. · [code](https://github.com/qureshinomaan/SplatSim)
- **[RialTo](https://arxiv.org/abs/2403.03949)** (MIT, University of Washington, TU Darmstadt, 2024-03-06) — Precursor. Phone-scan twin, then PPO with a BC term starting from 15 demos, then distill back. Real success averages 91%/77%/75% (poses/distractors/disturbances) vs 25%/11%/5% for BC. · [code](https://github.com/real-to-sim-to-real/RialToPolicyLearning)
- **[Parameter Space Noise](https://arxiv.org/abs/1706.01905)** (OpenAI; KIT; UC Berkeley, 2017-06-06) — Classic precursor. Episode-fixed weight noise for coherent exploration. DQN with parameter noise beats ES on 15 of 21 Atari games with 25x less data. Belongs with exploration / noise-space methods. · [code](https://github.com/openai/baselines)
