# On-policy policy gradients

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*How do I take a BC or SFT policy (diffusion, flow, ACT or a 7B VLA) that succeeds 50-90% of the time and push it toward 95-100% using only a success signal, a resettable simulator, and fresh on-policy rollouts?*

Roll out the pretrained policy as a stochastic policy, raise the log-probability of rollouts that beat a baseline, and clip each step so the pretrained skill survives.

- **Loop step:** Improve
- **Built on:** Likelihood-ratio policy gradient; Trust region by clipping (PPO); Where the baseline comes from: critic vs group; Getting a log-prob out of an expressive policy; Anchoring to the pretrained policy; Signal needs variance
- **Use it when:** Use it when you have (1) a pretrained policy that already succeeds sometimes, roughly 10-90%, (2) a simulator or digital twin with cheap resets and an automatic success check, and (3) GPUs for millions of rollouts. It is the right tool for pushing a competent SFT policy past its imitation plateau (RL4VLA: more demos stopped helping at 0.781, PPO reached 0.938).
- **What changed in 2025–26:** Twelve months ago (2025-05 to 2025-07), the question was whether PPO/GRPO can train a VLA at all. RIPT-VLA, VLA-RL, RL4VLA, TGRPO, ReinFlow and FPO said yes, in simulation. From 2025-09 to 2025-10 this became standard recipes and infrastructure.

**62 entries**, newest first. ⭐ marks must-know work.


## September 2026

- **[Cooperative Multi-Agent Vision-Language-Action Models via Reinforced Fine Tuning](https://arxiv.org/abs/2609.36588)** (Xu et al., 2026-09-29) — Three-stage RL fine-tuning for multi-robot VLAs: initialization-aware data collection, credit-filtered offline tuning, online latent-space fine-tuning; +23.1% RoboTwin, +16.4% RoboFactory, +44% real dual-Franka.
- **[Action Chunking Proximal Policy Optimization with Feedback Correction](https://arxiv.org/abs/2609.36250)** (Hahn et al., 2026-09-28) — ACPPO: PPO over action chunks, plus ACPPO-Corr stepwise feedback corrector adjusting planned actions within each chunk; best results across 25 simulated locomotion, manipulation and dexterous tasks (NeurIPS 2026). · [code](https://github.com/hshhahn/ACPPO)
- **[WAM-OPD](https://arxiv.org/abs/2609.34250)** (Liu et al., 2026-09-28) — On-policy distillation of task-specific teachers into a pretrained world action model via prefix-weighted trajectory replay over a fixed rollout pool, improving target tasks without extra environment interaction.
- **[SEES](https://arxiv.org/abs/2609.32698)** (Li et al., 2026-09-26) — Decomposes long-horizon tasks into atomic skills, tracks which fail most, replays failure states in simulation with LLM-generated success criteria, and RL-fine-tunes the VLA over evolution rounds without extra demonstrations.
- **[Uncertainty-gated exploration noise](https://arxiv.org/abs/2609.28838)** (KIT, Cukurova University, 2026-09-23) — Multi-task PPO on SmolVLA can collapse individual tasks. A rule-based sigma controller has pooled mean 0.579 vs 0.433 for fixed noise and 0.249 for learned noise. No arm beats the BC baseline of 0.648.
- **[SynthDemo-RL](https://arxiv.org/abs/2609.21650)** (Kingetsu et al., 2026-09-18) — Privileged-state teacher synthesizes demos; VLA is supervised-fine-tuned then PPO-refined with binary reward, rescuing all 27 zero-reward LIBERO-PRO tasks (97.8%/97.1%) without human demos.
- **[EmbodiedMind](https://arxiv.org/abs/2609.19659)** (Wang et al., 2026-09-17) — Three-stage training of embodied foundation models: rejection-sampling fine-tuning, difficulty-queued Iterative Rejection GRPO, and Trie-GRPO with action-prefix-tree step-level advantages; 70.02% across 18 benchmarks.
- **[GR2PO](https://arxiv.org/abs/2609.19850)** (Wang et al., 2026-09-17) — Critic-free GRPO-style RL for continuous control: group-normalized discounted returns from parallel trajectories with clipping; competitive with actor-critic methods, deployed on Jetson TX2.

## August 2026

- **[WAM-OPD](https://arxiv.org/abs/2608.22364)** (Yang et al., 2026-08-23) — Frozen teacher WAM densely relabels histories induced by an accelerated student's own rollouts; Flash-WAM improves 0.0% to 58.3% on Handover Mic, 16.7% to 33.3% on Put Object Cabinet (task-specific).
- ⭐ **[ADEPT](https://arxiv.org/abs/2608.19182)** (NVIDIA; University of Michigan, 2026-08-19) — Post-trains a generalist PPO arm-hand policy per task: BC-distill the actor into new observations, warm up the critic with the actor frozen, then update at LR 1e-5. 5/5 seeds reach ADR 50 vs 0/5 for standard PPO.
- **[Prism-GRPO](https://arxiv.org/abs/2608.17423)** (Purdue, AWS AI, 2026-08-18) — Reward = success + 0.2*quality (contacts, jerk, VLM judge) to break zero-variance groups, with an RLOO advantage. Theory: binary GRPO needs ~1.76x the rollouts per informative group at G=8 (p=0.1/0.9).
- ⭐ **[StructRL](https://arxiv.org/abs/2608.15139)** (Fudan University; Singapore Management University; Shanghai…, 2026-08-15) — Runs the flow policy deterministically and adds exploration noise only to the output action, AR(1)-correlated within a chunk with separate learned scales for position, rotation and gripper. ManiSkill pi0.5 OOD average: SFT 25.9, piRL 49.3, StructRL 70.3.
- ⭐ **[Temporal GRPO](https://arxiv.org/abs/2608.13026)** (Institute of Software, CAS; UCAS, 2026-08-13) — A frozen VLM proposes checkable stages compiled into predicates; group advantages are computed per stage over rollouts that reached it, so good early actions escape blame for later failures. RoboTwin 2.0 macro-average: 75.8 vs SimpleVLA-RL 68.8 (SFT 38.3).
- **[RefineFly](https://arxiv.org/abs/2608.09467)** (Wang et al., 2026-08-10) — Token-level PPO post-training of UAV VLA with dynamic failure memory, two-stage scene curriculum and drift regularization; +3.12-8.37 pp success over AerialVLA on TravelUAV using ~30% rollout budget.
- **[Beyond Flat Policies](https://arxiv.org/abs/2608.05999)** (Kong et al., 2026-08-06) — Hierarchical VLA post-training: a planner decomposes tasks into subgoals; the subgoal-conditioned executor is aligned to planner outputs, then improved with online RL. Outperforms flat baselines.
- **[RL Bootstrapping of OpenVLA-OFT for a Novel Robot Embodiment](https://arxiv.org/abs/2608.01013)** (Nurtdinov et al., 2026-08-02) — Demo-free RL fine-tuning of OpenVLA-OFT for a cable-driven parallel robot: PPO on directional primitives, then GRPO with expanded instructions; directional success 34.25% to 53.50%.

## July 2026

- **[LfH (RL from Hindsight)](https://arxiv.org/abs/2607.09042)** (MIT, MIT-IBM, Stanford, UCSD, 2026-07-10) — A VLM relabels all-fail GRPO groups with a hindsight instruction, so failures give gradient. Matches GRPO's final gain in ~5 vs ~30 steps. Real FR3: 56% vs 22% at 160 rollouts.
- ⭐ **[PAC-ACT](https://arxiv.org/abs/2607.09590)** (SUSTech, 2026-07-10) — Runs PPO over ACT's 8-action chunks with a small Gaussian head, anchored to the BC policy by a KL penalty and a reward penalty. Sim Contour / Square Assembly: ACT (BC) 60.0% / 51.2% → 100.0% / 98.2%.
- ⭐ **[PDE (Prompt-Driven Exploration)](https://arxiv.org/abs/2607.08837)** (MIT Improbable AI Lab; MIT-IBM Computing Research Lab, 2026-07-09) — Explores in language: a VLM diagnoses failed videos and proposes reworded per-episode instructions, and PPO consolidates successes from the prompt pool into the canonical-prompt weights. LIBERO-PRO from 0% start: 81.8 at step 120; action noise stays 0.
- **[Offline supervision for VLA PPO (RefKL)](https://arxiv.org/abs/2607.19399)** (MIRAI + anonymized lab, 2026-07-06) — PPO plus an annealed KL to a frozen SFT policy. At 1M steps it matches plain PPO at 2M (IND 0.93, OOD 0.77) and beats SFT warm-start and a BC loss. · [code](https://github.com/alstar8/offline-supervision-vla-rl)

## June 2026

- **[Z-1](https://arxiv.org/abs/2606.31846)** (Zioneer Robot Team, 2026-06-30) — GRPO on pi0.5 where each group branches from one cloned approach-phase state, so credit lands on the manipulation phase. RoboCasa 24 tasks 67.4 → 80.6. TurnOnSinkFaucet 29.2 → 98.4.
- **[T^2VLA](https://arxiv.org/abs/2606.29892)** (Fudan, Shanghai Innovation Institute, MoShen, 2026-06-29) — GRPO whose reward is DTW similarity to the policy's own most-confident rollouts, with no success signal. OFT LIBERO 91.0 → 97.2 (oracle-reward reference 99.1). ECCV 2026. Code at github.com/tenirsss/T2VLA-repo.
- **[ROAD-VLA](https://arxiv.org/abs/2606.25800)** (UNSW, Deakin, AFRL, 2026-06-24) — Replaces PPO's policy loss with distillation to a closed-form, token-level advantage-tilted teacher. OOD 69+/-4 → 73+/-4 vs PPO, but not better on every ID cell, and 40.8 h vs 25.4 h compute.
- **[dVLA-RL](https://arxiv.org/abs/2606.23623)** (Wu et al., 2026-06-22) — RL fine-tuning for discrete-diffusion VLAs: treats denoising as an MDP and optimizes the sampled path's joint probability; 99.7% on LIBERO, +30.6% over SFT on RoboTwin 2.0.

## May 2026

- **[PAPO-VLA](https://arxiv.org/abs/2605.19580)** (Guo et al., 2026-05-19) — Treats a VLA as planner plus executor: identifies planning actions via action variation and trajectory outcome, weights them by causal sufficiency/necessity, and injects weights into GRPO advantage estimation.
- **[D-VLA](https://arxiv.org/abs/2605.13276)** (Guo et al., 2026-05-13) — Distributed async RL system for large VLAs: decouples data and weight traffic, four-thread swimlane pipeline, dual-pool VRAM management; higher throughput on LIBERO with near-linear scaling.

## April 2026

- **[OmniVLA-RL](https://arxiv.org/abs/2604.17706)** (Jie et al., 2026-04-20) — Mix-of-Transformers VLA (reasoning, spatial, action experts) with Flow-GSPO: flow matching recast as an SDE and trained with Group Segmented Policy Optimization; evaluated on LIBERO and LIBERO-Plus.
- **[ScoRe-Flow](https://arxiv.org/abs/2604.10962)** (Zhejiang U., SJTU, Shanghai Innovation Institute,…, 2026-04-13) — Adds a closed-form score drift that pulls flow steps toward demo-like actions. Robomimic average 92.5 vs ReinFlow-S 85.9 and DPPO 75.9. The '5.4%' claim is measured against the authors' own ablation.

## March 2026

- **[SmoothVLA](https://arxiv.org/abs/2603.13925)** (Li et al., 2026-03-14) — GRPO fine-tuning of a VLA with a hybrid reward: sparse task success plus dense jerk-based intrinsic smoothness term; 13.8% smoother than standard RL on LIBERO.
- **[REFINE-DP](https://arxiv.org/abs/2603.13707)** (Georgia Tech, 2026-03-14) — DPPO on a humanoid diffusion planner, alternating with PPO on the low-level controller. Sim 50-70% → >90%. Booster T1 hardware 70% / 50% / 75%.
- **[Beyond Imitation](https://arxiv.org/abs/2603.12868)** (Sheng et al., 2026-03-13) — GRPO (value-network-free) fine-tuning of an imitation-pretrained diffusion navigation policy with selective layer updates; unseen-scene success 52.0% to 58.7%, zero-shot transfer to a real quadruped.
- ⭐ **[Simple Recipe Works (continual VLA RL)](https://arxiv.org/abs/2603.11653)** (UT Austin, UCLA, NTU, Sony AI, 2026-03-12) — Fine-tunes a VLA on each new task in sequence with GRPO on sparse 0/1 success and a rank-32 LoRA adapter, with no replay or regularizer. Sim LIBERO-Spatial: AVG 81.2, forgetting 0.3, vs 7.3 and 40.9 with full fine-tuning. · [code](https://github.com/UT-Austin-RobIn/continual-vla-rl)
- **[pi-StepNFT](https://arxiv.org/abs/2603.02083)** (GigaAI, CASIA, UCAS, Tsinghua, 2026-03-02) — Critic-free, likelihood-free step-wise ranking on Flow-SDE rollouts. pi0 LIBERO 57.6 → 90.5, trailing piRL PPO at 96.0. ManiSkill OOD 50.4 vs PPO 39.3. · [code](https://github.com/wangst0181/pi-StepNFT)

## February 2026

- **[RL-VLA³](https://arxiv.org/abs/2602.05765)** (Sun et al., 2026-02-05) — Fully asynchronous distributed RL framework for VLA post-training (decoupled simulation, inference, training; dynamic batching, environment sharding); up to 85.2% higher throughput than synchronous baselines, scaling 8 to 256 GPUs. · [code](https://github.com/Haoran0301/RL-VLA3)
- **[Reparameterization Flow Policy Optimization](https://arxiv.org/abs/2602.03501)** (Zhong et al., 2026-02-03) — RFO trains flow policies with reparameterization policy gradients, backpropagating through ODE action generation and differentiable dynamics; almost 2x SOTA baseline reward on a soft-body quadruped task.
- ⭐ **[FPO++](https://arxiv.org/abs/2602.02481)** (Amazon FAR, UC Berkeley, Stanford, HKU, CMU, 2026-02-02) — Flow policy gradient using exp(old CFM loss - new CFM loss) as the PPO ratio, clipped per (tau, eps) sample, with zero-noise test-time sampling. Zero-sampling alone lifts the Can base policy from 10.00% to 71.35% before RL. · [code](https://github.com/amazon-far/fpo-control)

## January 2026

- **[SA-VLA](https://arxiv.org/abs/2602.00743)** (Wuhan University; A*STAR IHPC; NUS; CAS; NTU, 2026-01-31) — RL fine-tuning of flow-matching VLAs that preserves spatial grounding: spatial tokens fused with visual tokens, geometric progress dense rewards, SCAN spatially-conditioned exploration; better zero-shot spatial generalization on LIBERO/LIBERO-PLUS. · [code](https://github.com/TwSphinx54/SA-VLA)
- **[DMPO / OGPO](https://arxiv.org/abs/2601.20701)** (Sun Yat-sen University, 2026-01-28) — One-step MeanFlow policy with a dispersive loss, making PPO's ratio exact. Square 0.83 at 1 NFE vs DPPO 0.78 at 20 NFE. Without the BC anchor, success collapses to near zero. · [code](https://github.com/ogpo-project/OGPO)

## December 2025

- **[EVOLVE-VLA](https://arxiv.org/abs/2512.14666)** (NUS Show Lab, 2025-12-16) — Test-time GRPO with a pretrained progress critic (VLAC) instead of an oracle reward. LIBERO 89.2 → 95.8. 1-shot 43.6 → 61.3. · [code](https://github.com/showlab/EVOLVE-VLA)

## November 2025

- **[Discover, Learn, and Reinforce](https://arxiv.org/abs/2511.19528)** (Yang et al., 2025-11-24) — DLR makes RL generate multiple distinct high-success behavior patterns per task for VLA pretraining; beats equal-sized standard-RL data downstream and scales positively where single-pattern RL data does not.
- **[SpeedAug](https://arxiv.org/abs/2512.00062)** (KAIST, UNIST, DeepAuto.ai, 2025-11-24) — Tempo-augmented demos plus a sparse reward with gamma < 1 make RL optimize speed. Sim uses DPPO. The real result (11.4 → 20.76 successes/min in 16 min on an arm labeled SO-101) uses PA-RL, not PPO, and was added in the May 2026 v2. Card had major corrections (Table 1 column mapping, v2-only hardware result). · numbers corrected after check

## October 2025

- ⭐ **[piRL](https://arxiv.org/abs/2510.25889)** (RLinf team: Tsinghua, PKU, CASIA, CMU, Infinigence AI,…, 2025-10-29) — Makes pi0/pi0.5 flow chunks PPO-trainable by adding a learned per-step noise net (Flow-Noise) or an equal-marginal SDE (Flow-SDE) for exact log-probs, dropped at deployment. Few-shot LIBERO (sim): pi0 57.6% → 97.6%, pi0.5 77.1% → 98.3%. · [code](https://github.com/RLinf/RLinf)
- ⭐ **[RL-100](https://arxiv.org/abs/2510.14830)** (Shanghai Qi Zhi Institute, SJTU, HKU, IIIS Tsinghua, UNC,…, 2025-10-16) — BC-trains a diffusion policy on about 100 demos, then applies PPO-clip at every denoising step, offline with IQL advantages and an OPE gate, then online with GAE. Real, 8 tasks: 100 mean vs DP3 67.8; 1000/1000 trials. · [code](https://github.com/Lei-Kun/RL-100)
- **[Refinery](https://arxiv.org/abs/2510.11019)** (USC, NVIDIA, UW, U, 2025-10-13) — A GP over start states plus Bayesian optimization picks where PPO trains, and a GMM steers deployment. Per-step assembly success 83.61% → 96.35% in sim.
- **[FPO for pi0](https://arxiv.org/abs/2510.09976)** (Institute of Automation, CAS, 2025-10-11) — Freezes the pi0 decoder, trains a small latent flow actor with a CFM-loss ratio proxy and Q-ensemble GAE. LIBERO average 87.2 (no plain-pi0 row to isolate the gain). Ablation without the ratio proxy: 32.4 vs 78.5 (ICRA 2026).
- ⭐ **[RLinf-VLA](https://arxiv.org/abs/2510.06710)** (Tsinghua, Zhongguancun Academy, Infinigence AI, PKU, UC…, 2025-10-08) — Infrastructure plus recipe for VLA RL: simulators, rollout and trainer share GPUs, split them or run pipelined, with chunk-aware PPO and GRPO tricks. ManiSkill 25 tasks, ID/OOD: OpenVLA 53.91/39.10 → GRPO 84.38/75.15 → PPO 96.09/81.93. · [code](https://github.com/RLinf/RLinf)

## September 2025

- **[VLA Model Post-Training via Action-Chunked PPO and Self Behavior Cloning](https://arxiv.org/abs/2509.25718)** (Wang et al., 2025-09-30) — Action-chunked PPO plus auxiliary self-BC loss on a dynamically refreshed buffer of high-quality self-collected trials; 0.93 success and 42.17 steps-to-success on MetaWorld, beating supervised fine-tuning.
- **[Beyond Human Demonstrations](https://arxiv.org/abs/2509.19752)** (Yang et al., 2025-09-24) — Diffusion-policy RL autonomously generates LIBERO training data; a VLA trained on it reaches 81.9% average success, +5.3 over human demos and +12.6 over Gaussian-RL data.
- **[Self-Improving Embodied Foundation Models](https://arxiv.org/abs/2509.15155)** (Google DeepMind, 2025-09-18) — REINFORCE with a steps-to-go head that serves as both shaped reward and success detector. Real LanguageTable ~62-63% → ~87-88% with ~3% extra episodes (NeurIPS 2025). · [code](https://github.com/self-improving-efms/self-improving-efms.github.io)
- ⭐ **[SimpleVLA-RL](https://arxiv.org/abs/2509.09674)** (Tsinghua University, Shanghai AI Lab, SJTU, PKU, HKU, 2025-09-11) — GRPO on a tokenized-action OpenVLA-OFT: 8 rollouts per scene, binary success reward on every action token, group-normalized advantages, no critic, and all-agree groups dropped. LIBERO average 91.0 → 99.1; with one demo per task, 48.9 → 96.9. · [code](https://github.com/PRIME-RL/SimpleVLA-RL)

## Before September 2025

- **[D2PPO](https://arxiv.org/abs/2508.02644)** (Sun Yat-sen University, 2025-08-04) — Adds dispersive-loss regularization against diffusion representation collapse to DPPO; +22.7% in pre-training and +26.1% after PPO fine-tuning on RoboMimic, with Franka real-robot validation (AAAI 2026). · [code](https://github.com/Guowei-Zou/d2ppo-release)
- **[FPO](https://arxiv.org/abs/2507.21053)** (UC Berkeley, 2025-07-28) — PPO ratio = exp(L_CFM_old - L_CFM_new), with no likelihoods. DM Control average 759.3 vs 667.8 for Gaussian PPO and 652.5 for DPPO. · [code](https://github.com/akanazawa/fpo)
- **[RLRC](https://arxiv.org/abs/2506.17639)** (Chen et al., 2025-06-21) — Structured pruning of a VLA, recovery via SFT then RL (critic warm-up, BC-loss regularization), then quantization: up to 8x memory reduction, 2.3x speedup at original success rate. · [code](https://rlrc-vla.github.io)
- **[TGRPO](https://arxiv.org/abs/2506.08440)** (Jilin University, 2025-06-10) — GRPO mixing step-level and trajectory-level group z-scores, with an LLM-written dense reward. LIBERO average 80.7 vs SFT 76.5 (v3), and not best on Goal.
- **[ReinFlow](https://arxiv.org/abs/2505.22094)** (CMU, Tsinghua, 2025-05-28) — A learned noise net makes each Euler step Gaussian, giving an exact chain likelihood for PPO. Can 57.83% → 98.50%, Transport 30.17% → 88.67%, with 62.82% less wall time than DPPO across tasks. · [code](https://github.com/ReinFlow/ReinFlow)
- **[RL4VLA (What Can RL Bring to VLA Generalization?)](https://arxiv.org/abs/2505.19789)** (Tsinghua, 2025-05-26) — Controlled study where PPO beats GRPO and DPO. ID 0.781 (SFT-16k, plateaued) → 0.938. Robot-pose OOD 0.339 → 0.797. Zero-shot Franka pick-and-place 0.00 → 0.27. · [code](https://github.com/gen-robot/RL4VLA)
- **[VLA-RL](https://arxiv.org/abs/2505.18719)** (Tsinghua SIGS, NTU, 2025-05-24) — Token-level PPO on OpenVLA-7B with an auto-labeled process reward model. LIBERO average 76.5 → 81.0, which is still 4.5 below pi0-FAST despite the abstract's 'matches' claim. · [code](https://github.com/GuanxingLu/vlarl)
- **[RIPT-VLA](https://arxiv.org/abs/2505.17016)** (UT Austin, 2025-05-22) — RLOO + PPO clip + dynamic sampling on binary success. QueST LIBERO average 82.7 → 93.6. OFT 96.7 → 97.5. 1-demo cross-scenario 3.5% → 97.2%. · [code](https://github.com/Ariostgx/ript-vla)
- **[Refined Policy Distillation](https://arxiv.org/abs/2503.05833)** (UTN Nuremberg, 2025-03-06) — A small PPO student plus an MSE term toward a frozen VLA teacher's action. Even 1-10% success teachers make sparse-reward PPO work in ManiSkill3 (sim only). · [code](https://github.com/Refined-Policy-Distillation/RPD)
- **[FLaRe](https://arxiv.org/abs/2409.16578)** (Ai2, UT Austin, UW, Sony AI, 2024-09-25) — PPO on a BC transformer in AI2THOR with four stabilizers (on-policy, 10x lower LR, no entropy bonus, separate critic copy). 79.5% average success on CHORES-S, +23.6 absolute over the best baseline. · [code](https://github.com/JiahengHu/FLaRe)
- **[DPPO](https://arxiv.org/abs/2409.00588)** (Princeton, MIT, TRI, CMU, 2024-09-01) — PPO over diffusion denoising steps. Furniture-Bench sim One-leg 57% → 97%, Lamp 12% → 87%, Round-table 1% → 86%. Zero-shot hardware One-leg 80% (16/20) vs 0% for Gaussian PPO. · [code](https://github.com/irom-lab/dppo)
- **[DAPG](https://arxiv.org/abs/1709.10087)** (UW, OpenAI, UC Berkeley, IIT, 2017-09-28) — BC on 25 demos, then NPG with a decaying demo-likelihood term. Reaches 90% on Adroit tasks in 30-55 iterations (3.33-6.1 robot-hours), where sparse-reward RL from scratch never gets there on 3 of 4 tasks. · [code](https://github.com/aravindr93/hand_dapg)
