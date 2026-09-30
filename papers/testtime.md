# Test-time search & verifiers

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*My policy sometimes does the right thing but not reliably. Can I raise its success rate without retraining it, just by choosing better among its own samples at run time?*

Leave the policy's weights alone and spend extra compute at deployment: sample several candidate actions, score them with a critic, verifier, world model or monitor, and run the best one (or nudge the sampler toward it).

- **Loop step:** Improve at deploy time (no weight updates)
- **Built on:** Best-of-N with a verifier (sample, score, pick); KL-regularized optimum = guidance; Offline critics that are safe to argmax over; Lookahead with a model (shooting, MPPI, MCTS); Over-optimization and support; Runtime monitors and adaptive compute; Search, then distill (expert iteration)
- **Use it when:** Use it when (1) your policy is stochastic (diffusion, flow or sampled tokens) and pass@k is clearly above pass@1, i.e. it sometimes does the right thing, (2) fine-tuning the policy is expensive or risky (a large VLA, a shared checkpoint), and (3) you can afford some extra latency or can batch samples off a shared prefill.
- **What changed in 2025–26:** Five shifts in the last 12 months. (1) From generic to on-policy verifiers. V-GPS (2024) used a generic offline-RL Q, and RoboMonkey (2025-06) a 7B verifier on synthetic preferences.

**89 entries**, newest first. ⭐ marks must-know work.


## September 2026

- **[RoboHarn-Evo](https://arxiv.org/abs/2609.37583)** (Bao et al., 2026-09-29) — Dual-loop harness evolves task- and action-level knowledge from rollouts with the VLM agent frozen; up to +24.2 points on RMBench, 70.0% to 88.3% held-out after 80 rollouts.
- **[Spotter](https://arxiv.org/abs/2609.36808)** (Li et al., 2026-09-29) — Policy runs autonomously while a VLM monitors in parallel and intervenes only on detected errors; +5.6-7.5 pts on RoboCasa, pi0.5 47.2% to 57.0% on RoboTwin 2.0 Hard, real robot 53% to 83%. · [code](https://github.com/zqc3117/Spotter)
- **[Taming VLAs under Robot Execution Errors](https://arxiv.org/abs/2609.37334)** (Lee et al., 2026-09-29) — Deployment-time adaptation lets VLAs pre-compensate for commanded-vs-executed motion errors; introduces RoboStress sim benchmark (friction, backlash, compliance); over 30-point real-robot gains on worn arms.
- ⭐ **[PreferenceFlow](https://arxiv.org/abs/2609.36872)** (Westlake University, 2026-09-29) — Pairs each human takeover window with a frozen pi0.5 chunk from the same state, trains a small Bradley-Terry scorer on the pairs, and uses it to guide sampling. Real Franka insertion: 90.5 vs frozen 69.0, full fine-tune 84.5.
- **[Test-Time Adaptation of Manipulation Policies Under Actuator Degradation](https://arxiv.org/abs/2609.36182)** (Sagar et al., 2026-09-28) — TeAR: lightweight Transformer rectifies a frozen policy's actions using live actuator telemetry (temperature, current, voltage); 31.8% vs 25.6% under degradation, +10-15% on hardware without on-robot fine-tuning.
- **[Scouting the Dynamics Gap](https://arxiv.org/abs/2609.36107)** (Li et al., 2026-09-28) — SCOUT meta-learns a belief latent shared by policy and forward dynamics model; at test time it updates the belief from dynamics prediction error, enabling fast adaptation and sim-to-real transfer (CoRL 2026). · [code](https://liy1shu.github.io/SCOUT/)
- **[AGF (Adjoint Guidance Flow)](https://arxiv.org/abs/2609.34944)** (KAIST; SKKU; GIST, 2026-09-28) — Amortized adjoint guidance net replaces per-step critic gradients. Real pooled 66.7 (base) / 68.0 (QGF) / 84.0 (AGF); best-of-N with Q collapsed on MolmoAct2 (e.g. 10.4 vs 75.4).
- **[Test-Time Spatial Reasoning for Robot Manipulation Using Generative Real-to-Sim](https://arxiv.org/abs/2609.33982)** (Kapelyukh et al., 2026-09-27) — Simify reconstructs sim-ready 3D assets from RGB-D and optimizes object placements in massively parallel physics simulation at test time for unseen-object rearrangement; IROS 2026.
- **[Recursive Harness Distillation across Agents for Robot Manipulation](https://arxiv.org/abs/2609.33378)** (Kim et al., 2026-09-27) — Strong agent distills VLA-intervention experience into a playbook for a light agent, refined recursively from execution feedback; real-world success 37.3% to 64.0%, SimplerEnv Bridge 41.7% to 66.7%.
- **[ActionGround](https://arxiv.org/abs/2609.33256)** (Chandra et al., 2026-09-27) — Training-free neuro-symbolic runtime layer (phase FSM plus inertia-weighted Euler-Lagrange term) on frozen VLAs; up to +6 points success, +19.3 stability, ~10x jerk robustness, <1 ms overhead.
- **[RoboFoundry](https://arxiv.org/abs/2609.32862)** (Liang et al., 2026-09-26) — Treats the embodied agent system (context management, hierarchical skills) as an evolving policy updated from execution; improves GPT-5.5 by 27.8% on EmbodiedBench, large gains on LIBERO-PRO. · [code](https://jingsongliang.com/robofoundry)
- **[RoboMonitor](https://arxiv.org/abs/2609.30715)** (Ajith et al., 2026-09-25) — Runtime execution monitor pretrained with action-conditioned future-feature prediction, inverse dynamics and masked prediction; 93.1% phase accuracy from 52 labeled episodes, 35/40 real trials with zero false recoveries.
- **[SafeLoop](https://arxiv.org/abs/2609.26313)** (Lou et al., 2026-09-22) — Non-invasive wrapper predicts collision/object-failure risk from vision and proprioception, then checkpoints or rolls back and re-queries the frozen VLA; ~70% fewer hazards on 24 LIBERO and 3 real tasks. · [code](https://github.com/Loule0-0/SafeLoop/tree/release/safeloop)
- **[D-JEPA](https://arxiv.org/abs/2609.24749)** (Liu et al., 2026-09-21) — Latent world model that learns ordinal relations among competing candidate futures from executed outcomes via a permutation-equivariant operator; 87.89% on PushT, +15.04 points on RoboTwin, +17 on real robots. · [code](https://nebulis-lab.com/D-JEPA)
- **[Beyond Visual Quality](https://arxiv.org/abs/2609.24745)** (Yuan et al., 2026-09-21) — Studies test-time candidate-action selection with world action models: oracle selection would lift success 68.9% to 79.2%, but tested selectors recover only modest gains; opportunity concentrates in few decisions.
- **[Latent Policy Steering](https://arxiv.org/abs/2609.22521)** (Wang et al., 2026-09-18) — Optical-flow embodiment-agnostic world model steers base policy via latent-space search; with 50 target demos, +16%/+8% sim and +62%/+14% real for Diffusion Policy/Pi0.5.
- **[VLA-Scope](https://arxiv.org/abs/2609.21246)** (Zhu et al., 2026-09-18) — Two-stage runtime monitor: detects and classifies input shift, then logistic regression on shift category, action-prefix features and progress predicts VLA failure; 0.945 OOD and 0.850 failure ROC-AUC on LIBERO-Spatial.
- ⭐ **[SeeQ](https://arxiv.org/abs/2609.22085)** (Carnegie Mellon University, 2026-09-18) — Trains a critic that only values finishing the active subtask (reward at subtask end, no bootstrapping across boundaries) and reranks 8 samples from a frozen pi0.5 with it. Real bimanual, 24 trials per task: 35.4% → 66.7% average. · [code](https://github.com/saksham002/generalist-value-functions)
- **[Catch Me If You Can](https://arxiv.org/abs/2609.21022)** (Ji et al., 2026-09-17) — VLA-Feedback: low-frequency diffusion planning with the final denoising step kept as a high-frequency visual feedback interface; dynamic sim tasks 27.5%→85.0%, real robot 51%→73% (CoRL 2026). · [code](https://vla-feedback.github.io)
- **[TraceFlow](https://arxiv.org/abs/2609.20646)** (HKU; SUSTech, 2026-09-17) — KDE scores from retrieved success and failure traces guide a frozen pi0.5, with no critic. Real ordered fruits TSR 21/50 → 39/50, then 47/50 after one more round. · [code](https://github.com/zhangjiaxuan-Xuan/TraceFlow)
- **[RAFAIL](https://arxiv.org/abs/2609.18324)** (Schneider et al., 2026-09-16) — Runtime failure detection via point-cloud anomaly detectors on task-relevant entity relationships, labeled offline by a VLM from successful demos only; 73.4% balanced accuracy on three real tasks.
- **[Causal-History Test-Time Scaling for Failure Recovery in Autoregressive World-Action…](https://arxiv.org/abs/2609.18016)** (Li et al., 2026-09-16) — Training-free failure recovery for autoregressive world-action models: progress-aware trigger detects stalls, rolls causal KV history back to a reliable prefix, then verifies three recovery hypotheses before committing.
- **[Lexicographic steering](https://arxiv.org/abs/2609.15014)** (MIT AeroAstro, 2026-09-14) — Dynamic-barrier guidance enforces priority-ordered costs on a frozen diffusion policy. Navigation success 44.3 → 56.0 vs selection-only 48.0.
- **[WorldAgen](https://arxiv.org/abs/2609.08162)** (Wan et al., 2026-09-08) — Shared transformer with world-model and action heads (mixed unidirectional attention mask); at deployment samples exploratory actions and test-time-trains the world model on collected transitions; CALVIN, LIBERO gains.
- **[VLA-Corrector](https://arxiv.org/abs/2609.06508)** (Song et al., 2026-09-06) — Learned verifier estimates stage-wise progress and risk from observations, proprioception and actions, triggering stage-conditioned recovery prompts for a frozen VLA; improves LIBERO/LIBERO-Plus robustness without parameter updates.
- **[FailureSpot](https://arxiv.org/abs/2609.04277)** (Ma et al., 2026-09-03) — Timestamp-level VLA failure detector trained with weak labels from anomalous action chunks (inconsistent, idle, random motions) plus active learning on uncertain trajectories, cutting annotation cost across multiple VLAs.
- **[World-Coherent Decoding](https://arxiv.org/abs/2609.02159)** (Zhang et al., 2026-09-02) — Samples futures from a frozen World Action Model, ranks by flow-based video surprisal and action effort, trains an online predictor from execution mismatch; RoboTwin 2.0 Hard 55.80% to 60.90%.
- **[Knowing When to Stop](https://arxiv.org/abs/2609.00908)** (Not specified on arXiv abs page, 2026-09-01) — Training-free adaptive execution horizon for VLAs: detects high-entropy plateaus in action-to-observation cross-attention to decide when to replan; improves success on RoboTwin 2.0 and LIBERO.

## August 2026

- **[R^3](https://arxiv.org/abs/2608.26053)** (Wu et al., 2026-08-26) — Post-trains a VLM as a robotic reasoner: mid-training on expert reasoning traces, then rubric-based RL on offline action data; language reasoning steers low-level policies at test time. · [code](https://robotic-reasoner.github.io/)
- **[RA-VLA](https://arxiv.org/abs/2608.25585)** (POSTECH, 2026-08-26) — Training-free VLA adaptation via behavior-aligned demonstration retrieval plus grounded execution to overcome pretrained-prior inertia; improves success and efficiency on LIBERO and a real UR5e (ICML 2026).
- ⭐ **[Q-Planning](https://arxiv.org/abs/2608.21204)** (Georgia Institute of Technology, 2026-08-21) — Keeps the policy frozen as a proposal, scores 32-64 candidate chunks with a separate ~1B Q-function, runs their softmax-Q-weighted average, and retrains only the critic. Real robot, 5 iterations: stack-cups 40% → 90% vs 55% for full-policy SFT. · [code](https://github.com/varungiridhar/qplanning-code)
- **[Reuse Before You Retrieve](https://arxiv.org/abs/2608.17484)** (Jeong et al., 2026-08-18) — Diagnoses frozen VLAs via recoverable headroom (resampling/retry selection) versus retrieval complementarity (external demos); gains up to 21 points on LIBERO track measured headroom.
- **[tau0-VLA](https://arxiv.org/abs/2608.16885)** (Shanghai Innovation Institute; Agibot; CUHK, 2026-08-17) — Confidence-gated beam search over next subtasks, scored with a world model and value model. Real Book Organization 6/10 → 9/10. · [code](https://github.com/sii-research/tau-0-vla)
- **[QWM](https://arxiv.org/abs/2608.17163)** (Stanford; Peking University, 2026-08-17) — World-model lookahead used only for action selection inside EXPO/RLPD online RL. Learning curves only; e.g. Tool Hang about 1.0 vs IDQL about 0.06 (read from figures).
- ⭐ **[CoRe](https://arxiv.org/abs/2608.14822)** (Case Western Reserve University, 2026-08-14) — Flags out-of-distribution states via a Mahalanobis test on the action-expert latent, rewinds to a recovery anchor, imagines the frozen policy's moves, and plans minimal object restorations; training-free. LangSwitch Switch average: pi0 18.3 → 87.5, pi0.5 35.5 → 92.7. · numbers corrected after check
- **[Self-Evolving Embodied Agents via Skill-Harness Evolution](https://arxiv.org/abs/2608.11350)** (Wang et al., 2026-08-11) — SHAPER: training-free self-improvement that keeps model weights frozen and evolves reusable skills plus a context-code harness; compared to fine-tuning and test-time scaling on VLABench and ESI-Bench. (adjacent)
- **[VANE](https://arxiv.org/abs/2608.09448)** (Ji et al., 2026-08-10) — Test-time prompt adaptation for VLAs: candidate updates are isolated, checked against future visual observations, and committed only if supported; +3.2 points over baseline TTT on SimplerEnv WidowX.

## July 2026

- **[RL2-VLA](https://arxiv.org/abs/2607.26991)** (NUS; University of Toronto, 2026-07-29) — Offline-RL head on VLA latents is blended into the velocity to diversify candidates for verifier best-of-N, gated by a SAFE failure detector. PolaRiS pi0.5 OOD average 14.3 → 42.7 (rephrase 31.8). · [code](https://github.com/marmotlab/RL2-VLA)
- **[GRACE](https://arxiv.org/abs/2607.21661)** (Park et al., 2026-07-22) — Steers pretrained diffusion policies with MPPI forward cost evaluations instead of gradients, handling non-differentiable constraints; avoids deployment-time obstacles on a real 7-DoF arm.
- **[PriGo](https://arxiv.org/abs/2607.07076)** (Li et al., 2026-07-08) — Lightweight PANet predicts motion primitives and applies differentiable guidance to pretrained diffusion/flow policies at inference, no retraining; gains on LIBERO, CALVIN, SIMPLER and real-robot tasks.
- **[SVA (Search, Value, and Act)](https://arxiv.org/abs/2607.03751)** (Nanjing University; ANU; CASIA, 2026-07-04) — Distills simulator MCTS into a critic for best-of-N over a frozen VLA. Motivating diagnostic: pass@1 33% vs pass@32 92%. EmbodiedBench +15.4 (Habitat) and +13.2 (Nav).
- **[DreamSteer](https://arxiv.org/abs/2607.02865)** (Meta FAIR; University of Minnesota, 2026-07-03) — DINOv2-space world model plus progress critic picks among pi0 samples and fixed motion primitives. Real OOD objects 23.75% → 66.25% (42.50% with policy samples only).
- **[VLA-Corrector](https://arxiv.org/abs/2607.01804)** (Pan et al., 2026-07-02) — Training-free plug-in for chunked VLAs: a latent-space vision monitor detects drift mid-chunk, truncates execution, and replans with online gradient guidance, giving an event-triggered adaptive action horizon.
- **[Guided Action Flow](https://arxiv.org/abs/2607.02092)** (UCL, 2026-07-02) — IQL critic on 100 logged rollouts of a frozen SmolVLA guides the flow sampler. Real bottle-to-box 19/40 → 34/40; no other baselines.
- **[FAR](https://arxiv.org/abs/2607.01111)** (CMU, 2026-07-01) — Failure-aware retry: an IQL value localizes the bad chunk, then a few Diffusion-DPO steps between attempts. Updates weights, so it fits better with preference fine-tuning (see misfiled).

## June 2026

- **[ELASTIC](https://arxiv.org/abs/2606.31132)** (CMU, 2026-06-30) — SAC meta-policy splits compute between denoising steps and number of samples, per state. Real pi0.5 matches Best-of-10 with 34% less latency; a hand-picked fixed schedule did about as well.
- **[E-TTS](https://arxiv.org/abs/2606.27268)** (CASIA, 2026-06-25) — Joint sampling of reasoning traces and actions scored by reasoning and action verifiers, with critique-and-resample. Real MolmoAct 22.13% → 48.75%; sim E-CoT 6.67 → 39.81 vs naive best-of-N 23.61.
- **[Simulator-in-the-loop cloth refinement](https://arxiv.org/abs/2606.24552)** (NUS; SJTU, 2026-06-23) — Real-to-sim cloth mesh estimate plus MPPI in a GPU cloth simulator around the base plan. Real single-arm fold 3/10 → 9/10, dual-arm 1/10 → 8/10.
- **[Robot Critics that Sweat the Small Stuff](https://arxiv.org/abs/2606.21572)** (Columbia; TRI, 2026-06-19) — Fine-tuned Qwen2.5-VL pairwise progress critic from binary success labels. Real Stacking 23/50 → 32/50, Push-Bowl mIoU 0.140 → 0.321. () · [code](https://github.com/SruthiSudhakar/robocritic) · numbers corrected after check
- **[VERITAS](https://arxiv.org/abs/2606.18247)** (Princeton, 2026-06-16) — Training-free verifier scores chunks against a Gemini-drawn 2D path, then success-filtered BC. Real bars average 36.5% → 58% (paper states 35%); the fine-tuned policy stays below steering. · [code](https://github.com/princeton-prism/veritas)
- **[ViTaL](https://arxiv.org/abs/2606.14981)** (CMU, 2026-06-12) — Vision picks the mode (world model plus VLM reward), then tactile-reward guidance refines contact. Appendix: wiping 30 → 90%, pipette 30 → 75%, insertion 40 → 80%. () · numbers corrected after check
- **[VeriSpace](https://arxiv.org/abs/2606.10568)** (CASIA, 2026-06-09) — 3D-aware 7B verifier over k=8 VLA samples. SimplerEnv OpenVLA 37.0 → 55.0 vs RoboMonkey 40.5, V-GPS 36.0, MG-Select 39.0.
- ⭐ **[QGF (Q-Guided Flow)](https://arxiv.org/abs/2606.11087)** (UC Berkeley; Physical Intelligence, 2026-06-09) — Freezes a BC flow policy and IQL critic; each flow step adds grad Q, evaluated at a one-step estimate of the finished action, to the velocity. OGBench sim, charts only: beats all tested test-time methods, slightly edges EDP. · [code](https://github.com/zhouzypaul/qgf)
- **[ProbeAct](https://arxiv.org/abs/2606.09740)** (Zhang et al., 2026-06-08) — Training-free runtime loop: feature probe for 3D object positions, kinematic failure state machine, and CBF action filter; lifts OpenVLA-OFT from 69.6% to 74.1% on LIBERO-plus.
- **[ActProbe](https://arxiv.org/abs/2606.08508)** (Huang et al., 2026-06-07) — Black-box failure detector using temporal consistency error between action chunks and chunk magnitude; +12.7% hypervolume, +9.0% early-detection ROC-AUC on unseen tasks, 2.9x fewer RL fine-tuning interactions. · [code](https://air-embodied-brain.github.io/actprobe)

## May 2026

- **[How VLAs Fail Differently](https://arxiv.org/abs/2605.28726)** (Gupta, 2026-05-27) — 450 PushT/ALOHA episodes across VQ-BeT, Diffusion Policy, ACT: action direction reversals predict failure (AUROC 0.79-0.93), velocity violations barely do; training-free SafeContract monitor with conformal calibration. · [code](https://github.com/krishnam94/vla-edge)
- **[TapSampling](https://arxiv.org/abs/2605.25547)** (Harbin Institute of Technology, 2026-05-25) — Action-VAE posterior makes cheap extra candidates and a progress verifier picks one. CALVIN avg length DP 2.41 → 2.58, OpenVLA 3.30 → 3.51; small gains. · [code](https://github.com/aipixel/TapSampling)
- **[Think Twice, Act Once](https://arxiv.org/abs/2605.12620)** (Singhi et al., 2026-05-12) — VeGAS: MLLM embodied agent samples multiple candidate actions and a verifier trained on LLM-synthesized failure cases picks one; up to 36% relative gain over CoT baselines on Habitat and ALFRED.
- ⭐ **[VLA-ATTC](https://arxiv.org/abs/2605.01194)** (University of Sydney; Central South University; USTC;…, 2026-05-02) — Samples two chunks and, when their DTW distance is high, samples 16 and picks one via a knockout tournament judged by a label-free relative action critic. LIBERO-LONG, 50 trials/task: pi0 82.8% → 90.6% gated (92.2% ungated); RoboMonkey 56.5%.

## April 2026

- ⭐ **[FASTER](https://arxiv.org/abs/2604.19730)** (Stanford University, 2026-04-21) — A critic predicts, from the raw noise seed, the denoised action's Q-value, so only the best seed gets denoised. pi0.5 on 5 libero_90 tasks vs EXPO: inference FLOPs 37.55 TF → 4.70 TF, success comparable. · [code](https://github.com/alexanderswerdlow/faster)

## March 2026

- ⭐ **[UF-OPS](https://arxiv.org/abs/2603.10282)** (University of Toronto; Google DeepMind; University of…, 2026-03-10) — Trains a small verifier on labeled evaluation rollouts of the deployed policy, then reranks 10 samples from the frozen diffusion or flow policy. Real ALOHA, 5 tasks, 20 trials each: mean success 38% → 87% (chart reading).
- **[OmniGuide](https://arxiv.org/abs/2603.10052)** (UPenn, 2026-03-09) — Test-time guidance of VLA action sampling via differentiable attractor/repeller energy fields from 3D foundation models, semantic reasoners and human pose estimators; improves success and safety in sim and real. · [code](https://omniguide.github.io/)

## February 2026

- ⭐ **[CoVer](https://arxiv.org/abs/2602.12281)** (Stanford University; NVIDIA Research, 2026-02-12) — Keeps pi0 frozen, samples K instruction rephrasings times M action chunks, and picks one with a CLIP-style contrastive verifier trained on BridgeV2. SIMPLER red-teaming, ID/OOD: pi0 41.5/29.7, pi0 fine-tuned on augmented data 44.0/48.7, pi0 (rephrase) + CoVer 65.5/62.0. · [code](https://github.com/cover-vla/cover-vla)
- **[VGAS](https://arxiv.org/abs/2602.07399)** (AAII, University of Technology Sydney, 2026-02-07) — Chunk-level critic with an expected-max-over-N backup and geometric regularization reranks 8 SmolVLA samples. LIBERO 5-shot 39.8 → 49.0; a plain Q-chunking critic (37.9) fell below BC.) · [code](https://github.com/Jyugo-15/VGAS)
- ⭐ **[VLS](https://arxiv.org/abs/2602.03973)** (University of Washington; AI2; University of Oxford; NUS, 2026-02-03) — Training-free: a VLM extracts 3D keypoints and writes differentiable per-stage rewards whose gradients guide each denoising step of a frozen diffusion or flow policy. CALVIN movable objects: 0.94 vs DynaGuide 0.26, base DP 0.13. · [code](https://github.com/Vision-Language-Steering/code)

## January 2026

- **[Cosmos Policy](https://arxiv.org/abs/2601.16163)** (NVIDIA; Stanford, 2026-01-22) — Video-model policy that is also a world model and value function. Best-of-N planning refit on 648 real rollouts raised hard ALOHA tasks from 59.5 to 72.0 average score. Mainly a policy architecture. · [code](https://research.nvidia.com/labs/dir/cosmos-policy/)
- **[CycleVLA](https://arxiv.org/abs/2601.02295)** (University of Oxford; University of Cambridge, 2026-01-05) — Progress/stop heads plus a VLM check at subtask boundaries; backtrack and retry with MBR consensus over 8 chunks. LIBERO-Plus pi0.5 65.0 → 86.0; LIBERO 96.9 → 98.5. · [code](https://github.com/dannymcy/cyclevla_code)
- **[V-VLAPS](https://arxiv.org/abs/2601.00969)** (Ren et al., 2026-01-02) — Value head trained on offline rollouts guides MCTS over VLA actions; matches baseline at default budget, +6 pts LIBERO-Object and +4 pts LIBERO-10 with larger search budgets.

## December 2025

- ⭐ **[TACO](https://arxiv.org/abs/2512.02834)** (TeleAI; USTC; Tsinghua, 2025-12-02) — Samples M chunks from a flow VLA and runs the one with the highest pseudo-count in the fine-tuning data, from a small Coin Flipping Network on internal features. RoboTwin 2.0 (pi0.5, 41 tasks, M=50): 59.3 → 64.0. · [code](https://github.com/breez3young/TACO)

## November 2025

- ⭐ **[JITI](https://arxiv.org/abs/2511.22555)** (Jilin University; Microsoft Research Asia, 2025-11-27) — Trains an offline Cal-QL critic on frozen VLA features from labeled demo segments; only when its Q-value jumps from its moving average are N chunks reranked. LIBERO-Elegant, SmolVLA: 49.8 → 67.2 ESR vs 53.8 reranking every step.
- **[VLA-Pilot](https://arxiv.org/abs/2511.14178)** (CUHK, 2025-11-18) — GPT-4o writes a keypoint scoring function, then evolutionary search with truncated re-noising inside a diffusion VLA's samples. Real DiVLA 0.31 → 0.62. · [code](https://rip4kobe.github.io/vla-pilot/)

## October 2025

- **[RISE](https://arxiv.org/abs/2510.19495)** (University of Washington; TRI, 2025-10-22) — SQIL-style 0/1 labels on expert vs play/failure data train an IQL critic that picks among 64 chunks. Real lampshade BC 17.5% → 82.5%. · [code](https://github.com/UWRobotLearning/RISE)
- **[RoVer](https://arxiv.org/abs/2510.10975)** (SIAT; Peng Cheng Lab, 2025-10-13) — Process reward model scores candidates and predicts a direction for local search, trained on perturbed expert actions only. Real Dobot 72.9% → 88.6%.
- **[MG-Select](https://arxiv.org/abs/2510.05681)** (KAIST; SNU; RLWRLD, 2025-10-07) — Verifier-free: picks the sample whose tokens diverge most (KL) from a masked-condition copy of the same VLA. RoboCasa 30 demos 5.3% → 14.2% (MG-Select*); SIMPLER 46.9 → 50.3.
- **[Compose Your Policies (GPC)](https://arxiv.org/abs/2510.01068)** (HKU; Shanghai AI Lab; SJTU, 2025-10-01) — Convex mix of two policies' scores at each denoising step, with the weight grid-searched on rollouts. E.g. 86.39 vs 78.84 for the better parent on Robomimic+PushT. · [code](https://github.com/SageCao1125/GPC)

## September 2025

- **[VLA-Reasoner](https://arxiv.org/abs/2509.22643)** (Nanyang Technological University, 2025-09-26) — MCTS with a KDE action prior, a video world model and a progress value on a frozen VLA. LIBERO OpenVLA 76.0 → 81.0; real OpenVLA 22% → 41%. · [code](https://github.com/wkguo/VLA-Reasoner)

## Before September 2025

- **[Improving Pre-Trained Vision-Language-Action Policies with Model-Based Search (VLAPS)](https://arxiv.org/abs/2508.12211)** (Neary et al., 2025-08-17) — Embeds VLA-prior-guided Monte Carlo Tree Search with an environment model into pretrained VLA (Octo) inference; raises success by up to 67 percentage points over VLA-only baselines. · [code](https://github.com/cyrusneary/vlaps)
- **[Latent Policy Steering](https://arxiv.org/abs/2507.13340)** (CMU, 2025-07-17) — Cross-embodiment-pretrained Dreamer world model plus critic ranks BC chunks. Real Franka, 30-50 demos: BC 21.2% → 36.2%.
- **[RoboMonkey](https://arxiv.org/abs/2506.17811)** (Stanford; UC Berkeley; NVIDIA, 2025-06-21) — 7B VLM verifier trained on 20M synthetic preferences reranks Gaussian-perturbed OpenVLA samples. Real OOD 60% vs OpenVLA 35% and V-GPS 30%. · [code](https://github.com/robomonkey-vla/RoboMonkey)
- **[DynaGuide](https://arxiv.org/abs/2506.13922)** (Stanford, 2025-06-16) — A DINOv2-space dynamics model's gradient guides a frozen diffusion policy toward goal images. CALVIN articulated parts 70% average, 8.7x over base. · [code](https://github.com/MaxDu17/DynaGuide)
- **[SAILOR](https://arxiv.org/abs/2506.05294)** (Mila; CMU; Cornell, 2025-06-05) — MPPI residual search in a learned world model with an IRL reward and critic, then distillation. Sim only; e.g. Square 0.29 → 0.78 at 50 demos. · [code](https://github.com/arnavkj1995/SAILOR)
- **[Hume](https://arxiv.org/abs/2505.21432)** (Shanghai AI Lab; SJTU; Fudan; INSAIT, 2025-05-27) — pi0-style VLA with a built-in value-query head that picks among 5 partly denoised chunks, plus a fast refiner. LIBERO 98.6% vs pi0 94.2%. · [code](https://github.com/hume-vla/hume)
- **[FOREWARN](https://arxiv.org/abs/2502.01828)** (CMU, 2025-02-03) — Latent world model plus fine-tuned VLM verify 6 plan modes in language. Real Franka Fork task 0.10 → 0.70. · [code](https://github.com/CMU-IntentLab/Forewarn)
- **[GPC (Generative Predictive Control)](https://arxiv.org/abs/2502.00622)** (Harvard; Georgia Tech, 2025-02-02) — World model plus reward predictor rank diffusion-policy chunks. State Push-T 0.812 → 0.934 (K=5000); vision Push-T 0.642 → 0.739 vs V-GPS 0.620. · [code](https://github.com/han20192019/gpc_code)
- **[ITPS](https://arxiv.org/abs/2411.16627)** (MIT CSAIL; NVIDIA, 2024-11-25) — Human clicks, sketches or nudges become a cost that steers a diffusion policy. Compares ranking, guidance and stochastic sampling. Steers intent, not competence. · [code](https://github.com/yanweiw/itps)
- **[V-GPS](https://arxiv.org/abs/2410.13816)** (UC Berkeley; CMU, 2024-10-17) — Offline Cal-QL Q reranks K samples from frozen generalists. Real WidowX Octo-small-1.5 0.24 → 0.44 (+82%); SIMPLER gains are small for some policies (OpenVLA 0.24 → 0.27). · [code](https://github.com/nakamotoo/V-GPS)
- **[Bidirectional Decoding (BID)](https://arxiv.org/abs/2408.17355)** (Stanford, 2024-08-30) — Training-free selection among N chunks by coherence with the last chunk and contrast with a weak checkpoint. 32% average relative gain over vanilla closed loop on 7 sim tasks. · [code](https://github.com/YuejiangLIU/bid_diffusion)
- **[IRASim](https://arxiv.org/abs/2406.14540)** (HKUST; ByteDance Seed, 2024-06-20) — Action-conditioned video world model picks among K chunks. Push-T IoU 0.637 → 0.961 (K=50, 1000 rollouts), but with no rollout data (P=0) more samples hurt (0.418). · [code](https://github.com/bytedance/IRASim)
- **[IDQL](https://arxiv.org/abs/2304.10573)** (UC Berkeley, 2023-04-20) — Frozen diffusion behavior policy plus IQL critic, with actions chosen by best-of-N. D4RL total 1213.8 vs IQL 1070.4. The template for propose-then-select. · [code](https://github.com/philippe-eecs/IDQL)
