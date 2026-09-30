# Reward, value & progress models

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*My robot only gets a success bit at the end of an episode, or a human has to watch every rollout. How do I get a dense, trustworthy score so RL, filtered BC or advantage conditioning can push the policy from about 60-80% toward 95%+?*

These methods leave the policy alone and build the score it climbs against: a learned or zero-shot judge that says how far along a rollout is, whether it succeeded, and where it went wrong.

- **Loop step:** Measure → Find failures
- **Built on:** Success classifier as reward; Progress from time (time-as-label); Preference and ranking rewards (Bradley-Terry); Potential-based shaping; Value model for advantage-weighted updates; VLM as a zero-shot judge; Reward hacking and pessimism; Measuring a reward model before spending robot time
- **Use it when:** Use this family when success is sparse or expensive to judge and the task is long or multi-stage. Examples: cloth folding, insertion sequences, anything where a rollout that gets 80% of the way looks the same as one that fails at once under a success bit.
- **What changed in 2025–26:** Four shifts stand out. (1) Scale and generality. Per-task rankers (Rank2Reward, VIP-style) gave way to VLM reward models trained on large pooled datasets: Robo-Dopamine on about 3,400 hours, Robometer on RBM-1M (over 1M trajectories), RynnValue on over 7,000 hours.

**65 entries**, newest first. ⭐ marks must-know work.


## September 2026

- **[TaRL](https://arxiv.org/abs/2609.36785)** (Wu et al., 2026-09-29) — Learns rewards by regressing task progress from tactile deformation maps of successful and failed demos; lifts nut threading 34% to 56% (sim) and real cube pickup 37% to 97%. · [code](https://embodiedai-ntu.github.io/tarl)
- **[Video2STL](https://arxiv.org/abs/2609.37519)** (Atasever et al., 2026-09-29) — VLM converts observation-only videos into parametric Signal Temporal Logic specs giving dense short-horizon and long-horizon progress rewards for RL; 85.8% success-once, 67.0% success-at-end on manipulation. · [code](https://video2stl.github.io/)
- **[StructRL](https://arxiv.org/abs/2609.36352)** (Penn State, Amazon AWS AI, 2026-09-28) — Prerequisite-gated, paced rewards from simulator-verified subtask predicates (not a learned model); RoboCasa365 GR00T-N1.5 49.1% vs 41.5% for the best terminal-reward baseline, and it beats Robometer on LIBERO-Long (96.6 vs 94.2). · [code](https://github.com/amazon-science/StructRL)
- **[ARS](https://arxiv.org/abs/2609.34484)** (AIRC, Midea Group, 2026-09-28) — An agentic VLM builds verified event timelines as offline progress labels; LIBERO-Long 57.9 vs 44.2 vanilla BC; exposes reward models that credit the wrong object.
- **[Learning Robot Policies from Sparse Success Signals via STL-Guided Stein Variational…](https://arxiv.org/abs/2609.31606)** (Zheng et al., 2026-09-25) — STL-SVPG: population-based Stein variational policy gradient using smooth STL robustness as a trajectory-level objective; best success on 5 of 6 quadcopter/manipulator benchmarks, with sim-to-real transfer.
- **[PHIRL](https://arxiv.org/abs/2609.31855)** (Yu et al., 2026-09-25) — IRL framework calibrating learned rewards with task-progress annotations; higher return and task success with only 20% of demonstrations annotated, and robust to reward hacking.
- **[HiRE](https://arxiv.org/abs/2609.27068)** (UC Berkeley, CUHK, 2026-09-22) — A training-free reward from DINOv2 success and failure buffers (KDE log-ratio); real bottle pick-and-place 20% to 67% where sparse and Robometer nearly fail. · [code](https://github.com/HiRE-Project/HiRE)
- **[Scaling Vision-Language Reward Learning for Robot Manipulation in Parallel Simulation…](https://arxiv.org/abs/2609.21767)** (Joualy et al., 2026-09-18) — RAPID: GPU-parallel IsaacLab rollouts plus single-request VLM preference labeling for reward learning; 8.0x end-to-end speedup, 95.5% fewer API calls, success 86.3% to 98.7% on five Franka tasks. · [code](https://github.com/rapid-vlm/rapid-vlm-rl)
- **[AnyviewMeter](https://arxiv.org/abs/2609.20106)** (Tu et al., 2026-09-17) — LoRA-adapts pretrained progress reward models with Plucker-ray camera conditioning and synchronized multi-view block attention; 41-69% lower progress error than averaged single-view, ~21% lower MAE on real tasks.
- **[FIERCE](https://arxiv.org/abs/2609.18651)** (NTU, Guangzhou Cloudbutterfly, 2026-09-16) — A progress plus failure-risk evaluator shapes SAC for a distilled specialist; real mean 92.0% vs 83.0% with Robometer-guided RL after 300 rollouts (single seed). · [code](https://github.com/ar-mine/FIERCE)
- **[Intrinsic Robot Rewarding](https://arxiv.org/abs/2609.17115)** (Schaffer et al., 2026-09-15) — Reuses the VLA's frozen visual encoder plus a reference bank of successful demonstration endpoints as a built-in reward/evaluator, avoiding a separate reward model; COMAU Racer 3 demonstrator at TRL 4.
- **[No Free Checker (survey)](https://arxiv.org/abs/2609.09250)** (Zhejiang University, CityU Hong Kong, 2026-09-08) — A survey of about 150 robot verifiers: credibility falls as availability rises, and no learned verifier has been tested against adversarial optimization. · [code](https://github.com/ZJUSCL/Awesome-Robot-Verifier)
- ⭐ **[Dyna: reward model as failure detector](https://daily.dev/posts/robot-demos-are-easy-reliability-is-hard-jason-ma-dyna-robotics-qiwax5jjh)** (Dyna Robotics, 2026-09) — Uses a learned video progress model as a monitor: progress dips flag errors during deployment and steer targeted recovery-data collection and retraining. Napkin folding held 99.4% success across four 24-hour trials, up from about 80% (talk, no paper).

## August 2026

- **[Beyond Pairwise Feedback](https://arxiv.org/abs/2608.25350)** (Katkuri et al., 2026-08-26) — VLM ranks K=3-5 candidate trajectories at once; a Plackett-Luce listwise loss trains the reward model, reaching 86% mean final success on Meta-World versus pairwise Bradley-Terry baselines.
- **[Dream2Reward](https://arxiv.org/abs/2608.18787)** (Zhang et al., 2026-08-19) — Learns reward from success-only demos by predicting expected transitions and scoring observed displacement on direction and magnitude agreement; penalizes wrong-direction and overshoot, reducing reward hacking in online/offline RL.
- **[Robo-Dopamine 2.0](https://arxiv.org/abs/2608.15680)** (Peking University, EvoPhys AI and others, 2026-08-16) — Adds rollout-history panels and OOD failure edits to the pairwise hop model; real repeated insertion 71/80 vs 48/80 for event-sparse reward.
- **[PRM-as-a-Judge 1.5](https://arxiv.org/abs/2608.14284)** (Liu et al., 2026-08-14) — Evaluation toolkit that turns rollout videos into dense progress curves with three metrics (failure-side progress, recovery, success-side execution quality); adds RoboPulse++ to validate process reward models. · [code](https://prm-as-a-judge.github.io)
- ⭐ **[RynnValue](https://arxiv.org/abs/2608.09853)** (DAMO Academy, Hupan Lab, 2026-08-10) — Predicts seconds until task completion from an instruction and a few frames, trained on over 7,000 hours of video with no preference labels. As reward for real online DSRL on frozen pi0.5: 72.5% vs Robometer 52.5%, sparse 48.8%. · [code](https://github.com/alibaba-damo-academy/RynnValue)
- **[ValueFormer](https://arxiv.org/abs/2608.02958)** (Chef Robotics, 2026-08-03) — A 3.5M-parameter causal critic on DINOv3 for AWR reweighting and mistake detection; sandwich completion 70% to 85% (n=20, Fisher p=0.45, not significant).

## July 2026

- **[WCM (World Critic Model)](https://arxiv.org/abs/2607.29613)** (Tongji, Shanghai Innovation Institute, Fudan, 2026-07-31) — A history-aware critic trained with latent next-state prediction; beats a single-frame VLM critic on all 7 real RECAP/AWR tasks (best 44/50) and reaches 99.0% on ManiSkill IND. · [code](https://github.com/sylvestf/WCM)
- **[Deployable Human Preference Alignment in Robotics: Learning Representative Rewards from…](https://arxiv.org/abs/2607.12466)** (Kim et al., 2026-07-14) — PREC clusters users into preference-coherent groups from binary preference labels and learns one reward model per cluster, beating a single shared policy in simulated locomotion.
- **[DenseReward](https://arxiv.org/abs/2607.13033)** (UNC Chapel Hill, CMU, SJTU, Amazon, 2026-07-14) — Failures synthesized in sim train a Qwen3-VL-4B progress model (MAE 0.081 vs 0.230 for RoboReward-8B); real DSRL stack cups 60% sparse to 80% (10 trials).
- **[UR-VC](https://arxiv.org/abs/2607.12892)** (The University of Hong Kong, 2026-07-14) — Training-free correction of t/T labels via cross-episode nearest neighbours; advantage-conditioned cloth folding 72.8% (131/180) to 78.9% (142/180), vs a no-advantage baseline.
- **[Rank-Then-Act](https://arxiv.org/abs/2607.01897)** (T-Tech, 2026-07-02) — A GRPO-trained VLM ranks shuffled frames and the reward is rank agreement; games and sim only (Catrap L2 1.00 vs GVL 0.47). · [code](https://github.com/corl-team/rank-then-act)

## June 2026

- **[WARP-RM](https://arxiv.org/abs/2606.28320)** (Yu et al., 2026-06-26) — Self-supervised relative-progress reward model trained on time-warped (speed-varied, reversed) demos; used to filter/reweight BC data, giving ~18x successful T-shirt-folding throughput over vanilla BC. · [code](https://uynitsuj.github.io/warp-rm/)
- **[RMTL](https://arxiv.org/abs/2606.26175)** (Ateş et al., 2026-06-24) — Splits long-horizon tasks into language-described micro-tasks, computing multi-view VLM rewards per active micro-task with hierarchical RL and reverse curriculum; faster learning than single-prompt VLM rewards on Fetch.
- **[Beyond Monotonic Progress](https://arxiv.org/abs/2606.24633)** (Qin et al., 2026-06-23) — ReTVL treats retry events in imperfect demos as supervision, combining global progress calibration with local pairwise preferences to learn a non-monotonic value model that reweights demonstration segments for BC.
- **[World Value Models for Robotic Manipulation](https://arxiv.org/abs/2606.24742)** (Wang et al., 2026-06-23) — World Value Model: uses a world model rather than a VLM for value estimation; SOTA Value-Order Correlation, adds Suboptimal-Value-Bench (800 annotated suboptimal trajectories), improves sim and real policy learning.
- ⭐ **[SVM (Success Visitation Matching)](https://arxiv.org/abs/2606.23640)** (UC Berkeley, 2026-06-22) — Trains a classifier on the policy's own successful vs failed rollouts and adds its clipped log-odds to the sparse success reward. Real WidowX with DSRL, Pick and Place: about 0.93 vs about 0.2 for outcome-only RL (plot reading).
- ⭐ **[RARM](https://arxiv.org/abs/2606.22027)** (National University of Singapore, Booking.com, University…, 2026-06-20) — A comparator trained on internet video matches rollout clips to one successful demo, advancing progress only on confident matches; no robot data. Real pi0 with DSRL, cloth folding at 90 rollouts: 25/30 vs binary reward 7/30, Robometer 9/30. · [code](https://rarm-robotics.github.io/)
- ⭐ **[SARM2 + SPIRAL](https://arxiv.org/abs/2606.10305)** (Stanford, UC Berkeley, SJTU, xdof.ai, 2026-06-09) — Classifies the active action primitive to gate a mixture-of-experts value head regressing normalized remaining steps, used as reward for a residual TD3+BC policy on frozen pi0.5. Real, Round 3 Whiteboard: 18/20 vs Robometer 13/20, sparse 11/20. · [code](https://github.com/Qianzhong-Chen/openspiral)

## May 2026

- **[Feat2Go](https://arxiv.org/abs/2605.30795)** (Shu et al., 2026-05-29) — Value model trained on progress targets from world-model patch similarity to subgoals reshapes rewards for PPO/GRPO VLA fine-tuning; OpenVLA-OFT OOD success 17.5% to 82.9% on ManiSkill3.
- **[ProcVLM](https://arxiv.org/abs/2605.08774)** (Renmin University, Zhipu AI, Tsinghua, 2026-05-09) — Procedure-grounded progress labels (subtask budgets plus visual change) in a 2B VLM; real stack-bowls soft success 37.5 to 62.5 at 5k steps (12 rollouts). · [code](https://github.com/RUCKBReasoning/ProcVLM)

## April 2026

- ⭐ **[ViVa](https://arxiv.org/abs/2604.08168)** (GigaAI, Sichuan University, Tsinghua University, 2026-04-09) — Builds the RECAP value model from the Wan2.2 video generator, jointly predicting future proprioception and a return separating successes [0,1) from failures [1,2). Real, three tasks, two rounds: 80.0% average success vs 63.3% with a VLM value. · [code](https://github.com/GigaAI-research/ViVa)
- **[ARM](https://arxiv.org/abs/2604.03037)** (LimX Dynamics, BUPT, Zhejiang University, 2026-04-03) — Tri-state (+1/0/-1) relative-progress labels credit recovery; advantage-weighted BC folds towels at 99.4% vs 78.5% with SARM weights and 62.1% plain BC (trial count not reported; CVPR 2026 Workshops). · [code](https://github.com/limxdynamics/FluxVLA)

## March 2026

- **[VLLR](https://arxiv.org/abs/2604.00055)** (CMU, Amazon Robotics, UT Austin, 2026-03-31) — A VLM progress reward used briefly to warm-start the critic, then policy self-certainty as a bonus; PickUp 97.0% vs 91.8% for sparse-reward FLaRe (sim).
- ⭐ **[SOLE-R1](https://arxiv.org/abs/2603.28730)** (MIT, RAI Institute, 2026-03-30) — An 8B video-language model reasons briefly each timestep, then outputs progress in [-100, 100] as the only reward for off-policy RL from scratch. Zero-shot online RL: solves 24 of 40 tasks (>=50% success) vs 7 for GPT-5. · [code](https://github.com/Philip-MIT/sole-r1-model)
- **[Large Reward Models (LRM)](https://arxiv.org/abs/2603.16065)** (USC PSI Lab, TRI, 2026-03-17) — Three Qwen3-VL-8B reward heads (contrastive, progress, completion); PPO on pi0.5 in ManiSkill goes from 56.88 to at most 60.93, vs 66.87 with privileged reward. · [code](https://github.com/physical-superintelligence-lab/Large-Reward-Models)
- ⭐ **[Robometer](https://arxiv.org/abs/2603.02115)** (USC, UT Dallas, MIT, University of Washington, Ai2, NVIDIA, 2026-03-02) — Qwen3-VL-4B reward model with progress, success and video-preference heads, trained on RBM-1M including failures; used as reward, terminator and stage switcher. Real DSRL on frozen pi0: 20% → 85% in about 40 minutes, 0 false-positive successes (RoboReward: 45). · [code](https://github.com/robometer/robometer)

## February 2026

- **[Keyframe-Guided Structured Rewards for Reinforcement Learning in Long-Horizon…](https://arxiv.org/abs/2603.00719)** (Qiu et al., 2026-02-28) — Extracts kinematics-aware keyframes from demos, diffusion-predicts stage targets, builds geometric progress rewards for online VLA fine-tuning; 82% after 40-60 min on four lab tasks vs HG-DAgger 42%, HiL-ConRFT 47%.
- ⭐ **[TOPReward](https://arxiv.org/abs/2602.19313)** (University of Washington, Allen Institute for AI, Amazon,…, 2026-02-22) — Reads a frozen Qwen3-VL-8B's log-probability of 'True' that a video prefix completes the task, giving a training-free progress curve. ManiRewardBench (113 tasks): VOC 0.942 vs 0.316 for GVL on the same model and 0.858 for the trained Robometer-4B. · [code](https://github.com/TOPReward/TOPReward)

## January 2026

- ⭐ **[RoboReward](https://arxiv.org/abs/2601.00675)** (Stanford University, UC Berkeley, 2026-01-02) — Fine-tunes Qwen3-VL to score full rollouts 1 to 5 against a rubric, creating negatives by LLM counterfactual relabeling; offline error predicts RL value (r=0.83). Real WidowX DSRL, open drawer, 20 trials: base 10%, RoboReward 8B 80%, oracle 90%. · [code](https://crfm.stanford.edu/helm/robo-reward-bench)

## December 2025

- ⭐ **[Robo-Dopamine](https://arxiv.org/abs/2512.23703)** (Peking University, BAAI, University of Sydney, CASIA, 2025-12-29) — A VLM reward model predicts signed progress 'hops' from multi-view images, fused into a potential used as policy-invariant shaping, not raw reward. 8 real tasks: 95.2% final success vs 68.0% with sparse reward (ConRFT) and 9.8% for BC. · [code](https://github.com/FlagOpen/Robo-Dopamine)

## November 2025

- **[SRPO](https://arxiv.org/abs/2511.15605)** (Tongji, Shanghai Innovation Institute, Fudan, 2025-11-19) — Scores failed rollouts by V-JEPA 2 latent distance to the batch's own successes; one-shot OpenVLA* on LIBERO goes from 48.9 to 99.2 with online GRPO. · [code](https://github.com/sii-research/siiRL)

## October 2025

- **[Reflective Self-Adaptation](https://arxiv.org/abs/2510.12710)** (Peking University, JD Explore, Tsinghua, 2025-10-14) — GPT-4o composes dense rewards from failure videos for PPO on OpenVLA; LIBERO average 83.6% vs 81.0% for VLA-RL; without success-filtered SFT it collapses to 0% from reward hacking.

## September 2025

- **[TimeRewarder](https://arxiv.org/abs/2509.26627)** (Tsinghua, Shanghai Qi Zhi, SJTU, UPenn, 2025-09-30) — Signed frame-pair temporal distance as dense reward; near-perfect success on 9/10 Meta-World tasks in 200k interactions (sim, from scratch). · [code](https://github.com/CowAndSheep/TimeRewarder)
- **[SARM](https://arxiv.org/abs/2509.25358)** (Stanford, UC Berkeley, xdof.ai, 2025-09-29) — Stage-aware progress labels from subtask segmentation; reward-weighted BC on pi0 gets 10/12 medium and 8/12 hard T-shirt folds vs 1/12 and 0/12 for plain BC; available in LeRobot. · [code](https://github.com/xdofai/opensarm)
- ⭐ **[VLAC](https://arxiv.org/abs/2509.15937)** (Shanghai AI Laboratory, 2025-09-19) — Fine-tunes InternVL into a pairwise critic outputting signed progress between two frames plus a done signal, labeled from video temporal order, and uses it for real-arm PPO. Four real tasks: about 30% → about 90% within 200 episodes. · [code](https://github.com/InternRobotics/VLAC)

## Before September 2025

- **[SAFE](https://arxiv.org/abs/2506.09937)** (University of Toronto, Vector, TRI, 2025-06-11) — A tiny failure-detector head on frozen VLA hidden states; unseen-task ROC-AUC 78.00 vs 73.93 for the best baseline in sim. · [code](https://github.com/vla-safe/SAFE)
- **[RFTF](https://arxiv.org/abs/2505.19767)** (Peking University, 2025-05-26) — A temporal-ranking value model as potential-based shaping in PPO on CALVIN; small gains, and sparse-reward PPO pushed strong BC below its start.
- **[ReWiND](https://arxiv.org/abs/2505.10911)** (USC, KAIST, 2025-05-16) — A language-conditioned progress reward with rewind augmentation from about 5 demos per task; real bimanual policy 12% to 68% in 1 hour of RL. · [code](https://github.com/rewind-reward/ReWiND)
- **[PROGRESSOR](https://arxiv.org/abs/2411.17764)** (UChicago, TTIC, 2024-11-26) — Predicts a Gaussian over progress from start, current and goal frames, with 'push-back' on the agent's own rollouts against reward hacking. · [code](https://github.com/ripl/progressor)
- **[GVL](https://arxiv.org/abs/2411.04549)** (Google DeepMind, UPenn, Stanford, 2024-11-07) — Zero-shot Gemini progress on shuffled frames; introduced VOC (RT-1 0.74); AWR gains on real ALOHA were small and uneven (4 wins, 1 tie, 2 losses). · [code](https://generative-value-learning.github.io/)
- **[GCR](https://arxiv.org/abs/2410.19989)** (RAI Institute, Northeastern, 2024-10-25) — A VIP-style progress embedding kept honest with hard negatives from the agent's own failed episodes; solves about twice as many tasks as baseline reward learners.
- **[Video-Language Critic (VLC)](https://arxiv.org/abs/2405.19988)** (Aalto University and others, 2024-05-30) — A CLIP video-text critic with a temporal ranking loss; on Meta-World, mean success 78% vs 58% with sparse reward alone. · [code](https://github.com/minttusofia/video_language_critic)
- **[Rank2Reward](https://arxiv.org/abs/2404.14735)** (MIT, UW, Bristol, 2024-04-23) — Temporal ranking of demo frames plus a GAIL discriminator for pessimism; real xArm tasks learned in under 2 hours. · [code](https://github.com/dxyang/rank2reward)
- **[RoboFuME](https://arxiv.org/abs/2310.15145)** (Stanford, 2023-10-23) — A MiniGPT-4 success detector as reward for reset-free Cal-QL fine-tuning; the 5-task average goes from 50.4% offline to 76.2% after 2-4 h.
- **[LIV](https://arxiv.org/abs/2306.00958)** (UPenn, Meta AI, 2023-06-01) — VIP plus CLIP InfoNCE gives language-conditioned value and reward from EPIC-KITCHENS. · [code](https://github.com/penn-pal-lab/LIV)
- **[SuccessVQA](https://arxiv.org/abs/2303.07280)** (DeepMind, UC Berkeley, 2023-03-13) — A fine-tuned Flamingo 3B as a yes/no success detector; strong in-domain (94-99% on gear insertion), weaker out of distribution.
- **[VIP](https://arxiv.org/abs/2210.00030)** (Meta FAIR, UPenn, 2022-09-30) — A goal-conditioned value embedding from Ego4D via a dual RL objective; VIP-weighted RWR gets 100/90/60/90% on 4 real tasks. · [code](https://github.com/facebookresearch/vip)
- **[DVD](https://arxiv.org/abs/2103.16817)** (Stanford, 2021-03-31) — A 'same task?' video classifier trained on human plus robot videos gives a reward from a single human demo; used with visual MPC, not RL. · [code](https://github.com/anniesch/dvd)
- **[Reward sketching + batch RL](https://arxiv.org/abs/1909.12200)** (DeepMind, 2019-09-26) — Human-drawn progress curves train a ranking-loss reward that relabels a 400+ hour robot log for offline distributional RL. · [code](https://github.com/deepmind/deepmind-research/tree/master/sketchy)
- **[VICE-RAQ](https://arxiv.org/abs/1904.07854)** (UC Berkeley, 2019-04-16) — A success classifier from 80 goal images plus 25-75 yes/no queries; 100% real success on pushing, draping and bookshelf, where a naive classifier gets 0%. · [code](https://github.com/avisingh599/reward-learning-rl)
- **[Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)** (OpenAI, DeepMind, 2017-06-12) — A Bradley-Terry reward from clip comparisons, refit online; 700 human labels nearly match true-reward RL on 8 MuJoCo tasks.
- **[Unsupervised Perceptual Rewards](https://arxiv.org/abs/1612.06699)** (Google Brain, 2016-12-20) — A staged progress reward from frozen Inception features and a few human demo videos drives real door-opening RL; the earliest 'frozen features + few demos' progress reward.
