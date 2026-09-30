# Black-box search

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*How do I improve a policy when all I can do is run it and get a score, for example a success flag, and I cannot or do not want to backprop through it?*

Treat the policy, or a few knobs around it, as a vector. Try perturbed copies, score each with one number per episode, and move toward the copies that scored well. No backprop through the policy, no critic, no action log-probs.

- **Loop step:** Improve
- **Built on:** Perturb, evaluate, step (random search / finite differences / ES); Reward-weighted averaging (PoWER, PI2, CEM, MPPI); Adapt the search distribution (CMA-ES); Bayesian optimization (BO); Quality-diversity (MAP-Elites) as a prior; Parameter-space vs action-space exploration; Black-box search inside modern methods
- **Use it when:** Use it when (1) you only have a scalar, trajectory-level score such as success, success times time, or pass/fail, (2) the policy's log-probs are unavailable or expensive (diffusion, flow, action chunks, deterministic controllers, closed APIs), and (3) you can shrink the search to a small space: controller gains, impedance, a noise vector, a residual or last-layer head, a prompt. Pick BO for tens of expensive real trials over a few knobs.
- **What changed in 2025–26:** Three things changed in the last 12 months. (1) Scale: ES went from 'works for small MLPs' to full-parameter fine-tuning of billion-parameter models. C155 (Sep 2025, ICML 2026) beat tuned PPO/GRPO on Countdown with a population of 30.

**37 entries**, newest first. ⭐ marks must-know work.


## September 2026

- **[Encore](https://arxiv.org/abs/2609.37359)** (Kang et al., 2026-09-29) — Coding agent analyzes demo evidence packs (keyframes, gripper events, trajectories), iteratively refines a policy program, then freezes it; 96.3% vs 89.3% prior agentic system on LIBERO-PRO; real bimanual tasks.
- **[Explore, Execute, Evolve](https://arxiv.org/abs/2609.37810)** (Xie et al., 2026-09-29) — RoboSkill: multimodal agent loop that explores, executes with vision/tactile feedback, and evolves a reusable code-skill library; LIBERO-10 first-episode success +12.5-25.0 points, real robot +8.3 points.
- **[Behavioral Foundation Models for Quality Diversity](https://arxiv.org/abs/2609.35615)** (Bendib et al., 2026-09-28) — BFM-QD runs quality-diversity search in a behavioral foundation model's compact latent space, giving a gradient-free, critic-free improvement operator that beats parameter-space QD on sparse/deceptive continuous control (NeurIPS 2026).
- **[Evolving Dexterous Robots from Scratch](https://arxiv.org/abs/2609.33101)** (Guo et al., 2026-09-27) — Evolution strategies plus RL co-design dexterous manipulator bodies from scratch, using a contrastively learned genetic embedding and autoregressive developmental decoder; manufactured real prototypes pick, hold, rotate, use tools.
- **[Multi-Objective Human-in-the-Loop Bayesian Optimization of a Lower-Limb Exoskeleton](https://arxiv.org/abs/2609.30695)** (Janwani et al., 2026-09-25) — MO-HILBO: multi-objective human-in-the-loop Bayesian optimization inferring personalized Pareto-optimal exoskeleton controllers over metabolic cost and comfort; Pareto orderings consistent with validation trials; open-source mohilo package. · [code](https://dynamicmobility.github.io/mohilo/)
- **[RACaP](https://arxiv.org/abs/2609.29394)** (Li et al., 2026-09-24) — Moves code generation to an offline curriculum self-evolution phase that improves typed Policy APIs and task memory; a ReAct loop calls frozen APIs at deployment; 54.4% on LIBERO-90.
- **[HarnessPAI](https://arxiv.org/abs/2609.29166)** (Wang et al., 2026-09-24) — Evolves a program-level harness across rollouts from execution feedback, no retraining; +61.6 points over pi0.5 on LIBERO-PRO; fine-tuning on harness-collected data adds 38.8 points. · [code](https://darwin-agent.github.io/HarnessPAI)
- **[AdaHVLA](https://arxiv.org/abs/2609.29204)** (Tang et al., 2026-09-24) — Multi-agent loop (analyze, revise, assess) refines the code harness around frozen VLAs from rollout experience via a stateful revision graph; NaVILA-LH 22.5% to 57.5%, manipulation up to +30.8 points.
- **[Privacy-Preserving Prompted Policy Search for Robotic Control](https://arxiv.org/abs/2609.30554)** (Irshayyid et al., 2026-09-24) — LLM-as-optimizer policy search that sends only encoded policy parameters and scaled rewards to cloud LLMs; beats the baseline prompted search on 7 of 10 control tasks.
- **[Context-Continuous Preference Learning for Exoskeleton Personalization](https://arxiv.org/abs/2609.28427)** (Sunin Baek, Sungwoo Park, Daekyum Kim, 2026-09-23) — Shares preference feedback across smoothly varying operating contexts to personalize exoskeleton assistance; about 37% (ankle) and 17% (elbow) less feedback than independent per-context learning, 9-participant study.
- **[RegenHarness](https://arxiv.org/abs/2609.27612)** (Wang et al., 2026-09-23) — Robot agent harness with isolated planning/supervision/verification/recovery contexts and a commit gate; evidence-gated self-improvement revises harness configuration from execution records without weight updates; quadruped warehouse demo.
- **[Know Your Body](https://arxiv.org/abs/2609.28530)** (Lou et al., 2026-09-22) — KnowBody: frozen-VLM control harness with an explicit, revisable body model seeded from one trajectory and refined by interaction; 75% vs 25% completion for native harness on four real-robot tasks. · [code](https://loule0-0.github.io/KnowBody/)
- **[Learning and Transferring Closed-Loop Robot Software](https://arxiv.org/abs/2609.19906)** (So Kuroki, Yujin Tang, 2026-09-17) — Coding agent iteratively refines closed-loop robot policy code with simulation feedback (RoboCasa 28.3% to 64.2%); archived optimized code transfers to new tasks, +15.6 points over unoptimized references.
- **[Accelerating Visual Policy Learning with Sampling-Based Model Predictive Control](https://arxiv.org/abs/2609.20575)** (Yale University, 2026-09-17) — SGPS: BC-initialized depth policies alternate sampling-based MPC action-target refinement with short-horizon first-order policy-gradient updates; learns Go2/G1 skills, zero-shot transfer to real Go2.
- **[Diverse and Adaptable Arm Coordination for Octopus-Crawling via Diffusion-Based…](https://arxiv.org/abs/2609.21138)** (Kim et al., 2026-09-17) — Diffusion-based optimization discovers and preserves diverse crawling coordination strategies for a simulated muscle-actuated soft multi-arm robot without demonstrations, adapting to new constraints without full retraining.
- ⭐ **[Decade of BO for control](https://arxiv.org/abs/2609.09403)** (RWTH Aachen, University of Technology Nuremberg, TU Munich, 2026-09-08) — Survey framing controller tuning as episodic policy search where a GP surrogate picks each trial; reviews 110 hardware papers and releases TuneControl. 57 of 110 use vanilla BO; sim 4-gain cart-pole: LogEI 4310±48 vs Sobol 4647±190 (lower better). · [code](https://github.com/Data-Science-in-Mechanical-Engineering/tunecontrol)
- **[Continual Field-Adaptive Models (CFAMs) for Post-Deployment Physical AI](https://arxiv.org/abs/2609.04552)** (Singh et al., 2026-09-03) — Frozen-plus-adaptive policy with Competence Capsules for gradient-free on-device post-deployment learning across five embodiments; parity using 40% of training data, +13.9 points success from field experience.

## August 2026

- **[You Don't Need To Stay in The Loop](https://arxiv.org/abs/2608.07555)** (Yu, 2026-08-02) — LLM controller runs disposable worker agents through train-evaluate-improve cycles with controller-owned measurement, crash recovery and evidence-gated promotion; false-promotion rate 0.001 per run vs 0.005-0.021.

## July 2026

- **[VLA Grounder](https://arxiv.org/abs/2607.04517)** (MIRAI, MISIS, 2026-07-05) — Keeps pi0/OpenVLA frozen and trains a Qwen3.5-9B grounder with GRPO to rewrite the instruction once per episode. pi0 VL-Think average 12.6 → 38.7, and it beats the evolutionary prompt optimizer GEPA (55.2 vs 63.9 on OpenVLA)., the venue is probably an ICML 2026 workshop, the released code uses a shaped reward, and train and test contexts may overlap. · numbers corrected after check

## June 2026

- ⭐ **[ENPIRE](https://arxiv.org/abs/2606.19980)** (NVIDIA, Carnegie Mellon University, UC Berkeley, 2026-06-18) — Coding agents edit training recipes, train, run real rollouts scored automatically, and keep what raises success. Pin insertion reached 100% (50 consecutive episodes) in about 40 min with 8 agent/robot pairs vs over 1.5 h with 1. · [code](https://github.com/NVlabs/ENPIRE)

## March 2026

- ⭐ **[Golden Ticket](https://arxiv.org/abs/2603.15757)** (Robotics and AI Institute, Arizona State University,…, 2026-03-16) — Replaces per-chunk Gaussian noise in a frozen diffusion or flow policy with one fixed noise vector found by black-box search. Real Franka block/cube pick: 80 → 98% success on 50 held-out episodes after 150 search episodes (~30 min). · [code](https://github.com/rai-opensource/lottery_tickets)

## November 2025

- ⭐ **[EGGROLL](https://arxiv.org/abs/2511.16652)** (FLAIR and WhiRL, Mila, NVIDIA AI Technology Center,…, 2025-11-20) — Makes each evolution-strategies perturbation a low-rank A_i B_i^T, so the population shares one big matmul plus cheap per-member corrections. On one GH200 (8192-dim bf16 layer, batch 1024): 91% of max batch-inference throughput vs PPO 34%, OpenES 0.41%. · [code](https://github.com/ESHyperscale/HyperscaleES)
- ⭐ **[TD-ES](https://arxiv.org/abs/2511.09923)** (The University of Sydney; NVIDIA, 2025-11-13) — Continues from a PPO checkpoint with antithetic evolution strategies whose bounded triangular per-coordinate noise keeps candidates in a small box around the weights. Sim IQM success, 3 Isaac Lab Franka tasks: PPO 67.2%, Gaussian ES 73.7%, TD-ES 85.0%.

## September 2025

- ⭐ **[ES at Scale (LLM)](https://arxiv.org/abs/2509.24372)** (Cognizant AI Lab, MIT, UCLA, UT Austin, 2025-09-29) — Perturbs all weights of 30 model copies with tiny Gaussian noise, scores each with an outcome-only reward, and steps toward high-scoring noise, with no backprop or log-probs. Countdown, Qwen-2.5-3B (language-only): ES 60.5 vs Dr.GRPO-v 43.8 (base 10.0). · [code](https://github.com/VsonicV/es-at-scale)

## Before September 2025

- ⭐ **[EvolSAC](https://arxiv.org/abs/2507.10030)** (University of Padova, 2025-07-14) — Pretrains SAC on a dense shaped reward, then runs SNES on the actor weights with the true competition score as fitness. Sim cartpole swing-up time: 2.085 s → 1.135 s, vs 1.565 s after 1000 more SAC-only episodes.
- ⭐ **[Zero-order optimization primer](https://arxiv.org/abs/2506.22087)** (Machines in Motion Laboratory, New York University, 2025-06-27) — Tutorial framing derivative-free robotics methods as one loop: sample Gaussian perturbations, score them, move toward better samples; the sample weighting recovers MPPI, CMA-ES or CEM-like updates. Plots only: on 4 Hydrax sim tasks, MPPI-CMA consistently outperforms MPPI. · [code](https://github.com/ajordana/zoo-rob)
- **[BOpt-GMM](https://arxiv.org/abs/2403.14305)** (University of Freiburg; UTN, 2024-03-21) — Imitate from about 10 demos with a GMM dynamical system, then BO (SMAC3) on sparse binary success. In sim it reaches 80% success 87 episodes earlier than SAC-GMM. · [code](https://github.com/robot-learning-freiburg/bopt_gmm)
- **[Diffusion-ES](https://arxiv.org/abs/2402.06559)** (Carnegie Mellon University, 2024-02-09) — Test-time evolutionary search over a frozen trajectory diffusion model, mutating by partial re-noise and denoise. It matches the hand-designed SOTA planner on nuPlan (driving score 92 vs 92). · [code](https://github.com/bhyang/diffusion-es)
- **[Table tennis BGS](https://arxiv.org/abs/2309.03315)** (Google DeepMind, 2023-09-06) — ARS-variant ES trains a ~1k-parameter policy in sim that transfers zero-shot to a real high-speed arm: average ball return 1.83 of 2.0 with measured latency modeled.
- **[CEM-RL](https://arxiv.org/abs/1810.01222)** (Sorbonne / CNRS ISIR, 2018-10-02) — CEM over actor weights, where half the population gets TD3 gradient steps before evaluation. HalfCheetah 10725 vs TD3 9630 at 1M steps. · [code](https://github.com/apourchot/CEM-RL)
- **[ARS](https://arxiv.org/abs/1803.07055)** (UC Berkeley, 2018-03-19) — Antithetic random search on linear policies, with reward-std scaling, state normalization and top-direction selection, is competitive on MuJoCo locomotion. · [code](https://github.com/modestyachts/ARS)
- **[OpenAI ES](https://arxiv.org/abs/1703.03864)** (OpenAI, 2017-03-10) — Weight-space ES as a scalable alternative to RL. Humanoid reached a score of 6000 in 10 minutes on 1,440 cores vs 657 minutes on one 18-core machine. · [code](https://github.com/openai/evolution-strategies-starter)
- **[CMA-ES tutorial](https://arxiv.org/abs/1604.00772)** (Inria, 2016-04-04) — Reference derivation of CMA-ES. Default population lambda = 4 + floor(3 ln n); covariance is accumulated across generations. · [code](https://github.com/CMA-ES/pycma)
- **[Intelligent Trial & Error](https://arxiv.org/abs/1407.3501)** (UPMC/CNRS, Inria, U, 2014-07-13) — A MAP-Elites archive built in sim, then BO over the archive on a damaged real hexapod. It recovers 3-7x the reference gait's speed in a few trials.
- **[PI2](https://www.jmlr.org/papers/v11/theodorou10a.html)** (USC, 2010-11) — Path-integral sample-perturb-reweight update with no gradient and no learning rate. The paper reports it learning about an order of magnitude faster than the gradient-based methods it compares against.
- **[PoWER](https://papers.nips.cc/paper_files/paper/2008/hash/7647966b7343c29048673252e490f736-Abstract.html)** (MPI for Biological Cybernetics, 2008-12) — Imitate once, then return-weighted parameter perturbation on the real robot. Ball-in-a-cup regularly succeeds after about 75 rollouts.
- **[Kohl & Stone Aibo gait](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf)** (UT Austin, 2004-05) — Finite-difference hill climbing of a 12-parameter gait on 3 real Aibos: 291 +/- 3 mm/s after about 3 hours. A roughly hand-tuned start worked best.
