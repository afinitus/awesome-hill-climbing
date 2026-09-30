# Broader RL advances that transfer

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*Which general RL results from the last year change how you should hill-climb a robot policy?*

Not robot-manipulation papers: scaling laws for value functions and RL compute, critic and actor-critic design studies, flow/diffusion policy theory, RL for language models (group baselines, off-policy and stale-data training, process rewards, test-time compute), and last-mile RL for driving policies. Robot work keeps borrowing from all of them.

- **Loop step:** Background for every step
- **Built on:** Everything in Foundations
- **Use it when:** Read these when you are choosing an algorithm family or a compute budget, or when a robot paper cites an idea from language-model RL and you want the original.
- **What changed in 2025–26:** RL for language models and RL for robot policies converged on the same toolkit in 2025-26: critic-free group baselines, KL-anchored fine-tuning from a pretrained model, learned verifiers, and spending compute at test time.

**47 entries**, newest first. ⭐ marks must-know work.


## September 2026

- **[EasyPPO](https://arxiv.org/abs/2609.36802)** (Zhou et al., 2026-09-29) — Stabilizes PPO's critic for LLM RL via actor-only truncation filtering, noise-normalized critic regression and smaller critic minibatches; relative gains of 14.89% coding, 2.28% math, 9.47% search over PPO.
- **[Learning from Teacher Continuations at Student States](https://arxiv.org/abs/2609.36246)** (Wang et al., 2026-09-28) — OLIVE: DAgger-like online LLM distillation; student generates prefix, teacher continues it, student trains by cross-entropy. +13% over offline SFT on ScienceWorld; async cuts training time 23.8%.
- **[RefineDrive](https://arxiv.org/abs/2609.35078)** (Sun et al., 2026-09-28) — Simulator-state failure diagnosis, minimum-correction retrieval from human trajectory bank, correction SFT, then GRPO with safety-layered reward; NAVSIM v1 PDMS 87.7 to 91.7, 89.4 EPDMS on v2. Driving VLA.
- **[Unified Trajectory Matching Policy Optimization: Diverse T2I Generation and VLA…](https://arxiv.org/abs/2609.34688)** (Ma et al., 2026-09-28) — Converts group-standardized rewards into a target distribution and matches it via forward KL over trajectory log-probs, avoiding mode collapse in diffusion/flow policy RL; improves VLA generalization, real-robot validated.
- **[Generate, Track, Improve](https://arxiv.org/abs/2609.31577)** (Olkin et al., 2026-09-25) — Off-policy RL refines a depth-conditioned flow-matching motion generator paired with a tracking policy on Unitree G1; up to +25 points terrain traversal, +80 points skill selection. · [code](https://zolkin1.github.io/generate-track-improve/)
- **[What Stops Recursive Self-Improvement in Robotics? Lessons from 123 Rounds of Agentic…](https://arxiv.org/abs/2609.31760)** (National University of Singapore, 2026-09-23) — Agent writes/integrates new manipulation skills over 123 improvement rounds; gains fail to compound due to relational perception limits, skill chains stuck on early steps, and flawed evaluators.
- **[Imperfection for Precision](https://arxiv.org/abs/2609.26672)** (Wei et al., 2026-09-22) — ε4P reuses two imperfect data sources by controlling where each contributes along the flow-matching trajectory; up to +31.7 points, replaces high-quality task data at a 4.2-point average drop. · [code](https://varepsilon4p.github.io/)
- **[OPTED](https://arxiv.org/abs/2609.20756)** (Col et al., 2026-09-17) — RL-trained privileged teacher on vectorized inputs supervises closed-loop fine-tuning of camera driving policies in 3DGS AlpaSim; 1.6x and 9.5x gains with ~1000x fewer interactions than direct RL. · [code](https://01dami23.github.io/opted/)
- **[When Validation Stops Learning](https://arxiv.org/abs/2609.10873)** (Ma et al., 2026-09-09) — Audit protocol for gating policy updates: paired-binomial checks admit 31.6% of updates at 2,000 episodes/stage vs zero for range-based gate; synthetic pushing only, no real-robot validation.
- **[Recursive Value Learning for Long-Horizon Offline Goal-Conditioned RL](https://arxiv.org/abs/2609.02237)** (Jeon et al., 2026-09-02) — DCRL: recursively splits trajectory segments into a balanced binary tree, cutting bootstrap depth from linear to logarithmic; lifts average on hardest long-horizon OGBench tasks from 55 to 64.

## August 2026

- **[Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO](https://arxiv.org/abs/2608.27351)** (Ba et al., 2026-08-27) — Analyzes ES post-training for LLM reasoning: broader coverage and better Pass@K than GRPO without entropy collapse; gains come from sparse large updates; proposes sequential GRPO-then-ES (adjacent, LLMs).
- **[RoG-DAgger](https://arxiv.org/abs/2608.24525)** (Zhong et al., 2026-08-25) — DAgger post-training for end-to-end driving using short-horizon kinematic rollouts to build expert corrections and trigger interventions; +5.3 driving score on Bench2Drive, Fail2Drive OOD success 55% to 66%.
- **[Le Critique](https://arxiv.org/abs/2608.16739)** (Venkatraman et al., 2026-08-17) — Privileged value functions add token-level signal to LLM RL baselines without biasing the policy objective; TETHER interpolates group-relative and value baselines, matching or exceeding mean-baseline GRPO on reasoning tasks.
- **[FIRE-VLA](https://arxiv.org/abs/2608.13395)** (Dou, 2026-08-13) — When all group rollouts fail, self-distills from a frozen privileged copy that sees future trajectories; nuScenes persistent failures 13.03% to 11.20%, L2 1.848m to 1.500m. Driving-adjacent.
- **[Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?](https://arxiv.org/abs/2608.02547)** (Lazzati et al., 2026-08-03) — Argues action chunking helps BC via implicit ensembling over diverse temporal relationships, not temporal consistency or horizon reduction; an explicit ensemble of delayed policies outperforms standard chunking.

## July 2026

- **[Deconstructing Actor-Critic](https://arxiv.org/abs/2607.13274)** (Shah et al., 2026-07-14) — 33,000+ experiments on a water-treatment-plant control task dissect actor-critic design choices; default Gaussian + pathwise gradients are least reliable, bounded distributions with adaptive schedules most robust.
- **[Reward-Aware Population Scaling of Evolutionary Strategies in LLM Fine-Tuning](https://arxiv.org/abs/2607.19408)** (Cho et al., 2026-07-08) — Shows z-score advantage normalization breaks small-population ES; without it, binary-reward ES fine-tunes 0.5B-7B LLMs with N=2 on GSM8K/TREC, with a closed-form zero-advantage probability.

## June 2026

- **[Depth over Fidelity in Fixed-Budget Noisy Evolution Strategies](https://arxiv.org/abs/2606.06555)** (Wang et al., 2026-06-04) — Probabilistic elite membership (RB-PEM) replaces hard rank weights with expected rank weights under noisy evaluations; gains on COCO bbob-noisy, RL policy search and HPO in high-misranking, fixed-budget settings (ICML 2026).

## April 2026

- **[LaST-R1](https://arxiv.org/abs/2604.28192)** (Chen et al., 2026-04-30) — RL post-training for VLAs via Latent-to-Action Policy Optimization, jointly optimizing latent reasoning and actions with adaptive reasoning depth; 99.9% LIBERO, +22.5% real-world over supervised fine-tuning.

## February 2026

- **[LLMs Can Learn to Reason Via Off-Policy RL](https://arxiv.org/abs/2602.19362)** (Ritter et al., 2026-02-22) — OAPL: off-policy LLM post-training that embraces lagged inference policies (400+ gradient steps) instead of importance-sampling; beats GRPO+IS on competition math, matches DeepCoder with 3x fewer generations.
- **[Stable Asynchrony](https://arxiv.org/abs/2602.17616)** (Huang et al., 2026-02-19) — VCPO scales learning rate by effective sample size and adds a critic-free minimum-variance baseline for stale-rollout policy gradients; stable asynchronous LLM RL, 2.5x faster training. · [code](https://github.com/mit-han-lab/vcpo)
- **[Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710)** (CMU; ICML 2026, 2026-02-02) — Shows expected-reward RL is a first-order approximation of maximum likelihood under binary terminal feedback; MaxRL interpolates between them as sampling compute scales, up to 20x test-time scaling efficiency over GRPO. · [code](https://zanette-labs.github.io/MaxRL/)

## January 2026

- **[Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)** (Hübotter et al., 2026-01-28) — SDPO: LLM conditioned on rich textual feedback (e.g. runtime errors) acts as self-teacher, distilling next-token predictions into the policy; matches best-of-k discovery with 3x fewer attempts. · [code](https://github.com/lasgroup/SDPO)

## December 2025

- **[Training Diffusion Policies via Prior-Mapping Co-Evolution](https://arxiv.org/abs/2512.02581)** (Zhang et al., 2025-12-02) — GoRL: online RL for generative policies by optimizing a tractable latent prior, then consolidating into an expressive prior-to-action mapping; HopperStand return over 870, 3x strongest baseline (ICML 2026). · [code](https://github.com/bennidict23/GoRL)

## November 2025

- **[The Path Not Taken](https://arxiv.org/abs/2511.08567)** (Zhu et al., 2025-11-11) — Three-Gate Theory for LLM RL with verifiable rewards: sparse-looking updates are a model-conditioned bias landing off principal weight directions with minimal spectral drift, unlike SFT.
- **[RL without TD learning](https://bair.berkeley.edu/blog/2025/11/01/rl-without-td-learning/)** (UC Berkeley, 2025-11-01) — Transitive RL: divide-and-conquer off-policy value learning over in-dataset subgoals with expectile regression; matches best TD-n on OGBench long-horizon (up to 3,000-step) goal-conditioned tasks without tuning n.

## October 2025

- **[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)** (Thinking Machines Lab, 2025-10-27) — Student samples its own trajectories; teacher grades every token via reverse KL. LLM-focused: 74.4% AIME'24 at 1,800 GPU hours vs 17,920 for RL, 9-30x cheaper overall. · [code](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)
- **[Transitive RL](https://arxiv.org/abs/2510.22512)** (Park et al., 2025-10-26) — Divide-and-conquer value update for offline goal-conditioned RL using the triangle inequality; needs O(log T) recursions versus TD's O(T), reducing bias accumulation on long-horizon tasks. ICLR 2026.
- **[The Art of Scaling Reinforcement Learning Compute for LLMs](https://arxiv.org/abs/2510.13786)** (Khatri et al., 2025-10-15) — 400k+ GPU-hour study of LLM RL scaling; recipe details mostly change compute efficiency, not peak performance; ScaleRL recipe predictably extrapolates to a 100k GPU-hour run.
- **[A Survey of Process Reward Models](https://arxiv.org/abs/2510.08049)** (Zheng et al., 2025-10-09) — Survey of step-level process reward models for LLMs: process data generation, PRM construction, use in test-time scaling and RL; applications span math, code, multimodal, robotics, agents.
- **[Prosperity before Collapse](https://arxiv.org/abs/2510.01161)** (CMU; Meta AI, 2025-10-01) — M2PO constrains the second moment of importance weights so LLM RL stays stable on data 256+ updates stale, matching on-policy; clipped tokens drop 1.22% to 0.06%. · [code](https://github.com/Infini-AI-Lab/M2PO)

## September 2025

- **[XQC](https://arxiv.org/abs/2509.25174)** (Palenicek et al., 2025-09-29) — SAC-based actor-critic whose BatchNorm + WeightNorm + distributional cross-entropy critic cuts condition numbers by orders of magnitude; SOTA sample efficiency on 55 proprioceptive and 15 vision tasks with fewer parameters. · [code](https://danielpalenicek.github.io/projects/xqc)
- **[floq](https://arxiv.org/abs/2509.06863)** (CMU, 2025-09-08) — Parameterizes the Q-function as a flow-matching velocity field trained with TD targets; more integration steps scale critic capacity, giving ~1.8x gains on offline RL and online fine-tuning. · [code](https://github.com/CMU-AIRe/floq)

## Before September 2025

- **[Compute-Optimal Scaling for Value-Based Deep RL](https://arxiv.org/abs/2508.14881)** (Fu et al., 2025-08-20) — Studies compute allocation across model size, batch size and UTD ratio in online value-based RL; identifies TD-overfitting, where large batches hurt small critics but not large ones.
- **[IRL-VLA](https://arxiv.org/abs/2508.06571)** (Jiang et al., 2025-08-07) — Driving VLA: imitation pretraining, then PPO fine-tuning against a lightweight inverse-RL reward world model instead of a simulator; top NAVSIM v2 results, 2nd in CVPR2025 Autonomous Grand Challenge.
- **[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)** (Alibaba, 2025-07-24) — LLM RL algorithm replacing GRPO's token-level importance ratios with sequence-likelihood ratios and sequence-level clipping; more stable and efficient than GRPO, stabilizes MoE training, used for Qwen3.
- **[Horizon Reduction Makes RL Scalable](https://arxiv.org/abs/2506.04168)** (Park et al., 2025-06-04) — Offline RL plateaus even with datasets up to 1000x larger than typical; long horizon is the bottleneck. Horizon reduction fixes scaling; SHARSA is a minimal method with best asymptotic performance. · [code](https://github.com/seohongpark/horizon-reduction)
- **[ProRL](https://arxiv.org/abs/2505.24864)** (NVIDIA, 2025-05-30) — Prolonged RL on LLMs with KL control, reference-policy resets and diverse tasks finds reasoning strategies base models cannot reach under extensive sampling; beats base across pass@k. · [code](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)
- **[The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617)** (Cui et al., 2025-05-28) — Shows LLM policy-gradient RL trades policy entropy for performance, driven by probability-logit covariance; Clip-Cov and KL-Cov restrict high-covariance token updates to prevent entropy collapse.
- **[Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711)** (Mukherjee et al., 2025-05-16) — Across 7 RL algorithms (PPO, GRPO, DPO...) and 10 LLMs, RL fine-tuning updates only 5-30% of parameters; these sparse updates are nearly full-rank. NeurIPS 2025.
- **[Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the…](https://arxiv.org/abs/2504.13837)** (Tsinghua University; Shanghai Jiao Tong University, 2025-04-18) — RLVR-trained LLMs win at pass@1 but base models win at large-k pass@k: RL improves sampling efficiency within base-model capability, unlike distillation; NeurIPS 2025 Oral. · [code](https://github.com/LeapLabTHU/limit-of-RLVR)
- **[Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)** (2025-04-10) — Position essay (preprint chapter of MIT Press book "Designing an Intelligence"): agents will gain superhuman ability by learning predominantly from their own experience via streams, grounded actions, grounded rewards, and planning.
- **[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)** (Sea AI Lab, 2025-03-26) — Finds a GRPO optimization bias that artificially lengthens (especially incorrect) responses; Dr. GRPO removes it for better token efficiency, reaching 43.3% on AIME 2024 with a 7B base model. · [code](https://github.com/sail-sg/understand-r1-zero)
- **[1000 Layer Networks for Self-Supervised RL](https://arxiv.org/abs/2503.14858)** (Wang et al., 2025-03-19) — Scaling contrastive goal-conditioned RL networks from 2-5 to 1024 layers gives 2x-50x gains on simulated locomotion and manipulation, with qualitatively new behaviors, without demos or rewards. · [code](https://wang-kevin3290.github.io/scaling-crl/)
- **[DAPO](https://arxiv.org/abs/2503.14476)** (Yu et al., 2025-03-18) — Decoupled Clip and Dynamic Sampling Policy Optimization: four techniques for large-scale LLM RL; 50 points on AIME 2024 with Qwen2.5-32B; code on verl plus dataset released. · [code](https://dapo-sia.github.io)
- **[The Promise of Generalist Robotic Policies](https://sergeylevine.substack.com/p/the-promise-of-generalist-robotic)** (Sergey Levine, 2024-10-06) — Essay: generalist robot foundation models bootstrap a deployment data flywheel; language feedback and RL from real-world outcomes then drive autonomous self-improvement beyond imitation.
- **[Self-Improving Robots and the Importance of Data](https://sergeylevine.substack.com/p/self-improving-robots-and-the-importance)** (Sergey Levine, 2022-08-22) — Levine essay arguing deployed robot fleets could autonomously gather ~10 billion hours of embodied experience per year, with offline RL as the way to learn from it.
