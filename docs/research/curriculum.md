# The Last Mile: taking robot policies from 50% to 95%

*A curriculum for an open-source course on hill-climbing robot policies. It starts in simulation on a Mac laptop and ends on a real SO-101 arm.*

Working repo name: `lastmile`. Tagline: *"One robot, one task, one base policy that works about half the time, and twelve ways to push it to 95%, each with its cost in robot-minutes and human-minutes."*

---

## 0. Design principles

1. **One task, one base policy, one scoreboard.** Every chapter starts from the same frozen checkpoint (`base_v1`). Every chapter is scored on the same held-out initial states. Methods can therefore be compared by success rate and by what they cost.
2. **The cost is part of the result.** Every run writes a ledger with robot-minutes (search, training and evaluation kept separate), human-minutes (demos, labels, takeovers, resets) and compute. The leaderboard sorts by success per robot-minute and per human-minute.
3. **Honesty is built into the harness.** Search sets and evaluation sets are separate. Every number has a Wilson 95% interval. Single-attempt success is reported next to any pass@k. Every chapter also shows one case where the method fails or makes things worse.
4. **Single-file algorithms.** Each method is one runnable file of about 150-450 lines, written in the style of CleanRL. Files import only `envs/`, `common/policy_flow.py` and `common/eval.py`.
5. **Laptop first, NVIDIA optional, hardware last.** Every sim chapter runs on a Mac CPU with native MuJoCo and PyTorch on CPU or MPS. Anything that needs NVIDIA (Isaac, ManiSkill, LeRobot's HIL-SERL learner, pi0/SmolVLA-scale training, Robometer inference) is labelled as a cloud-GPU stretch goal. On the SO-101, each chapter spends minutes to about an hour of robot time.

### LeRobot and SO-101 facts checked before relying on them (Sept 2026)

- **LeRobot HIL-SERL exists and supports SO-101.** It is an actor/learner SAC setup (in v0.6 the policy type is `gaussian_actor` and the algorithm is `algorithm.type=sac`), connected over gRPC. It includes a vision reward classifier (ResNet-10 backbone `helper2424/resnet10`), end-effector action space through IK from a URDF, 10 fps control and 128×128 image crops. Takeovers work with a gamepad, a keyboard, or the SO-101 leader arm (press `space` to take over, `s` for success, `esc` for failure). The docs say the SO-101 leader's gearing makes takeovers smoother than on the SO-100. **The docs list an NVIDIA GPU as a requirement.** The course therefore runs its own lightweight single-file SAC on CPU for state-based policies, and runs the LeRobot learner on a rented GPU when using it as the reference baseline.
- **`gym_hil` is a Franka Panda MuJoCo simulation (`PandaPickCube*-v0`), not an SO-101 simulation.** It is useful for testing the HIL stack, but it is not the course's twin.
- **SO-101 simulation assets.** TheRobotStudio publishes official URDF and MJCF files (`Simulation/SO101/scene.xml`, with `so101_new_calib.xml` and `so101_old_calib.xml`). Known caveats: LeRobot's 0-100 gripper mapping is not reflected in the model files, and base collision meshes were removed for stability. There is also a third-party plugin, `lerobot-env-so101` (Apache-2.0, ported from gym-hil). It provides an SO-101 pick-cube task with a 4-D `[dx,dy,dz,grasp]` end-effector action, is auto-discovered by LeRobot 0.6+, and documents about 17 mm of gravity sag at the end-effector. The course forks this plugin's structure for its own task.
- **Policies available in LeRobot:** ACT, Diffusion, VQ-BeT, Pi0 / Pi0-FAST / Pi0.5, SmolVLA, GR00T N1.7, XVLA and others. RL: HIL-SERL and TD-MPC, with **QC-FQL listed as coming soon**. Reward models: **SARM, TOPReward, Robometer**. v0.5.1 added HIL / DAgger / HG-DAgger data-collection support. v0.6.0 rebuilt the RL stack.
- **Implication.** ACT and SmolVLA are optional bases through LeRobot. The course's own base is a small flow-matching chunk policy (section 2), because most modern last-mile methods need direct access to the input noise. The course does not depend on unreleased LeRobot features: QC-FQL is implemented from scratch in Ch 5, and the LeRobot version becomes an optional comparison once it ships.

---

## 1. Curriculum: 12 chapters, from classic to modern

Each chapter lists: learning goal, the base RL idea, the 2025-26 papers it mirrors, a minimal implementation plan, the environment, the compute, the shareable plot or GIF, and the hardware step. "Paper reference" means a number from a paper. "Our target" is a hypothesis the repo has to confirm, not a promise.

### Ch 1 — Hill climbing with a success counter (finite differences, ARS)
- **Goal.** Improve a controller when all you can do is run it and count successes.
- **Base idea.** Perturb, evaluate, step. Gradient ascent on a Gaussian-smoothed objective with antithetic pairs: `g ≈ 1/(Nσ) Σ [R(θ+σε) − R(θ−σε)] ε`. ARS adds three small tricks: divide by the reward standard deviation, keep only the top directions, and normalize states.
- **Papers mirrored.** Kohl & Stone 2004 (original on-hardware hill climb); ARS (C012). Modern versions: **ES at Scale (C155, 2025-09)**, a population of 30 with only a scalar reward; **TD-ES (C186, 2025-11)**, which switches to ES after PPO plateaus, with bounded noise; **EvolSAC (C134, 2025-07)**, which pretrains on a proxy reward and then hill-climbs the true metric.
- **Implementation (`algos/01_hill_climb.py`, about 120 lines).** Use the scripted `KnobController`, the same controller that generates demos, with 8 knobs: grasp xy offset, grasp z offset, approach height, gripper-close delay, lift height, drop xy offset, drop height and speed scale. Run (a) greedy hill climbing, (b) antithetic finite differences, and (c) ARS with top-b directions. Use common random numbers: every candidate sees the same 16 start states from `S_search`. As an extra, run TD-ES-style triangular-noise ES on the last layer of `base_v1` as a "stage 2" and show variance across seeds, not just the mean.
- **Environment.** `CupDrop-v1` (section 2) with state observations. Knobs start from a deliberately mis-tuned setting (about 30-50%).
- **Compute.** Mac CPU, minutes. Each iteration is about 2N×16 episodes of 200 steps.
- **Shareable output.** An animated knob trajectory next to a success-vs-episodes curve, and a GIF of iteration 0 missing the cup beside the final iteration landing it.
- **Hardware step.** Tune 4 knobs of the scripted controller on the SO-101: 8 iterations × 2 antithetic candidates × 3 episodes ≈ 50 episodes (about 20 min). The scripted controller uses a color-blob estimate of the cube position. The lesson: the knob values that work in sim are wrong on the real arm (servo zero offsets, gravity sag), and 50 real episodes fix that.

### Ch 2 — One update, different weights: CEM, CMA-ES, PI2/MPPI and BO
- **Goal.** See that CEM, CMA-ES, PI2 and MPPI are the same weighted-mean update, and learn when Bayesian optimization is the right tool for tiny budgets.
- **Base idea.** Reward-weighted averaging: sample K candidates, weight them (top-k elites for CEM, ranks for CMA-ES, `exp(−cost/λ)` for PI2/MPPI), and move to the weighted mean. CMA-ES also adapts the covariance. BO fits a Gaussian process to (θ, score) pairs and picks the next trial with an acquisition function (logEI, UCB).
- **Papers mirrored.** **Zero-Order Optimization primer (C130, 2025-06)**, which shows MPPI-CMA, CMA-ES and CEM are one template; **A Decade of BO for controller tuning (C445, 2026-09)**, whose defaults are d+4 initial points and logEI, and which reports that most hardware uses tune fewer than 10 parameters. Classic: PI2 (C002), CMA-ES tutorial (C005), BOpt-GMM (C057).
- **Implementation (`algos/02_cem_cmaes_pi2.py`, one `weights=` switch; `algos/02b_bo.py` using BoTorch).** Same 8 knobs as Ch 1. Add an MPPI planner over action sequences, using MuJoCo state cloning as the model, to show the same loop running at every control step.
- **Environment.** `CupDrop-v1` knobs, plus an "MPPI with a perfect model" demo.
- **Compute.** Mac CPU, minutes. BoTorch runs on CPU.
- **Shareable output.** Four methods on one plot, success vs number of evaluations, with CMA-ES covariance ellipses animating in a 2-knob slice. Headline: "BO needs about 30 episodes for 4 knobs, CMA-ES needs about 300 for 8."
- **Hardware step.** Run BO on 3-4 real knobs with about 25-30 real episodes and compare with Ch 1's finite differences at the same robot-minutes. This produces the first leaderboard row where a method wins on robot-minutes.

### Ch 3 — The golden ticket: black-box search over one noise vector
- **Goal.** Improve a frozen generative policy without touching its weights, by searching a 32-dimensional noise vector. This is also the first hill climb of the shared `base_v1`.
- **Base idea.** Parameter-space search in a tiny space. A flow policy with deterministic Euler sampling is a function `a = π(s, w)`. Fix `w` for every chunk and search for it with random search, CEM or sequential halving, scoring each candidate by success on fixed start states. Search can only select behavior the base already has.
- **Papers mirrored.** **Golden Ticket (C236, 2026-03).** Paper reference: real Franka pick 80 → 98% on 50 held-out episodes, using 150 search rollouts and random search only; hardware average over 4 tasks 57.5 → 88.5. Robomimic Square regressed (52.2 → 32.9). Its released franka_sim setup uses a 32-dim noise and roughly 250 tickets × 25 envs. Also INSPO (2026-09), SDN, Diffusion-ES (C055).
- **Implementation (`algos/03_golden_ticket.py`, about 150 lines).** Random search, CEM, and random search with sequential halving ("racing"). **Choose the ticket by its search score and report held-out `S_eval` performance.** Also report the optimistic "best ticket measured on the eval set", which is the same optimism the C236 verification flagged in the SimplerEnv number, so readers can see the gap. Show the worst ticket too (the paper found one that succeeds 2 in 50).
- **Environment.** `CupDrop-v1` with `base_v1` (H=8 chunk × 4-D action = 32-dim noise, matching C236's franka_sim).
- **Compute.** Mac CPU. 250 tickets × 25 episodes ≈ 1M environment steps plus batched MLP inference, roughly tens of minutes (estimate, to be measured).
- **Shareable output.** A grid of 16 tickets as GIFs, colored by success, with the chosen ticket highlighted. A "search score vs held-out score" scatter plot showing regression to the mean.
- **Hardware step.** 10 tickets × 5 episodes on `R_search` (50 rollouts, about 15-20 min), then 30 held-out `R_eval` episodes. This is the cheapest real improvement in the course and the baseline every later method has to beat, which is C236's own recommendation.

### Ch 4 — Likelihood-ratio policy gradients with anchors (PPO/GRPO for a flow policy)
- **Goal.** Learn the equation behind most RL fine-tuning, `∇ = E[∇ log π(a|s) · A]`, and why pretrained policies collapse without anchors.
- **Base idea.** PPO clipping; GAE advantages from a critic versus GRPO/RLOO group baselines (G rollouts from an identical start state, which MuJoCo state restore makes easy). How to get a log-probability out of a flow policy: learned per-Euler-step noise as in ReinFlow/Flow-Noise, or noise only on the output as in StructRL. Anchors: KL or BC term to the frozen base, about 10× lower learning rate, no entropy bonus, critic warm-up.
- **Papers mirrored.** **SimpleVLA-RL (C140, 2025-09)**, GRPO with a binary reward, which cannot climb from 0%; **piRL (C157, 2025-10)**, where PPO beats GRPO for flow VLAs; **PAC-ACT (C352, 2026-07)**, PPO on ACT chunks, 60 → 100%, dropping to 72.7% without the KL anchor; **FPO++ (C208, 2026-02)**; **ADEPT (C400, 2026-08)**, 0/5 seeds survive at LR 1e-3; **Prism-GRPO (C385)**, tie-breakers for zero-variance groups.
- **Implementation (`algos/04_ppo_flow.py`, with flags `--adv {gae,grpo}` and `--noise {chain,output}`).** Chunk-level PPO with reward summed over executed actions and discount γ^c. Ablations: anchor off, LR ×10, and a 0%-success start (hard variant) to show that no signal means no climb. Log the fraction of zero-variance GRPO groups.
- **Environment.** `CupDrop-v1` state observations, with 8-32 parallel CPU processes.
- **Compute.** Heavier than any other sim chapter: about 1e5-1e6 steps per run. Hours on a Mac, or about 1 hour on a small cloud GPU. The repro notes put this family at 0.4M to tens of millions of steps.
- **Shareable output.** "The collapse plot": the same run with and without the anchor. Plus GRPO vs PPO curves with a zero-variance-group counter.
- **Hardware step.** **No real on-policy RL.** The on-policy repro notes say this is impractical at these sample counts. Instead: train in the domain-randomized sim twin, deploy zero-shot on the SO-101, and **measure and report the sim-to-real drop**. This is the honest lesson of the family.

### Ch 5 — Off-policy critics: SAC → RLPD → Q-chunking
- **Goal.** Build a critic that reuses every transition (demos, failures, old rollouts) and see why chunk critics matter for sparse rewards.
- **Base idea.** Bellman backups on replay; RLPD tricks (50/50 prior/online batches, LayerNorm critics, a Q ensemble, UTD above 1). A chunk critic `Q(s, a_{t:t+h})` with an unbiased h-step target `Σr + γ^h Q'`. Best-of-N extraction from a BC flow policy (QC), and the offline-to-online "dip". The Three Regimes check: is the policy or the dataset better?
- **Papers mirrored.** **Q-chunking (C133, 2025-07)**, OGBench 25 tasks: QC 52 → 86 vs FQL 37 → 58; **Three Regimes (C172, 2025-10)**; **IPE (C348, 2026-07)**, where naive critic pretraining helps little; **WSRL (C088)**; RLPD (C037); DQC (C191).
- **Implementation (`algos/05_rlpd_qchunk.py`).** SAC, with flags that turn it into RLPD and then into QC (chunk critic plus best-of-N from `base_v1`). Print the policy's success next to the demos' success (the regime check). Show the dip: CQL-style pretrain then online, vs a WSRL-style warm-up.
- **Environment.** `CupDrop-v1`, sparse 0/1 reward, 20-50 demos.
- **Compute.** Mac CPU/MPS, about 1e5-3e5 steps at UTD 2-4, 1-3 hours per seed (estimate).
- **Shareable output.** Three overlaid curves: SAC with demos, RLPD, QC. The dip in an inset.
- **Hardware step.** Offline first. Fit the chunk critic on **all real rollouts logged so far** (Ch 1-4 evaluations: a few hundred episodes with success labels) and run best-of-8 QC extraction. The only new robot time is 30 evaluation episodes. Optional: a 30-45 min online RLPD run with keypress success labels.

### Ch 6 — Residual RL on a frozen base
- **Goal.** Correct a frozen policy with a small add-on that starts at base performance and cannot drift far.
- **Base idea.** The residual MDP: `a = a_base + α·tanh(u)`, with the last layer zero-initialized. The base becomes part of the environment. Off-policy TD3/SAC with demos in the buffer, a critic on the summed action `Q(s, a_base + a_res)`, and replay warmed up with base rollouts (DAWN: the critic, not the actor, is the bottleneck).
- **Papers mirrored.** **ResFiT (C142, 2025-09)**, roughly 200× more sample-efficient than a PPO residual in sim, real 14 → 64% in about 15 min; **DAWN (C233, 2026-02)**, warm-up plus LayerNorm; **Res-HIL (C432, 2026-09)**, a zero-initialized TD3 residual on a 20-demo ACT with takeovers used as targets, reaching 100/64/50/66/92% in 10 min; **PLD (C158)**, which distills residual data back into the base; Policy Decorator (C086).
- **Implementation (`algos/06_residual_td3.py`, about 300 lines).** Flags: `--bound`, `--warmup_steps 20000`, `--layernorm`. Reproduce the DAWN ablation grid (2×2). Base options: `base_v1` or LeRobot ACT.
- **Environment.** `CupDrop-v1`, plus a "biased actuator" variant (constant joint offset), which residuals fix easily.
- **Compute.** Mac CPU, about 1e5 steps, around 1 hour.
- **Shareable output.** The 2×2 DAWN grid as small learning curves. A GIF of the base missing by 1 cm next to base-plus-residual landing it.
- **Hardware step.** A bounded residual on frozen real `base_v1`: 20-45 min online, keypress success labels, optional leader-arm takeovers logged as residual targets (Res-HIL). SO-101 end-effector deltas must go through the same clipping and rate limits as policy actions, the approximation to PAKT suggested in its verification notes.

### Ch 7 — DSRL: RL in the noise space of a frozen flow policy
- **Goal.** Turn the Ch 3 ticket into a state-dependent policy. RL picks the noise, not the action.
- **Base idea.** Latent-noise MDP: a small actor outputs `w = π_W(s)`, bounded in a box, and the frozen sampler decodes it. Exploration stays on the demo manifold. **Support versus reach**: steering can only reweight behaviors the base can already produce.
- **Papers mirrored.** **DSRL (C120, 2025-06)**, real Franka 2/10 → 9/10 in about 40 episodes; **PSS (C413, 2026-09)**, a top-k principal steering subspace found by finite differences; **RFS (C228, 2026-02)**, a noise-plus-residual hybrid (0.861 vs DSRL-only 0.483 vs residual-only 0.433); **SCORE (C324, 2026-06)**, support-constrained steering in a twin with zero real RL, 37.8 → 89.9; **BMD (C304)**, which warns about mode collapse.
- **Implementation (`algos/07_dsrl.py`, flags `--subspace k` and `--hybrid`).** SAC in noise space. Include the **support-limit experiment** on the hard variant: steering flatlines while the residual or hybrid still moves. Add a mode-coverage metric (grasp from left vs right of the cube).
- **Environment.** `CupDrop-v1` and `CupDrop-Hard-v1`.
- **Compute.** Mac CPU, about 1e5 steps.
- **Shareable output.** A plot of steering vs residual vs hybrid on the easy and hard variants (a 2×3 grid), and a histogram of grasp modes before and after.
- **Hardware step.** DSRL on real `base_v1`, 40-80 episodes (about 20-40 min). Report whether the policy lost a grasp mode.

### Ch 8 — Propose, edit, select: EXPO-style edit policy with Q selection
- **Goal.** Use the base as a proposer and a small bounded editor, and let a Q ensemble choose. This is the current reference recipe for real-robot fine-tuning.
- **Base idea.** Sample N chunks from the base, add a bounded edit to each (`|Δ| ≤ β`), and execute the argmax over the 2N candidates by Q. Use the same argmax inside the TD target. The edit policy is trained SAC-style to raise Q. No RL gradient passes through the base. Variants: noise-conditioned residual with best-of-K (DICE-RL); full-chunk editor that can leave the base's support (MiDAS).
- **Papers mirrored.** **EXPO-FT (C276, 2026-05)**, SFT pi0.5 20.5/30 → 30/30 on 8 real tasks with an average of 19.1 min of online data (30/30 is still consistent with a true rate around 88%); **Real-Time EXPO-FT (C412, 2026-09)**, 42% → 97% with a 10-minute cap; **DICE-RL (C235, 2026-03)**, real UR5 56.7 → 93.3%; **MiDAS (C402, 2026-08)**; **FASTER (C263)**, which picks the noise seed first to save compute.
- **Implementation (`algos/08_expo_edit.py`, about 350 lines).** N=8 proposals, tanh-bounded edit β ∈ {0.05, 0.1, 0.2}, REDQ-style ensemble, argmax used both to act and in the target. Ablations: β=0 (pure best-of-N with Q), frozen vs co-trained base, full-chunk editor.
- **Environment.** `CupDrop-v1` state observations; pixels as an option later.
- **Compute.** Mac CPU/MPS: 8 base samples per step makes it about 3-5× slower than Ch 6 (estimate).
- **Shareable output.** A "proposal fan" GIF: 8 translucent proposed chunks, their edits, and the chosen one in bold, with Q values shown.
- **Hardware step.** EXPO on real `base_v1` with at most 30 min online, keypress labels and optional leader-arm corrections in the buffer. Measure per-step latency on the Mac and report it.

### Ch 9 — Supervised policy improvement: filtered BC, AWR and RECAP-style advantage conditioning
- **Goal.** Improve the policy while keeping a purely supervised loss, and learn why the advantage signal matters more than the extraction rule.
- **Base idea.** The KL-regularized closed form `π* ∝ π_ref·exp(A/β)`, projected with weighted BC (AWR), 0/1 weights (filtered BC), or **conditioning** on a token `I = 1[A > ε]` with classifier-free guidance weight w. The advantage comes from a value network regressed on "negative steps to success". Retrain from the base checkpoint every round.
- **Papers mirrored.** **π*0.6 / RECAP (C179, 2025-11)**: on the same data, RECAP about 60 successes/hr at 96% vs AWR about 30/hr at 91% vs PPO about 22/hr at 75% (chart readings). **CFGRL (C110, 2025-05)**. **Dissecting advantage-guided post-training (C435, 2026-09)**: on a real robot, continuous weighting 0.74 > hard filter 0.50 > DAgger 0.45. **DEED (C369, 2026-07)**: round 2 regressed from 42% to 22%. **Hi-ORS (C159)**. **CUPID / RoboDrop** for curation.
- **Implementation (`algos/09_recap_lite.py`).** A small categorical value head (201 bins) on state, N-step advantage, and four extraction modes: filter, AWR, token plus CFG with w ∈ {1, 1.5, 2.5}, and soft weights with per-bin normalization. At least 3 rounds. Report success **and steps-to-success**, because RECAP's key observation is that AWR keeps success but loses speed.
- **Environment.** `CupDrop-v1`, with 30% sloppy demos mixed in (these also let a CUPID-lite curation pass show a real effect).
- **Compute.** Mac CPU: each round is about 200-500 rollouts plus a BC retrain of a small flow model (minutes).
- **Shareable output.** A 3-round plot of success and cycle time for the four extraction modes, including a regression if one happens.
- **Hardware step.** 2 rounds × 100-150 real rollouts with keypress labels. Leader-arm corrections are forced to "positive" as in RECAP; also try ROVE's rule, which penalizes the first seconds after a takeover. The value head trains on the Mac.

### Ch 10 — Humans in the loop: DAgger, HG-DAgger, RaC and takeovers as reward
- **Goal.** Spend human minutes where the policy actually fails, and measure those minutes.
- **Base idea.** Covariate shift (errors grow like T²ε). DAgger aggregation. Human-gated takeovers. Reweighting corrections to 50% of each batch (IWR/Sirius). RaC's recover-then-correct rule. RLIF, where a takeover counts as −1 reward.
- **Papers mirrored.** **RaC (C148, 2025-09)**; **SOP (C203, 2026-01)**, fleet online HG-DAgger with 3 h of on-policy data beating 80 h of extra demos; **FlowDAgger (C355, 2026-07)**, which inverts corrections into noise and keeps held-out skills (0.96 → 0.88 vs SFT 0.96 → 0.02); **TimelyDAgger (C443, 2026-09)**, where takeover timing moves results from 42% to 78% at the same budget; **HELP (C351)**, which reports throughput because recovery data makes policies slow; **PAKT (C462)**, which aligns the action pipeline.
- **Implementation (`algos/10_hg_dagger.py`).** In sim the "human" is the privileged scripted expert with a distance-based gate. Flags: `--reweight`, `--rac`, `--gate {human,ensemble,timely}`, `--rlif` (RLPD with −1 before takeovers and a deliberately weakened expert, to show RL beating its corrector), `--flowdagger` (noise inversion). Log intervention rate and "human steps".
- **Environment.** `CupDrop-v1`.
- **Compute.** Mac CPU, minutes per round.
- **Shareable output.** Intervention rate falling over rounds next to success rising. A split-screen GIF of the takeover (human-controlled segment tinted).
- **Hardware step.** HG-DAgger on the SO-101 with the leader arm, 3 rounds × 20 episodes, using LeRobot's HIL/DAgger collection tooling. Log human-seconds per round. **Reference baseline:** LeRobot HIL-SERL end to end (SAC plus ResNet-10 reward classifier plus leader-arm takeovers) with a rented NVIDIA GPU as the learner, compared on the same leaderboard.

### Ch 11 — Where the reward comes from: success detectors, progress models and hacking them
- **Goal.** Replace the human success key with learned signals, and see those signals get exploited.
- **Base idea.** Success classifier (with the policy's own states as negatives, VICE-style). Progress regression from time with ReWiND-style rewound clips as failures. SVM discriminator from the policy's own success and failure buffers. Potential-based shaping `γΦ(s') − Φ(s)` versus raw progress as reward. VOC as an offline check.
- **Papers mirrored.** **Robometer (C244, 2026-03)**, DSRL on a frozen pi0 20 → 85% in about 40 min, plus offline IQL on an SO-101; **TOPReward (C226, 2026-02)**, a zero-shot token log-prob reward; **SVM (C337, 2026-06)**; **Robo-Dopamine (C193)**, where removing shaping drops 85.0 → 41.3%; **SARM (C152)**; **RoboReward (C205)**, where offline accuracy correlates with downstream RL success at r = 0.83. Robometer, TOPReward and SARM are all available in LeRobot.
- **Implementation (`algos/11_progress_reward.py`).** Train a small head on frozen DINOv2 (or small CNN) features for t/T progress, with and without rewind negatives. Add an SVM discriminator. Plug each into Ch 6 residual RL three ways: sparse only, raw progress, potential shaping. **Staged hack:** a camera-occlusion case where the cube hovers over the cup and "looks" successful. The ground-truth MuJoCo success check catches it.
- **Environment.** `CupDrop-v1` with pixel observations (128×128 front and wrist cameras).
- **Compute.** Mac MPS for feature extraction (minutes). Robometer-4B and TOPReward (Qwen3-VL-8B) inference on a rented GPU; Mac or MPS support is unknown.
- **Shareable output.** A 4-panel "reward curves over one episode" figure (true success, naive progress, rewind progress, SVM), and the hack GIF showing the model at 0.9 while the cube misses.
- **Hardware step.** Train the success classifier on about 200 labeled real frames. Score the real videos logged so far zero-shot with Robometer and TOPReward (report VOC and false-positive rate against human labels). Rerun one Ch 6/7 hardware run with the learned reward and report **human-minutes saved and false positives caught**.

### Ch 12 — Test-time search with a critic, then the finale
- **Goal.** Spend compute at deployment instead of changing weights: best-of-N with a verifier, gradient guidance, and gating. Then stack the best methods.
- **Base idea.** Best-of-N as one greedy improvement step, with pass@N as its ceiling. The KL-regularized optimum written as guidance (`∇log π + (1/β)∇_a Q` at a one-step clean-action estimate). Over-optimization as N grows. Search only when samples disagree.
- **Papers mirrored.** **UF-OPS (C259, 2026-03)**, a verifier trained on the policy's own evaluation rollouts, real ALOHA mean 38% → 87% at best-of-10; **Q-Planning (C392, 2026-08)**; **SeeQ (C438, 2026-09)**; **QGF (C307, 2026-06)**; **VLA-ATTC (C288)**, gating on disagreement; **JITI (C189)**, SmolVLA on an **SO-100** 34.3 → 58.0% with reranking only at critical moments.
- **Implementation (`algos/12_best_of_n.py`).** First the SVA-style diagnostic: pass@1 vs pass@k with sim resets. Then an MC time-to-success verifier and a success classifier on logged (state, chunk) pairs, best-of-N for N ∈ {1, 2, 4, 8, 16, 32}, a random-pick control, an oracle verifier, QGF guidance with a β sweep, and a disagreement gate. Report latency.
- **Environment.** `CupDrop-v1`.
- **Compute.** Mac CPU. Batched N samples per step.
- **Shareable output.** "Success vs N" with oracle, learned-verifier and random-pick lines, where the learned verifier bends away from the oracle at large N (over-optimization).
- **Hardware step and finale.** Train the verifier on every labeled real rollout collected across the course, typically 500+. Rerank N=4-8 per chunk on the Mac. **Finale:** a combined stack (ticket or DSRL, plus EXPO edit or residual, plus gated best-of-N) evaluated on the extended real set with 60-100 trials, and a published "Solve card" (section 3).

---

## 2. Shared base policy and task: `CupDrop` on the SO-101

### Task
The SO-101 picks up a 2.5 cm colored cube placed at random in a 12 × 16 cm region and drops it into a cup taped at a fixed position. **Success** means the cube ends inside the cup cylinder within 20 s (200 steps at 10 Hz). Secondary metrics: time-to-success and near-miss distance.

**Why this task.**
- It is multi-step (reach, grasp, lift, align, drop), so failures are diverse: missed grasps, slips, off-target drops.
- It tolerates the SO-101's backlash and roughly 17 mm of gravity sag, and it has one precision knob, the **cup diameter**, used to set difficulty.
- Most failures are *near misses*, so the base succeeds sometimes and pass@k is well above pass@1. That is the condition selection, steering, residual and critic methods need.
- It resets quickly: the human lifts the cube out of the cup and puts it on the next marked spot.

**Variants.**
- `CupDrop-v1`: the main task.
- `CupDrop-Hard-v1`: smaller cup, or starts outside the demo region. Used to show support limits and the 0%-success cold start.
- `CupDrop-Bias-v1`: constant joint offsets, as caused by SO-101 servo zeros. Used for residual and knob chapters.

### Simulation
- MuJoCo scene built on the official SO-101 MJCF (`so101_new_calib.xml`). The gripper mapping gets fixed to match LeRobot's 0-100 convention, and that fix is documented. The environment structure follows `lerobot-env-so101`: a 4-D `[dx,dy,dz,grip]` end-effector action through IK at 10 Hz, the same action convention as LeRobot HIL-SERL, so sim and real share one interface.
- Domain randomization for the twin: friction, cube mass, command and observation delay, backlash, joint zero offsets, camera pose and lighting.
- Observation modes: `state` (joints, end-effector, gripper, cube xy, cup xy) for fast Mac training, and `pixels` (128×128 front and wrist cameras) from v0.3.

### Base policy `base_v1` = **FlowChunk-S**
A flow-matching policy over action chunks: MLP 4×256, chunk H=8, execute 4 then replan, 10-step Euler sampling. Its noise is 8×4 = 32-dim, the same size as Golden Ticket's released franka_sim. Given a fixed noise, it is deterministic. About 0.3M parameters, trains in minutes on a Mac.

**One policy serves every chapter:**

| Chapter | What FlowChunk exposes |
|---|---|
| 1-2 | The scripted `KnobController` is the demo expert and the "classic" policy. ES on FlowChunk's last layer is the extra. |
| 3, 7, 12 | The noise input `w`: fixed ticket, learned noise actor, sampling N candidates. |
| 4 | ReinFlow-style per-step noise or output noise gives an exact log-probability. |
| 5, 8 | The chunk is the critic's action. The base acts as the proposer or behavior prior. |
| 6 | Frozen base plus bounded residual. |
| 9, 10 | Plain supervised flow-matching loss, retrained with advantage tokens, weights or aggregated corrections. |
| 11 | Its rollouts provide success and failure buffers for learned rewards. |

Optional second bases through LeRobot: **ACT**, since ResFiT, Res-HIL and PAC-ACT all use ACT, and **SmolVLA** as a cloud-GPU stretch goal, since JITI runs SmolVLA on an SO-100.

### Calibrating to "about 40-60%, climbable to 95%+"
1. **Ceiling check.** The privileged `KnobController`, after BO tuning, must reach ≥98% on `S_eval` in sim. This proves the task is solvable.
2. **Set base success.** Train FlowChunk on n demos with 30% sloppy demos mixed in (off-center grasps, early releases), and sweep n (for example 10-60). Freeze the smallest n that gives 45-55% on `S_eval` as `base_v1`. Publish it with a dataset hash on the Hugging Face Hub.
3. **Headroom check.** Require pass@8 ≥ 90% with sim resets. If this fails, selection and steering methods cannot reach 95%, and the cup diameter is adjusted.
4. **Real calibration.** Collect about 50 leader-arm teleop demos (about 40 min). Tune demo count and cup diameter until real `base_v1` lands at 40-60% on `R_eval`. On the real arm the policy uses the **state** mode, with cube xy from a color-blob detector through a calibrated table homography, so Mac-trained MLP policies transfer. Pixel policies are optional from v1.0.

**Honest caveat.** 95%+ on the real SO-101 is a *target*. Hobby servos, no force sensing and blob-detector noise may cap it. The README will report where each method actually lands, with intervals.

---

## 3. The shared evaluation harness (`common/eval.py`, `common/ledger.py`)

### Initial-state sets, fixed and versioned
| Set | Sim | Real | Use |
|---|---|---|---|
| `S_search` / `R_search` | 64 seeds | 10 marked spots | Search, training selection, hyperparameters |
| `S_eval` / `R_eval` | 256 seeds | 30 marked spots on a printed mat, fixed shuffled order | Held-out score. Never used for any selection. |
| `S_eval_ext` / `R_eval_ext` | 1024 | 60-100 | Final claims and the finale |
| `S_stress` | shifted region, friction, lighting | lighting change, distractor | Robustness column |

- **Common random numbers.** Every episode's policy-sampling seed is fixed by (set, index), so method-vs-base comparisons are paired. Report deterministic-noise and stochastic results separately.
- **Seeds.** At least 3 training seeds in sim, reported both per seed and pooled. Real runs are usually single-seed and say so.
- **The real mat** is a printed PDF with numbered cube positions. A timed reset keypress logs human reset seconds.
- **Success detector.** Sim uses the true MuJoCo state. Real uses a top-camera color check ("cube pixels inside the cup mask"), **validated against about 100 human labels before first use** with its false-positive and false-negative rates published. The human key always overrides it.

### Statistics
- **Wilson 95% interval** on every success rate: center `(p̂ + z²/2n)/(1 + z²/n)`, half-width `z·sqrt(p̂(1−p̂)/n + z²/4n²)/(1 + z²/n)`.
- **Reference numbers printed in the README:**
  - 15/30 → [33.2, 66.8]
  - 27/30 → [74.4, 96.5]
  - **30/30 → [88.6, 100]**
  - 57/60 → [86.3, 98.3]
  - 243/256 → [91.5, 97.0]
  - A lower bound of at least 95% needs **73 consecutive successes**; at least 99% needs **381**.
- **Paired tests.** McNemar's exact test on shared initial states, plus a bootstrap confidence interval for the difference.
- **Rule:** "95%+" means a point estimate ≥95% on `S_eval` (n=256) in sim. On real hardware we report the interval and never write "100%".

### Metrics logged for every run (JSON to `results/`)
- success@1 with CI; pass@k in sim only
- time-to-success; near-miss distance
- intervention rate and human-seconds
- **robot-minutes**, split into `search`, `train`, `eval`. Sim is converted as steps / 10 Hz / 60, so sim and real use the same unit.
- **human-minutes**, split into `demos`, `labels`, `takeovers`, `resets`, `annotation`
- CPU-hours and GPU-hours; wall-clock time
- base checkpoint hash, init-set version, git SHA

### Leaderboard (auto-generated in the README by a GitHub Action)
| Method | Setting | Base SR [CI] | Final SR [CI] | Δ | Improvement robot-min | Human-min | Compute | n eval | Notes |
|---|---|---|---|---|---|---|---|---|---|

- The **improvement budget** counts every rollout except the final evaluation, so search episodes are not hidden.
- **Two plots:** success vs log(robot-minutes) and success vs log(human-minutes), each with a Pareto frontier. A derived column gives *robot-minutes to reach 90%*.
- **Solve card** for the finale (following Sunday's ACT-2 reporting proposal): performance with CI and n, declared scope (cube colors, positions, lighting), adaptation cost (robot-min, human-min), single-attempt success next to any retry number, and the longest success streak.

---

## 4. Famous results we cannot reproduce, and honest approximations

| Famous result | Why not at our scale | Honest approximation in the repo |
|---|---|---|
| **π*0.6 / RECAP**: espresso about 11 → 92%, 13 h runs (C179) | 4B+ VLA, tens of thousands of hours of pretraining, no code | Ch 9 RECAP-lite: tiny value head, token plus CFG on FlowChunk, same 4-way extraction comparison, restart from base each round |
| **SimpleVLA-RL / piRL**: LIBERO 91 → 99, pi0 57.6 → 97.6 | 7B/3B VLAs, 8× A800/H100, GPU-parallel sim | Ch 4 PPO/GRPO on a 0.3M-parameter flow policy; same lessons (anchors, zero-variance groups, 0% cold start) |
| **EXPO-FT / Real-Time EXPO-FT on pi0.5**: 30/30, 97% | 2× H200, JAX/OpenPI server, DROID stack | Ch 8 EXPO on FlowChunk; SmolVLA proposer as a cloud-GPU stretch; always report 30/30's [88.6, 100] |
| **RL-100**: 1000/1000 real trials, about 14 h robot time per task | Robot hours, not compute, rule it out | Do not claim. Report our budgets and intervals. |
| **HIL-SERL**: 100% on precise insertion in 1-2.5 h (C074) | Franka impedance control; SO-101 is position-controlled with hobby servos | LeRobot HIL-SERL on SO-101 as a reference row (cloud GPU learner). Expect lower precision and say so. PAKT/CR-DAgger compliant corrections do not transfer. |
| **LWD** (16-robot fleet), **SOP** (10 robots) | One arm | Emulate a "fleet" with parallel sim workers streaming to one learner; keep the ideas (streaming updates, on-policy data), not the numbers |
| **OpenAI ES** (1,440 cores), **EGGROLL**, **ES at Scale for LLMs** | Population and model scale | ES on FlowChunk's last layer in sim; triangular-noise TD-ES as a stage 2. No weight-space ES on hardware. |
| **Golden Ticket SimplerEnv 63 → 89.5** | Optimistic selection on the eval set | Select by search score and report held-out results; also show the optimistic number to expose the gap |
| **World-model RL** (WMPO, WoVR, RISE, Ctrl-World; 1.5-14B video models) | GPU-heavy video generators | Not a chapter. Optional appendix: tiny latent model plus success head in sim, with a rollout-horizon sweep to show exploitation. |
| **Sim-to-real in Isaac/ManiSkill** (RL-Co, TacCoRL, generative worlds at about 960 GPU-h) | NVIDIA only; massive parallelism | MuJoCo domain-randomized twin on CPU; Ch 4's zero-shot transfer with the measured drop |
| **Robometer / RoboReward / SOLE-R1 training** (1M+ trajectories, 4-8B VLMs) | Pretraining scale | Use released weights for zero-shot scoring (LeRobot integration, rented GPU); train small progress heads on the Mac |
| **Industry 99%s**: GEN-1 99%, Dyna 99.4% over 24 h, Sunday 99.1% over 785 attempts | Proprietary models and data, no ablations | Copy the reporting (Solve card, streaks, successes per hour), not the numbers |

**Standing honesty rules in CONTRIBUTING.md:** separate search and evaluation sets; CI on every number; single-attempt next to pass@k; include the unsteered and random-selection baselines; count search episodes in the budget; show one failure per chapter; state that search and steering can only select behavior the base already has.

---

## 5. Repo structure, roadmap and launch thread

### Structure
```
lastmile/
├── README.md                 # pitch, leaderboard (auto), plots, chapter index
├── pyproject.toml            # uv; core: mujoco, torch, numpy, gymnasium; extras: [lerobot], [bo], [cloud]
├── envs/
│   ├── cupdrop.py            # MuJoCo SO-101 CupDrop (+Hard, +Bias), DR, init-state sets
│   ├── assets/so101/         # vendored official MJCF (Apache-2.0) + gripper-mapping fix
│   ├── knob_controller.py    # scripted 8-knob expert (demos, Ch1-2)
│   └── real_so101.py         # LeRobot robot wrapper: same obs/action API, blob detector, success check
├── common/
│   ├── policy_flow.py        # FlowChunk-S: noise-in / chunk-out, ReinFlow noise option, advantage token option
│   ├── eval.py               # init sets, CRN seeding, Wilson, McNemar, bootstrap
│   ├── ledger.py             # robot-min / human-min / compute accounting
│   └── nets.py, buffers.py   # tiny MLPs, LayerNorm critics, replay
├── algos/                    # one runnable file per method (python algos/03_golden_ticket.py)
│   ├── 00_bc_flow.py   01_hill_climb.py   02_cem_cmaes_pi2.py   02b_bo.py
│   ├── 03_golden_ticket.py   04_ppo_flow.py   05_rlpd_qchunk.py   06_residual_td3.py
│   ├── 07_dsrl.py   08_expo_edit.py   09_recap_lite.py   10_hg_dagger.py
│   └── 11_progress_reward.py   12_best_of_n.py
├── hardware/
│   ├── bringup.md            # SO-101 + leader, calibration, cameras, homography
│   ├── mat_positions.pdf     # printed initial-state mat
│   ├── reset_protocol.md     # timed resets, keypress conventions (s / esc / space)
│   └── run_real.py           # runs any algos/* policy against real_so101 with the ledger
├── results/                  # one JSON per run -> leaderboard + plots
├── docs/chapters/ch01..12.md # theory, paper map, "what went wrong"
└── media/                    # GIFs and plots for each chapter
```
**Rules:** each algorithm file is at most about 450 lines, keeps a config dataclass at the top, has no inter-algorithm imports, prints a ledger summary at the end, and has a smoke test in CI (a 2-minute sim run on CPU).

### Staged roadmap
| Version | Scope | Exit criteria |
|---|---|---|
| **v0.1 — sim, state-only** | `CupDrop`, `KnobController`, FlowChunk `base_v1` (45-55%), eval harness and ledger, **Ch 1, 2, 3, 12** | Every file runs on a Mac CPU; leaderboard with CIs; ticket-vs-held-out plot; pass@k headroom check passes |
| **v0.2 — off-policy family** | **Ch 5, 6, 7, 8** in sim; DAWN, support-limit and proposal-fan figures | At least one method ≥95% on `S_eval` (n=256) with 3 seeds |
| **v0.3 — the rest in sim** | **Ch 4, 9, 10, 11**; pixel mode; domain-randomized twin; hack demo | All 12 chapters have sim rows; the Ch 4 collapse and Ch 9 regression reproduce or are honestly reported absent |
| **v0.4 — hardware bring-up** | SO-101 teleop demos, real `base_v1` at 40-60%, mat, blob success detector validated against human labels, **Ch 1-3 hardware steps** | First real hill climb (ticket) with Wilson CIs and a real ledger |
| **v0.5 — real HIL** | **Ch 6, 7, 10** on hardware; LeRobot HIL-SERL reference row (cloud GPU); Ch 4 zero-shot transfer measured | Real leaderboard sorted by robot-min and human-min |
| **v0.6 — critics and rewards on real** | **Ch 5, 8, 9, 11, 12** on hardware; learned success detector replaces the key | Human-minutes per improvement point reported |
| **v1.0 — finale** | Combined stack; 60-100 trial real evaluation; Solve card; docs and videos; optional SmolVLA/ACT bases | Real point estimate ≥95% with interval, or an honest statement of where it stalled and why |

### Twitter/X launch thread outline (v0.1, then a thread per release)
1. **Hook:** GIF of the SO-101 missing the cup, then 12 panels climbing. "One robot, one task, one base policy that works 50% of the time. 12 ways to push it to 95%, with the cost of each in robot-minutes and human-minutes. Open source; runs on a Mac."
2. **Why last-mile matters:** a 2025-26 pattern across labs is "pretrain, then hill-climb the last mile," but the recipes live in 200-page papers and closed stacks. This repo makes them small.
3. **The scoreboard:** a leaderboard screenshot with success vs robot-minutes. "Every row counts its search episodes."
4. **Ch 1-2:** the four-optimizers-one-update animation. "CEM, CMA-ES, PI2 and MPPI are the same weighted mean."
5. **Ch 3:** the golden-ticket grid. "Search 32 numbers and don't touch the weights," with the search-vs-held-out scatter plot. "Here is the regression to the mean nobody puts in the abstract."
6. **Ch 12:** "best-of-N with a verifier" curve showing where it breaks.
7. **Honesty panel:** "30/30 does not mean 100%. It means at least 88.6%." The Wilson table.
8. **What's hard:** "things we can't reproduce at small scale," the section 4 table as an image.
9. **Paper map:** each chapter tagged with the 2025-26 papers it mirrors (Golden Ticket, DSRL, EXPO-FT, Q-chunking, RECAP, ResFiT, UF-OPS, Robometer…).
10. **Hardware tease:** SO-101 + LeRobot, the v0.4-v1.0 roadmap, the mat PDF.
11. **Call to action:** contribute a method file that follows the budget rules; add your arm's Solve card.
12. Repo link plus credits to the paper authors and the LeRobot and MuJoCo teams.

---

### Open risks to check early
- SO-101 MJCF gripper and contact fidelity. Fix and validate before trusting sim numbers, and keep the Ch 4 transfer-drop measurement.
- Blob-detector noise on the real arm. Validate against human labels; fall back to pixel policies if needed.
- LeRobot HIL-SERL's NVIDIA requirement. Our own CPU SAC remains the primary path.
- QC-FQL in LeRobot is "coming soon". The course does not depend on it.
- The real 95% target may be capped by hardware. Report honestly.

### Sources checked for LeRobot and SO-101 claims
- [LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl)
- [LeRobot HIL-SERL in simulation (gym_hil)](https://huggingface.co/docs/lerobot/hilserl_sim)
- [LeRobot GitHub README](https://github.com/huggingface/lerobot) and [releases](https://github.com/huggingface/lerobot/releases)
- [Official SO-101 URDF/MJCF (TheRobotStudio)](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md)
- [lerobot-env-so101 plugin (third-party)](https://pypi.org/project/lerobot-env-so101/)

Method choices are based on the family syntheses in the task and on `papers.json (repro_notes field)`, specifically the notes for C012, C074, C088, C120, C133, C142, C152, C179, C186, C188, C189, C235, C236, C244, C259, C276, C319, C337, C352, C390, C412 and C432.