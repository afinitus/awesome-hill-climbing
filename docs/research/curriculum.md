# lastmile: course plan and future work

This is a planning document, last updated in October 2026. Released in v0.1: chapters 0-3, 5-8 and 12, in
simulation. Next: chapters 4 and 9-11, then the SO-101 hardware steps. The
[chapter index](../chapters/README.md) records what is implemented and the
[design contract](../DESIGN.md) defines the current experiment. Proposed work below is not a measured result,
a hardware tutorial, or a promise that every method reaches a particular success rate.

## What is available

Nine chapters are implemented: 0, 1, 2, 3, 5, 6, 7, 8 and 12. Chapter 2 has a separate Bayesian
optimization script, giving ten runnable algorithm files. Chapters 4, 9, 10 and 11 remain planned.
All published course measurements are simulated. The hardware steps in the chapter notes have not been run.

The SO-101 and AgileX PiPER scenes use vendored MuJoCo Menagerie models, with upstream licenses and commits
recorded in [the asset notes](../../lastmile/envs/assets/README.md). The task code is in
[`lastmile/envs/`](../../lastmile/envs/); it is not a fork of a LeRobot environment plugin. The current
CupDrop-v1 episode limit is 150 control steps at 10 Hz. Exact geometry, observations, variants and controller
limits are defined by the code and design contract, rather than by the earlier speculative dimensions in this plan.

Chapters 1–2 optimize a scripted controller with ten tunable knobs. The later chapters use the frozen
FlowChunk base or the explicitly described instructional adaptations. They do not all start with the same
controller, use the same task variant, or have the same training budget. Compare the declared settings and
costs in each chapter before comparing scores.

The implemented sequence is:

- [0: base calibration](../chapters/ch00.md): flow-matching behavior cloning, demonstration quality,
  calibration on search states, and pass@k diagnostics.
- [1: hill climbing and ARS](../chapters/ch01.md): greedy search, antithetic finite differences and
  reward-normalized search over controller knobs.
- [2: CEM, CMA-ES, PI² and BO](../chapters/ch02.md): related population updates, covariance adaptation,
  temperature, dimension and small-budget Bayesian optimization. These methods share aspects of a
  weighted-mean template; their complete algorithms are not identical. An MPPI action-sequence planner
  is not implemented here.
- [3: noise tickets](../chapters/ch03.md): random search, CEM and racing over a frozen policy's noise input,
  with search selection and explicit disclosure of the exploratory held-out peek.
- [5: off-policy critics](../chapters/ch05.md): SAC/RLPD adaptations and Q-chunk selection, with the corrected
  same-selector backup, controls, and an offline-pretraining ablation. The corrected experiment does not
  support the old claim that conservative pretraining always collapses.
- [6: residual TD3](../chapters/ch06.md): a bounded residual, critic warm-up and LayerNorm ablations. Its
  multi-step off-policy backup is an instructional approximation.
- [7: noise steering](../chapters/ch07.md): learned latent steering, steering subspaces and a residual
  hybrid, with finite-sample support diagnostics.
- [8: propose, edit, select](../chapters/ch08.md): an EXPO-inspired bounded editor and Q-based selection,
  controls, calibration and latency. The chapter explains the departures from the source method.
- [12: best-of-N](../chapters/ch12.md): verifiers trained from logged episodes, episode-separated validation,
  random selection, simulator lookahead and inference cost.

The [paper catalogue](../../papers/README.md) links the original methods and their reported results. Paper
results are not course results; the chapter notes explain which components were adapted. This plan avoids
maintaining a second, easily stale copy of paper numbers and software-version claims.

## Evaluation requirements

Use the rules in [CONTRIBUTING.md](../../CONTRIBUTING.md): choose hyperparameters on search data, charge
training and search, report controls and failures, and retain result files and provenance. Quick runs are
software diagnostics on search states. They are not new held-out experiments.

Report observed k/n and its Wilson interval under the binomial assumptions. An interval is not a guarantee
of reliability on new tasks, correlated deployment attempts or a changing policy. The current pooled
chapter intervals reuse fixed start states across training seeds and must carry the chapter's dependence
caveat. A non-significant paired test does not establish equivalence.

A finite pass@k result measures the particular sampling-and-reset protocol. It does not establish zero
policy support when all sampled attempts fail, and it does not bound a method that recombines candidate
chunks at successive decisions. A true support limit requires a stronger argument than a finite failure count.

CupDrop-v1 accepts a drop into an upright cup at its current position, including some episodes where the cup
was displaced or tipped and recovered. It is not a validated no-shoving benchmark. Historical adaptive reuse
of evaluation states also remains a limitation. These findings and their measurements are in the
[design contract](../DESIGN.md) (§2).

Before making stronger confirmatory benchmark claims, define a new metric/version, test adversarial cases,
freeze a fresh evaluation protocol and preserve complete training/pilot/checkpoint provenance. Do not quietly
reinterpret the existing outcome bitstrings under a stricter success definition.

## Planned chapters

These are experiment proposals. Their ablations should test hypotheses rather than assume the proposed
failure or improvement will occur. Runtime and robot-time budgets must be measured after implementation.

### 4: policy gradients for a flow policy

Compare likelihood-ratio policy-gradient recipes with well-defined action log probabilities. Candidate
constructions include stochastic flow steps and output noise. Compare critic-based advantages with group
baselines, correct executed-chunk discounting, and anchored versus unanchored updates. Track zero-variance
groups, learning-rate sensitivity and deterioration from the base. Sparse zero-reward samples can provide no
reward-gradient signal, but observing no successes does not prove that learning or exploration can never help.

Start in simulation. A later domain-randomized transfer experiment should measure the sim-to-real drop,
not assume zero-shot transfer. See the [on-policy family](../../papers/onpolicy.md) for source methods.

### 9: supervised policy improvement

Compare outcome filtering, advantage-weighted regression, and advantage conditioning with optional
classifier-free guidance. Keep data collection, value estimation and policy extraction separately testable.
Try multiple retraining rounds and report success, cycle time and regressions, including when the proposed
advantage signal fails. Restart-from-base and warm-start variants should have explicit cost accounting.
Outcome-driven data curation can be a separate ablation. See the [advantage-weighted family](../../papers/advbc.md).

### 10: corrections and interventions

Use a scripted expert as a simulated corrector, then compare aggregation, correction reweighting,
recover-then-correct behavior, intervention timing and intervention-derived rewards. Noise inversion is an
optional extension. Record intervention counts and durations, expert quality and task outcomes. Simulated
"human steps" are not measurements of real human effort. See the [human-corrections family](../../papers/hitl.md).

### 11: learned rewards and their failure modes

Compare success classifiers, progress regression, rewind negatives and discriminators. Keep potential-based
shaping separate from using raw progress as reward. Validate on held-out episodes and measure false positives
as well as downstream policy improvement. Investigate camera occlusion and other candidate reward exploits;
report whether they actually occur. This requires an appropriate image-data pipeline and is not implemented
by the current state-based course. See the [reward family](../../papers/reward.md).

## Other proposed extensions

- Weight-space search on part of the flow policy; an MPPI action-sequence planner; and explicit exploration
  budgets for comparison with knob search.
- Additional base policies such as ACT or SmolVLA, with their own checkpoints, input conventions and controls.
- Pixel policies and a domain-randomized twin with measured transfer error. Learned world-model experiments
  should test rollout-horizon effects and model exploitation rather than assume simulator equivalence.
- Gradient guidance, disagreement-gated search and a combined steering/editor/verifier stack, with ablations
  that separate each component's contribution and latency.
- A final reporting card with task scope, trial count, interval, adaptation cost, cycle time, interventions,
  retries, failures and checkpoint identity. A real success rate of 95% is an aspiration, not a release promise.

## Hardware work is future work

There is no implemented `hardware/` runner, real-arm environment wrapper or printed evaluation mat in this
repository. Create and validate those before describing the course as an end-to-end hardware workflow.

Start with calibration, a documented action interface, low-speed scripted motion, demonstrations and a
success detector checked against human labels. Freeze separate search and evaluation positions before
training. Record resets, takeovers and labeling time. When old evaluation rollouts are reused as training
examples, retire that evaluation set and use fresh data for subsequent claims.

**For every hardware experiment:** clear the workspace, provide an immediately reachable power cutoff or
e-stop, start with conservative speed and step limits, keep hands out while the policy runs, reset only after
motion stops, and supervise every run. A bounded policy output alone is not a hardware safety guarantee.

[LeRobot's HIL-SERL guide](https://huggingface.co/docs/lerobot/hilserl) is a possible external reference for
actor/learner training, demonstrations and human interventions; the current guide lists an NVIDIA GPU.
Its [simulation guide](https://huggingface.co/docs/lerobot/hilserl_sim) describes a separate environment.
Check the selected release's documentation before integrating it. This course does not currently run that
hardware reference baseline, and it does not depend on a promised future LeRobot feature.

## Public announcement scope

Describe the repository as a curated, source-linked guide to methods for improving robot policies, with a
simulation course and a roadmap. Link the current chapter index and the design contract. Use an implemented
chapter's measured result only with its task, baseline, evaluation protocol, interval and limitations.

Do not announce twelve completed methods that all reach 95%, real hardware results, exhaustive coverage of
the field, or independent replication of every listed paper. The illustrations show the goal of reliability
improvement; they are not evidence that the course reaches 99% deployment reliability.
