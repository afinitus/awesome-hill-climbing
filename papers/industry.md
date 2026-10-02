# Industry recipes & deployment reports

[← Awesome Hill Climbing](../README.md) · [all families](README.md)

*How do robot companies say they climb from ~60-80% to 99% on a paying task, and what do they actually disclose?*

Companies take a big pretrained robot model and push one real task from "works in a demo" to "works all shift" with deploy, grade, curate and retrain loops, sometimes with RL folded in.

- **Loop step:** The whole loop
- **Built on:** On-policy data aggregation (DAgger and its human-gated variant); Automatic success and progress detection; Policy improvement through supervised learning (advantage-weighted or advantage-conditioned BC); Broad pretraining as the reason local hill-climbing transfers (a company claim, e.g. ACT-2); Reliability compounds and evaluation is statistical; Sparse outcome reward plus self-play
- **Use it when:** Use these case studies as a template when you have one fixed, economically defined task, a pretrained base policy that already works sometimes (roughly 50-80%), and the ability to run many long evaluation sessions. A common order, pieced together from partial company disclosures: (1) define success with the customer's own quality bar and log everything; (2) automate grading and failure localization (progress model or auto-labeler); (3) collect targeted recovery or correction data where the policy actually fails (DAgger-style); (4) if success plateaus or speed matters, add a value function and advantage-weighted or advantage-conditioned retraining in the RECAP style; (5) report streaks, successes per hour and graded quality over hundreds of trials with a declared scope.
- **What changed in 2025–26:** RL entered industry recipes in supervised-friendly forms: π*0.6/RECAP (Nov 2025) used a value function and advantage-conditioned retraining and outperformed PPO and AWR baselines on laundry folding with π0.6 (a Gemma 3 4B backbone plus an 860M action expert), and Generalist lists learning from experience (RL) among GEN-1's ingredients. Large-scale human data is now presented as what makes post-training cheap: GEN-1 is pretrained on over 500,000 hours of wearable-device data and adapted with about an hour of robot data per task, and Sunday pretrains ACT-2 on sensorized human data. Reporting moved toward production metrics, such as ACT-2's 99.1% success over 785 laundry attempts and Dyna-2's 95 napkins per hour with 93% meeting the quality bar, but these posts give few details of the training updates behind the numbers. A Siemens factory case study is a rare published account of the effort and recurring failure modes of an iterative fine-tuning loop on π0.5.

**28 entries**, newest first. ⭐ marks must-know work.


## September 2026

- ⭐ **[Skild Physical Self-Play](https://www.skild.ai/blogs/physical-self-play)** (Skild AI, 2026-09-23) — Previews RL post-training via self-play: in simulated robot soccer the policy's one stated objective is to score, playing recent versions of itself, then is transferred to a real humanoid. No metrics; 140 simulated years of play, real match on video.
- **[The Robot Data Factory](https://arxiv.org/abs/2609.16705)** (Haddadin, Laptev, Reid et al., 2026-09-15) — Infrastructure for a continuous Deploy-Measure-Learn-Repeat robot-experience loop, with a mission-to-capability hierarchy, fleet/sensor/compute scaling laws, and three physical training grounds.
- **[Smooth Exponentials for Robotics](https://evjang.com/2026/09/10/smooth-exponential.html)** (Eric Jang, 2026-09-10) — Essay arguing robotics has entered smooth exponential progress via many players and standardized hardware-agnostic pretraining data; predicts the first 2-3 general-purpose home robots on the market in October 2027. No RL fine-tuning content.

## August 2026

- **[In-Context Learning Results Hint at a "GPT Moment" for Robotics](https://itcanthink.substack.com/p/in-context-learning-results-hint)** (Chris Paxton, 2026-08-28) — Chris Paxton reviews video-prompted in-context learning (Skild S1, GEN-1.5, Rhoda, Lingbot VA-2) as an alternative to retraining; notes wholly new skills at high success still need more data or on-robot RL.
- ⭐ **[Dyna: deployment is the eval](https://www.dyna.co/research/scaling-customer-deployments)** (Dyna Robotics, 2026-08) — Fleet workflow: an auto-labeler splits each deployed episode into SOP steps with outcome and failure mode, humans review uncertain cases, and engineers query the fleet to retrain. Napkin folding: 35/hour, 75% passing (Dyna-1) → 95/hour, 93% (Dyna-2).

## July 2026

- **[Gemini Robotics 2 brings whole body intelligence to robots](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)** (Google DeepMind, 2026-07-30) — Whole-body VLA plus ER 2 reasoning model and On-Device 2, which adapts to new bi-arm embodiments typically with under 200 examples; charted success ranges 32-92%, multi-finger dexterity still the hardest. No RL post-training described.
- **[Sunday Robotics ACT-2](https://physicalaiguide.com/guides/sunday-robotics-act-2-reliability-evidence/)** (Physical AI Guide, 2026-07-17) — Third-party evidence audit of ACT-2's 778/785 (99.1%) laundry-fold result: one fixed checkpoint across unseen homes, no per-home adaptation, but first-party, task-specific, not independently reproduced.
- ⭐ **[Sunday ACT-2 Preview](https://www.sunday.ai/blog/act-2-preview)** (Sunday Robotics, 2026-07-17) — Pretrains ACT-2 on sensorized human data from Sunday's own capture hardware, then post-trains on repeated in-house Memo runs and recoveries (update rule undisclosed). Laundry: 99.1% over 785 autonomous attempts in unseen homes, one fixed checkpoint, zero per-home adaptation.

## June 2026

- **[KinetIQ Ascend](https://thehumanoid.ai/technology/kinetiq-ascend/)** (Humanoid, 2026-06-25) — Real-world PPO on a flow-matching VLA (ODE-to-SDE exploration noise, sparse success reward, speed curriculum), with gains measured against a concurrent frozen-policy A/B baseline. Tote handling 77.6%→98.9%, bottle picking 80%→98%, throughput +42% to +129% in 3-5 robot-days.

## May 2026

- ⭐ **[Siemens factory VLA case study](https://arxiv.org/abs/2605.27461)** (Siemens, 2026-05-25) — Runs a pure-SFT loop on pi0.5 in a real Siemens factory: teleop demos, manual review, re-fine-tuning, and targeted recovery demos, relaxing constraints each round. 2,535 factory episodes; no final success rate given; authors say it fell short of expectations.
- **[What 10 Months in Production Taught Us About the Robotics "Bubble"](https://www.dyna.co/news/robotics-bubble)** (Dyna Robotics, 2026-05) — York Yang essay: pretraining alone is insufficient; capability matures through iterated post-training on deployment data. It says DYNA-1 shipped in April 2025 running 24h+ at 99%+ success on tasks like napkin folding; longest deployment ~10 months of daily use.

## April 2026

- **[Bessemer Predicts](https://www.bvp.com/atlas/bessemer-predicts-robotics-and-physical-ai)** (Bessemer Venture Partners, 2026-04-16) — Investor prediction piece with founder quotes: going from 80% to 99.9% task success is not a linear problem and needs different approaches; data is expensive and capital is the moat; sim-to-real manipulation remains open.
- **[Going Beyond World Models & VLAs](https://generalistai.com/blog/beyond-world-models)** (Generalist AI, 2026-04-07) — Position post: goals matter more than VLA/world-model labels; GEN-1 has ~99% of parameters trained from scratch on 500k+ hours, targeting 99%+ success from about one hour of robot data.
- ⭐ **[Generalist GEN-1](https://generalistai.com/blog/gen-1)** (Generalist AI, 2026-04-02) — Pretrained on 500K+ hours of human wearable-device data, adapted per task with about 1 hour of robot data; RL is one of five listed ingredients, not detailed. Three charted tasks: 99% average vs 64% for fine-tuned GEN-0 (whole-system comparison).
- **[Building a 100-Robot Data Factory Toward Factory-Ready AI](https://www.tutorintelligence.com/blog/building-a-100-robot-data-factory-toward-factory-ready-ai)** (Tutor Intelligence, 2026-04) — 100-robot Data Factory 1 trains the Ti0 VLA on teleop demos plus corrections of its own mistakes, then RL post-training from human-scored episodes; running one policy on 100 robots surfaces an 8-hour edge case in about 5 minutes.

## February 2026

- **[Where Autonomy Works](https://epoch.ai/blog/where-autonomy-works-evaluating-robot-capabilities-in-2026)** (Epoch AI, 2026-02-10) — Survey of reported robot reliability: warehouse picking 99%+ (per roboticist interviews), DYNA-1 napkin folding 99.4% over a 24-hour run, Figure package handling 95% (claimed, one-hour demo); autonomy works where environments are controlled or forgiving.

## December 2025

- **[State of Robot Learning](https://vedder.io/misc/state_of_robot_learning_dec_2025.html)** (2025-12) — Practitioner essay: nearly all robot learning today is BC plus laborious DAgger-style data iteration, and real-world RL remains hard; in the author's reading, pi*0.6's advantage-weighted method gives minor gains over BC and many tasks still needed human corrections.
- ⭐ **[DYNA-1i open-world dexterity](https://www.dyna.co/research/open-world-dexterity)** (Dyna Robotics, 2025-12) — Post-trains a VLA on 'only tens of hours' of office data to stay reliable in unseen venues; no objective or update rule is given. 30-minute shirt-folding trials (~40/hour): 22 consecutive failure-free folds in the office, 20 in unseen venues.

## November 2025

- **[AgiBot real-world RL deployment](https://www.prnewswire.com/news-releases/agibot-achieves-first-real-world-deployment-of-reinforcement-learning-in-industrial-robotics-302601935.html)** (AgiBot; Longcheer Technology, 2025-11-02) — AgiBot's RW-RL system learns assembly skills directly on a Longcheer pilot production line; company claims training cut from weeks to minutes and 100% task completion over extended operation. The article links no paper or code.
- **[DYNA-1 Pre-Training](https://www.dyna.co/research/pre-training)** (Dyna Robotics, 2025-11) — Company update, no paper: says the DYNA-1 base model folds laundry and sorts packages zero-shot in unseen environments, and that as little as one hour of demos fine-tunes it to roughly 100% success on new tasks. RL not discussed.

## Before September 2025

- ⭐ **[Figure Helix logistics](https://www.figure.ai/news/scaling-helix-logistics)** (Figure AI, 2025-06-07) — Improves one deployed parcel-sorting job with more in-domain demos, richer System-1 inputs (stereo, vision memory, state history, force) and a 50% larger decoder head; the post describes no RL. Over three months, barcode-orientation success rose ~70% → ~95%.
- **[1XWM (2025)](https://www.1x.tech/discover/redwood-ai-world-model)** (1X Technologies, 2025-06) — Action-conditioned video world model that also predicts task success from the final generated frame, used to rank humanoid policy checkpoints. Evidence is predicted success per checkpoint, checked by real runs on selected checkpoints; no correlation coefficient is reported.
- **[DYNA-1 blog](https://www.dyna.co/research/dyna-1)** (Dyna Robotics, 2025-06) — Reward-model-in-the-loop training over 6 weeks; self-reported 99.4% success over a 24-hour zero-intervention napkin run (850+ napkins, ~60% of human speed); method described only at a high level.
- **[DYNA-1 launch release](https://www.prnewswire.com/news-releases/dyna-robotics-unveils-dyna-1-the-first-commercial-ready-robot-foundation-model-offering-fully-autonomous-round-the-clock-dexterity-302441437.html)** (Dyna Robotics, 2025-04) — Press release for DYNA-1: self-reported 800+ napkins folded in a continuous 24-hour run with zero interventions, 99.4% success at 60% of human throughput; credits a proprietary reward model for error recovery and training-data generation.
- **[1X World Model (2024)](https://www.1x.tech/discover/1x-world-model)** (1X Technologies, 2024-09) — Blog and challenge arguing for a learned simulator as a repeatable evaluator. No quantitative link to real success. · [code](https://github.com/1x-technologies/1xgpt)

## Publication date not stated

- **[Cargo Cults, Data Flywheels, and Novelty Pumps](https://vedder.io/misc/novelty_pump.html)** — Essay: more robots repeating the same tasks yield redundant data; real improvement needs a labor-intensive "novelty pump" of ops teams seeking new states and tasks, not an automatic flywheel.
- **[Enact](https://www.ycombinator.com/companies/enact)** (Enact) — Startup that rolls out customer policies, finds failures and collects targeted recovery data; pi0.5 packing went from 90/100 (200 demos) to 99/100 with 50 added recovery demos (self-reported).
- **[Fern (Ishiki Labs)](https://www.ycombinator.com/companies/fern-bot)** (Fern / Ishiki Labs) — YC W26 startup: world-model RL environments replicating production scenes; adds RL heads atop customers' existing policies to raise speed and success weekly. No quantitative results published.
