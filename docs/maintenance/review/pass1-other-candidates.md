# Pass-1 candidate issues: narrative, family intros, foundations (UNVERIFIED)

Found by an automated audit that was interrupted before its independent verification step. Treat every item as a lead to check against the source, not as a confirmed error. Majors first.

## 1. [major] `data/explainer.json` — thesis, sentence 'Most recipes that work keep the large model frozen or tightly anchored and climb with something small: a noise vector, a residual or edit head, a critic, an advantage token, a verifier or a curated data slice.'

**Problem.** This overgeneralizes, and the repo's own examples contradict it. Several headline methods update the whole large model, some with no anchor at all. 'Advantage token' is also not a small thing being climbed: RECAP uses the token to condition a full retrain of the VLA.

**Evidence.** SimpleVLA-RL section 3.4 (arxiv.org/html/2509.09674) says 'we remove the KL divergence regularization following DAPO': this is full-parameter GRPO with no reference anchor. The papers.csv row C179 for RECAP says it 'retrains the whole flow VLA supervised'. PDE (C380) and F4R (C436) also PPO-update the VLA, and Temporal GRPO (C411) updates the full policy with GRPO. papers.csv family counts are onpolicy 69, advbc 58 and offpolicy 121, against residual 76, testtime 128 and blackbox 37, so 'most' cannot be checked.

**Proposed fix.** Replace with: "Many recipes that work keep the large model frozen and climb with something small: a noise vector, a residual or edit head, a critic, a verifier or a curated data slice. Others update the whole model but limit how far it can move. They restart each round from the pretrained checkpoint (RECAP), or they keep the supervised loss and change only which data or advantage label it copies. Full-model RL with no KL anchor also works when the starting policy is already decent (SimpleVLA-RL)."

## 2. [major] `data/explainer.json` — the_loop, Step 5: 'Others restart each round from the pretrained checkpoint to avoid drift (RECAP; DEED regressed from 42% to 22% in round 2).'

**Problem.** Putting the two side by side implies restarting prevents drift, with DEED as the warning about what happens otherwise. In fact DEED did restart every round from the base checkpoint, as RECAP does, and still regressed. The authors blame the data mix (self-generated rollouts crowding out the teleop demos), not the weights. The round-1 gain was also not statistically conclusive.

**Evidence.** DEED (arxiv.org/html/2607.20345): 'each refinement iteration is initialized from the original checkpoint (GR00T-N1.6-G1-PnPAppleToPlate) rather than the previous policy'. Table I: DE policy 32% (16/50), RECAP iteration 1 42% (21/50), iteration 2 22% (11/50). The authors call the gain 'measurable if not statistically conclusive' and attribute the round-2 drop to 'distributional drift, as self-generated rollouts come to dominate the training set'. RECAP (2511.14759): 'Both the value function and policy are finetuned from the pre-trained checkpoint ... useful for avoiding drift over multiple iterations'.

**Proposed fix.** Replace with: "Others restart each round from the pretrained checkpoint to avoid drift (RECAP). Restarting does not fix a skewed data mix, though. DEED, a small independent RECAP replication on a humanoid, restarted every round and still dropped from 42% (21/50) to 22% (11/50) in round 2. Its authors blame self-generated rollouts crowding out the teleoperation demos."

## 3. [major] `data/explainer.json` — industry_recipes[org=Physical Intelligence].evidence — 'Espresso about 11% -> about 92%'

**Problem.** Wrong baseline that overstates RECAP's effect. About 11% is the pretrained pi0.6 generalist with no task fine-tuning. The checkpoint RECAP's on-robot loop actually starts from is pi*0.6 offline RL + SFT at about 40%. The paper credits on-robot data with 'about a factor of two' fewer failures, but 11->92 implies about 11x.

**Evidence.** arXiv 2511.14759 Fig. 8 (Make_Espresso_success_rate_plot.png) shows bars of about 11% (pi0.6 Pretrain), about 27% (pi*0.6 OfflineRL Pretrain), about 40% (pi*0.6 OfflineRL+SFT) and about 92% (Ours). Sec. VI-C1: throughput 'more than doubles ... (the improvement from offline RL + SFT to the final pi*0.6 model), and the rate of failure reduces by about a factor of two.' Appendix A-F: cafe used a single iteration (429 correction + 414 autonomous episodes).

**Proposed fix.** Replace 'Espresso about 11% -> about 92%;' with 'Espresso: about 40% after offline RL + SFT -> about 92% after one RECAP round (about 11% for the pretrained pi0.6 with no task fine-tuning);'

## 4. [major] `data/explainer.json` — industry_recipes[org=AgiBot].disclosed — 'LWD does not disclose offline buffer size, intervention counts or compute.'

**Problem.** Factually wrong about a named lab's paper. LWD (v2, updated 2026-09-16) reports its offline buffer in full. Intervention counts and learner compute really are missing.

**Evidence.** arXiv 2605.00416 Table IV / Fig. 7: 'Offline data composition of the 652.5-hour buffer': 336.6 h demonstrations, 88.8 h successful rollouts, 39.2 h failed rollouts, 187.9 h play. Sec. V: 'each method is allocated a 4-hour wall-clock budget, corresponding to approximately 60 total hours of online data'. No GPU or compute figures appear anywhere in the paper.

**Proposed fix.** Replace with: 'LWD reports its 652.5-hour offline buffer (Table IV) and a ~60 robot-hour online budget, but not intervention counts or learner compute.'

## 5. [major] `data/explainer.json` — industry_recipes[org=1X Technologies].disclosed — 'Blogs only. The derivation of the alignment claim is not shown, and there is no published link to real success rates.'

**Problem.** Inaccurate. The 2025 1XWM blog links a 12-page technical report (1x.tech/1x-world-model.pdf). Its Figs. 7-9 plot world-model scores next to real Arcade task scores for checkpoint selection and for architecture A/B tests (ViT-B vs ViT-L, with vs without proprioception), and it reports alignment numbers (63.06% -> 71.17% with cross-task data). The real gap is that these comparisons are thin: no trial counts, no error bars, no correlation coefficient.

**Evidence.** 1X report Sec. 6.2: 'Given a true real-world success rate gap of 15% between two checkpoints, a World Model with 70% alignment can accurately predict the better policy with 90% success.' Fig. 7 caption: 'a checkpoint that scores significantly higher on the world model evaluation is also likely to score higher in real eval, although the margins are not necessarily similiar.' Blog: 'For more experiments, see our technical report.'

**Proposed fix.** Replace with: 'A blog plus a 12-page progress report. The report plots world-model scores against real Arcade task scores for a few checkpoints and architecture pairs, without trial counts, error bars or a correlation coefficient, and the 70%-alignment / 15-point / 90% claim is stated without derivation.'

## 6. [major] `data/explainer.json` — timeline[15] (CoVer, 2026-02) note

**Problem.** The note credits the 65.5/62.0 result to CoVer on a frozen pi0 and says it beats pi0 fine-tuned on the augmented data (44.0/48.7). In fact 65.5/62.0 is 'pi0 (rephrase) + CoVer', and pi0 (rephrase) IS pi0 fine-tuned on instruction-augmented data, the same model as the 44.0/48.7 baseline. That comparison sets 'fine-tuning + CoVer' against 'fine-tuning alone', not frozen pi0 + verifier against fine-tuning. The paper's frozen-pi0 + CoVer result is 57.0/61.0.

**Evidence.** arXiv 2602.12281 Sec. 5.3: '(2) pi0 (rephrase) [10] represents pi0 finetuned on instruction-augmented datasets'; Appendix 8.2: 'pi0 (rephrase) + CoVer: Combines training-time instruction augmentation with CoVer's inference-time optimization'. Table 3 (SIMPLER red-team, ID/OOD avg): pi0 41.5/29.7; pi0 w/ Inst. Aug. 44.0/48.7; pi0 + CoVer 57.0/61.0; pi0 (rephrase)+CoVer 65.5/62.0. Text: 'scaling verification (pi0 + CoVer) outperforms scaling policy learning ... 15% gains on ID tasks and 12% on OOD ... Combining pi0 (rephrase) and CoVer achieves the strongest overall performance'.

**Proposed fix.** Replace the note with: "Contrastive verifier picks among instruction rephrasings x sampled action chunks. SIMPLER red-team ID/OOD: frozen pi0 41.5/29.7 -> 57.0/61.0 with CoVer, beating pi0 fine-tuned on augmented instructions (44.0/48.7); stacking both gives 65.5/62.0."

## 7. [major] `data/explainer.json` — timeline[14] (RISE, 2026-02) note

**Problem.** 'with no physical rollouts' overstates the method. The self-improving loop runs in imagination, but the world model, value model and offline warm-up are trained on hundreds of logged real policy rollouts per task, and on human DAgger corrections for box closing.

**Evidence.** arXiv 2602.11075 App. XI-A: 'Dynamic Brick Sorting includes 3063 human demonstration data and 610 policy rollout data. Backpack Packing covers 2478 human demonstrations and 507 policy rollout data. Box Closing features 2286 human demonstrations, 524 policy rollouts, and 540 human corrections (DAgger) data.' App. XI-D: the policy is first warmed up 'following the offline RL approach [2]' with labels on 'the policy rollout data'. The headline numbers check out: Table: pi0.5 35/30/35 -> RISE 85/85/95; 'each of our evaluation results is based on an average score of 20 autonomous trials.' The same overclaim appears in data/papers.csv row C214 one_line ('Replaces physical rollouts ...').

**Proposed fix.** Replace the note with: "Compositional video world model plus value model drive advantage-conditioned pi0.5 training; the improvement loop runs in imagination (models are trained on demos plus ~500-600 logged real policy rollouts per task). Real: 35->85%, 30->85%, 35->95% (20 trials each)." Also soften data/papers.csv C214 one_line 'Replaces physical rollouts' to 'Replaces online physical rollouts'.

## 8. [major] `data/explainer.json` — timeline[21] (Siemens factory case study, 2026-05) note

**Problem.** 'A pure-SFT DAgger-style loop' mischaracterises the method. DAgger collects data in states the learner policy reaches. The paper says the recovery data came from teleoperating the robot into failure states instead of letting the policy reach them, and it lists human-in-the-loop takeover only as a future improvement.

**Evidence.** arXiv 2605.27461 Lessons: 'Implementing a human-in-the-loop workflow during evaluations (e.g., interrupting the execution and handing control to a teleoperator) would make collection of recovery behaviors more effective ... Currently, we first teleoperate the robot to a failure state instead of letting the policy do so, which may not match the distribution of actual failure states.' 'Between each round ... we ... collected Recovery data (242 episodes)'. The numbers check out: '2535 episodes (~10 hours) from the on-site factory settings'; earlier phase 'around 900 episodes were collected'; 'the final success rates of the finetuned policy did not meet expectations'.

**Proposed fix.** Replace the note with: "A pure-SFT iterate-and-retrain loop on pi0.5 in a real factory (teleop demos, manual curation, staged teleoperated recovery demos; no on-policy interventions): 2,535 factory episodes plus about 900 lab episodes. No success rate reported; the authors say final success 'did not meet expectations'."

## 9. [major] `data/explainer.json` — lessons[5].evidence (proxy-then-exact-metric lesson), sentence 'TD-ES after PPO: IQM 67.2% -> 85.0%.'

**Problem.** Wrong attribution. TD-ES is cited as an example of 'train on a convenient proxy first, then hill-climb the exact metric', but its second stage optimizes the same discounted return as PPO. The gain comes from switching optimizer (bounded ES), not from switching objective.

**Evidence.** arXiv 2511.09923 Sec 3.3: ES 'evaluate[s] returns J(θ+σδ) for the same objective (1)'. Algorithm rollout returns '∑γ^t r_t' (the RL discount). Success rate is only the reporting metric ('We report success rates rather than cumulative reward because...'). The numbers themselves (PPO 67.2%, TD-ES 85.0% IQM) are correct.

**Proposed fix.** Replace 'TD-ES after PPO: IQM 67.2% -> 85.0%.' with 'Related but not a metric switch: TD-ES refines a PPO checkpoint with bounded ES on the same return, IQM 67.2% -> 85.0%.' (or drop TD-ES from this lesson).

## 10. [major] `data/explainer.json` — lessons[5].evidence, EvolSAC sentence

**Problem.** Number given in the wrong setting. The sentence says SNES ran 'on the competition score' and then gives swing-up times of 2.085 s -> 1.135 s. Those times are from the simplified sim cartpole experiment, where the target metric is swing-up time, not from the AI Olympics competition score.

**Evidence.** arXiv 2507.10030 Sec 4.1 'Cartpole': 'Initially, the average swing-up times are 2.085 s for SAC ... SAC+SNES achieves ... 1.135 s ... continuing training with SAC ... additional 1000 episodes results in ... 1.565 s'. The competition (double pendulum acrobot/pendubot) is Sec 4.2.

**Proposed fix.** Replace the EvolSAC sentence with: 'EvolSAC: SAC on a shaped distance reward, then SNES directly on swing-up time; on sim cartpole 2.085 s -> 1.135 s, where 1000 more SAC episodes reached 1.565 s (the same recipe was then applied to the AI Olympics competition score).'

## 11. [major] `data/explainer.json` — lessons[6].evidence, PlayWorld sentence

**Problem.** Misattributed and incomplete. 'A demo-trained world model drove PlayWorld's policy to 25%' reads as if PlayWorld's own policy fell. It was the baseline: a policy fine-tuned inside a world model trained only on human demos. That baseline first rose to 45% before falling to 25% (carrot task). The values exist only in Fig. 9, so they are chart readings.

**Evidence.** arXiv 2603.09030 Fig. 9 (finetune.png), 'Remove Carrot From Bowl': base 0.30; Baseline 0.30, 0.30, 0.40, 0.45, 0.25 at 0/200/400/800/1200 steps. Text: the demo-trained baseline 'becomes unstable as the policy tries to "hack" the world model ... resulting in decrease in real-world success rates'. Appendix: 20 hardware trials per checkpoint.

**Proposed fix.** Replace with: 'In PlayWorld, a policy fine-tuned inside a world model trained only on human demos peaked at 45% and then fell to 25% on carrot removal, below its 30% base (chart reading, 20 trials per checkpoint).'

## 12. [major] `data/explainer.json` — lessons[0].evidence, last sentence 'The Golden Ticket paper notes search can only select or amplify existing behavior.'

**Problem.** Credits the paper with a statement it does not make. Nowhere does Golden Ticket say search can only select or amplify existing behavior. Its limitations section says the method needs a steerable policy and may need stronger search when the base is weak.

**Evidence.** arXiv 2603.15757 Sec 6: 'our method only works if the policy is steerable [45], which is not guaranteed' and 'reaching perfect performance may demand stronger search methods than ours, particularly when the base policy itself is weak'. No 'amplify'/'existing behavior' wording appears in the full text.

**Proposed fix.** Replace with: 'The Golden Ticket paper notes its noise search only works when the frozen policy is steerable, and may fall short when the base policy itself is weak.'

## 13. [major] `data/explainer.json` — lessons[9].evidence, last sentence 'R2S-Eval notes 20 binary trials cannot separate strong checkpoints.'

**Problem.** Unsupported paraphrase. R2S-Eval does not say this. It says real-world rankings can change across repeated evaluations and that success rate is too coarse. It does run 20 real trials per task, but it never claims those trials cannot separate strong checkpoints.

**Evidence.** arXiv 2609.03276 intro: 'The resulting rankings are difficult to repeat consistently ... the success rate is too coarse ... Policies with similar success rates may differ substantially in motion quality'. Sec on Tab. II: 'We run 20 and 200 trials per task in the real world and in simulation'. No statement about separating strong checkpoints.

**Proposed fix.** Replace with: 'R2S-Eval notes that real-world success-rate rankings can change across repeated evaluations, and that policies with similar success rates can differ substantially in execution quality.'

## 14. [major] `data/explainer.json` — lessons[3].evidence (off-policy vs on-policy robot time), 'By contrast, RL-100 used about 14 h of robot time per task, and FPO++ locomotion used about 147M-197M sim steps.'

**Problem.** Unfair comparison presented as evidence. EXPO-FT (19.1 min, starting from a pretrained pi0.5 SFT already at 20.5/30) and AutoSERL are set against RL-100. RL-100 used different tasks, trained from ~100-demo diffusion policies, and aimed for 100% over many consecutive trials. Most of its rollout time is iterated offline RL (PPO-style objective on stored data), so calling it simply 'on-policy' is a stretch. The 14 h also includes ~1.9 h of teleop demos. The FPO++ figure is not stated in the paper; it is derived arithmetic.

**Evidence.** RL-100 (arXiv 2510.14830, PDF) Table S3 averages per task: demos 1.9 h, iterative offline RL 6.8 h, online RL 5.3 h (text: 1.8 + 6.5 + 5.6 h). FPO++ (2602.02481): 4096 envs, 24 steps per update, 1500 (quadruped) / 2000 (humanoid) updates = 147.5M / 196.6M steps (computed). The ResFiT 200x figure is the only controlled comparison (sim BoxCleanup).

**Proposed fix.** Replace evidence with: 'ResFiT (sim BoxCleanup): 200k vs 40M steps against a PPO residual (about 200x). EXPO-FT: an average of 19.1 min of online data per task. AutoSERL: 50/50 in 8-45 min from one demo. For scale only (different tasks and starting policies, not a controlled comparison): RL-100, whose PPO-style objective runs over iterated offline and then online rollouts, logged about 14 h of data collection per task (about 1.9 h demos, 6.8 h offline-RL rollouts, 5.3 h online), and FPO++ locomotion trains on roughly 147M-197M sim steps (4096 envs x 24 steps x 1500-2000 updates).'

## 15. [major] `data/families.json` — trend field of blackbox, offpolicy, residual, hitl, testtime, reward, worldmodel, simreal, industry (rendered as 'What changed in 2025–26' in papers/*.md and site/index.html)

**Problem.** These trends are cut down to their first one or two sentences from docs/research/synthesis.json (trend_2025_2026). The published text announces a numbered list and shows only item (1), which looks broken and sloppy to readers.

**Evidence.** papers/offpolicy.md:12 'Five shifts happened in the last 12 months. (1) The target policy changed. The 2023-24 lineage ... from scratch.' and then nothing. The same happens with 'Three things changed' (blackbox), 'Four shifts' (residual, reward, simreal, industry), 'Five shifts' (hitl, testtime), and the worldmodel trend, which ends after '(1) ... The draw was on-policy RL without resets.' The full paragraphs are in docs/research/synthesis.json families[i].trend_2025_2026.

**Proposed fix.** Either restore the full paragraphs from synthesis.json after applying the corrections in the other findings, or remove the count and '(1)' and keep one complete statement. Suggested complete statements (each checked against papers.csv and the abstracts). offpolicy: "The target policy changed. The 2023-24 lineage (RLPD, SERL, HIL-SERL) trained small Gaussian SAC policies from scratch; in 2026 the same machinery is wrapped around pretrained flow VLAs such as pi0.5 (EXPO-FT)." testtime: "Verifiers moved from generic to policy-specific. V-GPS (2024) reranked samples with a generic offline-RL Q, and RoboMonkey (2025-06) with a 7B VLM verifier trained on synthetic preferences; newer work trains critics on the deployed policy's own rollouts with success labels." simreal: "The simulator changed jobs. Earlier sim-to-real trained small policies from scratch and hoped for zero-shot transfer; now it is also a practice gym for already-pretrained BC and VLA policies, anchored with real demos (for example RL-Co, Feb 2026)." industry: "RL entered industry recipes in supervised-friendly forms. π*0.6/RECAP (2025-11) showed a value function plus advantage-conditioned retraining beating PPO and AWR baselines on T-shirt folding with π0.6 (Gemma 3 4B backbone plus an 860M action expert)." For blackbox, residual, hitl, reward and worldmodel, see the separate findings.

## 16. [major] `data/families.json` — blackbox.trend

**Problem.** The text shows the internal catalog ID 'C155' to readers, who cannot decode it. It also never says that Countdown is a language-model reasoning task, so a robotics reader may think ES has beaten PPO/GRPO on robots. The full synthesis says 'Neither has robot results yet'. The count 'Three things changed' is dangling.

**Evidence.** site/index.html contains 'C155' twice. papers.csv C155 is 'Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning' (2509.24372; arXiv comment: 'Published at ICML 2026 main conference'). Paper Table 1: ES beats PPO and GRPO for all 7 Qwen/Llama models (0.5B-8B), population N=30; the RL baselines were grid-tuned and ES used one fixed setting.

**Proposed fix.** Replace with: "The biggest change in the last 12 months is scale. Evolution strategies went from 'works for small MLPs' to full-parameter fine-tuning of billion-parameter models: Evolution Strategies at Scale (Qiu et al., Sep 2025, ICML 2026) used a population of 30 to beat grid-tuned PPO and GRPO on the Countdown reasoning task for all seven LLMs it tried (0.5B-8B). This is a language-model result; robot results are still missing."

## 17. [major] `data/families.json` — onpolicy.trend

**Problem.** ReinFlow and FPO are listed among the papers that showed 'PPO/GRPO can train a VLA'. Neither paper trains a VLA: ReinFlow fine-tunes small flow-matching policies on locomotion and robomimic-style manipulation, and FPO trains diffusion/flow policies from scratch on continuous-control tasks.

**Evidence.** 2507.21053 abstract: 'FPO can train diffusion-style policies from scratch in a variety of continuous control tasks.' 2505.22094 abstract: 'fine-tunes a family of flow matching policies ... locomotion and manipulation tasks' (Rectified Flow and Shortcut models, compared with DPPO). Neither mentions a VLA.

**Proposed fix.** Replace with: "Twelve months ago (2025-05 to 2025-07), the question was whether PPO/GRPO can train a VLA at all. RIPT-VLA, VLA-RL, RL4VLA and TGRPO said yes, in simulation, while ReinFlow and FPO showed how to run PPO on flow-matching policies. From 2025-09 to 2025-10 this became standard recipes and infrastructure (SimpleVLA-RL, πRL, RLinf-VLA)."

## 18. [major] `data/families.json` — residual.trend

**Problem.** The text says freezing 'became the default because full RL fine-tuning of 3B models ... erodes recovery behavior (PPS, TRANSIC-style findings)'. Three problems: (a) PPS (Sep 2026) compares against supervised LoRA fine-tuning, not RL fine-tuning; it argues that keeping the base frozen preserves recovery behavior. (b) TRANSIC (2024) is about sim-to-real residuals learned from human corrections and found that BC fine-tuning forgets; it has nothing to do with RL on 3B VLAs. (c) PPS is from Sep 2026, so it cannot be the reason freezing 'became the default'. 'Default' is also an overstatement, since full RL fine-tuning (πRL, SimpleVLA-RL) is common. The count 'Four shifts' is dangling.

**Evidence.** 2609.09148 abstract: 'Because the base is never directly modified, its broad capabilities remain available ... including behaviors such as recovery from failure' and 'PPS outperforms LoRA fine-tuning'. 2405.10315 abstract: residual policies learned from human corrections for sim-to-real; no VLA or RL fine-tuning. Also, no residual-family one_line in papers.csv mentions OpenVLA-OFT as a frozen base.

**Proposed fix.** Replace with: "Bases moved from task-specific diffusion or ACT policies to frozen generalist VLAs (pi0, pi0.5, GR00T, SmolVLA). Freezing is attractive because full RL fine-tuning of multi-billion-parameter VLAs is costly, and directly fine-tuning the base can erode broad behaviors the new data never exercises, such as failure recovery (argued by PPS for supervised adaptation)."

## 19. [major] `data/families.json` — reward.trend

**Problem.** VIP is called a 'per-task ranker'. VIP is a universal visual reward and representation pretrained on Ego4D human video with no in-domain fine-tuning, which is the opposite of per-task. The count 'Four shifts stand out' is dangling.

**Evidence.** 2210.00030 abstract: 'Trained on large-scale Ego4D human videos and without any fine-tuning on in-domain, task-specific data, VIP's frozen representation can provide dense visual reward for ... unseen robotic tasks.' The numbers were checked and are correct: Robo-Dopamine '3,400+ hour', Robometer 'RBM-1M ... over one million trajectories', RynnValue 'over 7,000 hours'.

**Proposed fix.** Replace with: "Scale and generality. Per-task rankers (Rank2Reward) and small pretrained embedding rewards (VIP) gave way to VLM reward models trained on large pooled datasets: Robo-Dopamine on over 3,400 hours, Robometer on RBM-1M (over 1M trajectories), RynnValue on over 7,000 hours."

## 20. [major] `data/families.json` — worldmodel.trend

**Problem.** The text says all four papers ran RL 'inside pretrained video generators (Cosmos, OpenSora, Wan, SVD)'. VLA-RFT trained a 138M world model on LIBERO demos only, and World-Env does not name a pretrained base. Only WMPO (OpenSora backbone, pretrained on Open X) and ProphRL (Cosmos-Predict2 init, Wan2.1 autoencoder) clearly fit. The paper-to-generator mapping, including SVD, is not supported. The trend also stops after '(1)'.

**Evidence.** papers.json C163 (VLA-RFT): '138M world model', 'The world model is trained only on LIBERO demonstration data'. C180 (WMPO) builds_on: 'OpenSora-STDiT-v3 backbone'. C184 (ProphRL): 'Cosmos-Predict2-2B-Video2World init, Wan2.1 video autoencoder'. The World-Env paper names no pretrained generator.

**Proposed fix.** Replace with: "In the last 12 months the family went from 'world models as evaluators' to 'world models as RL environments for VLAs'. From Sept to Nov 2025, World-Env, VLA-RFT, WMPO and ProphRL ran GRPO/PPO-style RL for VLAs inside learned action-conditioned video models (WMPO on an OpenSora backbone, ProphRL initialized from Cosmos, VLA-RFT a 138M model trained on LIBERO demos), using learned or VLM success judges. The draw was on-policy RL without resets."

## 21. [major] `tools/build_site.py` — CHAPTER_HEADLINES["07"] = (r"noise-steering-best", "best steering run, chosen on the search set")

**Problem.** The site's Chapter 7 headline is 225/256 = 87.9% [83.3, 91.3]. That is one PSS seed, picked as the best of 9 noise-steering runs by search score. The chapter's own opening paragraph says PSS steering 'worked on one seed out of three' and leads with the pooled 622/768 = 81.0% [78.1, 83.6]. It also says the state-dependent noise policy 'did not beat the best fixed noise vector' (85.5%). The course table's lede promises 'pooled over seeds where there are seeds', so showing 87.9% overstates DSRL/PSS and contradicts the chapter.

**Evidence.** ch07.md line 3: 'Steering only the 8 most influential noise directions (PSS) worked on one seed out of three: pooled **622/768 = 81.0% [78.1, 83.6]**'. ch07.md line 101: 'PSS steering improved the zero ticket on one seed of three'. site/index.html renders 'best steering run, chosen on the search set · 225/256 · base 50.4%'. I re-ran chapter_results() with the regex r'dsrl-pss8_s\d+': it returns k=622, n=768, Wilson [78.1, 83.6].

**Proposed fix.** Change the entry to "07": (r"dsrl-pss8_s\d+", "PSS noise steering (k = 8), pooled over 3 seeds"). The site would then show 81.0% [78.1, 83.6], matching the chapter's opening. If the best-of-9 pick has to stay, make the label say 'single best of 9 runs by search score (1 of 3 PSS seeds improved; pooled PSS 81.0%)' and fix the lede (next finding).

## 22. [minor] `data/explainer.json` — thesis, sentence 'Pretrained VLA and BC policies now often start at 40-90% success, so the limit is no longer the optimizer.'

**Problem.** The range is too high for the papers this repo collects. The second half is an opinion stated as fact.

**Evidence.** I parsed 147 'X -> Y' before/after pairs from the papers.csv one_line fields. The median starting value is 39%: 67 start in 40-90%, 45 in 20-40%, 30 below 20% and 2 at 0. Repo examples that start low: Temporal GRPO SFT 38.3, CoRe pi0 18.3, F4R 26.25%, DEED 32%, Q-Planning wallet 25%.

**Proposed fix.** Replace with: "Pretrained VLA and BC policies, once fine-tuned on a task, now usually start well above zero. Most before/after numbers in this list start between 20% and 90%, with a median around 40%. So in many recent papers the optimizer is not the main bottleneck."

## 23. [minor] `data/explainer.json` — thesis, first sentence 'last-mile improvement of robot policies is one loop'

**Problem.** This is the author's framing, but it is stated as fact. On X it will draw 'not everything is this loop' replies.

**Evidence.** This is editorial. The explainer's own families include one-shot test-time methods with no retraining loop (CoRe, JITI).

**Proposed fix.** Replace 'In 2025-26, last-mile improvement of robot policies is one loop:' with 'In 2025-26, most last-mile improvement of robot policies can be read as one loop:'

## 24. [minor] `data/explainer.json` — the_loop, Step 0: 'If it never succeeds (0%), most methods have nothing to climb. SimpleVLA-RL stays at 0 -> 0 on RoboTwin 2.0 from a 0-demo SFT.'

**Problem.** The number is correct but the setting is vague. 'From a 0-demo SFT' means the OpenVLA-OFT base with no task fine-tuning, on five tasks. The step also leaves out an exception listed in this same repo: PDE climbs from a 0% start.

**Evidence.** SimpleVLA-RL Table 7 covers five RoboTwin 2.0 tasks. The '0 trajs SFT' row is 0 on every task both before and after RL, with the text 'every trajectory receives zero reward'. PDE row C380 reads 'LIBERO-PRO from 0% start: 81.8 at step 120', and the PDE abstract says 'PDE enables RL to learn successful policies even from zero-reward starts'.

**Proposed fix.** Replace with: "If it never succeeds (0%), most methods have nothing to climb: on five RoboTwin 2.0 tasks, SimpleVLA-RL on an OpenVLA-OFT with no task demonstrations stays at 0% after RL, because every rollout earns zero reward. The exceptions change how they explore instead. PDE has a VLM rewrite the instruction and climbs LIBERO-PRO from 0%."

## 25. [minor] `data/explainer.json` — the_loop, Step 2: 'Offline action MSE is a poor proxy: SIMPLER found average Pearson r 0.308, vs 0.924 for visual-matching sim.'

**Problem.** The numbers are correct but have no setting. They are an average over 3 Google Robot tasks for ranking 6 checkpoints, so a reader could take them as a general result.

**Evidence.** SIMPLER Table I (arxiv.org/html/2405.05941): Pearson r for Validation MSE is 0.464/0.230/0.231, average 0.308; for SIMPLER-VisMatch it is 0.976/0.855/0.942, average 0.924. The setting is 'ranking 6 common open-source policy checkpoints (3 RT-1 checkpoints, RT-1-X, RT-2-X, Octo-Base) on Google Robot tasks'.

**Proposed fix.** Replace with: "Offline validation action MSE is a poor proxy: when SIMPLER ranked 6 checkpoints on 3 Google Robot tasks, MSE averaged Pearson r 0.308 against real success, vs 0.924 for visual-matching sim."

## 26. [minor] `data/explainer.json` — the_loop, Step 3: 'or triggered by jumps in the critic's Q (JITI)'

**Problem.** JITI is not a failure detector. It flags 'decision-critical moments' for execution quality (elegance) in tasks that already succeed, and its trigger compares Q against a moving average.

**Evidence.** JITI (arxiv.org/html/2511.22555) defines the trigger as 'Δq_t = |q_t − q̄_t| ... q̄_t is the moving average over a short history window', firing on a 'sudden spike, whether reflecting entry into an ITC-relevant, high-value segment or a drop caused by uncertainty'. The goal is the 'Elegant Success Rate', not catching failures.

**Proposed fix.** Replace with: "or triggered when the critic's Q jumps away from its recent moving average (JITI, which targets lapses in execution quality rather than outright failures)"

## 27. [minor] `data/explainer.json` — the_loop, Step 4 bullet: 'test-time search changes no weights: it samples candidates and picks with a verifier or guides the sampler with a critic's gradient.'

**Problem.** Too broad. Methods the repo files under testtime train critics or verifiers. Q-Planning even retrains its Q-function every round, as Step 5 of this same text says. Only the policy weights stay fixed.

**Evidence.** Q-Planning (C392, family testtime) 'retrains only the critic'. JITI (C189, testtime) trains an offline Cal-QL critic.

**Proposed fix.** Replace with: "test-time search leaves the policy's weights unchanged: it samples candidates and picks with a verifier, or guides the sampler with a critic's gradient (the verifier or critic may itself be trained)."

## 28. [minor] `data/explainer.json` — industry_recipes[org=Physical Intelligence].evidence — 'T-shirt folding on identical data: RECAP ... vs AWR ... vs PPO ...'

**Problem.** The numbers match the chart, but 'identical data' suggests a clean method-only comparison. The AWR and PPO baselines start from pi0.6, while RECAP starts from the offline-RL-pretrained pi*0.6 + SFT, so initialization is confounded. The paper also notes that AWR's success (91%) is close to RECAP's; RECAP's real gap is throughput.

**Evidence.** 2511.14759 Sec. VI-B: 'AWR. Starting from the same pre-trained model pi0.6 (without advantage conditioning)'. Fig. 11 labels: 'pi0.6 +AWR', 'pi0.6 +PPO' vs 'pi*0.6 Ours'. Readings: about 60/30/22 per hr and 96/91/75%.

**Proposed fix.** Replace 'T-shirt folding on identical data:' with 'T-shirt folding, same data, different policy-extraction methods (AWR and PPO start from pi0.6, not the offline-RL pi*0.6 checkpoint):'

## 29. [minor] `data/explainer.json` — industry_recipes[org=Physical Intelligence].disclosed — 'the text's iteration claims do not clearly match the charts'

**Problem.** The claim is true but vague, and could read as an unexplained accusation. Make it specific.

**Evidence.** 2511.14759 Sec. VI-C2 text: laundry 'overall 50% improvement in throughput' and box assembly 'after the second iteration we see a 2x improvement in throughput'. Fig. 9 bars: laundry about 33 -> 42 -> 60/hr (i=0 -> i=2, about 1.8x); box about 10 -> 8.5 -> 13.3/hr (about 1.3x from i=0, about 1.6x from i=1).

**Proposed fix.** Replace 'and the text's iteration claims do not clearly match the charts' with 'and the text's iteration claims do not match Fig. 9 (box assembly is described as a 2x throughput gain but reads about 10 -> 13/hr from i=0 to i=2; laundry is described as +50% but reads about 33 -> 60/hr)'

## 30. [minor] `data/explainer.json` — industry_recipes[org=Physical Intelligence].evidence — '13 hours of continuous espresso operation'

**Problem.** This is not in the arXiv paper. It comes from PI's blog timelapse and press coverage. 'Continuous operation' can read as fully autonomous, but the paper says the system relies on human resets and interventions, and the blog gives no intervention count.

**Evidence.** Not present in the arXiv 2511.14759 text. Press coverage describes 'a timelapse video ... operating an espresso machine continuously for 13 hours'. Paper Sec. VII: 'our system is not fully autonomous: it relies on human labeling and effort for reward feedback, interventions, and episode resets.'

**Proposed fix.** Replace with: 'a 13-hour espresso timelapse in the blog video (intervention and reset counts not stated).'

## 31. [minor] `data/explainer.json` — industry_recipes[org=Physical Intelligence].recipe — 'a small TD3-style actor-critic on a compressed token from the frozen pi0.6'

**Problem.** Small precision issue. The VLA is first fine-tuned on 1-10 h of task demos together with the RL-token module, and only then frozen. As written, it sounds like the base pi0.6 is used as-is.

**Evidence.** 2604.23073: 'We first adapt the VLA on a small amount of task-specific [data] ... We then freeze the VLA and train lightweight off-policy actor and critic networks'. 'For each task, we collect 1–10 hours of teleoperated demonstrations. We then fine-tune the VLA model while training our R[L token]'.

**Proposed fix.** Replace 'from the frozen pi0.6' with 'from pi0.6 (task-fine-tuned on 1-10 h of demos, then frozen)'

## 32. [minor] `data/explainer.json` — industry_recipes[org=Dyna Robotics].evidence — 'but only 75% of folds at production grade'

**Problem.** Mislabels Dyna's metric. Dyna calls the same run 'high production-grade quality'. The 75% is the share meeting its strictest bar, while 98% reached near-perfect quality (grade 3 or higher).

**Evidence.** dyna.co/research/dyna-1: 'While 98% of folds reach near-perfect quality (grade ≥3), only 75% hit our rigorous quality bar.' Week 6: 'Sustained 24+ hours with ~800 folds and high production-grade quality'.

**Proposed fix.** Replace 'but only 75% of folds at production grade' with 'but only 75% of folds met Dyna's strictest quality bar (98% reached grade 3 or higher)'

## 33. [minor] `data/explainer.json` — industry_recipes[org=Dyna Robotics].disclosed — 'Dyna credits Dyna-2 gains to pretraining, new capabilities, tooling and systems together.'

**Problem.** Slight misattribution. Dyna names better pretraining, model steerability (language following), and improved tooling and systems. It does not say 'new capabilities'.

**Evidence.** dyna.co/research/scaling-customer-deployments credits Dyna-2 gains to better pretraining, enhanced model steerability (language following), and improved tooling and systems.

**Proposed fix.** Replace with: 'Dyna credits Dyna-2 gains to better pretraining, steerability (language following), and improved tooling and systems together.'

## 34. [minor] `data/explainer.json` — industry_recipes[org=Generalist AI].recipe — 'Five named post-training ingredients: pretraining advances, post-training techniques, ...'

**Problem.** Calling all five 'post-training' ingredients is self-contradictory and misquotes the blog. One of the five is pre-training advances.

**Evidence.** generalistai.com/blog/gen-1 describes GEN-1 as comprising 'innovations across pre-training advances, post-training techniques, learning from experience (RL), multimodal human guidance, as well as new inference-time techniques'.

**Proposed fix.** Replace 'Five named post-training ingredients:' with 'Five named ingredients, none detailed:'

## 35. [minor] `data/explainer.json` — industry_recipes[org=Generalist AI].evidence and disclosed — 'Box assembly in 12.1 s vs about 34 s for GEN-0 and pi0.' / 'No algorithm, ablation or trial counts.'

**Problem.** (a) The 12.1 s figure counts only folding time (first touch to fold complete), and the GEN-0 and pi0 times come from their own public videos, not a head-to-head run. (b) 'No trial counts' is slightly off. The blog gives consecutive-success streaks (e.g. 200 box folds in a row, 86 T-shirts in a row) but no denominators for the 99/64/19% averages.

**Evidence.** GEN-1 blog: 'we only count the time spent folding the box, from the moment of touching it ... For the previous SOTA, both GEN-0 (source) and π0 (source) used identical boxes and took roughly 34 seconds'. Also: 'Folding Boxes 200 times in a row without intervention', 'T-Shirt Folding ... 86 times in a row without intervention'.

**Proposed fix.** Evidence: 'Box folding time 12.1 s vs about 34 s for GEN-0 and pi0 (folding-only time, with baselines measured from their public videos).' Disclosed: 'Closed blog. No algorithm or ablation, and no trial counts behind the success percentages (only consecutive-success streaks such as 200 box folds in a row).'

## 36. [minor] `data/explainer.json` — industry_recipes[org=Sunday Robotics].recipe — 'pretrain on glove-based sensorized human data'

**Problem.** The ACT-2 post never says 'glove'. It says the data was collected with Sunday's proprietary hardware. The glove detail comes from the ACT-1 post, so here it is an inference stated as fact.

**Evidence.** sunday.ai/blog/act-2-preview: 'ACT-2 is pretrained on a high-quality, high-diversity, sensorized human dataset collected through Sunday's proprietary data collection hardware'. The words 'glove' and 'wearable' do not appear on the page.

**Proposed fix.** Replace 'pretrain on glove-based sensorized human data' with 'pretrain on sensorized human data from Sunday's proprietary capture hardware (a glove in ACT-1)'

## 37. [minor] `data/explainer.json` — industry_recipes[org=Sunday Robotics].disclosed — 'No ablation separates pretraining from the loop.'

**Problem.** Ambiguous, and it partly contradicts the evidence line. The post does ablate pretraining scale under a fixed post-training procedure (the 82->0 gap). What is missing is any measure of how much of the 99.1% comes from the hill-climbing loop.

**Evidence.** ACT-2 post: 'we apply the same post-training procedure at each pretraining scale' (Fig. 1). The post-training loop itself is 'cover[ed] ... in a separate technical post'.

**Proposed fix.** Replace with: 'No ablation isolates how much of the 99.1% comes from the hill-climbing loop (the pretraining-scale study holds post-training fixed).'

## 38. [minor] `data/explainer.json` — industry_recipes[org=Skild AI].disclosed — 'The post does not say the policy was initialized from S1.'

**Problem.** Technically true, but it omits that the post's framing implies S1 ('a strong base model like S1 can learn ... by competing against itself') while also saying the policy 'could barely walk' at first. State both, so this does not read as Skild hiding something it never claimed.

**Evidence.** skild.ai/blogs/physical-self-play: 'Our result shows that a strong base model like S1 can learn to complete extremely dexterous and dynamic tasks, like soccer, by competing against itself in a simulation.' 'In its first few simulated months inside NVIDIA's Isaac Sim, it could barely walk.'

**Proposed fix.** Replace with: 'The post frames this as post-training 'a strong base model like S1' but never states that the soccer policy was initialized from S1 (early on it 'could barely walk').'

## 39. [minor] `data/explainer.json` — industry_recipes[org=1X Technologies].evidence — 'The 2024 version reports only token-level baselines (GENIE_138M 8.79 CE loss).'

**Problem.** The 8.79 figure is not in the 2024 blog, which only states the 8.0 prize threshold. It comes from the 1X World Model Challenge GitHub leaderboard. Attribute it correctly.

**Evidence.** github.com/1x-technologies/1xgpt README leaderboard: '1x-technologies/GENIE_138M (--maskgit_steps 2) | 8.79' (temporally teacher-forced CE loss). The 2024 blog mentions only 'a loss of 8.0 on our private test set'.

**Proposed fix.** Replace with: 'The 2024 challenge reports only token-level baselines (GENIE_138M at 8.79 CE loss on its GitHub leaderboard).'

## 40. [minor] `data/explainer.json` — industry_recipes[org=Figure AI].recipe — 'Helix logistics: no RL.'

**Problem.** The blog never mentions RL, so 'no RL' is an inference from silence. Phrase it as what the blog describes. Everything else in this entry checks out. The ablation figure has no error bars or trial counts, and it also shows the 50% larger decoder lowering barcode success from 94.4% to 92.5%.

**Evidence.** figure.ai/news/scaling-helix-logistics: the improvements were 'achieved through both data scaling and model architectural improvements'; behaviors 'learned from demonstration'. UPS_GRAPH_6: 82.5 -> 84.0 -> 85.1 -> 94.4 -> 92.5% and 6.64 -> 5.24 -> 4.68 -> 4.31 -> 4.05 s, with no error bars.

**Proposed fix.** Replace 'Helix logistics: no RL.' with 'Helix logistics: no RL described.'

## 41. [minor] `data/explainer.json` — decision_guide[3].why (strong SFT VLA / on-policy PPO row)

**Problem.** The text says 'In three controlled comparisons PPO beat GRPO' but lists only two. Both sources come from the same RLinf team and codebase, so they are not independent replications. ADEPT is quoted as evidence in a VLA row, but it is sim PPO for a state-based dexterous-hand policy, not a VLA. The row's 'use' also recommends a 'BC or KL term', while ADEPT found that a KL penalty did not rescue the high-LR runs (at LR 1e-5 with beta=1 it still got 0/5 seeds).

**Evidence.** piRL Table 9 (Flow-SDE, LIBERO): pi0 SFT 57.6, +GRPO 90.0, +PPO 96.0; pi0.5 SFT 77.1, +GRPO 91.5, +PPO 97.9. RLinf-VLA Table I (ManiSkill): OpenVLA GRPO 84.38 / PPO 96.09 ID; OpenVLA-OFT GRPO 94.14 / PPO 97.66. ADEPT Table 3 on KUKA-Allegro FMB peg insertion: (d) Standard PPO at LR 1e-3 gives 0/5 and collapses at ADR 20; (a) full recipe gives 5/5. ADEPT text: 'A KL penalty ... does not rescue it, and at a learning rate of 10^-5, adding the penalty with beta=1 still yields 0% across all five seeds.' piRL used 8x H100 80GB, which supports the GPU claim.

**Proposed fix.** Replace why with: "piRL took pi0 from 57.6% to 97.6% on few-shot LIBERO (sim). In the RLinf team's own same-codebase comparisons, PPO beat GRPO every time (piRL Flow-SDE: pi0 96.0 vs 90.0, pi0.5 97.9 vs 91.5; RLinf-VLA ManiSkill ID: OpenVLA 96.09 vs 84.38). ADEPT (sim PPO for a dexterous hand, not a VLA): with standard PPO at LR 1e-3, 0/5 seeds reach ADR 50 (all collapse at ADR 20), while the low-LR recipe gets 5/5, and adding a KL penalty did not rescue the runs."

## 42. [minor] `data/explainer.json` — decision_guide[4].why (policy at or near 0% row)

**Problem.** 'LIBERO-PRO Regime 3' is a label from the paper's Table 2 that readers will not recognize, and it is a single task inside a multi-task run. PDE itself was still at 0% at step 30 on that task; the gain came from skills transferred from related tasks. 'The advantage is zero' is exactly true only for group-relative estimators (GRPO/RLOO). With a critic there is simply no success signal to learn from.

**Evidence.** PDE Table 2 (object-task perturbation, task 2 = Regime 3): Action Noise 0.0/0.0/0.0 at steps 0/30/120; PDE 0.0/0.0/81.8. Text: 'successful exploratory rollouts from related tasks can train reusable skills that transfer to harder tasks.' MiDAS Table 1 LIBERO-Long average: Zero-Shot 0.0, BC 33.5, DSRL 33.5, MiDAS 91.2 (correct as stated).

**Proposed fix.** Replace why with: "With 0/1 reward and every rollout in a group failing, group-relative advantages (GRPO/RLOO) are exactly zero and a critic sees no success to learn from. On one LIBERO-PRO task where action-noise PPO stayed at 0% for 120 steps, PDE reached 81.8% (skills transferred from prompts that worked on related tasks). Noise steering cannot leave the base's support: from a 1-demo BC base at 33.5% on LIBERO-Long (zero-shot 0%), DSRL stays at 33.5 while MiDAS reaches 91.2."

## 43. [minor] `data/explainer.json` — decision_guide[8].why (advantage-weighted BC row), RECAP sentence

**Problem.** The RECAP numbers (~60/hr vs ~30/hr) are read off a bar chart (Fig. 11) and are not stated in the text. The AWR baseline starts from pi0.6, while RECAP runs on pi*0.6 with advantage-conditioned offline-RL pretraining, so the gap is not purely conditioning vs AWR. 'Kept throughput at about 60/hr' is also the wrong verb: RECAP raised throughput from ~33/hr (offline RL + SFT). The Dissecting numbers are correct (one platform, 4 tasks, 20 trials per condition).

**Evidence.** pi*0.6 paper Fig. 11 (Laundry T-Shirts and Shorts throughput): pi*0.6 Ours ~60, pi0.6+AWR ~30.5, pi0.6+PPO ~22, pi*0.6 OfflineRL+SFT ~33. Success chart: Ours ~96%, AWR ~91%. Text: 'AWR can achieve a reasonable success rate, but leads to much slower policies with lower throughput.' Dissecting Table II success: SFT 0.11, DAgger 0.45, Weight 0.74, Filter 0.50.

**Proposed fix.** Replace the RECAP sentence with: "In RECAP's T-shirt/shorts comparison on the same data (read from the chart), RECAP on pi*0.6 reached about 60 successes/hr vs about 30/hr for AWR on pi0.6, at similar success rates (about 96% vs 91%)." Optionally write the first sentence as "On one real bimanual platform (4 tasks, 20 trials each)..."

## 44. [minor] `data/explainer.json` — decision_guide[10].why (HG-DAgger / interventions row), SOP sentence

**Problem.** The model is 'Base-1/2' (pretrained on half of ~160 h of data), not 'Base'. The '3 h' is wall-clock time on a multi-robot AgiBot fleet with human takeovers, so robot-hours and human effort are larger than '3 h' suggests next to '80 more hours of demos'.

**Evidence.** SOP Sec. V-E: 'For the Base-1/2 model, augmenting the offline demonstration dataset with an additional 80 hours of human-collected data results in only a modest improvement in success rate, from 0.576 to 0.612. In comparison, applying SOP yields ... from 0.571 to 0.800—using only 3 hours of on-policy interaction.' Sec. V-B: each experiment has a 3-hour (180 min) wall-clock budget; SOP+HG-DAgger; fleet of 10 dual-arm robots (4 in the ablations). RaC: 78.3% with 5 hours of data (correct).

**Proposed fix.** Replace the SOP sentence with: "SOP: about 3 h of wall-clock fleet interaction with human takeovers took the Base-1/2 model (pretrained on half of ~160 h of data) from 0.571 to 0.800, while 80 more hours of demos took it from 0.576 to 0.612."

## 45. [minor] `data/explainer.json` — decision_guide[15].why (rough simulator / sim PPO row)

**Problem.** The F4R gain is shown without its budget-matched baseline. Targeted BC already reaches 71.25% OOD, so the method-specific gain is +18.75 points, not 26→90. The TacCoRL no-anchor result comes from one task (Assembly #2) in an ablation. The RL-Co baseline (45.9) is SFT co-training, which should be stated.

**Evidence.** F4R table, OOD avg: Base 26.25%, Targeted BC 71.25%, F4R 90.0%. The abstract also frames the gain as 'outperforming the budget-matched Targeted BC baseline by 18.75 percentage points'. TacCoRL Sec. 4.5, Assembly #2: alpha=0.5, beta=0 gives 45% real; Table 2 visuo-tactile Assembly #2 Sim-Real Co-Training gives 0.55. RL-Co Table 1 pi0.5: SFT Co-Training 45.9, RL-Co 66.2.

**Proposed fix.** Replace why with: "RL-Co took real pi0.5 from 45.9 (SFT co-training) to 66.2 with zero real RL. F4R raised real OOD success from 26.25% to 90.0%, vs 71.25% for budget-matched targeted BC. On one TacCoRL task (Assembly #2), sim RL with no anchor (beta=0) fell below co-training alone in real success (45% vs 55%)."

## 46. [minor] `data/explainer.json` — decision_guide[13].why (stand-in evaluator row), Veo sentence

**Problem.** The text attributes MMRV 0.03 and Pearson 0.88 to '1600+ real trials'. Those metrics come from the nominal-scene comparison of 8 policy checkpoints (Fig. 4). The 1600+ figure is the total number of real evaluations across the whole report, including the OOD and safety studies.

**Evidence.** Veo paper Fig. 4 (nominal_correlation.png): 'MMRV = 0.03, Pearson = 0.88' across policies A-H. Abstract: 'We validate these capabilities through 1600+ real-world evaluations of eight Gemini Robotics policy checkpoints and five tasks.' The same plot shows predicted rates of 0-0.3 against real rates of 0.05-0.7, which supports the row's claim that absolute rates are wrong. PolaRiS r=0.9 and RoboWorld r=0.989 vs RoboArena (8 policies, 4,186 rollouts) verified.

**Proposed fix.** Replace with: "Veo eval: MMRV 0.03 and Pearson 0.88 across 8 policy checkpoints in nominal scenes (the paper reports 1600+ real trials in total). PolaRiS: average r = 0.9. RoboWorld: r = 0.989 with the RoboArena leaderboard (8 policies). Learned evaluators often get rankings right but absolute rates wrong."

## 47. [minor] `data/explainer.json` — decision_guide[14].why (world-model RL row)

**Problem.** The VLA-MBPO rollout-length numbers come from a LIBERO-Long simulation ablation, but the sentence follows a real-robot RISE claim without saying so. The RISE claim of no physical rollouts is true for the improvement loop. However, its world model and value model are trained on prior demos, policy rollouts and DAgger data, which readers should know.

**Evidence.** VLA-MBPO Table 3 (all ablations on LIBERO-Long suite): Branched 1/2/4 chunks = 63.9/66.8/62.9, Full Horizon = 52.8; Table 2 pi0.5 SFT LIBERO-Long = 54.6. RISE Table I: pi0.5 35/30/35% to RISE 85/85/95%. RISE Sec. III: offline data includes 'expert demonstrations and policy rollouts with success and failure ... a fraction of DAgger data'.

**Proposed fix.** Replace why with: "RISE took real pi0.5 from 30-35% to 85-95% with no physical rollouts during the improvement loop (its world and value models are trained on prior demos, rollouts and corrections). On LIBERO-Long (sim), VLA-MBPO shows full-horizon imagined rollouts falling below the SFT base (52.8 vs 54.6), while 2-chunk branches reach 66.8."

## 48. [minor] `data/explainer.json` — decision_guide[6].why (residual RL row), DAWN sentence

**Problem.** DAWN's 'warm-up' is not a separate critic pre-training phase. It means filling the replay buffer with base-policy transitions ('implicit warmup'), plus critic normalization. 'Critic warm-up' could be read as ADEPT-style frozen-actor critic training. 'Most of what makes residual RL efficient' is also stronger than the paper's 'simple yet principled solutions suffice'. The ResFiT numbers are correct (14% to 64%, 134 rollouts, ~15 min; ~200x = 200k vs 40M steps on sim BoxCleanup vs PPO residual).

**Evidence.** DAWN abstract: 'base-policy transitions serve as an essential value anchor for implicit warmup, and critic normalization effectively restores representation sensitivity'. DAWN Sec. on normalization compares LayerNorm and Hyperspherical Normalization. ResFiT: 'With 134 rollouts ... (≈15 minutes ...) ResFiT boosted the performance of the base model from 14% to 64%'; 'converging at 200k steps versus 40M steps, ... ∼200× boost'.

**Proposed fix.** Replace the DAWN sentence with: "DAWN finds that seeding replay with base-policy transitions plus critic normalization (LayerNorm) fixes the critic cold-start and scale-mismatch problems that slow residual RL."

## 49. [minor] `data/explainer.json` — decision_guide[11].why (best-of-N verifier row), UF-OPS sentence

**Problem.** The 38% to 87% figures are read from a bar chart (Fig. 5) and are not stated in the text, but they appear as exact numbers. They do match the abstract's 'average 49% improvement'. The SeeQ numbers (34/96 = 35.4%, 64/96 = 66.7%) are correct.

**Evidence.** UF-OPS: real results are only in Fig. 5; text: 'UF-OPS increases performance of the base policy on all instances, with gains spanning from 25% to 80%'; abstract: 'average 49% improvement in success rate over the base policy across 5 real tasks'; '60 rollouts are saved during evaluation for training the verifiers'; Best-of-N with N=10. papers.csv one_line already marks this as '(chart reading)'.

**Proposed fix.** Replace with: "UF-OPS raised the real ALOHA mean from about 38% to 87% (read from the chart; the abstract reports +49 points) with best-of-10, using a verifier trained on 60 labeled evaluation rollouts per task."

## 50. [minor] `data/explainer.json` — timeline[11] (Veo world-sim evaluation and PolaRiS, 2025-12) note

**Problem.** Wrong baseline/setting. MMRV 0.03 and Pearson 0.88 come from the nominal-scenario comparison (8 GROD checkpoints x 80 scene-instruction pairs, Fig. 4), not from all 1600+ real trials, which cover every experiment in the report (nominal, OOD, safety). Also, 'phone-scan' is not in the PolaRiS paper, which says 'short (2-5 minutes) monocular video'.

**Evidence.** arXiv 2512.10675 Sec. 3.2/Fig. 4: 'We compare predictions ... with real-world paired evaluations for the 80 scene-instruction combinations ... for eight variants of the GROD policy' with 'MMRV = 0.03 / Pearson = 0.88' (PDF). OOD panels give different values (e.g. MMRV 0.06 / Pearson 0.86). Abstract: '1600+ real-world evaluations of eight Gemini Robotics policy checkpoints and five tasks'. PolaRiS (2512.16881): 'We start by capturing a short (2–5 minutes) monocular video recording of the scene'; 'average Pearson correlation of r=0.9'; worst case over 'all six tested environments is r=0.81'.

**Proposed fix.** Replace the note with: "Two validated evaluators. Veo: MMRV 0.03 and Pearson 0.88 vs real success for 8 policy checkpoints on nominal scenes (the report has 1600+ real trials overall). PolaRiS video-scan twin: average Pearson r = 0.9 across 6 unseen scenes."

## 51. [minor] `data/explainer.json` — timeline[9] (WMPO, 2025-11) note

**Problem.** 'MimicGen with 128 real rollouts' reads as real-robot data. In this benchmark the 128 'real trajectories' were collected in the MimicGen simulator. WMPO's real-robot experiment is a separate one.

**Evidence.** arXiv 2511.09515: 'We conduct experiments in the Mimicgen simulation [23] ... Coffee_D0, StackThree_D0, ThreePieceAssembly_D0, and Square_D0'; Table 1, P=128: Base 33.6, GRPO 33.2, DPO 37.3, Ours 47.1.

**Proposed fix.** Replace the note with: "GRPO inside a video world model refit on the policy's own rollouts, failures included. MimicGen (sim), 128-rollout budget: 33.6 -> 47.1, vs 33.2 for online GRPO on the same budget."

## 52. [minor] `data/explainer.json` — timeline[17] (Robometer, 2026-03) note

**Problem.** The time figure is slightly off. The paper says ≤45 minutes (10k timesteps), not 'about 40 min'.

**Evidence.** arXiv 2603.02115 Sec. on online RL: 'DSRL+Robometer improves success from 20% to 85% in ≤ 45 minutes (10k timesteps), outperforming RoboReward's 55%.' Table XXIII: RoboReward TP 6 / FP 45; Robometer TP 18 / FP 0 (verified).

**Proposed fix.** Replace '20% -> 85% in about 40 min' with '20% -> 85% in at most 45 min (10k steps)'.

## 53. [minor] `data/explainer.json` — timeline[6] (PLD, 2025-10) note

**Problem.** The 91.8 -> 99.2 figure is the average over LIBERO Spatial/Object/Goal only; LIBERO-Long is not included. Without that, 'LIBERO' reads as the standard 4-suite average.

**Evidence.** arXiv 2511.00091 Table 1 columns: Spatial | Object | Goal | Avg for pi0 and OpenVLA; OpenVLA baseline 92.9/99.1/83.25 -> Avg 91.8; w/ PLD Avg 99.2.

**Proposed fix.** Replace 'LIBERO OpenVLA-OFT 91.8 -> 99.2.' with 'LIBERO (Spatial/Object/Goal average) OpenVLA-OFT 91.8 -> 99.2.'

## 54. [minor] `data/explainer.json` — timeline[12] (SOP, 2026-01) note

**Problem.** The note puts '10 robots' next to the 0.571 -> 0.800 result, which mixes two settings. The 10-robot fleet is the main multi-task run. The 0.571/0.800 vs 0.576/0.612 comparison is from the pretraining-scale analysis on the Base-1/2 model, and the paper does not say it used the 10-robot fleet.

**Evidence.** arXiv 2601.03044 Sec. V: 'partitioning the 10-robot actor fleet across tasks'; Sec. V-E: 'For the Base-1/2 model, augmenting the offline demonstration dataset with an additional 80 hours ... from 0.576 to 0.612. In comparison, applying SOP yields ... from 0.571 to 0.800—using only 3 hours of on-policy interaction.' Default algorithm: 'choose SOP+HG-DAgger'.

**Proposed fix.** Replace the note with: "Fleet-scale online HG-DAgger for pi0.5 (10 AgiBot G1s in the main run). Pretraining-scale ablation (half-data base): 3 h of on-policy interaction 0.571 -> 0.800 vs 80 more hours of demos 0.576 -> 0.612."

## 55. [minor] `data/explainer.json` — timeline[25] (RoboWorld and DEED, 2026-07) note

**Problem.** Calling DEED 'a RECAP replication' is unfair to both papers. DEED describes its method as 'adapted from RECAP' (text-based advantage prefix, VLM value function, GR00T N1.6, single GPU), and the iteration-1 gain over SFT was not significant. Placed in a timeline, 'RECAP replication regressed' can read as evidence against RECAP itself.

**Evidence.** arXiv 2607.20345 abstract: 'experience-driven refinement, adapted from RECAP via a text-based advantage prefix and a vision-language value function'. Results: data-efficient SFT 32% (16/50), RECAP iteration 1 42% (21/50), iteration 2 22% (11/50). RoboWorld (2607.01060): 'Pearson correlation of 0.989 ... with the original real-world leaderboard' over 8 open-sourced RoboArena policies (verified).

**Proposed fix.** Replace the note with: "RoboWorld neural evaluator: Pearson r = 0.989 vs the RoboArena leaderboard (8 policies). Separately, a low-budget RECAP-style adaptation on GR00T N1.6 (DEED) went 32% (SFT) -> 42% -> 22% over two iterations (50 trials each)."

## 56. [minor] `data/explainer.json` — timeline[24] (Sunday ACT-2 Preview, 2026-07) note

**Problem.** These are self-reported company blog numbers (no paper and no independent evaluation), but the note states them as plain fact. Other industry entries (GEN-1) say 'Reports'. The gap experiment also uses 50 evaluations per point.

**Evidence.** sunday.ai/blog/act-2-preview: 'Across 785 autonomous attempts spanning 9 major garment types, ACT-2 achieved an overall success rate of 99.1% (±0.3% standard error)'; gap table: 0% pretrain 96% in-domain / 14% out-of-domain (gap 82) ... 100% pretrain 100%/100% (gap 0), '50 evaluations behind each success rate'.

**Proposed fix.** Replace the note with: "Company-reported: 99.1% +/- 0.3% SE over 785 laundry attempts in unseen homes. The in- vs out-of-domain gap after the same post-training shrinks from 82 to 0 points with full pretraining (50 evals per point)."

## 57. [minor] `data/explainer.json` — timeline[2] (ResFiT, 2025-09) note

**Problem.** The ~200x sample-efficiency figure comes from a single simulated task (BoxCleanup: 200k vs 40M steps). The note reads as a general sim result.

**Evidence.** arXiv 2509.19301 Sec. V-A: 'On the simulated BoxCleanup task ... our approach converging at 200k steps versus 40M steps, we see a ~200x boost in sample efficiency'.

**Proposed fix.** Replace 'about 200x more sample-efficient than a PPO residual in sim.' with 'about 200x more sample-efficient than a PPO residual on the sim BoxCleanup task.'

## 58. [minor] `data/explainer.json` — timeline[8] (pi*0.6 / RECAP, 2025-11) note

**Problem.** The chart readings are correct, but 'T-shirt ablation on identical data' leaves out that the task is 'T-shirts and Shorts' laundry and that the AWR/PPO baselines start from pi0.6, not the offline-RL pi*0.6 checkpoint. The paper itself notes the shared data slightly favours the baselines.

**Evidence.** arXiv 2511.14759 Sec. VI: 'AWR. Starting from the same pre-trained model pi0.6 (without advantage conditioning)'; 'We use the T-shirts and Shorts task ... we use the same data ... This provides a slight advantage to the baselines'. Fig. 11 PNGs: Ours ~60/hr, ~96%; AWR ~30/hr, ~91%; PPO ~22/hr, ~75%.

**Proposed fix.** Replace the second sentence with: "Laundry (T-shirts and shorts) ablation on the same data: about 60/hr and 96% vs AWR about 30/hr and 91% and PPO about 22/hr and 75% (chart readings; baselines start from pi0.6)."

## 59. [minor] `data/explainer.json` — lessons[7].evidence, Skill-Space Shooting sentence

**Problem.** Overstated. 'beat matched extra human demos on all 4 tasks' is true only as point estimates. On two tasks the margin is 1/30 and 2/20 trials, with one run per task and the plateau taken as the best of the last three evaluations. That is exactly the small-evaluation issue lesson 10 warns about.

**Evidence.** arXiv 2609.38178 Table 1 (best of last three evals, one run per task): Stack-3 0.833 vs human 0.800 (N=30), Drawer 0.600 vs 0.500 (N=20), Coffee 71.25±7.50 vs 46.25±15.00, Sweeping 83.75±5.63 vs 63.13±11.88.

**Proposed fix.** Replace with: 'Skill-Space Shooting's machine corrector scored above duration-matched extra human demos on all 4 tasks: clearly on Coffee (71 vs 46 progress) and Sweeping (84 vs 63), narrowly on Stack-3 (0.83 vs 0.80) and Drawer (0.60 vs 0.50), one run each.'

## 60. [minor] `data/explainer.json` — lessons[8].evidence, HELP sentence

**Problem.** Mechanism slightly misattributed. 25 -> 42 tasks/hr is the round-2 Microplate result, HITL-only vs HELP at a matched recovery budget. The paper says throughput gains are driven mainly by higher success (60% -> 90%), not by removing slowness. The slowdown appeared in round 1, including for HELP itself.

**Evidence.** arXiv 2607.09776 Table 4, Microplate round 2: HITL 25/hr, 60%, 85.1 s; HELP 42/hr, 90%, 77.7 s. Test Tube: base 93 s, HELP(i=1) 105 s. Text: 'the throughput improvement is primarily driven by the increase in success rate ... the added HITL recovery data teaches the policy to recover ... at the cost of longer execution time.'

**Proposed fix.** Replace with: 'HELP: training on recovery data made round-1 policies slower (e.g. Test Tube 93 s -> 105 s per task); with rollout segmentation, round-2 Microplate reached 42 tasks/hr at 90% vs 25 at 60% for HITL-only at a matched recovery budget, mostly from higher success.'

## 61. [minor] `data/explainer.json` — lessons[4].evidence, VT-Refine sentence

**Problem.** Mixes sim and real numbers in one comparison. 0.48 -> 0.57 (30 -> 50 demos) is BC evaluated in simulation. 0.50 -> 0.85 is real-robot. The like-for-like sim comparison is 0.48 -> 0.94 after RL.

**Evidence.** arXiv 2510.14930 Table 3 ('results are only compared in simulation'): visuo-tactile 30 demos BC 0.48 / RL 0.94; 50 demos BC 0.57 / RL 0.92. Table 1 real table-top visuo-tactile means: pre-train 0.50, RL 0.85.

**Proposed fix.** Replace with: 'VT-Refine (visuo-tactile, evaluated in sim): going from 30 to 50 demos moved BC only from 0.48 to 0.57, while sim RL from the 30-demo policy reached 0.94; on the real robot, RL raised the 30-demo policy from 0.50 to 0.85.'

## 62. [minor] `data/explainer.json` — lessons[1].evidence, PAC-ACT, FlowDAgger, Sequential GRPO and TacCoRL clauses

**Problem.** Missing setting context. PAC-ACT's 72.7% comes from a sparse-reward ablation and is the final training-rollout window (cumulative 22.0%), not a main-table eval. 'Sequential GRPO' does not name the paper. FlowDAgger's 'keeps 0.88' hides that the frozen base was 0.96. The TacCoRL figures are for one real task (Assembly #2).

**Evidence.** PAC-ACT 2607.09590 Sec on sparse ablation: 'cumulative total success rate is only 22.0%, and the final rollout window also drops to 72.7%', with shortcut behavior; 100% with KL. Simple Recipe Works 2603.11653 table: Seq. FT 81.2 vs 'Without LoRA' 7.3. FlowDAgger 2607.08877 Table 3: base held-out mean 0.96, FlowDAgger 0.88, SFT 0.02. TacCoRL 2606.11743 Sec 4.5 (Assembly #2): beta=0 gives 45% real; Table 2 co-training 0.55; beta 0.1/1.0 gives 80%.

**Proposed fix.** Replace the evidence after the ADEPT sentence with: 'PAC-ACT (sparse-reward ablation): without its KL anchor the final training window drops to 72.7% (22.0% cumulative) as the policy learns a shortcut, vs 100% with it. Simple Recipe Works (continual VLA RL): sequential GRPO with LoRA gives AVG 81.2 vs 7.3 with full fine-tuning. FlowDAgger keeps held-out skills at 0.88 (frozen base 0.96) vs 0.02 for SFT. TacCoRL (real Assembly #2): with beta=0 the policy reaches 45%, below co-training alone (55%); beta=0.1 or 1.0 reaches 80%.'

## 63. [minor] `data/explainer.json` — lessons[0].lesson and evidence (select-and-amplify lesson)

**Problem.** One-sided framing. The cited sources contain counter-examples that the text leaves out. PDE itself goes from 0% to 81.8%, by eliciting latent behavior through reworded prompts. SimpleVLA-RL reports emergent 'pushcut'. ADEPT learns plate grasping and a flip-and-regrasp the base never showed. MiDAS, cited here, leaves the base's support. 'rarely create new behavior' is defensible only for zero-success starts. Also, 0-demo 'stays at 0 -> 0' is awkward phrasing.

**Evidence.** PDE 2607.08837: 'PDE reaches 81.8% success by step 120, while Action Noise, Robometer, and RND remain at 0%'. SimpleVLA-RL 2509.09674 Sec 6.1 'pushcut ... the VLA model autonomously discovers a more efficient solution'. ADEPT 2608.19182: 'a flip-and-regrasp strategy not successfully exhibited by the pretrained policy'. SimpleVLA-RL Table 7: 0-trajectory SFT 0% before and after RL.

**Proposed fix.** Lesson: 'These methods mostly select and amplify behavior the base policy already has. They rarely create competence from a zero-success start. Pick your method according to how often the base succeeds.' Evidence: 'SimpleVLA-RL: an OpenVLA-OFT base with no task demos starts at 0% on RoboTwin 2.0 and stays at 0% after RL. PDE: action noise, Robometer and RND all stay at 0 from a 0% start; PDE itself reaches 81.8% only by eliciting latent behavior through reworded prompts. From a 1-demo BC base at 33.5% on LIBERO-Long (zero-shot 0%), noise steering (DSRL) averages 33.5, while full-chunk editing (MiDAS) leaves the base's support and reaches 91.2. The Golden Ticket paper notes its search only works when the frozen policy is steerable. Exceptions exist: SimpleVLA-RL\'s emergent "pushcut" and ADEPT\'s plate flip-and-regrasp were not shown by the base.'

## 64. [minor] `data/explainer.json` — lessons[2].evidence, RoboReward clause

**Problem.** The setting is missing. r=0.83 was measured in simulated Robomimic, across checkpoints of the authors' reward model, not on real robots.

**Evidence.** arXiv 2601.00675 Sec 3: 'we use the Robomimic benchmark ... There is a clear correlation (r=0.83)'.

**Proposed fix.** Replace with: 'RoboReward: in simulated Robomimic, offline reward accuracy correlates with RL success at r=0.83.'

## 65. [minor] `data/explainer.json` — lessons[8].evidence, uncertainty-gated exploration noise clause; and Foresight clause

**Problem.** The uncertainty-gated clause overgeneralizes a narrow, heavily hedged study. It is one small-compute LIBERO-10 SmolVLA setup, collapse occurred in some seeds, and the bf16 training kept 96% of action-expert weights bit-identical. Separately, the Foresight subtask numbers are oracle-initialized.

**Evidence.** 2609.28838 abstract: 'under a matched small-compute budget on LIBERO-10 with a 450M-parameter SmolVLA ... fixed noise collapses tasks in two of three seeds ... No arm improves on the behavior-cloning baseline in this budget ... 96.02% of the action expert's elements stay bit-identical'. BC tare 136/210 = 0.648. Foresight 2607.16506 Table II: subtasks 'initialized from oracle states'.

**Proposed fix.** Replace with: 'Uncertainty-gated exploration noise (one small-compute LIBERO-10 study with SmolVLA): PPO fine-tuning collapsed individual tasks in most fixed- and learned-noise seeds while the aggregate looked healthy, and no arm beat the BC baseline of 0.648. Foresight: oracle-initialized subtasks reach 98.4/92.2/99.2 but the chained full task stays at 54.5%.'

## 66. [minor] `data/explainer.json` — lessons[6].evidence, VLA-MBPO and SOLE-R1 clauses

**Problem.** Small framing issues. VLA-MBPO attributes the full-horizon drop to compounding model error, not to the policy exploiting the model, and the result is on LIBERO-Long only. SOLE-R1's 42%/9% comes from the authors' own qualitative failure review.

**Evidence.** 2603.20607 Table 3 (LIBERO-Long): full horizon 52.8 vs SFT 54.6; 'significant performance degradation due to intolerable compounding model errors'. 2603.28730 Table 1: 'Quantitative summary of failure modes from qualitative review'; GPT-5 42% vs SOLE-R1 9% perceptual hallucination of success.

**Proposed fix.** Replace with: 'Full-horizon world-model rollouts compound model error and fall below SFT on LIBERO-Long (VLA-MBPO 52.8 vs 54.6).' and 'GPT-5 as reward: in the SOLE-R1 authors\' failure review, 42% of failures come from hallucinated success, vs 9% for SOLE-R1.'

## 67. [minor] `data/explainer.json` — lessons[7].evidence TimelyDAgger clause; lessons[5].evidence Q-Planning clause

**Problem.** Missing setting (sim vs real, task). TimelyDAgger's 42 vs 78 is a sim study (ManiSkill3 OpenDrawer, Grasp-OOD). Q-Planning's 90 vs 55 is the real stack-cups task, starting from 40%.

**Evidence.** 2609.33157 Sec V-B2: 'expert suffixes at six fixed takeover steps on OpenDrawer (Grasp-OOD) ... increases from 42% with immediate takeover to 78% at step 80'. 2608.21204 abstract: 'stack-cups 40% to 90% ... whereas SFT on successful rollouts alone stalls at 55%'.

**Proposed fix.** Use: 'TimelyDAgger (sim OpenDrawer) at a matched expert budget: 42% with immediate takeover vs 78% with takeover at step 80.' and 'Q-Planning: on real stack-cups, critic-only rounds on a frozen BC policy go from 40% to 90% where SFT on successful rollouts stalls at 55%.'

## 68. [minor] `data/families.json` — advbc.trend

**Problem.** Three problems. (a) It calls one paper 'the LeHome winner', but that paper placed 1st in the online simulation round and 2nd in the real-world final. (b) 'copied' implies the 12 named groups did derivative work; 'adopted or adapted' is fairer to them. (c) 'Advantage-weighted BC gave way to advantage-conditioned BC' is an overclaim: the LeHome entry combines AWR and RECAP, and the trend's own C435 result finds soft exponential weighting best. Separately, CFGRL as the theory is fine: RECAP says its formulation is 'most closely related to CFGRL'.

**Evidence.** 2606.27163 abstract: 'placed 1st of 62 teams in the online (simulation) round and 2nd in the real-world final' and 'AWR + RECAP combined'. papers.csv C369 DEED: iteration 2 regressed to 22%.

**Proposed fix.** Replace with: "The big shift came with π*0.6/RECAP in November 2025. Advantage-conditioned BC with classifier-free guidance, which suits flow VLAs whose likelihood is awkward to weight, joined advantage-weighted BC as a main option; CFGRL (May 2025) is its closest theoretical basis, as RECAP itself notes. In the next ten months at least a dozen groups adopted or adapted the recipe: GigaBrain RAMP, χ0, DexPIE, DEED, ROVE, STEAM, Robo-ValueRL, PACE, DistAL, Facet-0, CLIFT and a prizewinning LeHome Challenge entry."

## 69. [minor] `data/families.json` — hitl.trend

**Problem.** Being-H0.5 and 'Gr-Dexter' (correct spelling GR-Dexter) are named as policies being corrected in HITL loops. Neither is in papers.csv, both are base-model or system technical reports rather than correction-loop papers, and the HITL family synthesis cites no HITL paper that uses them. GO-1 is supported only by Genie Centurion (May 2025), which is outside the 'last 12 months' window. The count 'Five shifts' is dangling.

**Evidence.** There is no 'Being' or 'Dexter' VLA entry in papers.csv. arXiv 2512.24210 (GR-Dexter) is a hardware, teleop and training-recipe report, and 2601.12993 (Being-H0.5) is a cross-embodiment VLA. SOP (2601.03044) and HELP (2607.09776) claims were checked and are correct: fleet to cloud learner; teleoperator plus floor operator; throughput.

**Proposed fix.** Replace with: "The policy being corrected is now a VLA: pi0/pi0.5-class models replace small BC-RNNs, and the loops are built as systems: SOP streams fleet corrections to a cloud learner, and HELP splits operator roles (a teleoperator and a floor operator) and reports tasks per hour."

## 70. [minor] `data/families.json` — offpolicy.built_on[1]

**Problem.** RLPD is called SAC's 'real-robot variant'. RLPD (ICML 2023) was evaluated on D4RL, Adroit and similar simulated benchmarks with no robot experiments; SERL and HIL-SERL later used it on real robots.

**Evidence.** 2302.02948 abstract describes only benchmark results; papers.csv C037: 'The engine of SERL and HIL-SERL.'

**Proposed fix.** "Max-entropy actor-critic (SAC) and its offline-data variant RLPD (the engine of SERL/HIL-SERL)"

## 71. [minor] `data/families.json` — simreal.use_when

**Problem.** The text ends on 'Pick by situation.' with nothing after it, because the cut removed the numbered list of situations.

**Evidence.** synthesis.json simreal.when_to_use continues with '(1) A pretrained BC/VLA policy ... (5) Precision where the actuator gap dominates ...'.

**Proposed fix.** Either delete the last sentence ('Pick by situation.') or append: "With a pretrained BC/VLA policy, a few real demos and a rough sim, use BC-anchored sim PPO (RL-Co); for recurring deployment failures, rebuild the failing scenes (F4R). Avoid it for deformables, fluids or contact physics your simulator cannot fake."

## 72. [minor] `data/families.json` — industry.question, industry.use_when, industry.built_on[3]

**Problem.** The entry states company claims as fact. 'How do real robot companies actually climb ... to 99%' assumes the 99% claims are established, but the family's own trend says the strongest claims 'come without update rules or ablations'. 'The order that shows up across companies' and 'Pretraining scale makes local hill-climbing trustworthy' overstate what partial disclosures support.

**Evidence.** synthesis.json industry.trend_2025_2026: 'The strongest claims (99% GEN-1, 99.1% ACT-2, Dyna-2's 95/hr) come without update rules or ablations ... The field's sense of the loop is sharper than its evidence about which ingredient does the work.'

**Proposed fix.** question: "How do robot companies say they climb from ~60-80% to 99% on a paying task, and what do they actually disclose?" use_when: replace "The order that shows up across companies:" with "A common order, pieced together from partial company disclosures:" built_on[3]: "Broad pretraining as the reason local hill-climbing transfers (a company claim, e.g. ACT-2)"

## 73. [minor] `data/families.json` — industry.trend (π*0.6 size)

**Problem.** '4B flow VLA' understates π0.6. The 4B is the Gemma 3 backbone alone; the action expert adds 860M parameters.

**Evidence.** arXiv 2511.14759: 'The base VLM is Gemma 3 4B model' and 'The size of the action expert is increased to 860M parameters.' The comparison against PPO and AWR on T-shirt folding was checked and is correct.

**Proposed fix.** "... beating PPO and AWR baselines with π0.6 (Gemma 3 4B backbone plus an 860M action expert)."

## 74. [minor] `data/families.json` — generalrl.trend

**Problem.** 'converged on the same toolkit' is an overclaim. The repo's own offpolicy and residual trends say real-robot work runs mainly on off-policy critics (SAC/TD3, RLPD-style), not critic-free group baselines.

**Evidence.** synthesis residual trend: 'nearly all real-robot papers now use SAC or TD3 with RLPD-style demo mixing'.

**Proposed fix.** "RL for language models and RL for robot policies increasingly share a toolkit in 2025-26: critic-free group baselines, KL-anchored fine-tuning from a pretrained model, learned verifiers, and spending compute at test time. Real-robot work still leans on off-policy critics."

## 75. [minor] `data/families.json` — onpolicy.use_when

**Problem.** 'GPUs for millions of rollouts' overstates the budget. Papers report millions of environment steps but thousands to tens of thousands of episodes; for example, RL4VLA overtakes SFT after about 0.4M environment steps. Also, the RL4VLA numbers 0.781 and 0.938 are correct (Table 1, in-distribution success), but the paper's own text calls RL 'comparable to SFT-16k in the training setting', so the table should be cited.

**Evidence.** rl4vla PDF: 'RL overtakes the strongest SFT checkpoint (SFT-16k) after roughly 0.4 M environment steps ... At convergence, it performs comparably to SFT-16k in the training setting'; Table 1 IND Suc.: SFT 0.781 (.013), RL 0.938 (.013).

**Proposed fix.** "(3) GPUs for many thousands of parallel simulated rollouts (millions of environment steps). ... (RL4VLA, Table 1: more demos stopped helping at 0.781 in-distribution success, PPO reached 0.938)."

## 76. [minor] `data/foundations.csv` — row 'Perturb and score', equation column

**Problem.** The equation 'θ' = θ + σε; keep it if R improves' describes greedy hill climbing only. Most of the cited references do something else. Kohl & Stone, OpenAI ES and ARS estimate a finite-difference gradient from the perturbations. CMA-ES refits a search distribution. Chapter 1 teaches exactly this distinction (greedy vs FD/ARS on a smoothed objective).

**Evidence.** ch01.md lines 11-13 separate greedy (keep if strictly better) from antithetic FD and ARS. ES (1703.03864) and ARS (1803.07055) step along Σ R(θ+σε_i)ε_i.

**Proposed fix.** Change the equation to: "θ' = θ + σε; keep the best (hill climbing) or step along Σ R(θ+σεᵢ)·εᵢ (ES/ARS)"

## 77. [minor] `data/foundations.csv` — row 'Reward-weighted averaging', equation column

**Problem.** 'π_new ∝ π_old · exp(R/λ)' fits RWR-style methods. AWR exponentiates the advantage A/β. PI² uses exp(−S/λ) with a path cost S. PoWER weights by return without exponentiating. The notation is loose for 2 of the 3 cited references.

**Evidence.** AWR abstract 1910.00177 (advantage-weighted). PI² (JMLR v11, theodorou10a) uses an exp(−cost/λ) reweighting, which is also how ch02.md describes it.

**Proposed fix.** Change to: "π_new(a) ∝ π_old(a) · exp(R(a)/λ)  (AWR: advantage A; PI²: −cost)"

## 78. [minor] `data/foundations.csv` — row 'Residual learning on a frozen base', equation column

**Problem.** 'a = a_base + α · tanh(u)' presents the tanh bound as the classic form. Both cited classics (Johannink et al. 1812.03201, Silver et al. 1812.06298) add an unbounded learned residual, π = π_base + f_θ. The tanh bound comes from recent work (EXPO-FT, ResFiT-style) and this course.

**Evidence.** Residual RL for Robot Control (1812.03201) and Residual Policy Learning (1812.06298) both use an additive residual with no tanh bound.

**Proposed fix.** Change to: "a = a_base + f_θ(s)  (bounded form: a_base + α·tanh(u))"

## 79. [minor] `data/foundations.csv` — row 'Greedy selection & guidance', equation and refs

**Problem.** The only equation is the argmax over samples, but the first reference is classifier-free guidance, which reweights the score/velocity rather than picking an argmax. The 'guidance' half of the row has no equation.

**Evidence.** Classifier-Free Diffusion Guidance (2207.12598) mixes conditional and unconditional scores with a guidance weight. It does not select samples.

**Proposed fix.** Change the equation to: "a* = argmax over samples of Q(s,a); or guide: (1+w)·∇log p(a|s,c) − w·∇log p(a|s)"

## 80. [minor] `data/foundations.csv` — row 'Bellman backups on replay', equation column

**Problem.** 'Q(s,a) ← r + γ Q(s', a')' leaves a' undefined. In the cited methods it is max over a' (DQN), a deterministic actor (DDPG/TD3), or a sample from the policy with an entropy term (SAC/RLPD). The terminal mask is also omitted.

**Evidence.** DQN (nature14236) uses max_a' Q. SAC (1801.01290) uses a' ~ π(·|s').

**Proposed fix.** Change to: "Q(s,a) ← r + γ Q(s', a'),  a' ~ π(·|s') (DQN: max over a')"

## 81. [minor] `docs/chapters/ch08.md` — line 3 ('This is the recipe behind EXPO-FT's real-robot fine-tuning') and the EXPO-FT row of 'Papers it mirrors'

**Problem.** The chapter's main arm keeps the base frozen and calls that 'the recipe behind EXPO-FT'. It never says that EXPO-FT's default co-trains the VLA, or that freezing the VLA in their ablation failed to reach 30/30. Line 58 notes that EXPO trains the base, but the intro framing misdescribes the mirrored paper. A careful reader on X could call it out.

**Evidence.** Same EXPO-FT HTML quotes as the previous finding. ch08 line 3: 'leaves the base's weights alone (except in one ablation)... This is the recipe behind EXPO-FT's real-robot fine-tuning'.

**Proposed fix.** Add to line 3: "(EXPO-FT also keeps training the VLA with its own supervised loss by default, and its frozen-VLA ablation misses 30/30. Our main arm freezes the base, and the co-trained ablation below is the closer copy.)"

## 82. [minor] `tools/build_awesome.py` — START_HERE entry C276 (EXPO-FT)

**Problem.** 'Off-policy RL around a VLA without backprop through it' reads as if the VLA stays frozen. EXPO-FT's default jointly updates the base VLA with its original supervised objective. Only the RL/Q gradient stays out of the VLA. The paper's ablation that freezes the VLA does not reach 30/30 within the same interaction budget.

**Evidence.** arxiv.org/html/2605.25477: 'The base VLA is updated using its original training objective, without modification.' 'This avoids gradient backpropagation through the VLA model'. 'By default, EXPO-FT jointly adapts the base VLA together with the edit policy... we freeze the base VLA... EXPO-FT does not reach 30/30 successes within the same interaction budget.'

**Proposed fix.** Replace with: "Off-policy RL around a VLA with no RL gradient through it: propose chunks, apply a small bounded edit, execute the best under a Q-ensemble. The VLA itself keeps training on its own supervised loss."

## 83. [minor] `tools/build_awesome.py` — START_HERE entry C179 (pi*0.6 / RECAP)

**Problem.** 'guidance pushes it toward positive' suggests classifier-free guidance is part of the method. As far as I can verify from the paper's HTML, the default inference just conditions on 'Advantage: positive' (equivalent to β = 1). CFG with β > 1 is presented as an option. Verdict: plausible. The HTML summary I got was not fully verbatim on which experiments use CFG. papers.csv's one_line ('using CFG at test time') has the same issue.

**Evidence.** arxiv.org/html/2511.14759: 'we can directly sample from the policy with I_t=True (which corresponds to setting β=1...)' ... 'or to use both a conditional and unconditional model to implement classifier-free guidance (CFG), which enables inference with β>1.'

**Proposed fix.** Replace the end with: "...the model is retrained with its supervised loss, and at test time it is conditioned on 'positive' (optionally with classifier-free guidance)."

## 84. [minor] `tools/build_awesome.py` — START_HERE entry C259 (UF-OPS)

**Problem.** 'No weight updates' is loose wording. The method trains a verifier network. What it leaves untouched is the base policy's parameters.

**Evidence.** Abstract 2603.10282: 'training verifier functions using policy rollout data... without changing the base parameters'.

**Proposed fix.** Replace 'No weight updates.' with 'No policy weight updates.'

## 85. [minor] `tools/build_awesome.py` — CHAPTERS entry 2: base idea 'Reward-weighted averaging'

**Problem.** The chapter covers CEM, CMA-ES, PI² and BO. The chapter itself says BO 'takes a different route' (a surrogate model plus an acquisition function, with no weighted averaging), so the base-idea label does not cover a quarter of the chapter.

**Evidence.** ch02.md line 19: 'Bayesian optimization (BO) takes a different route. It keeps every trial, fits a probabilistic model...'

**Proposed fix.** Change the base idea to: "Reward-weighted averaging (plus BO's surrogate model)"

## 86. [minor] `tools/build_site.py` — sec_course lede ('pooled over seeds where there are seeds, and never the best seed picked by its held-out score') and the comment above CHAPTER_HEADLINES

**Problem.** The lede says headlines are pooled over seeds wherever seeds exist. That is false for Chapter 3 (racing_s1, a single run, picked as the best of 9 random/cem/racing runs by search score) and for Chapter 7 (a single best-of-9 run). Both chapters ran 3 seeds per method. Chapter 3's doc does lead with this peek-free pick, so the label itself is fine. The general claim is what's wrong.

**Evidence.** CHAPTER_HEADLINES['03'] = r'racing_s1' gives 219/256. Pooling racing over its 3 seeds gives 616/768 = 80.2% [77.2, 82.9] (recomputed). In ch03.md the headline is 'best of the 9 random/cem/racing runs by search'.

**Proposed fix.** Change the lede to: 'The result column is the headline each chapter leads with, on the held-out set with its 95% Wilson interval: pooled over seeds for per-method results, or, for Chapters 3 and 7, the single run picked by search score (it is charged the cost of every run it was picked from). Never the best seed picked by its held-out score.' Update the code comment the same way.

## 87. [minor] `tools/build_site.py` — sec_course: Wilson interval shown on pooled k/768 headlines (ch01, ch02, ch05, ch06, ch08, ch12)

**Problem.** The site shows pooled Wilson intervals over 768 episodes as if they were independent trials. Chapter 1 itself warns that these pooled intervals are too narrow, because the 3 seeds share the same 256 start states, and says to quote the per-seed rows. The site does the opposite of what its own chapter recommends.

**Evidence.** ch01.md line 88: 'These pooled intervals are too narrow. The three runs of a method share the same 256 start states, so pooling them does not give three times the evidence. Quote the per-seed rows.'

**Proposed fix.** Add one sentence to the lede: 'Pooled intervals treat the 3 seeds' episodes as independent; the seeds share the same 256 start states, so the true uncertainty is wider (see each chapter's per-seed rows).' A better option is to show a start-state cluster-bootstrap interval, as Chapter 0 does.
