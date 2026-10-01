# Chapters

Twelve ways to climb one base policy from about 50% toward 95%+, ordered from classic to modern. Every chapter starts from the same frozen base (`base_v1`, or the mis-tuned scripted controller in chapters 1 and 2), is scored on the same held-out initial states, and reports its cost in robot-minutes and human-minutes. The full plan is in [docs/research/curriculum.md](../research/curriculum.md) §1, and the contract every chapter follows is in [docs/DESIGN.md](../DESIGN.md). These simulation scores use the current CupDrop-v1 success rule, which can count a cube deposited after the cup moves or is tipped and righted. Three of the base’s 129 successes moved the cup by more than 1 cm; the review does not establish a no-shoving guarantee.

Everything runs in MuJoCo on a laptop CPU. The SO-101 is the default robot and the real-hardware target; the AgileX PiPER runs in sim through the same environment API.

Paper ids (`C###`) match [data/papers.csv](../../data/papers.csv).

| # | Chapter | The base idea | Papers mirrored | Code | Status |
|---|---|---|---|---|---|
| 0 | [The base policy](ch00.md) | Behavior-clone a flow-matching chunk policy (rectified flow) from mixed-quality demos. At a fixed training budget the sloppy fraction sets the success rate; more demos of the same mix help only if training grows with them. Calibrated on the search set: `base_v1` scores 129/256 = 50.4% on the eval set, with pass@8 of 95.3%. | Rectified flow, flow matching, ACT, Diffusion Policy; Golden Ticket ([C236]), RINSE ([C274]) | [`algos/00_bc_flow.py`](../../algos/00_bc_flow.py) | available |
| 1 | [Hill climbing with a success counter](ch01.md) | Perturb, evaluate, step: greedy hill climbing, antithetic finite differences on a Gaussian-smoothed success rate, and ARS (reward-std normalization, top-b directions). On the mis-tuned scripted controller all three go from 48.0% to 99-100% on the eval set, at 280-420 search robot-minutes per run before tuning costs; a 2.5× larger FD step falls off cliffs, and a random candidate is a lottery. | Kohl & Stone 2004 ([C000]), ARS ([C012]), ES at Scale ([C155]), TD-ES ([C186]), EvolSAC ([C134]) | [`algos/01_hill_climb.py`](../../algos/01_hill_climb.py) | available |
| 2 | [One update, different weights: CEM, CMA-ES, PI2/MPPI and BO](ch02.md) | Sample candidates, weight them (elites, ranks or `exp(-cost/λ)`) and move to the weighted mean; BO fits a GP and picks the next trial with an acquisition function. CEM, CMA-ES, PI2 and BO all reach 99.7-99.9% on the eval set (pooled over 3 seeds). BO was not more sample-efficient than local sampling here, but it beat global Sobol search; local random search matches the final score at higher robot-minute cost, and a hardware-sized BO budget failed on 2 of 3 seeds. | Zero-order optimization primer ([C130]), A decade of BO for controller tuning ([C445]), PI2 ([C002]), CMA-ES tutorial ([C005]), BOpt-GMM ([C057]) | [`algos/02_cem_cmaes_pi2.py`](../../algos/02_cem_cmaes_pi2.py), [`algos/02b_bo.py`](../../algos/02b_bo.py) | available |
| 3 | [The golden ticket: black-box search over one noise vector](ch03.md) | Freeze `base_v1` and search its 32-dim input noise (random search, CEM, racing). Select on the search set, report the eval set: the best of the methods fixed in advance (racing, seed 1) goes from 50.4% to 85.5%; a method added after peeking at results reaches 87.5%, which we report but do not quote. The free zero ticket already gives 75.8%, random tickets are mostly worse than fresh noise, and small searches overstate their pick. | Golden Ticket ([C236]), DSRL ([C120]), Diffusion-ES ([C055]), FPO++ ([C208]) | [`algos/03_golden_ticket.py`](../../algos/03_golden_ticket.py) | available |
| 4 | Likelihood-ratio policy gradients with anchors | `∇ = E[∇ log π(a\|s) · A]` with PPO clipping; GAE vs GRPO group baselines; KL/BC anchors to the frozen base so it does not collapse. | SimpleVLA-RL ([C140]), piRL ([C157]), PAC-ACT ([C352]), FPO++ ([C208]), ADEPT ([C400]), Prism-GRPO ([C385]) | `algos/04_ppo_flow.py` | planned |
| 5 | [Off-policy critics: SAC → RLPD → Q-chunking](ch05.md) | Bellman backups on replay; RLPD's 50/50 batches and ensemble critics; a chunk critic whose bootstrap uses the same selection as acting. Corrected Q-chunking rerun: 50.4% → 93.9% [92.0, 95.4] pooled over three seeds, with 95.9–101.0 fresh training robot-minutes per seed. Scratch-actor SAC and RLPD score 3.0% and 16.3%; demos-only QC-CQL scores 56.6% [53.1, 60.1]. Prior development costs are charged separately. | Q-chunking ([C133]), Three Regimes ([C172]), IPE ([C348]), WSRL ([C088]), RLPD ([C037]), DQC ([C191]) | [`algos/05_rlpd_qchunk.py`](../../algos/05_rlpd_qchunk.py) | available |
| 6 | [Residual RL on a frozen base](ch06.md) | `a = a_base + α·tanh(u)` with a zero-initialized last layer, trained off-policy (TD3) with a warmed-up critic. With α = 0.1 it goes from 50.4% to 89.7% [87.4, 91.7] on the eval set (pooled over 3 seeds), and from 18.8% to 71.1% [67.8, 74.2] on the mis-zeroed `bias` variant (one seed stops at 58.2%), at 134 robot-minutes of training per run. Without the critic warm-up, training success first drops below the base; the large bound (α = 1, one seed) dips early and then ends above the default. | ResFiT ([C142]), DAWN ([C233]), Res-HIL ([C432]), PLD ([C158]), Policy Decorator ([C086]) | [`algos/06_residual_td3.py`](../../algos/06_residual_td3.py) | available |
| 7 | [DSRL: RL in the noise space of a frozen flow policy](ch07.md) | A small actor picks the noise, the frozen sampler decodes it, SAC trains the actor in that latent-noise MDP. Steering all 32 noise numbers never beat its zero-ticket start; steering the top 8 PSS directions reached 81.0% pooled on the eval set (622/768, one seed of three improved), below Chapter 3's best fixed ticket (85.5%). A 4-number bounded residual control reached 94.4% (725/768). On hard states the base never solved in 16 tries, steering solved 6.8% and the residual 30.2%; finite failed trials do not establish absence of support. | DSRL ([C120]), PSS ([C413]), RFS ([C228]), SCORE ([C324]), BMD ([C304]), Golden Ticket ([C236]) | [`algos/07_dsrl.py`](../../algos/07_dsrl.py) | available |
| 8 | [Propose, edit, select: EXPO-style edit policy with Q selection](ch08.md) | Sample N = 8 chunks from the frozen `base_v1`, add a bounded edit (β·tanh) to each, execute the argmax of the 16 candidates under a Q ensemble, and use the same argmax in the TD target. β = 0.1, chosen on the search set, goes from 50.4% to 93.8% on the eval set (pooled over 3 seeds), with 100 robot-minutes of online interaction plus 39-50 of search scoring per run (the β sweep that chose it costs 15 times that). Controls split the gain: the editor alone reaches 85.8%, the critic's pick adds the rest. The unbounded editor (β = 2) collapses to 0.4%. | EXPO-FT ([C276]), Real-Time EXPO-FT ([C412]), DICE-RL ([C235]), MiDAS ([C402]), FASTER ([C263]), RL Token ([C264]) | [`algos/08_expo_edit.py`](../../algos/08_expo_edit.py) | available |
| 9 | Supervised policy improvement: filtered BC, AWR and RECAP-style conditioning | Project `π* ∝ π_ref·exp(A/β)` with weighted BC, 0/1 filtering, or an advantage token plus classifier-free guidance; retrain from the base each round. | π\*0.6 / RECAP ([C179]), CFGRL ([C110]), Dissecting advantage-guided post-training ([C435]), DEED ([C369]), Hi-ORS ([C159]) | `algos/09_recap_lite.py` | planned |
| 10 | Humans in the loop: DAgger, HG-DAgger, RaC and takeovers as reward | Covariate shift makes errors compound; aggregate corrections where the policy actually goes, gate takeovers by a human, or treat a takeover as −1 reward (RLIF). | RaC ([C148]), SOP ([C203]), FlowDAgger ([C355]), TimelyDAgger ([C443]), HELP ([C351]), PAKT ([C462]) | `algos/10_hg_dagger.py` | planned |
| 11 | Where the reward comes from: success detectors, progress models and hacking them | Replace the human success key with a learned classifier or progress model, use potential-based shaping, and watch the policy exploit the learned reward. | Robometer ([C244]), TOPReward ([C226]), SVM ([C337]), Robo-Dopamine ([C193]), SARM ([C152]), RoboReward ([C205]) | `algos/11_progress_reward.py` | planned |
| 12 | [Test-time search with a verifier](ch12.md) | Best-of-N with a verifier is one greedy improvement step: sample N chunks from the frozen `base_v1`, execute the one a learned Monte-Carlo value ranks highest. With 4096 labeled rollouts per verifier, best-of-32 goes from 50.4% to 93.9% on the eval set (pooled over 3 verifier seeds; one seed reaches 95%). With a few hundred labeled rollouts the gains are small, depend on the seed and fade as N grows. v0.1 implements best-of-N only; QGF guidance, disagreement gating and the finale stack are planned. | UF-OPS ([C259]), Q-Planning ([C392]), SeeQ ([C438]), V-GPS ([C075]), IDQL ([C042]), VLA-ATTC ([C288]); planned: QGF ([C307]), JITI ([C189]) | [`algos/12_best_of_n.py`](../../algos/12_best_of_n.py) | available |

**Status.** `available` chapters run in sim on state observations on a Mac, each with a chapter doc and full-run results in `results/`. Chapters 0-3 and 12 made up the v0.1 release; chapters 5-8 make up v0.2. The v0.2 exit criterion (at least one method at 95% or more on eval, n = 256 per seed, with 3 seeds) is not met by any pooled headline among Chapters 5–8. The chosen headline settings reach 93.9% (Ch 5), 89.7% (Ch 6), 81.0% (Ch 7, steering the top 8 PSS directions; its bounded-residual control pools 94.4%) and 93.8% (Ch 8), all pooled over three seeds. The best single seed of a chosen setting is 96.1% (Ch 8). Two Ch 8 ablations that were not selected pool above 95% (β = 0.2 at 98.3%, which lost the search-set selection by 2 episodes, and the co-trained base at 97.0%). `planned` chapters follow the roadmap in the curriculum (§5): chapters 4 and 9–11 in v0.3, then hardware steps on the SO-101.

Each chapter doc (`chNN.md`) covers the idea in plain words, the math, the papers it mirrors, our actual results with intervals and plots, what went wrong, and the hardware step (DESIGN.md §8).

[C000]: https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf
[C002]: https://www.jmlr.org/papers/v11/theodorou10a.html
[C005]: https://arxiv.org/abs/1604.00772
[C012]: https://arxiv.org/abs/1803.07055
[C037]: https://arxiv.org/abs/2302.02948
[C042]: https://arxiv.org/abs/2304.10573
[C055]: https://arxiv.org/abs/2402.06559
[C057]: https://arxiv.org/abs/2403.14305
[C075]: https://arxiv.org/abs/2410.13816
[C086]: https://arxiv.org/abs/2412.13630
[C088]: https://arxiv.org/abs/2412.07762
[C110]: https://arxiv.org/abs/2505.23458
[C120]: https://arxiv.org/abs/2506.15799
[C130]: https://arxiv.org/abs/2506.22087
[C133]: https://arxiv.org/abs/2507.07969
[C134]: https://arxiv.org/abs/2507.10030
[C140]: https://arxiv.org/abs/2509.09674
[C142]: https://arxiv.org/abs/2509.19301
[C148]: https://arxiv.org/abs/2509.07953
[C152]: https://arxiv.org/abs/2509.25358
[C155]: https://arxiv.org/abs/2509.24372
[C157]: https://arxiv.org/abs/2510.25889
[C158]: https://arxiv.org/abs/2511.00091
[C159]: https://arxiv.org/abs/2510.26406
[C172]: https://arxiv.org/abs/2510.01460
[C179]: https://arxiv.org/abs/2511.14759
[C186]: https://arxiv.org/abs/2511.09923
[C189]: https://arxiv.org/abs/2511.22555
[C191]: https://arxiv.org/abs/2512.10926
[C193]: https://arxiv.org/abs/2512.23703
[C203]: https://arxiv.org/abs/2601.03044
[C205]: https://arxiv.org/abs/2601.00675
[C208]: https://arxiv.org/abs/2602.02481
[C226]: https://arxiv.org/abs/2602.19313
[C228]: https://arxiv.org/abs/2602.01789
[C233]: https://arxiv.org/abs/2602.10539
[C235]: https://arxiv.org/abs/2603.10263
[C236]: https://arxiv.org/abs/2603.15757
[C244]: https://arxiv.org/abs/2603.02115
[C259]: https://arxiv.org/abs/2603.10282
[C263]: https://arxiv.org/abs/2604.19730
[C264]: https://arxiv.org/abs/2604.23073
[C274]: https://arxiv.org/abs/2604.23000
[C276]: https://arxiv.org/abs/2605.25477
[C288]: https://arxiv.org/abs/2605.01194
[C304]: https://arxiv.org/abs/2605.11387
[C307]: https://arxiv.org/abs/2606.11087
[C324]: https://arxiv.org/abs/2606.27475
[C337]: https://arxiv.org/abs/2606.23640
[C348]: https://arxiv.org/abs/2607.27203
[C351]: https://arxiv.org/abs/2607.09776
[C352]: https://arxiv.org/abs/2607.09590
[C355]: https://arxiv.org/abs/2607.08877
[C369]: https://arxiv.org/abs/2607.20345
[C385]: https://arxiv.org/abs/2608.17423
[C392]: https://arxiv.org/abs/2608.21204
[C400]: https://arxiv.org/abs/2608.19182
[C402]: https://arxiv.org/abs/2608.11363
[C412]: https://arxiv.org/abs/2609.18207
[C413]: https://arxiv.org/abs/2609.33765
[C432]: https://arxiv.org/abs/2609.30023
[C435]: https://arxiv.org/abs/2609.28161
[C438]: https://arxiv.org/abs/2609.22085
[C443]: https://arxiv.org/abs/2609.33157
[C445]: https://arxiv.org/abs/2609.09403
[C462]: https://arxiv.org/abs/2609.25630
