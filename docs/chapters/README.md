# Chapters

Twelve ways to climb one base policy from about 50% toward 95%+, ordered from classic to modern. Every chapter starts from the same frozen base (`base_v1`, or the mis-tuned scripted controller in chapters 1 and 2), is scored on the same held-out initial states, and reports its cost in robot-minutes and human-minutes. The full plan is in [docs/research/curriculum.md](../research/curriculum.md) §1, and the contract every chapter follows is in [docs/DESIGN.md](../DESIGN.md).

Everything runs in MuJoCo on a laptop CPU. The SO-101 is the default robot and the real-hardware target; the AgileX PiPER runs in sim through the same environment API.

Paper ids (`C###`) match [data/papers.csv](../../data/papers.csv).

| # | Chapter | The base idea | Papers mirrored | Code | Status |
|---|---|---|---|---|---|
| 1 | Hill climbing with a success counter | Perturb, evaluate, step: antithetic finite differences on a Gaussian-smoothed success rate. ARS adds reward-std normalization and top-b directions. | Kohl & Stone 2004 ([C000]), ARS ([C012]), ES at Scale ([C155]), TD-ES ([C186]), EvolSAC ([C134]) | `algos/01_hill_climb.py` | v0.1 |
| 2 | One update, different weights: CEM, CMA-ES, PI2/MPPI and BO | Sample candidates, weight them (elites, ranks or `exp(-cost/λ)`) and move to the weighted mean. For tiny budgets, BO fits a GP and picks the next trial with an acquisition function. | Zero-order optimization primer ([C130]), A decade of BO for controller tuning ([C445]), PI2 ([C002]), CMA-ES tutorial ([C005]), BOpt-GMM ([C057]) | `algos/02_cem_cmaes_pi2.py`, `algos/02b_bo.py` | v0.1 |
| 3 | The golden ticket: black-box search over one noise vector | Freeze the flow policy and search its 32-dim input noise. Select by search score, report the held-out score. Search can only select behavior the base already has. | Golden Ticket ([C236]), INSPO, SDN, Diffusion-ES ([C055]) | `algos/03_golden_ticket.py` | v0.1 |
| 4 | Likelihood-ratio policy gradients with anchors | `∇ = E[∇ log π(a\|s) · A]` with PPO clipping; GAE vs GRPO group baselines; KL/BC anchors to the frozen base so it does not collapse. | SimpleVLA-RL ([C140]), piRL ([C157]), PAC-ACT ([C352]), FPO++ ([C208]), ADEPT ([C400]), Prism-GRPO ([C385]) | `algos/04_ppo_flow.py` | planned |
| 5 | Off-policy critics: SAC → RLPD → Q-chunking | Bellman backups on replay that reuse every transition; RLPD's 50/50 batches and LayerNorm critics; a chunk critic with an unbiased h-step target. | Q-chunking ([C133]), Three Regimes ([C172]), IPE ([C348]), WSRL ([C088]), RLPD ([C037]), DQC ([C191]) | `algos/05_rlpd_qchunk.py` | planned |
| 6 | Residual RL on a frozen base | `a = a_base + α·tanh(u)` with a zero-initialized last layer, trained off-policy (TD3) with a warmed-up critic. | ResFiT ([C142]), DAWN ([C233]), Res-HIL ([C432]), PLD ([C158]), Policy Decorator ([C086]) | `algos/06_residual_td3.py` | planned |
| 7 | DSRL: RL in the noise space of a frozen flow policy | A small actor picks the noise, the frozen sampler decodes it. Steering can reweight behaviors the base can already produce, not add new ones. | DSRL ([C120]), PSS ([C413]), RFS ([C228]), SCORE ([C324]), BMD ([C304]) | `algos/07_dsrl.py` | planned |
| 8 | Propose, edit, select: EXPO-style edit policy with Q selection | Sample N chunks from the base, add a bounded edit to each, execute the argmax by a Q ensemble, and use the same argmax in the TD target. | EXPO-FT ([C276]), Real-Time EXPO-FT ([C412]), DICE-RL ([C235]), MiDAS ([C402]), FASTER ([C263]) | `algos/08_expo_edit.py` | planned |
| 9 | Supervised policy improvement: filtered BC, AWR and RECAP-style conditioning | Project `π* ∝ π_ref·exp(A/β)` with weighted BC, 0/1 filtering, or an advantage token plus classifier-free guidance; retrain from the base each round. | π\*0.6 / RECAP ([C179]), CFGRL ([C110]), Dissecting advantage-guided post-training ([C435]), DEED ([C369]), Hi-ORS ([C159]) | `algos/09_recap_lite.py` | planned |
| 10 | Humans in the loop: DAgger, HG-DAgger, RaC and takeovers as reward | Covariate shift makes errors compound; aggregate corrections where the policy actually goes, gate takeovers by a human, or treat a takeover as −1 reward (RLIF). | RaC ([C148]), SOP ([C203]), FlowDAgger ([C355]), TimelyDAgger ([C443]), HELP ([C351]), PAKT ([C462]) | `algos/10_hg_dagger.py` | planned |
| 11 | Where the reward comes from: success detectors, progress models and hacking them | Replace the human success key with a learned classifier or progress model, use potential-based shaping, and watch the policy exploit the learned reward. | Robometer ([C244]), TOPReward ([C226]), SVM ([C337]), Robo-Dopamine ([C193]), SARM ([C152]), RoboReward ([C205]) | `algos/11_progress_reward.py` | planned |
| 12 | Test-time search with a critic, then the finale | Best-of-N with a verifier is one greedy improvement step, capped by pass@N; guidance and disagreement gating spend compute only where it helps. Then stack the best methods. | UF-OPS ([C259]), Q-Planning ([C392]), SeeQ ([C438]), QGF ([C307]), VLA-ATTC ([C288]), JITI ([C189]) | `algos/12_best_of_n.py` | v0.1 |

**Status.** `v0.1` chapters are the first release: sim, state observations, runnable on a Mac CPU. `planned` chapters follow the roadmap in the curriculum (§5): chapters 5–8 in v0.2, chapters 4 and 9–11 in v0.3, then hardware steps on the SO-101.

Each chapter doc (`chNN.md`) covers the idea in plain words, the math, the papers it mirrors, our actual results with intervals and plots, what went wrong, and the hardware step (DESIGN.md §8).

[C000]: https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf
[C002]: https://www.jmlr.org/papers/v11/theodorou10a.html
[C005]: https://arxiv.org/abs/1604.00772
[C012]: https://arxiv.org/abs/1803.07055
[C037]: https://arxiv.org/abs/2302.02948
[C055]: https://arxiv.org/abs/2402.06559
[C057]: https://arxiv.org/abs/2403.14305
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
