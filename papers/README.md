# All papers and posts, by family

[← Awesome Hill Climbing](../README.md)

| Family | Question it answers | Entries | Must-know |
| :--- | :--- | ---: | ---: |
| [Black-box search](blackbox.md) | How do I improve a policy when all I can do is run it and get a score, for example a success flag, and I cannot or do not want to backprop through it? | 37 | 8 |
| [On-policy policy gradients](onpolicy.md) | How do I take a BC or SFT policy (diffusion, flow, ACT or a 7B VLA) that succeeds 50-90% of the time and push it toward 95-100% using only a success signal, a resettable simulator, and fresh on-policy rollouts? | 69 | 11 |
| [Off-policy critics & offline-to-online RL](offpolicy.md) | How do I take a pretrained robot policy that works 40-80% of the time to 95-100% with minutes-to-hours of real robot data, reusing every demo and rollout I already have? | 121 | 12 |
| [Residual, edit & steering policies](residual.md) | How do I push a big imitation, diffusion or VLA policy from 50% to 95%+ when I can't cheaply fine-tune or backprop through it, without breaking what it already does well? | 76 | 12 |
| [Advantage-weighted & filtered BC](advbc.md) | How do I push a big imitation policy (a diffusion policy or a flow VLA) up on one task using its own rollouts, human corrections and success labels, without running policy-gradient RL through it? | 58 | 12 |
| [Human corrections & DAgger](hitl.md) | My imitation policy works most of the time but fails in states the demos never covered. How do I spend a small amount of human time so the next version fixes exactly those failures? | 77 | 12 |
| [Test-time search & verifiers](testtime.md) | My policy sometimes does the right thing but not reliably. Can I raise its success rate without retraining it, just by choosing better among its own samples at run time? | 128 | 12 |
| [Reward, value & progress models](reward.md) | My robot only gets a success bit at the end of an episode, or a human has to watch every rollout. How do I get a dense, trustworthy score so RL, filtered BC or advantage conditioning can push the policy from about 60-80% toward 95%+? | 89 | 12 |
| [World models & evaluation](worldmodel.md) | How do I tell whether checkpoint k+1 really beats checkpoint k, and how do I improve a policy, when every real robot trial is slow, noisy and needs a reset? | 101 | 12 |
| [Sim-to-real & real-to-sim](simreal.md) | How do I get the trial-and-error gains of RL for a robot policy without paying for thousands of risky, hand-reset real-world episodes? | 53 | 12 |
| [Industry recipes & deployment reports](industry.md) | How do real robot companies actually climb from ~60-80% to 99% on a paying task, and what do they disclose about it? | 28 | 7 |
| [Broader RL advances that transfer](generalrl.md) | Which general RL results from the last year change how you should hill-climb a robot policy? | 75 | 0 |

The same data as one searchable table: [`data/papers.csv`](../data/papers.csv).
