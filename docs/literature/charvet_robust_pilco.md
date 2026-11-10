# Learning Robust Controllers via Probabilistic Model-Based Policy Search
Reviewed 2026-11-10. [Full paper v1](https://arxiv.org/pdf/2110.13576).
Read §§1–5 and appendices A–B, including the proof and parameter table.
## Paper
- §3: PILCO moment matching, a primal-dual expected accumulated-risk constraint, and a lower bound on GP likelihood noise.
- Hypothesis: forcing predictive noise above a floor regularizes policy search under small parameter/observation changes.
- §4/B: simulated pendulum; ten training seeds and ten evaluation rollouts; vary mass, pole length, gravity and observation noise. Report return, violation count and dispersion. The main effect is reduced variation across trained controllers.
- Appendix B implements the floor as softplus(raw_noise)+floor. Sparse GP and RBF policy; horizon 40, 20 learning episodes.
- Appendix A assumes smooth parameter-dependent dynamics and attempts a robustness radius proportional to the noise floor.
## Code and evidence boundary
No author implementation link found in the full paper, arXiv page, or exact-title/code search performed in this review. The cited GPyTorch/PyBullet packages are dependencies, not this algorithm's released code. No implementation was executed.
## Independent assessment
This is noise regularization, not calibrated uncertainty. It supplies no reason to choose a force-dependent noise floor in our cutting system.
The appendix's displayed implication from equation (5) to (6) reverses a bound: f_hat <= f+sigma implies f >= f_hat-sigma, not an upper bound on f. Do not import that displayed proof as our safety theorem.
The empirical robustness result remains distinct from that mathematical concern.
Known prior art for cautious model-based policy search; not a good central mechanism for our contact peaks.
