# Visuotactile world models for force-constrained lifting
Reviewed 2026-11-10. [Paper, arXiv:2609.09597](https://arxiv.org/abs/2609.09597), version 2.
Read the method, all experiments, calibration/action-branch diagnostics and limitations (PDF pp. 2–7), and the source's reproducibility section.

## Release status — paper evidence, not code evidence
The abstract page advertises code/tabulated results as ancillary material. The retrieved v2 source archive contains main.tex, class/figure/table files, but no implementation; no downloadable code link was exposed on the retrieved abstract page. Therefore **code advertised, implementation not retrieved**. This corrects both “no code exists” and “code audited” overstatements. Assets/source were kept on GPU7, not the Mac.

## Method and actual task
A MuJoCo/robosuite Panda lifts a rigid box at 20 Hz, with 25 internal 2 ms steps. Inputs combine visual/tactile/proprioceptive history. A latent dynamics model predicts motion plus endpoint forces and within-control-interval peak normal forces. BPTT policy fitting uses model rewards, a terminal value and BC; it is compared to IQL.

The task includes a shared scripted approach and later lowering; only part of the episode is learned. Offline data comprise 240 training, 48 validation and 48 calibration episodes from scripted and force-feedback behavior. Do not equate this with fully online end-to-end robot learning.

## Especially relevant mechanisms
- An explicit peak-force prediction target, separate from endpoint force.
- Episode-level maximum positive underprediction residuals and one-sided conformal corrections, with sampling/shift caveats.
- Replaying physical prefixes and branching actions to compare predicted versus realized consequences.
- A model-based selector of nine action candidates, current-force anchoring and a measured-force guard. The anchoring correction is not itself a conformal safety guarantee.

The paper reports that better prediction need not yield better imagined-policy learning: the pooled joint task result for WM-RL is worse than IQL (11.9% versus 25.0% in the reported setting). Model-based selection benefits are conditional on the task phase and comparison. No real-robot result is established.

## Consequence for our novelty
“Add a substep-peak head, calibrate an upper bound and train through a WM” already has a very close non-cutting precedent. Neither that combination nor physical branch replay is a new mechanism by itself. It remains useful as a reference for separating endpoint/peak, calibration/feedback anchoring, and predictive fit/actual action quality. Our existing WM already predicts peak wrench, so merely adding such an output is not a new implementation either.
