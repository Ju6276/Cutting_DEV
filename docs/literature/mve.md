# MVE — model-based value expansion

Reviewed 2026-11-10. Paper: [Feinberg et al., 1803.00101](https://arxiv.org/abs/1803.00101).
Read scope: method, error argument, TD-k correction and main experiments (§§3–4);
not a line-by-line proof audit. Author code inspected at
[vlad17/mve, 19835bba87a2e5abc9bca653d011bf2eed68dd62](https://github.com/vlad17/mve/tree/19835bba87a2e5abc9bca653d011bf2eed68dd62).
No reproduction executed.

## Mechanism and evidence

Short model rollouts build value targets from predicted rewards plus a terminal
critic. The TD-k trick also fits the critic at intermediate imagined states.
This addresses the mismatch between physical replay states and states where the
model-expanded target queries the critic. The error argument depends on model
accuracy, regularity and value error under the relevant state distribution;
a perfect model alone does not guarantee useful targets.

Experiments compare return versus real interactions and value error on MuJoCo
tasks, including oracle-model and TD-k/horizon ablations. Longer rollouts are
not uniformly better. Reward computation is known, with fully observable
environment wrappers; this differs from hidden cutting damage.

## Code evidence

`mve/ddpg/ddpg.py::_tf_compute_model_value_expansion` (lines 442–520)
uses target actions and a target critic, generates states on the fly, then
recursively computes imagined reward targets. `drop_tdk` controls intermediate
critic fitting. The inspected implementation masks already-terminal physical
transitions but explicitly assumes its model does not predict new terminal states.
It cannot be copied unchanged into force-limit cutting.

## Relevance and boundary

MVE is not the implementation of our `real` critic: ours uses episode-bounded
physical lambda returns. Nor does TD-k justify assigning a physical-state target
to a different predicted state in `real_wm`. Small physical-return error does
not certify critic extrapolation inside MPPI. Short horizons and imagined-value
training are established ideas, not a new contribution here.
