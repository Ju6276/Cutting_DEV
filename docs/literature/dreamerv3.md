# DreamerV3 — actual gradient paths
Reviewed 2026-11-10.

## Sources and scope
- [Paper](https://arxiv.org/abs/2301.04104), methods/results/ablations pp. 3–9 of the expanded version. All 40 pages and supplemental experiments have not been individually audited.
- [Official code](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c), pinned e01491fad6434b2245a3b8ca201dd7faedcc458c.
- Read `dreamerv3/agent.py` loss, imagined/replay losses and lambda-return code, together with the relevant configuration defaults. No training.

## Method
A recurrent latent state-space model uses discrete stochastic latents, reconstruction/reward/continuation predictions and balanced KL losses. The actor learns from imagined trajectories; value targets use lambda returns and categorical symlog/two-hot prediction. Return normalization uses running percentile statistics. Real replay value training supplements imagination.

Crucially, this version uses REINFORCE actor gradients, including continuous-action settings. It is not the same actor gradient estimator as SuperMimic's pathwise BPTT.

## Source findings that constrain our claims
The continuation head reads the representation without a stop-gradient switch. A configurable reward gradient is a separate feature. Therefore A (detaching terminal BCE from our physical-state model during fitting) is not a direct copy of DreamerV3's continuation training.

The slow critic exists, but the supplied default configuration has `slowtar: False`; slow-value regularization and using a slow network as bootstrap target are different operations. Reading only a function's default argument would misdescribe the executed configuration. Actor loss multiplies log-policy by a stopped advantage.

## Reuse boundaries
Latent-state regularization and normalization offer references for implementation, not a guarantee against value extrapolation on our imagined physical states. Dreamer's broad return benchmarks do not validate force peaks, force-limit endings or our reward scales. Its stop-gradient choices cannot be transplanted into a BPTT chain without checking which derivatives they remove.
