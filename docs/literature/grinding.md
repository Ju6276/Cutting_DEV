# Learning to Shape by Grinding — cutting-surface-aware MBRL
Reviewed 2026-11-10.

## Sources and scope
- [Paper](https://arxiv.org/abs/2308.02150), central model/controller pp. 3–5 and real-robot experiments pp. 6–7.
- [Author project](https://t-hachimine.github.io/csambrl/) provides paper/video; no algorithm/data download was verified in this review. Code not inspected.

## Method
An analytic geometric removal model is combined with a learned GP residual for the effective cutting surface. Inputs encode removed volume, object bounds and action geometry. Random-shooting MPC evaluates shape error/overcut and model uncertainty. Data are collected and the model refit iteratively. This is model-based reinforcement learning using planning; MBRL need not train a separate actor network.

A UR5e holds the object against a grinding belt, with a wrist F/T sensor and 3-D surface sensing. Hybrid force/position control executes the operation. The learning target is material-removal geometry, rather than all particles or a free-standing force classifier.

## Evidence/limits
Simulation compares learning from hundreds of interactions against much larger model-free training budgets; robot results use shape-reconstruction error. Protective-stop/high-removal data are excluded from fitting, so the work is not evidence of calibrated prediction of unsafe contact-force events.

## Transfer to our design
Keep reliable geometric/control transformations explicit and learn uncertain material response is an established strategy. It is a useful engineering direction for a smaller, physically consistent cutting WM, not a novelty claim by itself. The sensor, workpiece arrangement and output objective differ from food slicing; do not quote its shape-error result as force-risk validation.
