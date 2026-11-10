# MBPO — model samples, not actor BPTT
Reviewed 2026-11-10.

## Sources
[Paper](https://arxiv.org/abs/1906.08253), §§3–6 including model-bias bounds, branched-rollout construction and design ablations.
[Official implementation](https://github.com/jannerm/mbpo/tree/ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129), pinned ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129.
Read `mbpo/models/fake_env.py`, `MBPO._rollout_model` and `_training_batch`. No training and no independent theorem proof.

## Algorithm/code
Fit a bootstrap ensemble of probabilistic next-state/reward models. Start many short rollouts at physical replay states; sample an ensemble member per transition; put simulated transitions into replay and train SAC on a real/model mixture.
The inspected execution is NumPy model sampling. It does not propagate actor gradients through a physical or learned rollout. The fake environment computes termination with a task function on predicted next observations; rollouts remove terminal entries before subsequent simulation.

## Evidence/limits
MuJoCo return versus real interaction, horizon and update-ratio ablations. Short rollouts mitigate compounding error. The theoretical bound's practical variant needs model-error behavior under the updated policy; in-distribution validation loss alone is not a universal guarantee. The paper explicitly discusses model exploitation and policy shift.

## Our use
Short model rollout is shared background, not proof of our H8 choice. Known task predicates can replace a classifier only when the predicted state contains the correct relevant event information. Applying a correct force threshold to an underestimated peak still misses the event. Online collection and a learned model have existed long before recent WM naming.
