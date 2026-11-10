# SuperMimic — paper and community reproduction
Reviewed 2026-11-10.

## Sources and scope
- [Paper](https://arxiv.org/abs/2610.28480): method/results/limitations pp. 3–9 and observation, action, randomization and architecture details pp. 13–15.
- [Community reproduction](https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2), pinned ba6105349069bf040f1fdad60873d13138d505a2. This is not official author training code.
- Inspected imagination `objective.py`, `rollout.py` and `policy_trainer.py`. These are the actor/critic gradient paths; this note does not claim a full runtime reproduction of all environments.

## Algorithm and data
The current policy interacts with the physical simulator, producing state/action/next-state transitions. A deterministic model fits short sequences (reported fitting horizon 8). Actor updates backpropagate through a frozen learned-model rollout (horizon 32) with discounted rewards and terminal value. “Frozen” means model parameters are not optimized during the actor step, not that derivatives with respect to action/state are removed. Imagined GAE targets train the critic.

Motion references are targets for imitation, not replacement dynamics transitions. The paper's broad motion set and its controlled LAFAN comparison must be distinguished. LAFAN is a dataset. Policy actions are 29 joint setpoints, not sampled recorded LAFAN actions. Privileged simulator context conditions the model; a cutting model not observing material properties has a different identification problem.

## Code details
The objective adds horizon rewards and terminal value. GAE target construction is no-grad. The rollout preserves the actor–model chain and uses predicted survival. The policy trainer disables WM/critic parameter gradients, performs actor BPTT and subsequently trains the critic on detached imagined states with fixed targets for that update. The inspected path has no EMA target critic. Gradient clipping and source settings should be attributed, not called local theoretical fixes.

## Evidence and limitations
The paper evaluates motion tracking, learning efficiency and model rollout speed on its own hardware/task. It does not establish food cutting, contact-force calibration or our multi-material controller. Copying the framework plus BC/shorter horizon is not a new generic algorithm. The community code is a concrete implementation reference, with provenance separate from the paper.
