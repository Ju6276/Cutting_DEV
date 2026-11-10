# SRL-VIC: A Variable Stiffness-Based Safe Reinforcement Learning for Contact-Rich Robotic Tasks

- Paper: [arXiv:2406.13744v1](https://arxiv.org/abs/2406.13744v1), 8 pages. Read on 2026-11-10: all Sections I–V, equations, experimental setup, hyperparameters, simulation/real-robot results and discussion. No separate appendix was provided in this version.
- Remote paper: `/dataHDD/dong/Research/cutting_literature_20261110/srl-vic.pdf`.
- Author implementation: no SRL-VIC task/controller repository verified after checking the paper, arXiv links and exact-title/GitHub searches. The paper references [Recovery RL](https://github.com/abalakrishna123/recovery-rl), which is a **dependency/prior implementation, not the authors' released VIC experiment**.
- Dependency code inspected: Recovery RL commit `084c968538850e82776564e07b9dcdffca369936`. Relevant functions and the scope of that evidence are listed below.

## Sections I–III: task and algorithm

The task policy controls contact-rich robot motion through a variable impedance controller. Safety is handled by a separate learned risk critic and a recovery policy. The task SAC policy proposes an action; a risk threshold decides whether it is executed or replaced by the recovery action.

The paper's risk value predicts discounted future constraint events. Its Bellman target has the form `c + (1-c) gamma_risk Q_risk(s_next,a_next)`; a terminal failure must not keep bootstrapping future failures. The recovery policy minimizes predicted risk, while the task policy maximizes the task/entropy objective.

The task observation contains TCP position and six-dimensional wrench, totaling nine dimensions. The safety/recovery input emphasizes wrench rather than encoding maze position, to discourage memorizing specific wall locations. Actions change two reference-position coordinates and two Cartesian stiffness values. This is task learning plus compliance selection, not end-to-end joint-torque generation.

The method is model-free: a risk critic is not a physical next-state world model. Policy gradients pass through critics, not through MuJoCo or a learned multi-step dynamics rollout.

## Sections IV–V: data and evaluation

Offline pretraining uses 40,018 transitions, including 185 violations. Data come from exploration at multiple initial positions with random directions/actions; they are not successful demonstrations to imitate. The safety critic/recovery policy receive 10,000 pretraining updates with batch size 256.

Online experiments report three training runs of 1,500 episodes, a maximum of 500 steps per episode and ten updates per simulation step. The risk discount and switching threshold are design hyperparameters (reported as 0.65 and 0.7), not universal safety standards.

The simulated system uses a Franka robot in MuJoCo 2.3.3 with a contact-rich maze and an object to manipulate. Baselines remove recovery or use fixed stiffness (including 300 and 1000 N/m). Low fixed stiffness can reduce violations while failing to make progress: force reduction must be assessed together with completion.

Final simulation evaluations include successful nominal trials and changed conditions. Real-robot deployment uses robot proprioception/force sensing and ROS-based control, not camera images as policy input. The initial physical deployment succeeds in four of five trials; after simulation training with observation noise, the reported physical trial set succeeds in six of six. Thus “no real-robot fine-tuning” must not be rewritten as “no simulator-side adaptation after initial deployment.”

Safety-critic transfer and task-policy transfer differ: a transferable recovery behavior does not imply the task policy solves arbitrary new maze shapes. The paper reports a new-shape difficulty for the task policy.

## Dependency source: what was actually checked

At Recovery RL commit `084c968538850e82776564e07b9dcdffca369936`:

- `recovery_rl/qrisk.py:QRiskWrapper.update_parameters`: samples real transitions, computes a detached target using the **maximum** of two risk critics, fits both with MSE, and trains the recovery policy to minimize the maximum predicted risk. The target mask comes from stored episode termination; this code must not be assumed to reproduce SRL-VIC's exact masks without its environment.
- `QRiskWrapper.get_value` runs under `torch.no_grad()`; `select_action` returns a detached action for environment execution.
- `recovery_rl/experiment.py:get_action`: obtains a task action, checks `critic_val > eps_safe`, then substitutes a recovery-policy action if needed. The switching branch is not a differentiable physical model.
- `experiment.py:pretrain_critic_recovery` uses offline constraint transitions. Its rollout loop stores executed actions and constraint signals for risk learning.
- `recovery_rl/sac.py:SAC.update_parameters`: ordinary off-policy SAC targets are computed under `no_grad`; task-policy gradients flow through task Q-functions. Additional constrained variants exist in this upstream code and should not be conflated with the specific SRL-VIC paper configuration.

These paths establish the upstream algorithm pattern. They do **not** verify the authors' force observation transform, VIC implementation, data collection or exact neural network choices. Source files were read statically; no experiment was rerun.

## What this changes for our direction

Force-aware policy learning, adaptive stiffness, safety critics and recovery switching are established mechanisms. A new cutting paper cannot use that collection alone as its novelty.

The current cutting task has irreversible internal damage and contact-mode changes. A wrench-only safety critic may alias states with different future force responses; adding history is plausible but is not by itself proof of a new algorithm. Route B should show that a specific model mechanism changes reliable action choice, not merely that a classifier predicts old trajectories better.

Do not copy the paper's 40 N-style limit, risk discount or threshold into the cutting task. Existing task definitions and units must remain explicit. A learned risk score below a threshold is not a hard per-step force guarantee.
