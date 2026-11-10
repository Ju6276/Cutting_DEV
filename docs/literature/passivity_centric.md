# Passivity-Centric Safe Reinforcement Learning for Contact-Rich Robotic Tasks

- Paper: [arXiv:2503.00287v3](https://arxiv.org/abs/2503.00287v3), accepted Robotics and Autonomous Systems version, 27 pages. Read on 2026-11-10: complete Sections 1–4, including controller equations, all training/deployment experiments, limitations and supplementary-material statement. No separate appendix is present in the downloaded version.
- Remote paper: `/dataHDD/dong/Research/cutting_literature_20261110/passivity.pdf`.
- Author release: [Hugging Face passiveRL](https://huggingface.co/Anonymous998/passiveRL/tree/6f7468ee71565f13a6a6c462a87f53b5eb725bd9), revision `6f7468ee71565f13a6a6c462a87f53b5eb725bd9`.
- Release audit: read README and repository tree; downloaded and inspected the ZIP directory on GPU7. It contains SAC and safety-critic checkpoint files for several configurations; an offline-data pickle is separately listed. **The checked release does not contain training/controller source.** No checkpoint or pickle was executed or deserialized. This is paper-read plus artifact-inventory evidence, not implementation verification.
- Search scope for code: paper/arXiv outbound links, the author-linked Hugging Face release, exact-title/GitHub searches. No author training repository was verified.

## Sections 1–2: problem and mechanism

The paper separates three properties often conflated in contact control: completing a task, keeping a learned risk low, and ensuring passivity of the robot–environment interaction. A successful RL policy can inject energy aggressively; a force-aware reward alone does not ensure an energetically passive controller.

The framework combines a task SAC policy, a learned safety critic and a recovery policy with variable impedance control. The safety critic receives measured six-dimensional wrench. The task policy additionally receives TCP position. The action changes two Cartesian reference coordinates and two stiffness values; the controller aligns stiffness axes with the movement direction. Reported action limits include reference changes up to 3 cm and stiffness in the 300–1000 N/m range.

Known impedance-controller dynamics and an energy tank provide the physical structure. The tank tracks an energy budget and bounds energy-flow rate. A passivity filter scales the requested movement when its energy flow exceeds the permitted rate and can freeze stiffness updates when the tank is depleted. The paper gives the energy/tank and scaling equations rather than treating the entire controller as an unknown neural transition model.

Training includes force-limit, energy-budget and energy-flow violation signals. The **passivity filter is used at deployment, not simply inserted into every training step**: otherwise it can mask unsafe proposed actions and change credit assignment. Passivity-aware training teaches the policy to work within the energetic constraints, while passivity-ensured deployment adds the physical filter.

There is no learned full-state world model and no BPTT through learned dynamics in this method. The task and recovery policies learn through value functions.

## Sections 3.1–3.5: data, experiments and metrics

The contact-rich task is robot maze traversal, first simulated and then performed by a Franka Panda with a force/torque sensor. The system uses a fast low-level controller; policy learning is separate from that servo rate. The policy is not an image-based vision policy.

Offline pretraining data contain roughly 40,000 transitions, including 1,363 constraint violations. These are exploration transitions for safety learning, not a library of successful task demonstrations. The reported force limit is 40 N. The training study uses ten seeds and studies several energy budgets and flow constraints; these are deliberately selected experimental parameters, not universal safe values.

Comparisons include passivity-agnostic policies, passivity-aware training under different energy constraints, and deployment with/without passivity filtering. Metrics include task success, constraint violations, energy budget/flow behavior and interaction behavior. Changing the deployment energy constraints can substantially alter success even when nominal training success is high.

Physical experiments use selected trained models with three trials per setting. Passivity-aware configurations complete three of three in the reported table, while the agnostic comparison completes two of three. Such small physical sample counts support a demonstration, not a precise population reliability claim. The model-selection description and the table's seed wording should be checked when reproducing exact selection: the text discusses selecting strong models, while the table also refers to selected/random-seed models.

## Assumptions and limits

- Passivity constrains energy exchange under the specified controller/tank assumptions. It is **not** a mathematical guarantee that instantaneous contact force stays below a chosen force threshold. The paper itself reports force violations in a passivity-filtered configuration.
- Energy-budget state is history dependent. The listed task/safety observations do not simply contain the full tank state, so a compact wrench/position observation should not automatically be called fully Markov for all constrained dynamics.
- Learned safety/recovery behavior and analytical passivity filtering are distinct mechanisms; their guarantees must be stated separately.
- Maze contact does not include irreversible cutting damage, material separation or changing internal topology.
- Source code was not available in the checked author release, so exact discretization, numerical tank clamping and safety-network target handling were not verified.

## Relation to cutting and Route B

This is a strong non-cutting precedent for **structured physical constraints plus learned compliant control**. It disproves a broad claim that prior safe learning ignores contact forces or energy.

A cutting adaptation could use controller/energy structure, but passivity alone cannot replace the project's actual force-limit event definition or a substep peak-force predictor. Conversely, a statistically calibrated force bound cannot establish passivity. These are different claims.

The useful lesson is methodological: a contribution needs an explicit physical or statistical mechanism with assumptions, not just another auxiliary loss. Copying an energy tank into the current controller would be an additional design change requiring its own cutting-specific justification; it is not recommended here as a spontaneous replacement for the approved S0–S4 route.
