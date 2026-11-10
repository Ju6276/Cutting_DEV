# TopoCut — learned multi-step topology planning
Reviewed 2026-11-10.

## Sources and scope
- [Paper](https://arxiv.org/abs/2509.19712): main method/results pp. 3–8 and technical appendices pp. 15–22, 25–29.
- [Official project](https://topocut.github.io/) links code as “Coming Soon” at this review. No implementation inspected; simulated behavior cannot be audited from paper equations alone.

## Method/data
A damage-enabled MPM process supplies material trajectories. The system reconstructs separated geometry/topology, including a trajectory-dependent gap and connected components. A topology encoder/model predicts the next object configuration under a cut. Spectral information from graph Laplacians supplies rewards for desired piece structure. MPPI and teleoperation provide expert trajectories; a goal-conditioned discrete-diffusion imitation policy predicts a cut mask, which is converted into a cutting plane.

The paper reports roughly 1,000 demonstration trajectories / 6,000 cuts, nine shapes (five training, four unseen) and slice/stick/dice goals, with 405 evaluation episodes. This is high-level multi-cut geometric planning, not our 50-ms compliant controller learning from a wrist-force signal.

## What is evaluated
Geometric/topological prediction and task-goal achievement, with Chamfer/EMD/Hausdorff-style geometric losses and matched topology supervision. There is no sensor-force waveform or force-limit prediction target in the inspected model. A visual/topological split produced with an explicit reconstruction gap must not be treated as independent validation of physical fracture mechanics.

## Consequence for our paper
Multiple cuts, multiple shapes, a learned cutting-transition model and topology-aware goals are already covered. Our focused question can be force-sensitive feedback during a cut, but merely switching from one prism to several objects is not sufficient novelty. Public code availability remains a separate, unresolved reproduction constraint.
