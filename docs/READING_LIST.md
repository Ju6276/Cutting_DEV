# Reading list

Papers to read for an RA-L submission that combines a cutting benchmark with a
force-constrained, multi-object policy. Each entry says why it matters here.
arXiv identifiers were checked against the arXiv API on 2026-11-07. Notes marked
"partial" are based on an incomplete read and must be checked against the full
text before the paper relies on them.

Priority: **A** decides novelty or the paper's framing; **B** method background;
**C** context.

## Robotic cutting: closest competitors

| P | Paper | Why read |
|---|---|---|
| A | Luo, Yuan. *A Perception-Manipulation Robotics System for Food Cutting*. IROS 2026. [arXiv:2607.04367](https://arxiv.org/abs/2607.04367), [project](https://robotproject8.github.io/robocutting/) | Real UR5e food cutting. PPO tunes target downward force and sawing speed. Reward is downward progress minus mechanical work. Ten foods, one policy per food, plus knife selection from trial cuts. Their actions bound force to [-30, 0] N. Closest real-robot RL work. |
| A | Beltran-Hernandez, Erbetti, Hamaya. *SliceIt! A Dual Simulator Framework for Learning Robot Food Slicing*. ICRA 2024. [arXiv:2404.02569](https://arxiv.org/abs/2404.02569) | Our task, reward and SAC baseline. Paper force-penalty formula differs from the code; see METHOD. |
| A | Wang, Bian, Heiden, Garg. *TopoCut: Learning Multi-Step Cutting with Spectral Rewards and Discrete Diffusion Policies*. arXiv 2025. [arXiv:2509.19712](https://arxiv.org/abs/2509.19712) | Cutting benchmark (MLS-MPM, Taichi): slice, stick and dice goals, seen and novel shapes. Kinematic knife; no knife force or force limit found (partial). Goal-conditioned behavior cloning. Defines the benchmark gap we would fill. |
| A | Xu et al. *RoboNinja: Learning an Adaptive Cutting Policy for Multi-Material Objects*. RSS 2023. [arXiv:2302.11553](https://arxiv.org/abs/2302.11553) | Differentiable MPM; closed-loop policy trained from optimized trajectories; collision and energy metrics. |
| A | Heiden et al. *DiSECt: A Differentiable Simulator for Parameter Inference and Control in Robotic Cutting*. Autonomous Robots. [arXiv:2203.10263](https://arxiv.org/abs/2203.10263) | Our material model; gradient trajectory optimization baseline; Ansys force data for sim validation. |
| B | Heiden et al. *DiSECt: A Differentiable Simulation Engine for Autonomous Robotic Cutting*. RSS 2021. [arXiv:2105.12244](https://arxiv.org/abs/2105.12244) | Original DiSECt; mean-force and time objective. |
| B | Lou et al. *DexNinja: Learning Robust Dexterous Cutting Policy with a Real-to-Sim-to-Real Data Engine*. arXiv 2026. [arXiv:2610.17208](https://arxiv.org/abs/2610.17208) | Dexterous-hand cutting with RGB-D and fingertip tactile input; ACT-style behavior cloning; differentiable MPM data engine built on RoboNinja; category-level generalization on real robots (partial). |
| B | Yang et al. *BladeMaster: Real-Time Robotic Cutting Simulation with Online-Generated Persistent Discontinuities*. arXiv 2026. [arXiv:2609.27342](https://arxiv.org/abs/2609.27342) | GPU TLMPM cutting with tool reaction forces and persistent cuts; no learning. Candidate second simulator for cross-simulator checks (partial). |
| C | *CRESSim-MPM: A Material Point Method Library for Surgical Soft Body Simulation with Cutting and Suturing*. IROS 2025. [arXiv:2502.18437](https://arxiv.org/abs/2502.18437) | GPU MPM cutting, surgical domain. |
| C | *Sashimi-Bot: autonomous tri-manual advanced manipulation and cutting of deformable objects*. npj Robotics 2026. [link](https://www.nature.com/articles/s44182-026-00098-9) | Sim-trained manipulation with zero-shot transfer for slicing. |
| C | *Learning to Shape by Grinding: Cutting-surface-aware Model-based Reinforcement Learning*. [arXiv:2308.02150](https://arxiv.org/abs/2308.02150) | One of few model-based RL papers on material removal. |

## Safe and force-constrained learning for contact

| P | Paper | Why read |
|---|---|---|
| A | Kuo et al. *Uncertainty-aware Contact-safe Model-based Reinforcement Learning*. RA-L 2021. [arXiv:2010.08169](https://arxiv.org/abs/2010.08169) | Closest prior work to force-constrained MBRL: model uncertainty for contact safety, mixing and scooping on real robots. Must state our difference. |
| A | *Passivity-Centric Safe Reinforcement Learning for Contact-Rich Robotic Tasks*. [arXiv:2503.00287](https://arxiv.org/abs/2503.00287) | Model-free constrained RL with a measured-force constraint and energy tanks. |
| A | *Safe Learning for Contact-Rich Robot Tasks: A Survey from Classical Learning-Based Methods to Safe Foundation Models*. [arXiv:2512.11908](https://arxiv.org/abs/2512.11908) | Map of the field; check for cutting-specific work we missed. |
| B | As et al. *Constrained Policy Optimization via Bayesian World Models* (LAMBDA). ICLR 2022. [arXiv:2201.09802](https://arxiv.org/abs/2201.09802) | Constrained MBRL with world-model uncertainty. |
| B | Huang et al. *SafeDreamer: Safe Reinforcement Learning with World Models*. ICLR 2024. [arXiv:2307.07176](https://arxiv.org/abs/2307.07176) | Lagrangian constraints in a world-model agent. |
| B | Achiam et al. *Constrained Policy Optimization*. ICML 2017. [arXiv:1705.10528](https://arxiv.org/abs/1705.10528) | Constrained MDP baseline formulation. |
| B | Dalal et al. *Safe Exploration in Continuous Action Spaces*. [arXiv:1801.08757](https://arxiv.org/abs/1801.08757) | Safety layer; the usual alternative to a learned constraint. |
| C | *SRL-VIC: A Variable Stiffness-Based Safe Reinforcement Learning for Contact-Rich Robotic Tasks*. [arXiv:2406.13744](https://arxiv.org/abs/2406.13744) | Safe RL through variable impedance, like our compliance actions. |

## Model-based RL used in our critic

| P | Paper | Why read |
|---|---|---|
| A | *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*. CoRL 2026. [arXiv:2610.28480](https://arxiv.org/abs/2610.28480) | Our base method; no official code. |
| B | Janner et al. *When to Trust Your Model: Model-Based Policy Optimization* (MBPO). [arXiv:1906.08253](https://arxiv.org/abs/1906.08253) | Short model rollouts from real states (actor horizon 8). |
| B | Feinberg et al. *Model-Based Value Estimation* (MVE). [arXiv:1803.00101](https://arxiv.org/abs/1803.00101) | Real-transition critic targets (`real`, `real_wm`). |
| B | Buckman et al. *STEVE*. [arXiv:1807.01675](https://arxiv.org/abs/1807.01675) | Ensemble-weighted value expansion. |
| B | Fujimoto, Gu. *A Minimalist Approach to Offline RL* (TD3+BC). [arXiv:2106.06860](https://arxiv.org/abs/2106.06860) | Our behavior regularization scale. |
| C | Yu et al. *MOPO: Model-based Offline Policy Optimization*. [arXiv:2005.13239](https://arxiv.org/abs/2005.13239) | Uncertainty-penalized model rewards. |
| C | *Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation*. RA-L. [arXiv:2508.21065](https://arxiv.org/abs/2508.21065) | Differentiable simulation plus residual dynamics; RA-L format reference. |

## Object and material data

DiSECt material parameters are not only Young's modulus and Poisson ratio; the
cut-spring parameters must be fitted to a knife force profile. Literature moduli
alone therefore do not define a DiSECt food.

| Source | Contents | Status |
|---|---|---|
| DiSECt dataset ([repo](https://github.com/NVlabs/DiSECt), 96 MB, CC BY-NC 4.0) | Meshes plus ANSYS LS-DYNA simulated knife forces and nodal motion: prism, cylinder (jello), sphere (apple). Configs in `examples/config/`. | Used; prism is our L1 object. |
| SliceIt OSX dataset (link in `SliceIt/disect/README.md`, CC BY-NC 4.0) | Real knife-force profiles (`calibrated/*.npy`) referenced by `examples/config/osx/` for cucumber (E 2.6 MPa), potato (2.0 MPa) and tomato (40 kPa). The three configs share default cut-spring parameters (ke 200, softness 200, surface softness 300), so they are not fitted DiSECt materials. | Google Drive link returned "page not found" on 2026-11-07; ask the authors. A cucumber mesh is in `catkin_ws/src/o2ac_gazebo/models/cucumber`. |
| Sawhney et al. *Playing with Food*. [arXiv:2101.02252](https://arxiv.org/abs/2101.02252) | Real robot cutting of multiple foods with force, audio, images and proprioception; publicly downloadable. | Check food list and F/T frame. |
| Luo, Yuan (robocutting). [arXiv:2607.04367](https://arxiv.org/abs/2607.04367) | Force, velocity and displacement from trial cuts on 14 foods with a UR5e. | Release not confirmed. |
| Lenz et al. *DeepMPC*. RSS 2015 | 1488 PR2 cutting examples over 20 materials. | Release and modalities not confirmed. |
| Zhang et al. *Leveraging Multimodal Haptic Sensory Data for Robust Cutting*. [arXiv:1909.12460](https://arxiv.org/abs/1909.12460) | More than 20 foods, joint-torque knife force estimates. | Release not confirmed. |
| *CulinaryCut-VLAP*. [arXiv:2601.06451](https://arxiv.org/abs/2601.06451) | Simulated MPM cutting data with force annotations, 7 fruits. | Simulated, not measured. |
| Jia lab, Iowa State (via DiSECt) | Real apple and potato knife forces with laser-scanned meshes. | Provided to DiSECt; not public. |
| GenVegeFruits3D ([LIRIS](https://datasets.liris.cnrs.fr/genvegefruits3d-version1)) | Meshes for 100 fruit and vegetable categories. | Geometry only; partial release. |

## Still to do

- Citation tracking: papers that cite SliceIt, DiSECt and RoboNinja.
- Full reads of the robocutting paper, Kuo et al. and the second half of
  TopoCut before any novelty claim.
- Search for cross-object force-aware cutting policies and for real-robot
  validation of cutting simulators with F/T sensors.
