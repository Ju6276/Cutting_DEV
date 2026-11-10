# Reading list

Literature and data entry points for the RA-L paper (research question in
[PLAN](PLAN.md)). Each entry says why it matters here. Individual paper/code records and explicit unread scopes are in
[literature/README](literature/README.md); synthesis is in [NOVELTY](NOVELTY.md).
The 2026-11-10 review distinguishes retrieved versions, author code, community
implementations and unavailable releases. A source inspection is not a run.

Priority: **A** decides novelty or the paper's framing; **B** method background;
**C** context.

## Robotic cutting: closest competitors

| P | Paper | Why read |
|---|---|---|
| A | Luo, Yuan. *A Perception-Manipulation Robotics System for Food Cutting*. IROS 2026. [arXiv:2607.04367](https://arxiv.org/abs/2607.04367), [project](https://robotproject8.github.io/robocutting/) | Real UR5e food cutting. PPO tunes target downward force and sawing speed. Reward is downward progress minus mechanical work. Ten foods, one policy per food, plus knife selection from trial cuts. Their actions bound force to [-30, 0] N. Closest real-robot RL work. |
| A | Beltran-Hernandez, Erbetti, Hamaya. *SliceIt! A Dual Simulator Framework for Learning Robot Food Slicing*. ICRA 2024. [arXiv:2404.02569](https://arxiv.org/abs/2404.02569) | Our task, reward and SAC baseline. Paper force-penalty formula differs from the code; see METHOD. |
| A | Wang, Bian, Heiden, Garg. *TopoCut: Learning Multi-Step Cutting with Spectral Rewards and Discrete Diffusion Policies*. arXiv 2025. [arXiv:2509.19712](https://arxiv.org/abs/2509.19712) | Cutting benchmark (MLS-MPM, Taichi): slice, stick and dice goals, seen and novel shapes. Kinematic knife; no sensor-force prediction target in the inspected method. Goal-conditioned diffusion behavior cloning; code still advertised as forthcoming. See the individual note; this does not by itself establish our novelty. |
| A | Xu et al. *RoboNinja: Learning an Adaptive Cutting Policy for Multi-Material Objects*. RSS 2023. [arXiv:2302.11553](https://arxiv.org/abs/2302.11553) | Differentiable MPM; closed-loop policy trained from optimized trajectories; collision and energy metrics. |
| A | Heiden et al. *DiSECt: A Differentiable Simulator for Parameter Inference and Control in Robotic Cutting*. Autonomous Robots. [arXiv:2203.10263](https://arxiv.org/abs/2203.10263) | Our material model; gradient trajectory optimization baseline; Ansys force data for sim validation. |
| B | Heiden et al. *DiSECt: A Differentiable Simulation Engine for Autonomous Robotic Cutting*. RSS 2021. [arXiv:2105.12244](https://arxiv.org/abs/2105.12244) | Original DiSECt; mean-force and time objective. |
| B | Lou et al. *DexNinja: Learning Robust Dexterous Cutting Policy with a Real-to-Sim-to-Real Data Engine*. arXiv 2026. [arXiv:2610.17208](https://arxiv.org/abs/2610.17208) | Dexterous-hand cutting with RGB-D and fingertip tactile input; ACT-style behavior cloning; differentiable MPM data engine built on RoboNinja; category-specific policies and real-robot generalization; assets release differs from algorithm code. |
| B | Yang et al. *BladeMaster: Real-Time Robotic Cutting Simulation with Online-Generated Persistent Discontinuities*. arXiv 2026. [arXiv:2609.27342](https://arxiv.org/abs/2609.27342) | GPU TLMPM cutting with tool reaction forces and persistent cuts; no learning. Forward mechanics reference; no new simulator migration is planned. |
| C | *CRESSim-MPM: A Material Point Method Library for Surgical Soft Body Simulation with Cutting and Suturing*. IROS 2025. [arXiv:2502.18437](https://arxiv.org/abs/2502.18437) | GPU MPM cutting, surgical domain. |
| C | *Sashimi-Bot: autonomous tri-manual advanced manipulation and cutting of deformable objects*. npj Robotics 2026. [link](https://www.nature.com/articles/s44182-026-00098-9) | SAC transfers shape-pushing control; cutting uses geometric trajectories with tactile board-contact correction. Do not call cutting itself an RL policy. |
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

## Model-based RL and value-learning precedents

| P | Paper | Why read |
|---|---|---|
| A | *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*. CoRL 2026. [arXiv:2610.28480](https://arxiv.org/abs/2610.28480) | Our base method; no official code. |
| B | Janner et al. *When to Trust Your Model: Model-Based Policy Optimization* (MBPO). [arXiv:1906.08253](https://arxiv.org/abs/1906.08253) | Short model rollouts augment SAC replay; this is not our BPTT actor implementation. |
| B | Feinberg et al. *Model-Based Value Estimation* (MVE). [arXiv:1803.00101](https://arxiv.org/abs/1803.00101) | Model-expanded targets and TD-k at imagined states; not an implementation or justification of our `real`/`real_wm` targets. |
| B | Buckman et al. *STEVE*. [arXiv:1807.01675](https://arxiv.org/abs/1807.01675) | Ensemble-weighted value expansion. |
| B | Fujimoto, Gu. *A Minimalist Approach to Offline RL* (TD3+BC). [arXiv:2106.06860](https://arxiv.org/abs/2106.06860) | Precedent for actor BC regularization; original offline objective includes Q scaling and is not identical to our fixed online coefficient. |
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
| Sawhney et al. *Playing with Food*. [arXiv:2101.02252](https://arxiv.org/abs/2101.02252) | Real robot cutting of multiple foods with force, audio, images and proprioception; publicly downloadable. | See the data/module review in literature/playing_with_food.md; dataset access and task labels differ from our online transitions. |
| Luo, Yuan (robocutting). [arXiv:2607.04367](https://arxiv.org/abs/2607.04367) | Force, velocity and displacement from trial cuts on 14 foods with a UR5e. | Release not confirmed. |
| Lenz et al. *DeepMPC*. RSS 2015 | 1488 PR2 cutting examples over 20 materials. | Release and modalities not confirmed. |
| Zhang et al. *Leveraging Multimodal Haptic Sensory Data for Robust Cutting*. [arXiv:1909.12460](https://arxiv.org/abs/1909.12460) | 25 materials/conditions, force and audio fusion for six cutting phases; DMP execution. | Release not confirmed. |
| *CulinaryCut-VLAP*. [arXiv:2601.06451](https://arxiv.org/abs/2601.06451) | Simulation cutting demonstrations and force annotations; versions differ. | See literature/culinarycut.md for v1 versus current project release, and actual code/data accessibility. |
| Jia lab, Iowa State (via DiSECt) | Real apple and potato knife forces with laser-scanned meshes. | Provided to DiSECt; not public. |
| GenVegeFruits3D ([LIRIS](https://datasets.liris.cnrs.fr/genvegefruits3d-version1)) | Meshes for 100 fruit and vegetable categories. | Geometry only; partial release. |

## Still to do

Open reading items are tracked in [NOVELTY](NOVELTY.md) ("Still open").
