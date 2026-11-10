# CulinaryCut-VLAP / CulinaryCut: versioned paper and public-resource audit

- Full text read: [arXiv:2601.06451v1](https://arxiv.org/abs/2601.06451v1), “CulinaryCut-VLAP: A Vision–Language–Action–Physics Framework for Food Cutting via a Force-Aware Material Point Method,” 16 pages. Read on 2026-11-10: complete main Sections 1–6 and supplementary A–D, including physics, language templates, trajectory examples and all ablations.
- Remote paper: `/dataHDD/dong/Research/cutting_literature_20261110/culinarycut.pdf`.
- Current author [project page](https://deepvelop99.github.io/culinary.github.io/) presents a later ECCV 2026 work, “CulinaryCut: A Physics-Grounded Cutting Benchmark for Vision-Language-Action Models.” It still links the old arXiv text and shows “Paper TBD” for the later paper. **The old full text and current project claims differ; they must not be merged into one experiment record.**
- Author “Code” link: [deepvelop99/ECCV26-Culinary](https://github.com/deepvelop99/ECCV26-Culinary). Live page access and GitHub API both returned 404. The user's public repository listing did not contain that repository, although a search index/profile cache still advertised it. No source commit or training/physics implementation could be verified.
- [Sample dataset](https://drive.google.com/drive/folders/1SwtYeItEToyYlJ2Sn1m8qqSIndDWxIHN) is publicly listable. Its root contains banana, melon and peach sample folders. No bulk dataset or weights were downloaded.

## Old full text: task, data and policy

The old paper couples ManiSkill robot simulation with a Taichi MLS-MPM deformable-food solver. It uses a 7-DoF Franka arm with a knife, multi-view vision, language instructions and continuous robot actions.

Tasks combine a cutting style with a target specification: ratio/middle cuts or multiple cuts that create several pieces. Teleoperated seeds are augmented by AABB-based planning under object pose/scale and cutting variations. Instructions come from templates and language augmentation.

The old data-statistics paragraph claims 325,000 trajectories and seven foods. It also says five cut styles, whereas the task definition and template table explicitly list four. The stated combinatorial description does not by itself reconcile the total. These are unresolved reporting details, not quantities independently verified from released data.

VLA baselines are RDT-1B, Octo and OpenVLA. A contact classifier and behavior-cloned cut-style transfer module are proposed. There is **no learned physical WM with BPTT actor training** in this pipeline.

The old manipulation safety module learns `F_hat = R(v, material)` and chooses a speed satisfying its predicted force bound. Material parameters are provided to this predictor. This is already a learned force-response limiter for cutting; it is not a calibrated uncertainty bound and not proof of physical hard safety.

## Supplement A: actual stated physics and assumptions

The solver uses MLS/APIC particle-grid transfers, corotated elasticity, optional J2 plasticity and a scalar damage variable that softens Lamé moduli near the blade. Knife/board contact uses SDFs and frictional grid-velocity correction. Tool force is accumulated from momentum impulses over an output interval.

Damage is enabled by explicit proximity/contact-strength, approach-speed and downward-stroke conditions. Additional stabilization includes damping, speed caps, determinant clamping, a small separating tip force, and connectivity/color-based separation processing. Blade slowdown also uses a material-scaled quadratic speed-resistance model.

These details matter: the implementation is not described as a parameter-free, fully physical differentiable fracture solver. The paper does not verify gradients through those branches or through the coupled robot–MPM system. “Uses MPM” and “reports forces” do not establish differentiability or force calibration.

The stated force metric is an impulse-averaged output quantity, distinct from the maximum filtered sensor component or a high-frequency substep maximum in our environment.

## Old experiments and limits

Evaluation emphasizes geometric instruction grounding, object/scene variation, and style/topology effects. Success uses a tolerance relative to object length and an orientation condition; the exact angle-unit interpretation is not fully specified in the text. It is not our target-pose-count success definition.

All principal experiment descriptions in this old version are simulated. A Young's-modulus sweep produces increasing simulated force and lower blade speed; that monotonic relation alone does not establish sim-to-real quantitative validity. The introduction's real-consistency claim should not be substituted for an actual measured-real-force experiment.

The old supplementary safety-module table reports lower average maximum speed (3.63 to 0.89 m/s) and simulated maximum force (129.31 to 37.78 N). These belong to its own model and limit convention. The stated 100 N hardware limit is not a universal certified robot-safety threshold.

All three VLA baselines have 0% complete multi-cut success in the supplementary strict all-boundaries evaluation, although some individual boundaries are correctly placed. Topology updates improve matched visual/behavioral performance modestly and help avoid redundant recutting.

Do not copy the old dataset comparison table uncritically: it marks earlier work such as Playing with Food as lacking force data, which conflicts with that work's original data-collection description.

## Current project and concrete public sample

The newer author project reports **21,000 cutting trajectories, approximately 1.57 million training instances, fifteen foods and fifteen scenes**. It explicitly says force signals are dataset annotations, not VLA policy inputs, and frames the result as physical grounding through data without changing the policy architecture. These are current project claims, not details established by the older full text.

The sample `banana/auto_0` contains:
- `trajectory.h5` and `trajectory.json`;
- `alignment.json`;
- `mpm_render.npz`, `ms_cam.npz`;
- `mpm.mp4` and `ms_base.mp4`.

I read the JSON metadata through the public download endpoint. `trajectory.json` specifies `obs_mode=rgb`, `control_mode=pd_ee_delta_pose`, `reward_mode=none`, robot UID `pkour`, a scripted MPM co-simulation source and a 272-step sample marked successful. This is not an RL replay record from our UR5e/FDCC environment.

`alignment.json` includes object variation, stage boundaries and separate robot/MPM force summaries. In this sample, the MPM and robot peak-force summaries differ (about 11.61 versus 28.32 N), while its own alignment flags pass. Without source/metric definitions this cannot be interpreted as physically identical wrench feedback. No HDF5/NPZ contents or force units/frame synchronization were independently audited.

## Relation to our paper

This is an important multi-object cutting benchmark and data resource, and an existing cutting force-limiter precedent. It narrows any claim that multi-food cutting, language goals or force-aware simulated data alone are novel.

It is not an immediately verified replacement for the current simulation platform: the advertised source is inaccessible, the current full paper is unavailable at the project link, and the public data are samples from a different action/robot interface. The useful next step, if later needed, is resource verification, not an unapproved migration or claiming complete reproduction.
