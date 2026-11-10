# Playing with Food: Learning Food Item Representations through Interactive Exploration

- Paper: [arXiv:2101.02252v1](https://arxiv.org/abs/2101.02252v1), 12 pages, ISER 2020 proceedings. Read on 2026-11-10: complete Sections 1–6, data-collection/processing details, embedding objective, both experimental tables and limitations. No separate appendix in this version.
- Author [project page](https://sites.google.com/view/playing-with-food) and [public dataset](https://drive.google.com/drive/folders/1QNMkOip22Wly3MGWlhvlxKTmc39bkfZl) were checked live. The paper's tinyurl still redirects to this Drive folder.
- Remote paper: `/dataHDD/dong/Research/cutting_literature_20261110/playing-food.pdf`.
- Code: no author embedding-training repository verified from the paper/project links and title/GitHub searches. Franka control-stack and sounddevice ROS dependencies are cited, but their presence is not release of this paper's embedding implementation.

## Sections 1–3: what data exist

The goal is to learn food material representations through multimodal interaction. This is neither a learned transition model nor RL cutting-policy training.

Two real-robot setups are used. Cutting uses two Franka Panda arms: one holds a knife and the other holds tongs. Four cameras provide overhead, side and wrist observations; contact microphones are mounted on the cutting board and tools. Playing uses a single Panda with overhead/side cameras and fingertip cameras, plus contact audio.

Cutting motions are DMPs fitted by ridge regression to kinesthetic demonstrations and chained until a cut completes. The collection includes 21 food types, ten slice instances per food type, with slice categories drawn from 14 types. Thicknesses range from 3–50 mm, angles include ±30 degrees, and some cuts are made while the food is held by tongs. Therefore the data contain different motions and outcomes, not one fixed simulated cut plane.

Each slice then receives five playing trials. The robot presses until measuring 10 N, grasps and releases from 15 cm, and records vision, force/proprioception and audio. The additional cooked-food study uses seven food categories boiled for ten minutes; these cooked slices were hand cut rather than robot cut.

The public root currently exposes `cutting_data`, `playing_data`, `old_playing_data` and `silhouette_data`. I followed the cutting/apple/slice/trial hierarchy without downloading the full dataset. A sample folder lists WAV audio, RGB PNG images, depth NPY arrays and RGB/depth AVI videos. The paper also states that robot force/proprioception are recorded. The sampled HTML listing does not independently establish the full schema, units, sampling rates or action alignment of those numerical channels.

This is a much richer real-robot interaction dataset than DiSECt mesh/material files, but it is not automatically a compatible UR5e–FDCC transition dataset.

## Sections 3.3–4: representation-learning method

Audio is converted to MFCC features and reduced with PCA. Proprioceptive features include the end-effector's final vertical position at 10 N, deformation since first contact, and gripper width after a 60 N grasp.

A pretrained ResNet34 plus MLP maps an overhead image to an embedding. Triplets are constructed using neighbors in audio/proprioceptive feature space: positive examples come from nearby interaction outcomes, negative examples from more distant outcomes. Food/slice-label triplets are also studied.

At inference, the learned representation can use **only an overhead image**. Multimodal training supervision does not mean all those sensors feed the inference network. Subsequent supervised MLPs predict food class, slice type, slice width, hardness or juiciness.

There is no action-conditioned dynamics rollout, model-based planner, critic, or BPTT control objective. The paper suggests dynamics modeling and improved manipulation as future uses, not completed results.

## Section 5: metrics and what the results establish

The main table reports 21-way food classification, 14-way slice classification, three-way hardness/juiciness classification, and slice-width RMSE. Hardness and juiciness use leave-one-food-class-out evaluation.

Benefits are task-dependent. The visual ResNet baseline is strongest for food identity (98.9% in the main table), while proprioceptive-supervised embeddings improve hardness and slice-width prediction; audio contributes other material distinctions. The main leave-one-food-out hardness/juiciness accuracies remain modest. This is not evidence of reliable force-limit prediction.

The auxiliary cooked/uncooked subset yields much stronger classification results, including approximately 99% hardness/cooked recognition for the combined audio/proprioceptive embedding and roughly 71% juiciness recognition. These numbers belong to that smaller, distinct subset and cannot replace the main 21-food results.

The paper acknowledges that material similarity and slice-shape differences limit generalization. Similar commanded cuts can produce different shapes on different foods.

## Relation to our cutting direction

Useful: material properties can be inferred from interaction rather than given as privileged simulator labels. Force, deformation and sound can support a history-dependent representation; apparent food identity alone is not the physical state.

Not established: that these embeddings predict action-dependent peak forces, distinguish irreversible cutting damage, or improve a closed-loop safe policy. Those would require aligned data and task-specific validation.

The paper is a dataset/representation precedent, not a ready-made WM+MPC baseline. Its sensors, robot, DMP action semantics and collection protocol differ from the current unified environment. Do not silently treat the public trajectories as transitions from our controller.
