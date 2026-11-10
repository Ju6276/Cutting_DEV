# DexNinja — data densification and category-specific imitation
Reviewed 2026-11-10.

## Sources/scope
[Paper](https://arxiv.org/abs/2610.17208), formulation/pipeline §§3–5, experiments §§6–8. Supplementary proofs and all simulator implementation details are not independently verified.
[Official project](https://dex-ninja.github.io/) was retrieved directly after browser failure. It links [simulated assets](https://huggingface.co/datasets/DexNinja/DexNinja_Simulated_Assets), paper and videos, but no algorithm-code link at this review. Assets are not the training algorithm or pretrained policy.

## Algorithm
Reconstruct real objects and knife transforms from teleoperation. Sample category-conditioned geometry and physical parameters, refine/check simulated cutting trajectories, and retain feasible successful demonstrations. Train arm/hand ACT-style chunked imitation with visual, tactile and proprioceptive signals. Shorter execution chunks are used in contact-sensitive stages. The differentiable simulator supports data generation; deployment is learned imitation, not our learned-WM BPTT training.

## Results and exact scope
Main table: four foods, 25 held-out instances each, 5 real / 50 simulated sequences in the stated protocol. Sim+real totals are 76% progress, 50% success and 60% integrity. Do not replace this table with the abstract's “more than 60%” phrase without stating the scaling setting. Integrity is manually judged. The paper explicitly says base policies are **category-specific**; it does not establish one universal policy for all categories. Lemons provide additional multi-cut tests with declining success as complexity rises.

## Implication
Real-to-sim augmentation, food variation, tactile feedback and cutting imitation are already covered. Shared multi-material feedback may differ in setting, but novelty needs the modeling/control mechanism. This work is useful for eventual real-data acquisition, not a reason to add vision, dexterous hands or multi-cut planning now.
