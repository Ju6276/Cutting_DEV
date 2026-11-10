# Sashimi-Bot — distinguish shape RL from cutting control

Reviewed 2026-11-10. [Published article](https://doi.org/10.1038/s44182-026-00098-9);
retrieved publisher [accepted manuscript](https://www.nature.com/articles/s44182-026-00098-9_reference.pdf).
Read scope: cutting and tactile results, shape-learning method, cutting planner,
contact classifier, and data/code availability. Other modules are contextual;
this is not a full supplementary implementation audit.

## Method and evidence

Three robot arms straighten salmon, cut while stabilizing it, and pick/place
slices. SAC learns **shape manipulation by pushing**, using binary segmentation
of current/goal shape and an IoU reward. Sim-to-real policy transfer concerns
that module.

Cutting itself uses a geometry-based blade-tangent trajectory planner with
configured tilt, depth and stroke lengths. A GelSight image classifier detects
board contact; the remaining trajectory is vertically corrected to avoid
penetrating below the detected board. This is not a learned cutting WM,
WM+BPTT cutting policy, or numerical force-limit controller.

Four cutting configurations each show five non-initial 7 mm cuts with successful
separation. Contact classifier data comprises 12,397 tactile readings from
157 trajectories; held-out trajectory evaluation reports accuracy, precision
and recall. Recognition latency/ambiguity at initial contact remains relevant.
System-level outcomes must not be mislabeled cutting-policy RL comparisons.

## Release status

The inspected publisher manuscript explicitly says the full source code will
not be publicly released; core code was provided to editors/reviewers and
additional pseudocode can be requested. Summary source data and videos are
publisher supplements. No public algorithm implementation was inspected.

## Relevance

An integrated cutting system can be a valid contribution without a new generic
RL optimizer. But our claimed contribution must match the actual learned module.
This paper does not establish that simply adding objects to our current fixed
task constitutes a comparable system contribution.
