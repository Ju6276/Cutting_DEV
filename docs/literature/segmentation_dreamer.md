# Segmentation Dreamer — auxiliary gradients with the correct scope
Reviewed 2026-11-10.

## Sources/status
[Author project and final RLJ paper](https://indylab.github.io/SD/), paper method §§3–4, experiments §5.1 and Appendix F; other benchmark appendices not exhaustively audited.
[Preprint](https://arxiv.org/abs/2410.09972). The official project still labels code “Coming Soon”; no source audit claimed.

## Method
Train a Dreamer representation by reconstructing task-relevant masked RGB targets, while deployment still consumes original RGB. Obtain masks from simulator truth or a foundation segmentation model adapted with few annotations. A second binary-mask decoder identifies likely false negatives in the RGB reconstruction target; corresponding pixel losses are suppressed.
Its binary decoder receives a stopped latent representation. Appendix F compares against allowing that auxiliary loss into the WM and reports poorer downstream control without the stop.

## Data and evidence
Visual DMC/Meta-World with distracting backgrounds; reported evaluation averages across four training seeds. The final paper also includes robot lane following. Masks are training supervision, not a mandatory policy input. Removing task-relevant ground pixels can hurt contact modeling, illustrating that the choice of “irrelevant” information matters.

## Relevance
This is an actual precedent for isolating an auxiliary prediction head's fitting gradient. It concerns mask prediction, not success/force termination, and does not establish that detaching terminal BCE improves cutting control. The actor's gradient estimator and physical-state versus latent-state representation also differ. The current cutting method retains the terminal BCE gradient; this paper is background, not a selected component.
