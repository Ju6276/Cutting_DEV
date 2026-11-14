# BladeMaster forward integration patches

External repository: `git@github.com:Ju6276/BladeMaster-Lab.git`.
Base: `4b2d983`. Apply `0001-intact-reference-cloud.patch`, then
`0002-batched-worlds.patch` with `git am`, or check out the published
`codex/ur5e-batched` branch at `e2e5221`.

The first patch accepts intact uniform-volume samples. The second generates
world-indexed forward kernels and implements independent active masks, contact
history and reset. It retains the original scalar forward equations.

Set the training configuration's `source_root` to the external checkout; keep
its license with the source. Dependencies use the project's configured Warp,
NumPy, Torch and MuJoCo environment. Validation details and GPU capacity are in
`docs/BLADEMASTER_BATCHING_REVIEW.md`. Large mesh samples remain external assets.
