# Source versions

The main branch retains both original histories through a two-parent merge:

- CuttingWorldModel: `57286e4` (66 source commits before integration).
- CuttingGpuBackend: `5279f77` (14 source commits before integration).
- SuperMimic-Reproduced: `ba610534`, source subset and license in
  `third_party/supermimic/`.
- ndm-gym: `0171f5e`, the 500-round WM preparation stage is recorded in METHOD.
- Optional MPPI optimizer: [UM-ARM-Lab/pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi),
  `e04a569cd4215e705f9013145c496fc59cb25ed6`, MIT; installed by
  `requirements-mpc.txt`. Its sampling and update code is unchanged.

The exact SliceIt and nested repository versions are in
[`sources.json`](../integrations/sliceit/patches/sources.json). Each entry records
the public base commit, original local revision and expected Git tree, with a
binary-capable source patch where needed. The checkout script verifies that
applying each patch reproduces its source tree. These patches include the
material bridge, controller state export, synchronized stepping and SAC fixes.

```bash
python3 scripts/checkout_sliceit.py --output external/SliceIt
python3 scripts/checkout_sliceit.py --output external/SliceIt --check
```

The external checkout retains its patches in the Git index. Its Git status
therefore records local source changes; the parent repository tracks their
patches and version lock. This script replaces the upstream README's clone
and recursive submodule commands. Continue with dependency installation and
ROS workspace builds in the pinned SliceIt README. Optional uninstalled
upstream submodules are not fetched by this script.

Original code: [SliceIt](https://github.com/omron-sinicx/sliceit),
[DiSECt](https://github.com/NVlabs/DiSECt),
[SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced),
[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym).

Historical RESULTS entries retain their original project paths and commits.
The current commands use this repository root and `backends/mujoco`.


2026-11-12 local source correction: bundled
`backends/mujoco/third_party/disect/disect/dflex/sim.py` now uses a consistent
second-endpoint contact coordinate and transpose interpolation weights for force
distribution (METHOD, `edge_coordinate_consistent_v2`). This changes the bundled
physical solver, not the separately pinned Gazebo/SliceIt checkout patches.
The old bundled behavior is retained in Git at `38b85a5`. Attribution and licenses
are unchanged; this is our correction, not a claim of an upstream release.

## Official SuperMimic adaptation

The separately adapted author implementation (HybridRobotics/supermimic at
`5efbf11`) is documented in [the component mapping](OFFICIAL_SUPERMIMIC.md).
It is distinct from the community reproduction used by the active learner.
That document's path is retained because the official adapter writes it into
run manifests; the unsuccessful historical cutting run is not an upstream
algorithm verdict.

## Local DFlex execution changes

The free-cut reset acceleration also changed bundled DFlex launches to use the
current PyTorch CUDA stream, and repaired extension-cache freshness detection
(`1b287d5`). These are local execution/cache changes affecting every bundled path,
including legacy tasks; they are not upstream releases. See [METHOD](METHOD.md)
for regression scope and the unresolved long-horizon damage-equivalence check.
