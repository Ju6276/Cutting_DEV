# Source versions

The main branch retains both original histories through a two-parent merge:

- CuttingWorldModel: `57286e4` (66 source commits before integration).
- CuttingGpuBackend: `5279f77` (14 source commits before integration).
- SuperMimic-Reproduced: `ba610534`, source subset and license in
  `third_party/supermimic/`.
- ndm-gym: `0171f5e`, the 500-round WM preparation stage is recorded in METHOD.
- Optional MPPI optimizer: [UM-ARM-Lab/pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi),
  `e04a569cd4215e705f9013145c496fc59cb25ed6`, MIT; installed by
  `requirements/baselines.txt`. Its sampling and update code is unchanged.

The exact SliceIt and nested repository versions are in
[`sources.json`](../../integrations/sliceit/patches/sources.json). Each entry records
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


2026-11-16 shared physical knife: optional BladeMaster reproduction source
[BladeMaster-Lab](https://github.com/Ju6276/BladeMaster-Lab),
commit `0106c14e85dc738d372981d87abb0cede4ba6b72`,
branch `codex/shared-disect-knife`. GPU7 checkout:
`/dataHDD/dong/Projects/BladeMaster-Warp-shared-knife-20261116`.
This supplies the two-slope finite blade contact/constraint kernels; Cutting_DEV
supplies the common UR5e rigid mount. The task identity is
`disect_knife_rigid_mount_v1`; see the shared-knife section of
[场景](../METHOD.md#scene) for geometry and commands.


2026-11-16 historical UR5e BladeMaster integration source: commit
`c51ef733364aa9cbbbceb6ba5b422cabf67ddefc` on `codex/shared-disect-knife`,
same GPU7 checkout as above. Adds persistent per-world material coefficient
buffers, masked raw/settled snapshots and generator support. Scalar physical
formulae are unchanged. Full GPU7 regression: 2165 passed / 8 skipped /
18 subtests passed (2140.56 s). The current method guide defines the paired episode distribution.


The UR10e curved-CAD forward integration was reviewed against
[BladeMaster-Lab 166a269](https://github.com/Ju6276/BladeMaster-Lab/commit/166a269),
based on c51ef733364aa9cbbbceb6ba5b422cabf67ddefc, in
/dataHDD/dong/Projects/BladeMaster-Warp-ur10e-ikea.
Current configurations select this separate source root; run manifests retain
the exact native revision used. Do not substitute the old straight-wedge
checkout merely because both expose a BladeMaster API. This records the
integration source, not acceptance of native differentiation.

## Historical backend probes

The original UR5e import/coupling method and 2026-11-02 robot-only measurements
are historical source evidence, not current UR10e runtime documentation:
[import and coupling](https://github.com/Ju6276/Cutting_DEV/blob/7cc9d1d/backends/mujoco/docs/METHOD.md),
[measurements](https://github.com/Ju6276/Cutting_DEV/blob/7cc9d1d/backends/mujoco/docs/RESULTS.md).
Current method/operation information is maintained only in the root docs guides.
