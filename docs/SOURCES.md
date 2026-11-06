# Source versions

The main branch retains both original histories through a two-parent merge:

- CuttingWorldModel: `57286e4` (66 source commits before integration).
- CuttingGpuBackend: `5279f77` (14 source commits before integration).
- SuperMimic-Reproduced: `ba610534`, source subset and license in
  `third_party/supermimic/`.
- ndm-gym: `0171f5e`, the 500-round WM preparation stage is recorded in METHOD.
- [supermimic_manipulation](https://github.com/emoPointer/supermimic_manipulation):
  `eda7da480c6d92a68cc41c9628b7efdae08cae23`, MIT. The separate
  `train_manipulation.py` entry imports its runner, agent, replay and networks
  directly from a clean external checkout. Its Python 3.13 environment uses
  the upstream `uv.lock`; physics retains the existing Python 3.12 environment.

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
