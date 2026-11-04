# Sources and licenses

This repository combines the CuttingWorldModel and CuttingGpuBackend source
histories. Vendored and patched dependencies retain their upstream licenses:

- SuperMimic-Reproduced: `third_party/supermimic/LICENSE`.
- SliceIt task code: `third_party/sliceit/LICENSE`.
- DiSECt/DFlex: `backends/mujoco/third_party/disect/LICENSE.md`.
- Controller/KDL implementation: `backends/mujoco/licenses/`.
- External SliceIt, controller, ROS and tf2rl checkouts: licenses in each pinned
  upstream repository. Their source changes are in `integrations/sliceit/patches/`.

The public DiSECt dataset is distributed separately under CC BY-NC 4.0.
Robot meshes, datasets, generated scenes and trained weights are prepared
outside Git. Source links and revisions are in [SOURCES](docs/SOURCES.md).
