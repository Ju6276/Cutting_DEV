# Headless DiSECt core

Source: SliceIt's [DiSECt fork](https://github.com/omron-sinicx/DiSECt), local
revision `db7de0f`. Files under `disect/dflex` retain the NVIDIA Source Code
License in [LICENSE.md](LICENSE.md).

The FEM, cutting-spring, contact and integrator equations are copied unchanged.
The package initializer omits the optional USD renderer. Compiler compatibility
changes replace removed Python `imp` loading, handle Python 3.9+ AST subscript
nodes, remove the unsupported GCC `-Z` flag and put generated extensions in
`TORCH_EXTENSIONS_DIR/disect_headless`. Original geometry construction runs
through `scripts/export_disect_scene.py` in the source DiSECt environment;
the runtime loads its model/state without ROS, Qt or geometry-generation tools.
