#!/bin/bash
# Create the offline renderer environment for this checkout and link it as .venv-render.
# A clean venv (no learning packages) from the same Python 3.12 as the base .venv, with the
# exact package set of requirements/render.txt. The base .venv is not modified.
#   scripts/env/make_render_env.sh [ENV_DIR]   (default /dataHDD/dong/Envs/cutting-render)
set -euo pipefail
cd "$(dirname "$0")/../.."
env_dir="${1:-/dataHDD/dong/Envs/cutting-render}"
base_py="$(readlink -f .venv)/bin/python"
[ -x "$base_py" ] || { echo "base .venv python missing" >&2; exit 1; }
interp="$("$base_py" -c 'import sys; print(sys._base_executable)')"
[ -x "$env_dir/bin/python" ] || "$interp" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install -q -r requirements/render.txt
ln -sfn "$env_dir" .venv-render
.venv-render/bin/python -c "import vtk, mujoco, collada, matplotlib, PIL; print('render env ok', 'vtk', vtk.vtkVersion.GetVTKVersion(), 'mujoco', mujoco.__version__)"
