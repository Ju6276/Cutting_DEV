#!/bin/bash
# Create the SAC baseline environment for this checkout and link it as .venv-sac.
# Recipe (same as GPU9 cutting-baselines-20261116): a venv made from the base .venv
# Python, a .pth that exposes the base GPU packages (torch/warp/mujoco unchanged), and
# requirements/baselines.txt installed on top. The base .venv is not modified.
#   scripts/env/make_sac_env.sh [ENV_DIR]   (default /dataHDD/dong/Envs/cutting-sac)
set -euo pipefail
cd "$(dirname "$0")/../.."
env_dir="${1:-/dataHDD/dong/Envs/cutting-sac}"
base_py="$(readlink -f .venv)/bin/python"
[ -x "$base_py" ] || { echo "base .venv python missing" >&2; exit 1; }
if [ ! -x "$env_dir/bin/python" ]; then
    "$base_py" -m venv "$env_dir"
    base_site="$("$base_py" -c 'import site; print(site.getsitepackages()[0])')"
    env_site="$("$env_dir/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    echo "$base_site" > "$env_site/existing_gpu_packages.pth"
fi
"$env_dir/bin/python" -m pip install -q -r requirements/baselines.txt
ln -sfn "$env_dir" .venv-sac
.venv-sac/bin/python - <<'PY'
import importlib.metadata as md, json, torch, warp, stable_baselines3, pytorch_mppi
import re
MPPI_REVISION = re.search(r"MPPI_REVISION = '([0-9a-f]{40})'",
    open('cutting_world_model/shared_baselines.py').read()).group(1)
vcs = json.loads(md.distribution('pytorch-mppi').read_text('direct_url.json') or '{}').get('vcs_info', {})
if vcs.get('commit_id') != MPPI_REVISION:
    raise SystemExit(f"pytorch_mppi revision {vcs.get('commit_id')} != pinned {MPPI_REVISION}")
print('sac/planner env ok', torch.__version__, warp.__version__, stable_baselines3.__version__,
      'pytorch_mppi', MPPI_REVISION[:7])
PY
