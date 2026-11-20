#!/bin/bash
# Install the native BladeMaster runtime extras (requirements/blademaster.txt) into a
# target directory that cutting_env.sh puts on PYTHONPATH. --no-deps keeps the base numpy.
# Replaces the undocumented /dataHDD/dong/Envs/blademaster-extra-20261114 used on A6000.
#   scripts/env/make_bm_extra.sh [TARGET]   (default /dataHDD/dong/Envs/cutting-bm-extra)
set -euo pipefail
cd "$(dirname "$0")/../.."
target="${1:-/dataHDD/dong/Envs/cutting-bm-extra}"
pip_py="$(readlink -f .venv-sac)/bin/python"   # the base .venv has no pip
[ -x "$pip_py" ] || { echo "run scripts/env/make_sac_env.sh first (provides pip)" >&2; exit 1; }
"$pip_py" -m pip install -q --no-deps --target "$target" -r requirements/blademaster.txt
PYTHONPATH="$target" .venv/bin/python - <<'PY'
import numpy, scipy
from scipy.spatial.transform import Rotation, Slerp
assert scipy.__version__ == '1.18.1', scipy.__version__
print('bm extra ok', 'scipy', scipy.__version__, 'numpy', numpy.__version__, scipy.__file__)
PY
