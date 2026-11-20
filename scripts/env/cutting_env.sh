# Runtime environment for Cutting_DEV on the GPU servers. Source from the repository root:
#   source scripts/env/cutting_env.sh          # WM+BPTT, evaluation, tests (.venv)
#   CUTTING_ENV=sac source scripts/env/cutting_env.sh      # SAC baseline (.venv-sac)
#   CUTTING_ENV=planner source scripts/env/cutting_env.sh  # WM+MPPI / analytic MPC (.venv-sac)
#   CUTTING_ENV=render source scripts/env/cutting_env.sh   # offline studio/VTK rendering (.venv-render)
# .venv is the host's base GPU environment (torch/warp/mujoco); .venv-sac layers
# requirements/baselines.txt on top of it (scripts/env/make_sac_env.sh). The same
# environment holds SB3 and the pinned pytorch_mppi; the base
# .venv has neither (evaluate_pose_policy.py --method wm_mppi|analytic_mpc fails there).
# New base installs include SciPy. For existing base environments missing it, make_bm_extra.sh installs
# requirements/blademaster.txt into CUTTING_BM_EXTRA (default
# /dataHDD/dong/Envs/cutting-bm-extra), added to PYTHONPATH when present.
# Nonempty overrides take precedence over host defaults. An explicitly empty
# CUDA_VISIBLE_DEVICES remains empty (CPU-only). GPU defaults are not reservations.
_root="$(pwd)"
if [ ! -f "$_root/scripts/env/cutting_env.sh" ]; then
    echo "cutting_env.sh: source it from the repository root" >&2; return 1 2>/dev/null || exit 1
fi
case "${CUTTING_ENV:-base}" in
    base) _venv="$_root/.venv" ;;
    sac|planner) _venv="$_root/.venv-sac" ;;
    render) _venv="$_root/.venv-render" ;;
    *) echo "cutting_env.sh: CUTTING_ENV must be base, sac, planner or render" >&2; return 1 2>/dev/null || exit 1 ;;
esac
if [ ! -x "$_venv/bin/python" ]; then
    echo "cutting_env.sh: $_venv/bin/python missing (sac/planner: scripts/env/make_sac_env.sh; render: scripts/env/make_render_env.sh)" >&2
    return 1 2>/dev/null || exit 1
fi
case "$(hostname -s)" in
    tamsGPU5|tamsgpu5) _arch=8.6; _gpu=1 ;;   # A6000 launch default; check current occupancy
    *) _arch=8.9; _gpu=0 ;;                    # GPU7 / GPU9: RTX 4090
esac
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$_venv/bin:$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dataHDD/dong/.cache}"
export TMPDIR="${TMPDIR:-/dataHDD/dong/tmp}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch_extensions_cutting_gpu}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor_cutting}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton_cutting}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$XDG_CACHE_HOME/warp/cutting_gpu_backend}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$_arch}"
export MAX_JOBS="${MAX_JOBS:-1}"   # parallel nvcc/gcc builds hit a gcc ICE on GPU9 (2026-11-18)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-$_gpu}"
export SLICEIT_ROOT="${SLICEIT_ROOT:-/dataHDD/dong/Projects/SliceIt}"
export CUTTING_LOGGER_PYTHON="${CUTTING_LOGGER_PYTHON:-$_root/.venv/bin/python}"
export NETRC="${NETRC:-/dataHDD/dong/.config/wandb/netrc}"
# Current checkout takes precedence; retain explicitly supplied extra modules.
_bm_extra="${CUTTING_BM_EXTRA:-/dataHDD/dong/Envs/cutting-bm-extra}"
if [ -d "$_bm_extra" ]; then _bm_path=":$_bm_extra"; _bm_state="$_bm_extra"; else _bm_path=""; _bm_state="missing (BladeMaster needs scripts/env/make_bm_extra.sh)"; fi
export PYTHONPATH="$_root:$_root/backends/mujoco$_bm_path${PYTHONPATH:+:$PYTHONPATH}"
echo "cutting_env: $(hostname -s) env=${CUTTING_ENV:-base} python=$_venv/bin/python commit=$(git -C "$_root" rev-parse --short HEAD) gpu=$CUDA_VISIBLE_DEVICES bm_extra=$_bm_state"
unset _root _venv _arch _gpu _bm_extra _bm_path _bm_state
