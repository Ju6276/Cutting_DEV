# Runtime environment for Cutting_DEV on the GPU servers. Source from the repository root:
#   source scripts/env/cutting_env.sh          # WM+BPTT, evaluation, tests (.venv)
#   CUTTING_ENV=sac source scripts/env/cutting_env.sh   # SAC baselines (.venv-sac)
# .venv is the host's base GPU environment (torch/warp/mujoco); .venv-sac layers
# requirements-baselines.txt on top of it (scripts/env/make_sac_env.sh).
# Host-specific values are chosen here so all servers use one tracked file.
_root="$(pwd)"
if [ ! -f "$_root/scripts/env/cutting_env.sh" ]; then
    echo "cutting_env.sh: source it from the repository root" >&2; return 1 2>/dev/null || exit 1
fi
case "${CUTTING_ENV:-base}" in
    base) _venv="$_root/.venv" ;;
    sac)  _venv="$_root/.venv-sac" ;;
    *) echo "cutting_env.sh: CUTTING_ENV must be base or sac" >&2; return 1 2>/dev/null || exit 1 ;;
esac
if [ ! -x "$_venv/bin/python" ]; then
    echo "cutting_env.sh: $_venv/bin/python missing (sac: run scripts/env/make_sac_env.sh)" >&2
    return 1 2>/dev/null || exit 1
fi
case "$(hostname -s)" in
    tamsGPU5|tamsgpu5) _arch=8.6; _gpu=1 ;;   # A6000 host: GPU0 belongs to another user
    *) _arch=8.9; _gpu=0 ;;                    # GPU7 / GPU9: RTX 4090
esac
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$_venv/bin:$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME=/dataHDD/dong/.cache TMPDIR=/dataHDD/dong/tmp
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions_cutting_gpu" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor_cutting"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton_cutting" WARP_CACHE_PATH="$XDG_CACHE_HOME/warp/cutting_gpu_backend"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$_arch}"
export MAX_JOBS="${MAX_JOBS:-1}"   # parallel nvcc/gcc builds hit a gcc ICE on GPU9 (2026-11-18)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$_gpu}"
export SLICEIT_ROOT=/dataHDD/dong/Projects/SliceIt
export CUTTING_LOGGER_PYTHON="$_root/.venv/bin/python"
export NETRC=/dataHDD/dong/.config/wandb/netrc
export PYTHONPATH="$_root:$_root/backends/mujoco"
echo "cutting_env: $(hostname -s) env=${CUTTING_ENV:-base} python=$_venv/bin/python commit=$(git -C "$_root" rev-parse --short HEAD) gpu=$CUDA_VISIBLE_DEVICES"
unset _root _venv _arch _gpu
