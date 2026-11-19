"""Exercise the sourced server environment without changing any real installation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/env/cutting_env.sh'
KEYS = ('CUDA_HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'TORCH_EXTENSIONS_DIR',
        'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'WARP_CACHE_PATH',
        'TORCH_CUDA_ARCH_LIST', 'MAX_JOBS', 'CUDA_VISIBLE_DEVICES',
        'SLICEIT_ROOT', 'CUTTING_LOGGER_PYTHON', 'NETRC', 'PYTHONPATH')


def source_env(tmp_path, host, overrides=None, selected='base'):
    root = tmp_path / 'checkout'
    (root / 'scripts/env').mkdir(parents=True)
    shutil.copyfile(SCRIPT, root / 'scripts/env/cutting_env.sh')
    for venv in ('.venv', '.venv-sac', '.venv-render'):
        (root / venv / 'bin').mkdir(parents=True)
        (root / venv / 'bin/python').symlink_to(sys.executable)
    (root / 'hostname').write_text('#!/bin/sh\nprintf "%s\n" "' + host + '"\n')
    (root / 'hostname').chmod(0o755)
    (root / 'dump.py').write_text(
        'import os,json,sys\nprint("RESULT="+json.dumps({k:os.environ.get(k) for k in sys.argv[1:]}))')
    env = {k: v for k, v in os.environ.items() if k not in KEYS and k != 'CUTTING_ENV'}
    env.update(PATH=str(root) + os.pathsep + env['PATH'], CUTTING_ENV=selected)
    env.update(overrides or {})
    result = subprocess.run(
        ['bash', '-c', 'source scripts/env/cutting_env.sh && python dump.py "$@"',
         'env-test', *KEYS, 'PATH'], cwd=root, env=env, capture_output=True, text=True, check=True)
    row = next(line for line in result.stdout.splitlines() if line.startswith('RESULT='))
    return root, json.loads(row[len('RESULT='):])


@pytest.mark.parametrize('host,gpu,arch', [('tamsGPU5', '1', '8.6'), ('tamsGPU7', '0', '8.9'),
                                          ('tamsGPU9', '0', '8.9')])
def test_host_defaults_and_checkout_imports(tmp_path, host, gpu, arch):
    root, out = source_env(tmp_path, host)
    assert (out['CUDA_VISIBLE_DEVICES'], out['TORCH_CUDA_ARCH_LIST']) == (gpu, arch)
    assert out['PYTHONPATH'] == f'{root}:{root}/backends/mujoco'
    assert out['MAX_JOBS'] == '1'
    assert out['PATH'].split(':')[0] == str(root / '.venv/bin')


def test_explicit_paths_and_device_selection_survive_sourcing(tmp_path):
    overrides = {k: '/tmp/custom_' + k.lower() for k in KEYS}
    overrides.update(CUDA_VISIBLE_DEVICES='7', TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='2',
                     PYTHONPATH='/tmp/external_modules')
    root, out = source_env(tmp_path, 'tamsGPU9', overrides, selected='sac')
    for key, val in overrides.items():
        assert out[key] == (f'{root}:{root}/backends/mujoco:{val}' if key == 'PYTHONPATH' else val)
    assert out['PATH'].split(':')[0] == str(root / '.venv-sac/bin')


@pytest.mark.parametrize('selected', ['sac', 'planner'])
def test_baseline_and_planner_names_select_the_baselines_env(tmp_path, selected):
    root, out = source_env(tmp_path, 'tamsGPU7', selected=selected)
    assert out['PATH'].split(':')[0] == str(root / '.venv-sac/bin')


def test_render_name_selects_the_render_env(tmp_path):
    root, out = source_env(tmp_path, 'tamsGPU9', selected='render')
    assert out['PATH'].split(':')[0] == str(root / '.venv-render/bin')


def test_unknown_env_name_is_rejected(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        source_env(tmp_path, 'tamsGPU7', selected='mppi')


def test_explicit_cpu_only_selection_is_not_replaced_by_host_gpu(tmp_path):
    _, out = source_env(tmp_path, 'tamsGPU5', {'CUDA_VISIBLE_DEVICES': ''})
    assert out['CUDA_VISIBLE_DEVICES'] == ''
