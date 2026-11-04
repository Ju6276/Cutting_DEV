#!/usr/bin/env python3
"""Build backend assets from a recorded SliceIt reset and exported DiSECt scene."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True,
                        help='Directory containing task.json and episode_000.npz')
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True,
                        help='Directory containing scene.pt and its export manifest')
    parser.add_argument('--source-root', type=Path, action='append', required=True,
                        help='ROS package source roots used to resolve robot meshes')
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    args = parser.parse_args()
    backend = args.backend_root.resolve()
    assets = backend / 'runs'
    copies = [
        (args.capture / 'task.json', assets / 'source/task.json'),
        (args.capture / 'episode_000.npz', assets / 'source/episode_000.npz'),
        (args.controller, assets / 'controller_source/controller_params.json'),
        (args.scene / 'scene.pt', assets / 'disect_source_v4/scene.pt'),
        (args.scene / 'manifest.json', assets / 'disect_source_v4/manifest.json'),
    ]
    for source, target in copies:
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists():
            raise FileExistsError(target)
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    command = [sys.executable, str(backend / 'scripts/prepare_robot.py'),
               '--task', str(assets / 'source/task.json'),
               '--output', str(assets / 'model_coupled')]
    for source in args.source_root:
        command.extend(['--source-root', str(source.resolve())])
    subprocess.run(command, check=True)
    revision = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(
        ['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip())
    (assets / 'preparation.json').write_text(json.dumps({
        'git': {'commit': revision, 'dirty': dirty}, 'command': sys.argv,
        'inputs': [str(source.resolve()) for source, _ in copies],
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
