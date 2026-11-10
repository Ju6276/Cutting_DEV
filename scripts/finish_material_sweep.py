#!/usr/bin/env python3
"""Add the renderer's task/report fields to material-sweep recordings and render them."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTCOME = {'success': 'success', 'outside': 'outside', 'steps': 'truncated', 'terminated': 'terminated'}


def main():
    sweep, task = Path(sys.argv[1]), Path(sys.argv[2])
    for report_path in sorted(sweep.glob('*/report.json')):
        run = report_path.parent
        report = json.loads(report_path.read_text())
        report.setdefault('label', report['scene'])
        report['outcome'] = OUTCOME[report['ending']]
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        if not (run / 'task.json').exists():
            shutil.copy(task, run / 'task.json')
        if (run / f"{report['label']}.mp4").exists():
            continue
        title = (f"{report['scene'].replace('_', ' / ')}: reference trajectory, force limit off, "
                 f"{report['released_springs']}/{report['total_springs']} springs cut")
        subprocess.run(['/dataHDD/dong/Envs/render-20261108/bin/python', str(ROOT / 'scripts/render_episode.py'),
                        '--recording', str(run), '--title', title,
                        '--source-root', '/opt/ros/noetic/share',
                        '--source-root', '/dataHDD/dong/Projects/SliceIt/catkin_ws/src',
                        '--speed', '0.25', '--fps', '30'], check=True)


if __name__ == '__main__':
    main()
