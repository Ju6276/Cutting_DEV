#!/usr/bin/env python3
"""Wait for a budgeted training run, then evaluate its final checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--sliceit-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-hours', type=float, default=7.)
    parser.add_argument('--evaluation-hours', type=float, default=1.)
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()
    training = Path(args.training_run).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    status_path = output.parent / (output.name + '.followup.json')
    start = time.monotonic()

    def status(state, **values):
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'state': state, 'training_run': str(training),
                                         'wall_seconds': time.monotonic() - start, **values}, indent=2) + '\n')
        temporary.replace(status_path)

    try:
        while True:
            if time.monotonic() - start > args.wait_hours * 3600:
                raise TimeoutError('Training did not finish within the follow-up wait budget')
            try:
                record = json.loads((training / 'status.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                record = {}
            if record.get('state') == 'failed':
                raise RuntimeError(record.get('error', 'Training supervisor failed'))
            if record.get('state') == 'completed':
                break
            status('waiting')
            time.sleep(5)
        checkpoint = training / 'data/last.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        task = json.loads((training / 'data/task.json').read_text())
        layout = task.get('prediction_layout')
        if layout not in (None, 'control_history_v2'):
            raise ValueError('Unsupported checkpoint prediction layout: ' + str(layout))
        model_state = 'control-history' if layout == 'control_history_v2' else 'native'
        command = [sys.executable, str(Path(__file__).with_name('run_coupled_collection.py')),
                   '--sliceit-root', args.sliceit_root, '--output', str(output),
                   '--mode', 'evaluate', '--resume', str(checkpoint), '--episodes', str(args.episodes),
                   '--model-state', model_state,
                   '--max-hours', str(args.evaluation_hours),
                   '--wall-seconds', str(round(args.evaluation_hours * 3600 + 300))]
        status('evaluating', command=command)
        subprocess.run(command, check=True)
        progress = json.loads((output / 'data/progress.json').read_text())
        status('completed' if progress['state'] == 'completed' else progress['state'],
               evaluation_progress=progress)
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
