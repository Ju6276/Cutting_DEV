"""JSON-line logger process using the independent modern Python environment."""
import contextlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys


class SidecarTracker:
    def __init__(self, output, config, job_type='debugging'):
        environment = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
            environment.pop(key, None)
        environment['NETRC'] = '/dataHDD/dong/.config/wandb/netrc'
        environment['WANDB_CONFIG_DIR'] = '/dataHDD/dong/.config/wandb'
        environment['WANDB_CACHE_DIR'] = '/dataHDD/dong/.cache/wandb'
        environment['WANDB_DATA_DIR'] = '/dataHDD/dong/.cache/wandb-data'
        environment['XDG_CACHE_HOME'] = '/dataHDD/dong/.cache'
        environment['TMPDIR'] = '/dataHDD/dong/tmp'
        executable = '/dataHDD/dong/Projects/CuttingGpuBackend/.venv/bin/python'
        self.process = subprocess.Popen(
            [executable, str(Path(__file__).resolve()), '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1, env=environment)
        try:
            self.info = self.request(dict(op='init', output=str(Path(output).resolve()),
                                          config=config, job_type=job_type), timeout=60)
        except BaseException:
            self.close()
            raise

    def request(self, message, timeout=60):
        if self.process.poll() is not None:
            raise RuntimeError('W&B sidecar exited with code ' + str(self.process.returncode))
        self.process.stdin.write(json.dumps(message) + '\n')
        self.process.stdin.flush()
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError('W&B sidecar timed out during ' + message['op'])
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('W&B sidecar closed before acknowledging ' + message['op'])
        reply = json.loads(line)
        if not reply.get('ok'):
            raise RuntimeError('W&B sidecar ' + message['op'] + ' failed: ' + reply.get('error', 'unknown'))
        return reply

    def log(self, values):
        self.request(dict(op='log', values=values))

    def finish(self, status=None):
        try:
            self.request(dict(op='finish', status=status))
            self.process.stdin.close()
            code = self.process.wait(timeout=10)
            if code:
                raise RuntimeError('W&B sidecar finish exit code ' + str(code))
        finally:
            self.close()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def worker():
    protocol = sys.stdout
    tracker = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            operation = message['op']
            with contextlib.redirect_stdout(sys.stderr):
                if operation == 'init':
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                    from cutting_world_model.tracking import ExperimentTracker
                    tracker = ExperimentTracker(message['output'], message['config'],
                                                job_type=message['job_type'])
                    reply = dict(ok=True, op=operation, id=tracker.run.id, url=tracker.run.url)
                elif operation == 'log':
                    tracker.log(message['values'])
                    reply = dict(ok=True, op=operation)
                elif operation == 'finish':
                    tracker.finish(status=message.get('status'))
                    reply = dict(ok=True, op=operation)
                else:
                    raise ValueError('Unknown logger operation: ' + operation)
            protocol.write(json.dumps(reply) + '\n')
            protocol.flush()
            if operation == 'finish':
                return
        except BaseException as error:
            protocol.write(json.dumps(dict(ok=False, error=repr(error))) + '\n')
            protocol.flush()
            raise
    if tracker is not None:
        with contextlib.redirect_stdout(sys.stderr):
            tracker.finish(status={'state': 'failed', 'error': 'Training pipe closed without finish'})


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Use SidecarTracker from the training wrapper')
    worker()
