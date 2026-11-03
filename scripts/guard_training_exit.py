#!/usr/bin/env python3
"""Reap one run after its supervisor exits, retaining worker and logger grace periods."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import psutil


def identity(process):
    return {'pid': process.pid, 'create_time': process.create_time()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config['output'])
    uid = os.getuid()
    own_pid = os.getpid()
    markers = config['markers']
    if not markers or any(not key or not value for key, value in markers.items()):
        raise ValueError('Nonempty unique run environment markers are required')
    status_path = output / 'guard_status.json'
    event_path = output / 'guard_events.jsonl'
    seen = {}
    supervisor = config['supervisor']
    worker = config['worker']
    poll_seconds = config.get('poll_seconds', 2.)
    status = {'state': 'watching', 'guard': identity(psutil.Process()),
              'supervisor': supervisor, 'worker': worker, 'signals': 0}

    def emit(event, **values):
        row = dict(event=event, unix=time.time(), **values)
        with event_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def persist():
        status.update(updated_unix=time.time(), observed=list(seen.values()))
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_path)

    def current(record, require_marker=False):
        try:
            process = psutil.Process(record['pid'])
            if process.pid == own_pid or process.uids().real != uid:
                return None
            if process.create_time() != record['create_time']:
                return None
            if not process.is_running() or process.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                return None
            if require_marker:
                environment = process.environ()
                if not any(environment.get(key) == value for key, value in markers.items()):
                    return None
            return process
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            return None

    def scan():
        matched = []
        for process in psutil.process_iter():
            try:
                record = identity(process)
                process = current(record, require_marker=True)
                if process is None:
                    continue
                command = process.cmdline()
                names = [Path(part).name for part in command]
                logger = any(name in ('wandb_json_tracker.py', 'sidecar_tracking.py',
                                      'wandb-core', 'wandb-service')
                             or name.startswith('wandb-service') for name in names)
                record['role'] = 'logger' if logger else 'run_process'
                if record['pid'] == worker['pid'] and record['create_time'] == worker['create_time']:
                    record['role'] = 'worker'
                record['name'] = process.name()
                key = (record['pid'], record['create_time'])
                seen[key] = record
                matched.append(record)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                continue
        return matched

    def send(record, signum):
        process = current(record, require_marker=True)
        if process is None:
            return
        try:
            # psutil checks process identity again before sending the signal.
            process.send_signal(signum)
            status['signals'] += 1
            emit('signal', target=record, signal=signal.Signals(signum).name)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            pass

    def wait_worker(seconds):
        deadline = time.monotonic() + seconds
        while current(worker) is not None and time.monotonic() < deadline:
            scan()
            persist()
            time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))
        return current(worker) is None

    emit('started', supervisor=supervisor, worker=worker)
    while current(supervisor) is not None:
        scan()
        persist()
        time.sleep(poll_seconds)
    status['state'] = 'worker_grace'
    emit('supervisor_exited')
    scan()
    persist()
    if current(worker) is not None:
        send(worker, signal.SIGINT)
        if not wait_worker(config.get('worker_grace_seconds', 120.)):
            send(worker, signal.SIGTERM)
            if not wait_worker(config.get('worker_term_seconds', 10.)):
                send(worker, signal.SIGKILL)
                wait_worker(config.get('worker_kill_seconds', 5.))
    status['state'] = 'logger_grace'
    emit('worker_exited', alive=current(worker) is not None)
    deadline = time.monotonic() + config.get('logger_grace_seconds', 60.)
    while time.monotonic() < deadline:
        remaining = scan()
        persist()
        if not any(record['role'] == 'logger' for record in remaining):
            break
        time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))
    status['state'] = 'cleanup'
    for signum, grace in ((signal.SIGINT, config.get('cleanup_int_seconds', 10.)),
                          (signal.SIGTERM, config.get('cleanup_term_seconds', 5.)),
                          (signal.SIGKILL, config.get('cleanup_kill_seconds', 5.))):
        deadline = time.monotonic() + grace
        signaled = set()
        quiet_since = None
        while time.monotonic() < deadline:
            remaining = scan()
            persist()
            for record in remaining:
                key = (record['pid'], record['create_time'])
                if key not in signaled:
                    send(record, signum)
                    signaled.add(key)
            if remaining:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= min(2., grace / 2.):
                break
            time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))
        if not scan():
            break
    status['remaining'] = scan()
    status['state'] = 'cleaned' if not status['remaining'] else 'cleanup_failed'
    persist()
    emit('finished', state=status['state'], remaining=status['remaining'])
    if status['remaining']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
