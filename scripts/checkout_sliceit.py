#!/usr/bin/env python3
"""Restore the pinned SliceIt sources and local integration patches."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / 'integrations/sliceit/patches'


def git(directory, *arguments):
    return subprocess.check_output(
        ['git', '-C', str(directory), *arguments], text=True).strip()


def verify(directory, entry):
    if git(directory, 'write-tree') != entry['source_tree']:
        raise RuntimeError('Source tree differs from the lock: ' + str(directory))
    # Parent gitlinks refer to the original local commits recorded in the lock.
    changed = git(directory, 'diff', '--raw').splitlines()
    if any(not line.startswith(':160000 160000 ') for line in changed):
        raise RuntimeError('Unstaged source changes: ' + str(directory))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'external/SliceIt')
    parser.add_argument('--local-source', type=Path,
                        help='Read Git objects from an existing SliceIt checkout')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    entries = json.loads((PATCHES / 'sources.json').read_text())
    output = args.output.resolve()
    for entry in entries:
        target = output / entry['path']
        if not args.check:
            if target.exists() and any(target.iterdir()):
                raise FileExistsError('Choose an empty destination: ' + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.local_source:
                source = (args.local_source / entry['path']).resolve()
                subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout',
                                str(source), str(target)], check=True)
                git(target, 'remote', 'set-url', 'origin', entry['url'])
            else:
                subprocess.run(['git', 'init', str(target)], check=True)
                git(target, 'remote', 'add', 'origin', entry['url'])
                git(target, 'fetch', '--depth=1', 'origin', entry['base_commit'])
            git(target, 'checkout', '--detach', entry['base_commit'])
            if entry.get('patch'):
                git(target, 'apply', '--index', str(PATCHES / entry['patch']))
        verify(target, entry)
        print(entry['path'] + ': matches pinned source tree', flush=True)


if __name__ == '__main__':
    main()
