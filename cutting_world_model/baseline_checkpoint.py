"""Integrity-checked SAC checkpoints; refuse mixed generations after interrupted saves."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save_sac_checkpoint(model, output, metadata):
    output = Path(output)
    # Finish each payload before replacing it. The manifest is replaced LAST.
    # This detects an interrupted multi-file replacement, not automatic recovery.
    with tempfile.TemporaryDirectory(prefix='.sac_save_', dir=output) as staging:
        staging = Path(staging)
        model.save(staging / 'model.zip')
        model.save_replay_buffer(staging / 'replay.pkl')
        metadata = dict(metadata, checkpoint_payload_sha256={
            name: digest(staging / name) for name in ('model.zip', 'replay.pkl')})
        (staging / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        for name in ('model.zip', 'replay.pkl', 'metadata.json'):
            os.replace(staging / name, output / name)


def read_sac_metadata(output, require_replay=False):
    output = Path(output)
    metadata = json.loads((output / 'metadata.json').read_text())
    hashes = metadata.get('checkpoint_payload_sha256')
    if hashes is None:
        warnings.warn('Legacy SAC checkpoint has no payload integrity record', RuntimeWarning)
        return metadata
    names = ('model.zip', 'replay.pkl') if require_replay else ('model.zip',)
    for name in names:
        if name not in hashes or digest(output / name) != hashes[name]:
            raise ValueError('SAC checkpoint payload mismatch: '+name+
                             '; refuse mixed or interrupted checkpoint')
    return metadata
