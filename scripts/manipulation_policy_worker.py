"""Frozen upstream manipulation actor for the shared cutting benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cutting_world_model.benchmark import check_contract
from cutting_world_model.manipulation import SOURCE_REVISION
from cutting_world_model.sac_gpu_bridge import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--protocol-fd', type=int, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.source_root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.source_root), 'status', '--porcelain'], text=True)
    if revision != SOURCE_REVISION or dirty:
        raise ValueError('Use the clean, pinned manipulation source checkout')
    sys.path.insert(0, str(args.source_root / 'src'))
    from clab.algorithms.supermimic.networks import Policy, ACTOR_INPUT_NORMALIZATION
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location='cuda', weights_only=True)
    if (saved.get('algorithm') != 'supermimic' or saved.get('task') != 'ur5e_apple'
            or saved.get('actor_input_normalization') != ACTOR_INPUT_NORMALIZATION):
        raise ValueError('Expected a manipulation SuperMimic cutting checkpoint')
    cfg, mdp = saved['config'], saved['mdp_contract']
    torch.backends.cuda.matmul.allow_tf32 = cfg['tf32']
    policy = Policy(mdp['obs_dim'], mdp['action_dim'], cfg['policy_hidden'], amp=cfg['amp']).cuda().eval()
    policy.load_state_dict(saved['agent']['policy'])
    policy.validate_normalization()
    policy.requires_grad_(False)
    weights = {key: value.clone() for key, value in policy.state_dict().items()}
    contract = json.loads((args.checkpoint.parent / 'params/benchmark.json').read_text())

    def handle(message):
        if message['op'] == 'init':
            check_contract(contract, message['contract'])
            return {'policy_updates': saved['agent']['policy_updates']}
        if message['op'] == 'act':
            observation = torch.tensor(message['observation'], device='cuda', dtype=torch.float32)
            if observation.ndim != 2 or observation.shape[-1] != mdp['obs_dim']:
                raise ValueError('Unexpected actor observation shape')
            if not torch.isfinite(observation).all():
                raise ValueError('Non-finite actor observation')
            with torch.no_grad():
                return {'action': policy(observation).cpu().tolist()}
        if message['op'] == 'unchanged':
            return {'unchanged': all(torch.equal(value, weights[key])
                                     for key, value in policy.state_dict().items())}
        if message['op'] == 'close':
            return {}
        raise ValueError('Unknown actor operation')

    serve(args.protocol_fd, handle)


if __name__ == '__main__':
    main()
