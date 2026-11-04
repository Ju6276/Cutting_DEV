"""Compare headless DFlex stepping against an exported source transition."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_gpu_backend.disect import DiSECt
from provenance import git_revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.scene / 'manifest.json').read_text())
    sim = DiSECt(args.scene / 'scene.pt')
    start = time.perf_counter()
    sim.advance(metadata['reference_steps'] * sim.dt)
    torch.cuda.synchronize()
    reference = torch.load(args.scene / 'reference.pt', weights_only=False)
    differences = {}
    for key, target in reference.items():
        actual = getattr(sim.state, key)
        error = (actual - target).abs()
        differences[key] = {'max_abs': float(error.max()) if error.numel() else 0.,
                            'rms': float(error.square().mean().sqrt()) if error.numel() else 0.}
    report = {'git_revision': git_revision(), 'command': sys.argv,
              'source': metadata, 'seconds': time.perf_counter()-start,
              'differences': differences,
              'released_springs': int((sim.state.cut_spring_ke == 0).sum())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
