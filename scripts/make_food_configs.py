#!/usr/bin/env python3
"""DiSECt scene configs for scanned food meshes with their official DiSECt materials.

Scene layout (knife, static bottom layer, ground, motion) is the fixed-bottom prism
benchmark's; the object is a prepared tetrahedral mesh (scripts/prepare_food_mesh.py)
and its material keys come from the matching official DiSECt config.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_material_sweep_configs import MATERIAL_KEYS, MATERIALS  # noqa: E402

mesh_dir, out = Path(sys.argv[1]), ROOT / 'configs/food_scenes'
base = json.loads((ROOT / 'configs/public_prism_fixed_bottom/disect_scene.json').read_text())
out.mkdir(parents=True, exist_ok=True)
for mesh in sorted(mesh_dir.glob('*.msh')):
    food = mesh.stem.split('_')[0]
    material = json.loads(MATERIALS[food].read_text())
    config = json.loads(json.dumps(base))
    config['generator'] = 'meshio'
    config['generators']['meshio'] = {'filename': str(mesh.resolve())}
    for key in MATERIAL_KEYS:
        if key in material:
            config[key] = material[key]
    (out / f'{mesh.stem}.json').write_text(json.dumps(config, indent=2) + '\n')
    print(mesh.stem, food)
