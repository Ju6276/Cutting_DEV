#!/usr/bin/env python3
"""DiSECt scene configs for scanned food meshes with their official DiSECt materials.

Scene layout (knife, static bottom layer, ground, motion) is the fixed-bottom prism
benchmark's; the object is a prepared tetrahedral mesh (scripts/prepare_food_mesh.py,
file stem ``<food>_<id>``) and its material keys come from the matching official DiSECt
config (examples/config: sphere_apple, osx/cucumber, osx/potato, osx/tomato).

The object is held by its bottom layer: vertices below ``HOLD_HEIGHT`` are static,
except near the cut plane (the benchmark's exclusion box), as a real object is placed
or held on the board; a few static vertices let round scans slide away from the knife.

Usage: make_food_configs.py <tet mesh directory> [<SliceIt root>]
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MATERIAL_KEYS = ('young', 'poisson', 'mu', 'lambda', 'density',
                 'cut_spring_ke', 'cut_spring_kd', 'cut_spring_softness',
                 'surface_cut_spring_ke', 'surface_cut_spring_kd', 'surface_cut_spring_softness',
                 'sdf_ke', 'sdf_kd', 'sdf_kf', 'sdf_mu',
                 'surface_sdf_ke', 'surface_sdf_kd', 'surface_sdf_kf', 'surface_sdf_mu')
HOLD_HEIGHT = .006  # m above the object's lowest point (the object rests at y = 0.0008)


def main():
    mesh_dir = Path(sys.argv[1])
    official = Path(sys.argv[2] if len(sys.argv) > 2 else '/dataHDD/dong/Projects/SliceIt') / 'disect/examples/config'
    materials = {'apple': official / 'ansys_sphere_apple.json', 'cucumber': official / 'osx/ansys_cucumber.json',
                 'potato': official / 'osx/ansys_potato.json', 'tomato': official / 'osx/ansys_tomato.json'}
    base = json.loads((ROOT / 'configs/public_prism_fixed_bottom/disect_scene.json').read_text())
    out = ROOT / 'configs/food_scenes'
    out.mkdir(parents=True, exist_ok=True)
    for mesh in sorted(mesh_dir.glob('*.msh')):
        food = mesh.stem.split('_')[0]
        material = json.loads(materials[food].read_text())
        config = json.loads(json.dumps(base))
        config['generator'] = 'meshio'
        config['generators']['meshio'] = {'filename': str(mesh.resolve())}
        for key in MATERIAL_KEYS:
            if key in material:
                config[key] = material[key]
        config['static_vertices']['include'][0]['max'][1] = config['geometry']['position'][1] + HOLD_HEIGHT
        (out / f'{mesh.stem}.json').write_text(json.dumps(config, indent=2) + '\n')
        print(mesh.stem, food)


if __name__ == '__main__':
    main()
