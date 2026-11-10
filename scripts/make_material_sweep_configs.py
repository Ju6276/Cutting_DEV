#!/usr/bin/env python3
"""Write DiSECt scene configs: each prepared geometry with each official DiSECt material.

Geometry (mesh, placement, knife, ground, motion) comes from the geometry's own config;
the material keys below come from the material's official config. Official sources:
DiSECt examples/config (prism, cylinder_jello, sphere_apple, osx/cucumber, potato, tomato).
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SLICEIT = Path(sys.argv[1] if __name__ == '__main__' and len(sys.argv) > 1
               else '/dataHDD/dong/Projects/SliceIt')
OUT = ROOT / 'configs/material_sweep'
OFFICIAL = SLICEIT / 'disect/examples/config'
GEOMETRIES = {
    'prism': ROOT / 'configs/public_prism_fixed_bottom/disect_scene.json',
    'cylinder': OFFICIAL / 'ansys_cylinder_jello.json',
    'sphere': Path('/dataHDD/dong/tmp/ansys_sphere_apple_free.json'),
}
MATERIALS = {
    'prism': SLICEIT / 'configs/public_prism.json',
    'jello': OFFICIAL / 'ansys_cylinder_jello.json',
    'apple': OFFICIAL / 'ansys_sphere_apple.json',
    'cucumber': OFFICIAL / 'osx/ansys_cucumber.json',
    'potato': OFFICIAL / 'osx/ansys_potato.json',
    'tomato': OFFICIAL / 'osx/ansys_tomato.json',
}
MATERIAL_KEYS = ('young', 'poisson', 'mu', 'lambda', 'density',
                 'cut_spring_ke', 'cut_spring_kd', 'cut_spring_softness',
                 'surface_cut_spring_ke', 'surface_cut_spring_kd', 'surface_cut_spring_softness',
                 'sdf_ke', 'sdf_kd', 'sdf_kf', 'sdf_mu',
                 'surface_sdf_ke', 'surface_sdf_kd', 'surface_sdf_kf', 'surface_sdf_mu')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    for g, gpath in GEOMETRIES.items():
        base = json.loads(gpath.read_text())
        for m, mpath in MATERIALS.items():
            material = json.loads(mpath.read_text())
            config = dict(base)
            for key in MATERIAL_KEYS:
                if key in material:
                    config[key] = material[key]
                elif key in config:
                    del config[key]
            name = f'{g}_{m}'
            (OUT / f'{name}.json').write_text(json.dumps(config, indent=2) + '\n')
            index[name] = {'geometry': g, 'geometry_config': str(gpath), 'material': m,
                           'material_config': str(mpath),
                           'material_values': {k: material[k] for k in MATERIAL_KEYS if k in material}}
    (OUT / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(len(index), 'configs')


if __name__ == '__main__':
    main()
