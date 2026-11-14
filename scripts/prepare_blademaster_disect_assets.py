"""Prepare BladeMaster-inspired geometries and elastic parameters for DiSECt.

Runs in the food-mesh CPU environment. Geometry comes from the pinned local
BladeMaster reproduction. DiSECt cutting springs and contact remain the explicit
template values; no fracture-energy-to-spring conversion is performed.
"""
import argparse
import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "4b2d983daeaf9ba1f6a66bc0bb01712b1eb602c5"
PRESETS = {
    "banana": dict(shape="banana", size=[.12, .06, .03], young=400000., poisson=.45,
                   density=1100., x_offset=.00037, source="presets.banana_config + geometry.levelset",
                   omitted=dict(yield_stress_pa=30000., cutting_energy_j_m2=700., sharpness=.9)),
    "cake": dict(shape="cylinder", size=[.16, .16, .056], young=10000., poisson=.4,
                 density=1100., x_offset=.04037, source="scenarios._scene_config(fig4)",
                 omitted=dict(cutting_energy_j_m2=10.)),
    "tofu": dict(shape="block", size=[.08, .08, .07], young=6000., poisson=.35,
                 density=1100., x_offset=.00037, source="scenarios._scene_config(fig7-shallow)",
                 omitted=dict(cutting_energy_j_m2=10.)),
    "luncheon": dict(shape="block", size=[.10, .05, .042], young=20000., poisson=.4,
                     density=1200., x_offset=.00037, source="scenarios._scene_config(fig8)",
                     omitted=dict(yield_stress_pa=2000., cutting_energy_j_m2=10.)),
}


def source_geometry(source):
    spec = importlib.util.spec_from_file_location("bm_source_geometry", source / "blademaster_warp/geometry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def surface(preset, geometry):
    import trimesh
    size = np.asarray(preset["size"])
    if preset["shape"] == "block":
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation([0, 0, size[2]/2])
    elif preset["shape"] == "cylinder":
        mesh = trimesh.creation.cylinder(radius=size[0]/2, height=size[2], sections=64)
        mesh.apply_translation([0, 0, size[2]/2])
    else:
        from skimage.measure import marching_cubes
        spacing = .001
        lower = np.array([-size[0]/2, -size[1]/2, 0.]) - 2.137*spacing
        upper = np.array([size[0]/2, size[1]/2, size[2]]) + 2*spacing
        axes = [np.arange(lo, hi+spacing/2, spacing) for lo, hi in zip(lower, upper)]
        xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        values = geometry.levelset(xyz, size, "banana")
        vertices, faces = marching_cubes(values, level=0., spacing=(spacing,)*3, allow_degenerate=False)[:2]
        mesh = trimesh.Trimesh(vertices + lower, faces, process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("Source surface must be closed and consistently oriented")
    # Proper rotation: BladeMaster z-up to DiSECt y-up; preserve all dimensions.
    points = np.asarray(mesh.vertices)[:, [0, 2, 1]].copy()
    points[:, 2] *= -1
    points[:, 1] -= points[:, 1].min()
    points[:, 0] += preset["x_offset"]
    return points, np.asarray(mesh.faces, dtype=np.int32)


def volumes(points, tetra):
    xyz = points[tetra]
    return np.linalg.det(xyz[:, 1:] - xyz[:, :1]) / 6


def disect_config(template, preset, path, height):
    cfg = copy.deepcopy(template)
    cfg["generators"]["meshio"]["filename"] = str(path)
    cfg.update(young=preset["young"], poisson=preset["poisson"], density=preset["density"])
    young, nu = cfg["young"], cfg["poisson"]
    cfg["mu"] = young/(2*(1+nu))
    cfg["lambda"] = young*nu/((1+nu)*(1-2*nu))
    cfg["static_vertices"] = dict(include=[], exclude=[])
    # Knife half-height + tip height is 29.54 mm for retained GAZEBO blade.
    cfg["initial_y"] = height + cfg["geometry"]["position"][1] + .01 + .02954
    cfg["knife_motion"]["position"][1] = cfg["initial_y"]
    cfg["velocity_y"] = -.05
    cfg["knife_motion"]["velocity"] = [0, -.05, 0]
    cfg["sim_duration"] = (height + cfg["geometry"]["position"][1] + .02)/.05
    return cfg


def pose_config(name):
    cfg = json.loads((ROOT/"configs/free_cut/pose_cucumber.json").read_text())
    cfg["description"] = "BladeMaster-inspired "+name+" in DiSECt; development only"
    cfg["status"] = "candidate_not_training_validated; source plasticity/fracture not mapped; ground contact not calibrated."
    group = cfg["groups"][0]
    group.update(name="bm_"+name, scene="runs/sweep/bm_"+name+"_20261114/scene.pt",
                 worlds=2, clearance=[.03,.03],
                 material={key:[1.,1.] for key in group["material"]},
                 target_speed=[.03,.03],
                 settle_seconds=3.0 if name in {"cake", "tofu"} else .75, sim_dt=1e-5)
    return cfg


def main():
    import meshio
    import wildmeshing
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--configs", type=Path, default=ROOT/"configs/blademaster_disect")
    p.add_argument("--objects", nargs="+", choices=tuple(PRESETS), default=list(PRESETS))
    a = p.parse_args()
    revision = subprocess.check_output(["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError("Review a new BladeMaster source revision before migrating its presets")
    dirty = subprocess.check_output(["git", "-C", str(a.source), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError("Use the clean pinned BladeMaster source for geometry migration")
    a.assets.mkdir(parents=True, exist_ok=True)
    a.configs.mkdir(parents=True, exist_ok=True)
    geometry = source_geometry(a.source)
    template_path = ROOT/"configs/free_scenes/potato_003_half.json"
    template = json.loads(template_path.read_text())
    catalog = {}
    for name in a.objects:
        preset = PRESETS[name]
        path = a.assets / (name+".msh")
        if path.exists():
            raise FileExistsError(path)
        points, faces = surface(preset, geometry)
        old = Path.cwd()
        try:
            os.chdir(a.assets)
            tet = wildmeshing.Tetrahedralizer(stop_quality=8, max_its=40, edge_length_r=.06, epsilon=.008)
            tet.set_mesh(points, faces)
            tet.tetrahedralize()
        finally:
            os.chdir(old)
        vertices, cells = tet.get_tet_mesh()[:2]
        before = volumes(vertices, cells)
        if preset["shape"] != "banana":
            envelope = .008*np.linalg.norm(np.ptp(points, axis=0))
            support = np.abs(vertices[:, 1]) <= envelope
            vertices[support, 1] = 0.
        vertices[:, 1] -= vertices[:, 1].min()
        after = volumes(vertices, cells)
        if not np.isfinite(vertices).all() or np.any(before*after <= 0):
            raise ValueError("Invalid tetrahedralization or support projection")
        meshio.write(path, meshio.Mesh(vertices, [("tetra", cells)]), file_format="gmsh22")
        cfg = disect_config(template, preset, path, float(np.ptp(vertices[:, 1])))
        (a.configs/(name+".json")).write_text(json.dumps(cfg, indent=2)+"\n")
        (a.configs/("pose_"+name+".json")).write_text(json.dumps(pose_config(name), indent=2)+"\n")
        edges = np.concatenate([np.linalg.norm(vertices[cells[:, i]]-vertices[cells[:, j]], axis=1)
                                for i, j in itertools.combinations(range(4), 2)])
        record = dict(source_revision=revision, source_preset=preset, mesh=str(path),
                      config="configs/blademaster_disect/"+name+".json", vertices=len(vertices), tetrahedra=len(cells),
                      extents_m=np.ptp(vertices, axis=0).tolist(), volume_m3=float(np.abs(after).sum()),
                      min_edge_m=float(edges.min()), fixed_particles=0, cut_plane="DiSECt x=0",
                      centre_offset_x_m=preset["x_offset"],
                      inherited_disect_template=str(template_path.relative_to(ROOT)),
                      mapped_fields=["young", "poisson", "density", "mu", "lambda"],
                      omitted_source_parameters=preset["omitted"],
                      surface_method="analytic source boundary; fTetWild tetrahedralization",
                      support_projection_volume_change_fraction=float((np.abs(after).sum()-np.abs(before).sum())/np.abs(before).sum()))
        path.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
        catalog[name] = record
        print(name, len(vertices), "vertices", len(cells), "tetrahedra", flush=True)
    (a.configs/"catalog.json").write_text(json.dumps(catalog, indent=2)+"\n")


if __name__ == "__main__":
    main()
