"""Build user-authorized estimated tool inertials and a nominal UR10e robot bundle.

Run in the separate requirements-cad environment. Inputs and outputs stay outside git.
This prepares robot/FDCC assets, not a replacement cutting-material scene.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.tool_inertia import mesh_mass, combine, transform, validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def usd_mesh(path):
    stage = Usd.Stage.Open(str(path))
    result = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        points = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3] * UsdGeom.GetStageMetersPerUnit(stage)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        ids = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        faces, offset = [], 0
        for count in counts:
            for k in range(1, count - 1):
                faces.append([ids[offset], ids[offset + k], ids[offset + k + 1]])
            offset += count
        result.append(trimesh.Trimesh(points, faces, process=True))
    if len(result) != 1:
        raise ValueError("Expected the supplied single closed IKEA knife mesh")
    mesh_mass(result[0], 1)
    return result[0]


def register_knife(knife, assembly):
    # User CAD and supplied USD have the same 259.691 mm length; do not rescale.
    rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]])
    points = knife.vertices @ rotation.T
    translation = np.array([.075 - (points[:, 0].max() + points[:, 0].min()) / 2,
                           assembly.bounds[0, 1] - points[:, 1].min(),
                           assembly.bounds[1, 2] - points[:, 2].max()])
    # Only exposed blade is used: holder occludes the handle in the union STL.
    visible = knife.vertices[knife.vertices[:, 0] < -.04]
    if len(visible) < 10:
        raise ValueError("Unexpected supplied knife coordinate frame")
    def residual(x):
        moved = visible @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:]
        nearest, _, _ = trimesh.proximity.closest_point(assembly, moved)
        return (moved - nearest).ravel()
    fit = least_squares(residual, np.r_[Rotation.from_matrix(rotation).as_rotvec(), translation],
                        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=80, diff_step=1e-5)
    error = np.linalg.norm(residual(fit.x).reshape(-1, 3), axis=1)
    if not fit.success or error.max() > 1e-6:
        raise ValueError("Knife and assembly do not match within 1 micrometre")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(fit.x[:3]).as_matrix()
    matrix[:3, 3] = fit.x[3:]
    registered = knife.copy()
    registered.apply_transform(matrix)
    return registered, dict(assembly_from_usd=matrix.tolist(), max_surface_error_m=float(error.max()))


def inertial(link, part):
    node = ET.SubElement(link, "inertial")
    ET.SubElement(node, "origin", xyz=" ".join(map(str, part["com_m"])))
    ET.SubElement(node, "mass", value=str(part["mass_kg"]))
    tensor = np.asarray(part["inertia_kg_m2"])
    ET.SubElement(node, "inertia", **{"i"+a+b: str(tensor[i, j])
                  for i, a in enumerate("xyz") for j, b in enumerate("xyz") if i <= j})


def fixed(robot, name, parent, xyz=(0, 0, 0), rpy=(0, 0, 0), part=None, mesh=None, cylinder=None, color=".55 .57 .59 1", collision_mesh=None):
    link = ET.SubElement(robot, "link", name=name)
    if part:
        inertial(link, part)
    if mesh or cylinder:
        for kind in ("visual", "collision"):
            element = ET.SubElement(link, kind)
            geometry = ET.SubElement(element, "geometry")
            if mesh:
                ET.SubElement(geometry, "mesh", filename=str(collision_mesh if kind == "collision" and collision_mesh else mesh))
            else:
                ET.SubElement(geometry, "cylinder", radius=str(cylinder[0]), length=str(cylinder[1]))
            if kind == "visual":
                material = ET.SubElement(element, "material", name=name+"_material")
                ET.SubElement(material, "color", rgba=color)
    joint = ET.SubElement(robot, "joint", name=parent+"__"+name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=name)
    ET.SubElement(joint, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))


def build(args):
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    knife = usd_mesh(args.assets / "IKEA_365_Knife.usdz")
    assembly = trimesh.load(args.assets / "knife_adapter_original.stl", force="mesh")
    flange = trimesh.load(args.assets / "ur_adapter_original.stl", force="mesh")
    for mesh in (assembly, flange):
        mesh.apply_scale(.001)
        mesh_mass(mesh, 1)
    knife, registration = register_knife(knife, assembly)
    # Check actual containment/union volume before interpreting subtraction as PLA.
    outside = trimesh.boolean.difference([knife, assembly], engine="manifold")
    outside_volume = abs(outside.volume) if outside is not None else 0.
    if outside_volume > 1e-9:
        raise ValueError(f"Registered knife not contained: outside {outside_volume} m3; {registration}")
    holder = trimesh.boolean.difference([assembly, knife], engine="manifold")
    # Discard only boolean surface-roundoff slivers, bounded by 1e-9 m3 total.
    pieces = holder.split(only_watertight=False)
    kept = [p for p in pieces if abs(p.volume) > 1e-10]
    rejected = sum(abs(p.volume) for p in pieces if abs(p.volume) <= 1e-10)
    if rejected > 1e-9 or not kept:
        raise ValueError("Boolean subtraction produced significant ambiguous fragments")
    holder = trimesh.util.concatenate(kept)
    for mesh in (knife, holder):
        mesh.apply_translation(-np.asarray(cfg["assembly_mount_CAD_m"]))
    flange.apply_translation(-np.asarray(cfg["flange_mount_CAD_m"]))
    meshes = dict(knife=knife, holder=holder, flange=flange)
    parts = {name: mesh_mass(mesh, cfg["knife_mass_kg"] if name == "knife"
                           else mesh.volume * cfg["pla_density_kg_m3"]) for name, mesh in meshes.items()}
    for name, mesh in meshes.items():
        mesh.export(args.output / (name+".stl"))
    # Partition the original union's visible triangles, avoiding coincident
    # knife/holder surfaces from roundoff-sized boolean slivers. Geometry is not moved.
    display=assembly.copy()
    display.apply_translation(-np.asarray(cfg["assembly_mount_CAD_m"]))
    _,distance,_=trimesh.proximity.closest_point(knife,display.triangles_center)
    steel=distance < 1e-6
    if not steel.any() or steel.all():
        raise ValueError("Unable to separate original visible steel and PLA faces")
    for name,mask in (("knife",steel),("holder",~steel)):
        visual=trimesh.Trimesh(display.vertices.copy(),display.faces[mask],process=False)
        visual.export(args.output/(name+"_visual.stl"))

    distal = combine([parts["knife"], parts["holder"]])
    sensor_z = cfg["adapter_height_m"] + cfg["sensor_height_m"]
    r, h, m = cfg["sensor_radius_m"], cfg["sensor_height_m"], cfg["sensor_mass_kg"]
    sensor = validate(m, [0, 0, 0], np.diag([m*(3*r*r+h*h)/12]*2+[m*r*r/2]))
    total = combine([parts["flange"],
                     transform(sensor, np.eye(3), [0, 0, sensor_z-h/2]),
                     transform(distal, np.eye(3), [0, 0, sensor_z])])
    clock = cfg["sensor_clocking_rad"]
    rotation = Rotation.from_euler("z", clock).as_matrix()
    sensor_payload = transform(distal, rotation.T, [0, 0, 0])
    tree = ET.parse(args.urdf)
    robot = tree.getroot()
    fixed(robot, "user_ur_adapter", "tool0", part=parts["flange"], mesh=args.output/"flange.stl", color=".35 .36 .38 1")
    fixed(robot, "ati_mini45_body", "user_ur_adapter", [0, 0, sensor_z-h/2],
          part=sensor, cylinder=(r, h), color=".30 .32 .34 1")
    # Sensor origin must be an ancestor of the distal load, not a visual-only sibling.
    fixed(robot, "ati_sensing_frame", "ati_mini45_body", [0, 0, h/2], [0, 0, clock])
    fixed(robot, "knife_mount", "ati_sensing_frame", rpy=[0, 0, -clock])
    fixed(robot, "user_knife_holder", "knife_mount", part=parts["holder"], mesh=args.output/"holder_visual.stl", collision_mesh=args.output/"holder.stl", color=".35 .36 .38 1")
    fixed(robot, "ikea_knife", "knife_mount", part=parts["knife"], mesh=args.output/"knife_visual.stl", collision_mesh=args.output/"knife.stl")
    # Control origin is the center-plane projection of the lowest CAD vertex in
    # the nominal blade-down pose, not necessarily the edge midpoint. The
    # separate knife_tip frame marks the foremost CAD vertex.
    tip = knife.vertices[np.argmax(knife.vertices[:, 1])]
    lowest = knife.vertices[np.argmax(knife.vertices[:, 2])]
    control = lowest.copy()
    control[0] = 0.
    fixed(robot, "knife_control", "knife_mount", xyz=control)
    fixed(robot, "knife_tip", "knife_mount", xyz=tip)
    if hasattr(ET, "indent"): ET.indent(tree)
    tree.write(args.output/"robot.urdf", encoding="utf-8", xml_declaration=True)
    manifest = dict(revision=cfg["revision"], status="robot_assets_estimated_not_cutting_validated",
                    configuration=cfg, parts=parts, sensor=sensor, robot_payload_tool0=total,
                    sensor_distal_payload=sensor_payload, registration=registration,
                    outside_knife_volume_m3=outside_volume, discarded_roundoff_volume_m3=rejected,
                    volumes_m3={name: mesh.volume for name, mesh in meshes.items()},
                    robot_urdf=str(args.output/"robot.urdf"),
                    robot_base=cfg["robot_base"], robot_tip=cfg["robot_tip"],
                    sensor_frame=cfg["sensor_frame"], q_initial_rad=cfg["q_initial_rad"],
                    joint_names=[j.get("name") for j in robot.findall("joint") if j.get("type")=="revolute"],
                    source_hashes={p.name: sha(p) for p in [args.urdf, args.config,
                        args.assets/"IKEA_365_Knife.usdz",args.assets/"knife_adapter_original.stl",args.assets/"ur_adapter_original.stl"]},
                    limitations=["Solid PLA assumption, not slicer infill; knife uniform-volume mass distribution",
                                 "ATI cylinder inertia estimate, axis clocking unverified",
                                 "MuJoCo mesh collisions use convex hulls; not a cutting SDF",
                                 "New physical blade needs dedicated DiSECt/BM material-contact integration"])
    (args.output/"platform.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(masses_kg={k:p["mass_kg"] for k,p in parts.items()},
                          total_payload_kg=total["mass_kg"], distal_payload_kg=distal["mass_kg"],
                          registration=registration),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True, help="Expanded official nominal UR10e URDF")
    parser.add_argument("--config", type=Path, default=Path("configs/robots/ur10e_ati_mini45.json"))
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())
