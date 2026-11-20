"""Compile the estimated UR10e bundle, preserving the sensor's distal inertial subtree."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backends/mujoco/scripts"))
from prepare_robot import adjacent_collision_pairs


def compile_bundle(root, controller_source, ros_package):
    root = Path(root)
    info = json.loads((root / "platform.json").read_text())
    from cutting_world_model.compliance_frame import configure
    xml, compliance = configure((root / "robot.urdf").read_text(), info, info["q_initial_rad"])
    (root / "robot.urdf").write_text(xml)
    info["fdcc_compliance"] = compliance
    tree = ET.parse(root / "robot.urdf")
    robot = tree.getroot()
    ext = ET.SubElement(robot, "mujoco")
    ET.SubElement(ext, "compiler", fusestatic="false", discardvisual="true",
                  strippath="false", inertiafromgeom="false")
    # Preserve masses on their own links: aggregating them into wrist_3 would
    # remove the distal load from the ATI frame's subtree wrench.
    for link in robot.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if filename.startswith("package://ur_description/"):
            mesh.set("filename", str(Path(ros_package) / filename.split("package://ur_description/")[1]))
        if not Path(mesh.get("filename")).exists():
            raise FileNotFoundError(mesh.get("filename"))
    tree.write(root / "robot_physics.urdf")
    spec = mujoco.MjSpec.from_file(str(root / "robot_physics.urdf"))
    spec.option.timestep = .001
    spec.option.gravity[:] = [0, 0, -9.8]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for first, second in adjacent_collision_pairs(robot, info["robot_base"]):
        spec.add_exclude(bodyname1=first, bodyname2=second)
    joints = {j.get("name"): j for j in robot.findall("joint")}
    for name in info["joint_names"]:
        effort = float(joints[name].find("limit").get("effort"))
        spec.add_actuator(name=name+"_torque", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gainprm=[1]+[0]*9, gear=[1,0,0,0,0,0],
                          ctrllimited=True, ctrlrange=[-effort,effort])
    model = spec.compile()
    mujoco.mj_saveModel(model, str(root / "robot.mjb"))
    params = json.loads(Path(controller_source).read_text())
    params.update(end_effector_link=info["robot_tip"], robot_base_link=info["robot_base"],
                  joints=info["joint_names"], ft_sensor_ref_link=info["sensor_frame"],
                  compliance_ref_link=compliance["frame"], gravity=dict(x=0.,y=0.,z=-9.8))
    payload = info["sensor_distal_payload"]
    params["tool"] = dict(mass=payload["mass_kg"],
                          **{"com_"+a: value for a,value in zip("xyz",payload["com_m"])})
    (root / "controller.json").write_text(json.dumps(params, indent=2)+"\n")
    info["runtime"] = dict(nq=model.nq, nv=model.nv, nu=model.nu,
                          fixed_inertias="preserved for ATI subtree sensing",
                          sensor_sign=-1, sensor_convention="distal load on sensor; negate MuJoCo parent-on-child wrench",
                          actuator_model="source position-command emulation; not UR10e servo identification",
                          controller_gains="source FDCC gains, not hardware-tuned",
                          sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest()
                                  for f in ("robot.mjb","robot.urdf","robot_physics.urdf","controller.json",
                                            "knife.stl","holder.stl","flange.stl","knife_visual.stl","holder_visual.stl")})
    (root / "platform.json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps(info["runtime"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", required=True)
    p.add_argument("--controller-source", required=True)
    p.add_argument("--ur-description", default="/opt/ros/noetic/share/ur_description")
    a=p.parse_args()
    compile_bundle(a.platform, a.controller_source, a.ur_description)
