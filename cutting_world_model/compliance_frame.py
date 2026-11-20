"""Explicit FDCC stiffness axes for a mounted robot tool (no visual geometry)."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .robot_platform import frame_poses

REVISION = "cutting_compliance_axes_v1"
# Source SliceIt/UR5e convention at the nominal cut: x down, y along base y,
# z along base x. Translation x and rotation y are the policy's stiffness channels.
BASE_FROM_COMPLIANCE = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])

def configure(xml, metadata, q, parent="knife_mount", child="knife_compliance"):
    tree = ET.fromstring(xml)
    def origin(j):
        o=j.find("origin");t=np.eye(4)
        if o is not None:
            t[:3,3]=np.fromstring(o.get("xyz","0 0 0"),sep=" ")
            t[:3,:3]=Rotation.from_euler("xyz",np.fromstring(o.get("rpy","0 0 0"),sep=" ")).as_matrix()
        return t
    def set_origin(j,t):
        o=j.find("origin")
        if o is None:o=ET.SubElement(j,"origin")
        o.set("xyz"," ".join(map(str,t[:3,3])))
        o.set("rpy"," ".join(map(str,Rotation.from_matrix(t[:3,:3]).as_euler("xyz"))))
    old=[j for j in tree.findall("joint") if j.find("child").get("link")==child]
    if old:
        for j in tree.findall("joint"):
            if j.find("parent").get("link")==child:
                set_origin(j,origin(old[0])@origin(j))
                j.find("parent").set("link",parent)
    for joint in list(tree.findall("joint")):
        if joint.find("child").get("link") == child:
            tree.remove(joint)
    for link in list(tree.findall("link")):
        if link.get("name") == child:
            tree.remove(link)
    clean = ET.tostring(tree, encoding="unicode")
    poses = frame_poses(dict(metadata, robot_xml=clean), q)
    parent_pose = poses[parent]
    local_rotation = parent_pose[:3,:3].T @ BASE_FROM_COMPLIANCE
    ET.SubElement(tree, "link", name=child)
    joint = ET.SubElement(tree, "joint", name=parent+"__"+child, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz="0 0 0",
                  rpy=" ".join(map(str, Rotation.from_matrix(local_rotation).as_euler("xyz"))))
    # Put these axes on the task-TCP ancestry; preserve the physical TCP pose.
    for j in tree.findall("joint"):
        if j.find("child").get("link")==metadata["robot_tip"] and metadata["robot_tip"]!=parent:
            if j.find("parent").get("link")!=parent:raise ValueError("Unexpected tool TCP ancestry")
            t=np.eye(4);t[:3,:3]=local_rotation
            set_origin(j,np.linalg.inv(t)@origin(j))
            j.find("parent").set("link",child)
    return ET.tostring(tree, encoding="unicode"), dict(
        revision=REVISION, frame=child, parent=parent,
        nominal_base_rotation=BASE_FROM_COMPLIANCE.tolist(),
        origin="unchanged knife_mount origin; no change to wrench moment point",
        axes="x: down; y: base +y; z: base +x at nominal cutting pose",
        behavior="tool-fixed axes; rotate with tool away from nominal pose")
