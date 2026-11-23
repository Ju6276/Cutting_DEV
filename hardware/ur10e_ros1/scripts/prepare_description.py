#!/usr/bin/env python3
"""Append the fixed simulated tool subtree to a saved calibrated real URDF.
Offline only. Never reads or overwrites a live ROS parameter.
"""
import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

def combine(actual_text, tool_text, mesh_root):
    actual, tool = ET.fromstring(actual_text), ET.fromstring(tool_text)
    original = [ET.tostring(e) for e in actual]
    names = {e.get("name") for e in actual if e.tag in ("link", "joint")}
    descendants, selected = {"tool0"}, []
    while True:
        added = [j for j in tool.findall("joint")
                 if j.find("parent").get("link") in descendants
                 and j.find("child").get("link") not in descendants]
        if not added:
            break
        for joint in added:
            if joint.get("type") != "fixed":
                raise ValueError("Tool subtree must be fixed")
            descendants.add(joint.find("child").get("link"))
            selected.append(joint)
    if not selected:
        raise ValueError("Missing tool0 tool subtree")
    links = [e for e in tool.findall("link") if e.get("name") in descendants - {"tool0"}]
    for original_element in links + selected:
        if original_element.get("name") in names:
            raise ValueError("Tool already present or duplicate name: " + original_element.get("name"))
        element = copy.deepcopy(original_element)
        for mesh in element.findall(".//mesh"):
            path = Path(mesh_root).resolve() / Path(mesh.get("filename")).name
            if not path.is_file():
                raise FileNotFoundError(path)
            mesh.set("filename", str(path))
        actual.append(element)
    if [ET.tostring(e) for e in list(actual)[:len(original)]] != original:
        raise RuntimeError("Calibrated arm changed")
    link_names = [e.get("name") for e in actual.findall("link")]
    if len(set(link_names)) != len(link_names):
        raise ValueError("Duplicate links")
    for joint in actual.findall("joint"):
        for side in ("parent", "child"):
            if joint.find(side).get("link") not in link_names:
                raise ValueError("Joint references missing link")
    if not {"knife_control", "knife_compliance", "ati_sensing_frame"} <= set(link_names):
        raise ValueError("Missing controller frames")
    return ET.tostring(actual, encoding="unicode")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-urdf", type=Path, required=True)
    ap.add_argument("--tool-urdf", type=Path, required=True)
    ap.add_argument("--mesh-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    result = combine(a.arm_urdf.read_text(), a.tool_urdf.read_text(), a.mesh_root)
    # Do not silently overwrite a calibrated description.
    with a.output.open("x") as f:
        f.write(result + "\n")
