"""Pinned presentation assets; simulation geometry and platform identity stay authoritative."""
import hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_PROFILE=Path(__file__).resolve().parents[1]/"configs/render/studio_v1.json"

def load_profile(path=DEFAULT_PROFILE):
    p=Path(path);d=json.loads(p.read_text())
    if d.get("revision")!="food_studio_v1":raise ValueError("Unsupported render contract")
    return d,hashlib.sha256(p.read_bytes()).hexdigest()

def appearance_reference(profile,name):
    if name not in profile["appearance"]:raise ValueError("No approved appearance for object: "+name)
    a=profile["appearance"][name]
    if a.get("kind")=="procedural":return None
    root=Path(a["directory"])
    for filename,sha in a["sha256"].items():
        p=root/filename
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise ValueError("Missing or changed approved appearance: "+str(p))
    return root

def validate_platform(task, expected):
    actual="ur10e" if task.get("robot_base")=="base_link" and task.get("robot_platform") else "ur5e" if task.get("robot_base")=="b_bot_base_link" else None
    if expected not in ("ur5e","ur10e") or actual!=expected:
        raise ValueError("Recorded robot platform does not match requested platform")
    if actual=="ur10e":
        from .robot_platform import RobotPlatform
        root=task["robot_platform"].get("asset_root")
        if not root:raise ValueError("Recording has no actual CAD bundle identity")
        loaded=RobotPlatform.load(root).metadata["robot_platform"]
        if loaded["runtime_sha256"]!=task["robot_platform"]["runtime_sha256"]:
            raise ValueError("Robot bundle changed since recording")
    return actual

def camera(profile,center,view):
    c=profile["camera"][view];target=np.asarray(center)+c["target_offset_m"]
    return target+np.asarray(c["offset_m"]),target,float(c["ortho_scale_m"])

def _fixed_transform(xml,target,source):
    parents={j.find("child").get("link"):j for j in ET.fromstring(xml).findall("joint")}
    def ancestors(link):
        names=[link]
        while link in parents:
            link=parents[link].find("parent").get("link");names.append(link)
        return names
    target_ancestors=ancestors(target)
    common=next(n for n in ancestors(source) if n in target_ancestors)
    def from_common(link):
        joints=[]
        while link!=common:
            j=parents[link]
            if j.get("type")!="fixed":raise ValueError("Material frame must be fixed relative to robot base")
            joints.append(j);link=j.find("parent").get("link")
        t=np.eye(4)
        for j in reversed(joints):
            o=j.find("origin");x=np.eye(4)
            if o is not None:
                x[:3,3]=np.fromstring(o.get("xyz","0 0 0"),sep=" ")
                x[:3,:3]=Rotation.from_euler("xyz",np.fromstring(o.get("rpy","0 0 0"),sep=" ")).as_matrix()
            t=t@x
        return t
    return np.linalg.inv(from_common(target))@from_common(source)

def material_coordinates(d,task,points):
    if "base_position_world" in d:
        p=d["base_position_world"];r=d["base_rotation_world"].reshape(3,3)
    else:
        ids=list(d["geom_body"])
        for i,body in enumerate(ids):
            try:t=_fixed_transform(task["robot_xml"],task["robot_base"],str(body))
            except (ValueError, StopIteration):continue
            body_r=d["body_mat"][0,i].reshape(3,3)
            r=body_r@t[:3,:3].T
            p=d["body_pos"][0,i]-r@t[:3,3]
            break
        else:raise ValueError("Recording lacks a body fixed to the robot base")
    local=(np.asarray(points)-p)@r
    t=_fixed_transform(task["robot_xml"],task["robot_base"],"cutting_board_disect")
    return (local-t[:3,3])@t[:3,:3]

def procedural_colors(kind, points):
    """Deterministic synthetic food appearance, in initial material coordinates."""
    x=np.asarray(points,dtype=float)*1000.
    if not np.isfinite(x).all():raise ValueError("Nonfinite appearance coordinates")
    grain=(np.sin(x[:,0]*3.1+x[:,1]*1.7)*np.sin(x[:,2]*2.3-x[:,1]*4.1))
    fine=np.sin(x[:,0]*11.3+x[:,2]*13.7)*np.cos(x[:,1]*9.1)
    if kind=="tofu":
        base=np.array([.91,.89,.77]);shade=.018*grain+.008*fine
    elif kind=="luncheon_meat":
        base=np.array([.72,.42,.40]);shade=.055*grain+.025*fine
    elif kind=="banana":
        base=np.array([.93,.88,.69]);shade=.015*grain+.006*fine
    else:raise ValueError("Unknown synthetic food appearance: "+kind)
    return np.clip(base+shade[:,None],0,1)


def recorded_clip(evaluation, group, allow_partial=False):
    """Return a real completed episode or explicitly labelled saved prefix."""
    evaluation=Path(evaluation)
    case=int(group["first_world"])
    rows_path=evaluation/"episodes.jsonl"
    rows=[json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    row=next((r for r in rows if r["case"]==case),None)
    if row is not None:
        return row
    if not allow_partial:
        raise ValueError("Recorded case has no completed episode; use --allow-partial for a labelled prefix")
    root=evaluation/"geometry"
    last=-1
    while (root/f"{last+1:04d}"/group["path"]/"world_states.npz").is_file():
        last+=1
    if last<1:
        raise ValueError("No recorded motion prefix")
    return dict(case=case,steps=last,ending="evaluation_incomplete",success=None,
                completed=False,scope="Saved trajectory prefix only; evaluation ended before this case terminated")
