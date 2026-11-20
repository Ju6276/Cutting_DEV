"""Export an actual compiled robot platform through the existing shared studio geometry path."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import hashlib
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import render_episode as r
from cutting_world_model.robot_platform import RobotPlatform

def export(platform,output,verification_path):
    bundle=RobotPlatform.load(platform)
    m=bundle.metadata
    verification=json.loads(Path(verification_path).read_text())
    if verification["robot_xml_sha256"] != hashlib.sha256(m["robot_xml"].encode()).hexdigest():
        raise ValueError("Visual poses are from another robot asset")
    poses=verification["initial_visual_poses"]
    names=[l.get("name") for l in ET.fromstring(m["robot_xml"]).findall("link") if l.get("name") in poses]
    actors,sources=r.visual_actors(dict(geom_body=np.array(names)),bundle.metadata,[Path("/opt/ros/noetic/share/ur_description")])
    out=dict(points=np.zeros((0,3)),faces=np.zeros((0,3),dtype=int),colors=np.zeros((0,3)),
             knife=np.zeros((0,3)),knife_tri=np.zeros((0,3),dtype=int),
             ground=np.array([[.35,-.27,.37],[1,-.27,.37],[1,.27,.37],[.35,.27,.37]]),
             center=np.array([.62,0,.46]))
    for j,(k,actor,local) in enumerate(actors):
        actor.GetMapper().Update()
        triangles=vtk.vtkTriangleFilter();triangles.SetInputData(actor.GetMapper().GetInput());triangles.Update()
        poly=triangles.GetOutput();v=vtk_to_numpy(poly.GetPoints().GetData())
        f=vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1,4)
        assert np.all(f[:,0]==3)
        pose=np.asarray(poses[names[k]]["xyz"])
        rotation=np.asarray(poses[names[k]]["rotation"])
        out[f"robot_v{j}"]=(v@local[:3,:3].T+local[:3,3])@rotation.T+pose
        out[f"robot_f{j}"]=f[:,1:];out[f"robot_c{j}"]=actor.GetProperty().GetColor()
        out[f"robot_tool{j}"]=names[k]=="ikea_knife"
    out["metadata"]=np.array(json.dumps(dict(format="static_platform",robot_meshes=len(actors),
                    platform=str(platform),sources=sources,geometry="Compiled robot FK and the URDF visual geometry; no material-cutting claim")))
    Path(output).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**out)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--platform",required=True);p.add_argument("--output",required=True)
    p.add_argument("--verification",required=True)
    a=p.parse_args();export(a.platform,a.output,a.verification)
