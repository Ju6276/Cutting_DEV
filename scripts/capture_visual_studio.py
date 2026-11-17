"""Render metric depth from the shared physical snapshot, then select a visual cut."""
import argparse,json
from pathlib import Path
import numpy as np
import vtk
from PIL import Image,ImageDraw
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
import render_episode as r
from cutting_world_model.visual_cut import DepthCamera,visual_midpoint_plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",type=Path,required=True);p.add_argument("--camera",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    d=dict(np.load(a.scene,allow_pickle=False));meta=json.loads(str(d["metadata"]))
    cfg=json.loads(a.config.read_text());cam=DepthCamera(**json.loads(a.camera.read_text()))
    if not np.allclose(d["ground"].mean(0),cfg["board_origin_base"],atol=1e-6):
        raise ValueError("This snapshot adapter requires verified world==robot base")
    points=d["points_0"] if bool(d.get("dynamic_topology",False)) else d["points"][0]
    faces=d["faces_0"] if bool(d.get("dynamic_topology",False)) else d["faces"]
    surface=r.polydata(points,faces)
    cols=numpy_to_vtk(np.asarray(d["colors"]*255,dtype=np.uint8),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)
    surface.GetCellData().SetScalars(cols)
    food=r.actor(surface,(1,1,1),smooth=True)
    food.GetMapper().ScalarVisibilityOn();food.GetMapper().SetScalarModeToUseCellData()
    food.GetMapper().SetColorModeToDirectScalars()
    win=vtk.vtkRenderWindow();win.SetOffScreenRendering(True);win.SetMultiSamples(0);win.SetSize(cam.width,cam.height)
    ren=vtk.vtkRenderer();ren.SetBackground(.91,.93,.95);win.AddRenderer(ren);ren.AddActor(food)
    g=d["ground"];board=r.box(g.mean(0)-[0,0,.006],[np.ptp(g[:,0]),np.ptp(g[:,1]),.012],(.58,.42,.25));ren.AddActor(board)
    for j in range(meta["robot_meshes"]):
        act=r.actor(r.polydata(d[f"robot_v{j}"],d[f"robot_f{j}"]),d[f"robot_c{j}"],smooth=True)
        t=d[f"robot_transform{j}"][0];act.SetUserMatrix(r.matrix(t[:3,3],t[:3,:3]));ren.AddActor(act)
    ren.AddActor(r.actor(r.polydata(d["knife"][0],d["knife_tri"]),(.44,.47,.50),specular=.5))
    c=ren.GetActiveCamera();T=cam.base_from_camera
    c.SetPosition(*T[:3,3]);c.SetFocalPoint(*(T[:3,3]+T[:3,2]));c.SetViewUp(*(-T[:3,1]))
    c.SetViewAngle(np.degrees(2*np.arctan(cam.height/(2*cam.fy))));c.SetClippingRange(.01,2)
    def read(depth=False):
        win.Render();cap=vtk.vtkWindowToImageFilter();cap.SetInput(win);cap.ReadFrontBufferOff()
        if depth:cap.SetInputBufferTypeToZBuffer()
        else:cap.SetInputBufferTypeToRGB()
        cap.Update();x=vtk_to_numpy(cap.GetOutput().GetPointData().GetScalars())
        if not depth:return x.reshape(cam.height,cam.width,3)[::-1].copy()
        x=x.reshape(cam.height,cam.width)[::-1]
        return np.where(x<1-1e-7,.02/(2-x*1.99),0.)
    try:
        depth=read(True);rgb=read();food.SetVisibility(False);board.SetVisibility(False)
        known=read(True)
    finally:win.Finalize()
    mask=(depth>0)&(known>0)&(np.abs(depth-known)<.0002)
    np.savez_compressed(a.output/"depth.npz",depth_m=depth,robot_mask=mask)
    Image.fromarray(rgb).save(a.output/"rgb.png")
    plan=visual_midpoint_plan(depth,cam,cfg,mask);object_mask=plan.pop("object_mask")
    np.savez_compressed(a.output/"points.npz",object_points_base=cam.points(depth)[object_mask])
    (a.output/"plan.json").write_text(json.dumps(plan,indent=2,allow_nan=False))
    overlay=Image.fromarray(rgb);draw=ImageDraw.Draw(overlay)
    normal=np.asarray(plan["selected_plane_normal_base"]);along=np.cross(normal,[0,0,1])
    centre=np.asarray(plan["selected_plane_point_base"])
    line=cam.project(np.array([centre-.075*along,centre+.075*along]))
    draw.line([tuple(x) for x in line],fill=(245,185,45),width=3)
    overlay.save(a.output/"perception.png")
    print(json.dumps(plan,indent=2))
if __name__=="__main__":main()
