"""Render metric depth from the shared physical snapshot, then select a visual cut."""
import argparse,json,subprocess,sys,os
from pathlib import Path
import numpy as np
import vtk
from PIL import Image,ImageDraw
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
import render_geometry as r
from cutting_world_model.visual_cut import DepthCamera,visual_midpoint_plan
from cutting_world_model.camera_sim import projection_matrix,range_depth,validate_sensor_capture
from cutting_world_model.render_contract import load_profile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sensor-profile",type=Path,help="Optional approximate camera profile with explicit depth range")
    p.add_argument("--blender",default="/usr/bin/blender")
    p.add_argument("--samples",type=int,default=32)
    p.add_argument("--capture-only",action="store_true",help="Save RGB/depth without selecting a plane")
    p.add_argument("--scene",type=Path,required=True);p.add_argument("--camera",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    d=dict(np.load(a.scene,allow_pickle=False));meta=json.loads(str(d["metadata"]))
    cfg=json.loads(a.config.read_text());cam=DepthCamera(**json.loads(a.camera.read_text()))
    sensor=json.loads(a.sensor_profile.read_text()) if a.sensor_profile else None
    if sensor:validate_sensor_capture(cam,sensor,cfg.get("world_from_base"))
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
    g=d["ground"];profile,_=load_profile();thickness=profile["board_thickness_m"]
    board=r.box(g.mean(0)-[0,0,thickness/2],[*profile["board_size_m"],thickness],(.58,.42,.25));ren.AddActor(board)
    for j in range(meta["robot_meshes"]):
        act=r.actor(r.polydata(d[f"robot_v{j}"],d[f"robot_f{j}"]),d[f"robot_c{j}"],smooth=True)
        t=d[f"robot_transform{j}"][0];act.SetUserMatrix(r.matrix(t[:3,3],t[:3,:3]));ren.AddActor(act)
    if not bool(d.get("mounted_cad_knife",False)):
        ren.AddActor(r.actor(r.polydata(d["knife"][0],d["knife_tri"]),(.44,.47,.50),specular=.5))
    c=ren.GetActiveCamera();T=cam.base_from_camera
    c.SetPosition(*T[:3,3]);c.SetFocalPoint(*(T[:3,3]+T[:3,2]));c.SetViewUp(*(-T[:3,1]))
    c.SetClippingRange(.01,2)
    proj=vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):proj.SetElement(i,j,float(projection_matrix(cam)[i,j]))
    c.SetExplicitProjectionTransformMatrix(proj);c.SetUseExplicitProjectionTransformMatrix(True)
    def read(depth=False):
        win.Render();cap=vtk.vtkWindowToImageFilter();cap.SetInput(win);cap.ReadFrontBufferOff()
        if depth:cap.SetInputBufferTypeToZBuffer()
        else:cap.SetInputBufferTypeToRGB()
        cap.Update();x=vtk_to_numpy(cap.GetOutput().GetPointData().GetScalars())
        if not depth:return x.reshape(cam.height,cam.width,3)[::-1].copy()
        x=x.reshape(cam.height,cam.width)[::-1]
        return np.where(x<1-1e-7,.02/(2-x*1.99),0.)
    try:
        depth=read(True);food.SetVisibility(False);board.SetVisibility(False)
        known=read(True)
    finally:win.Finalize()
    if sensor:depth=range_depth(depth,sensor)
    mask=(depth>0)&(known>0)&(np.abs(depth-known)<.0002)
    np.savez_compressed(a.output/"depth.npz",depth_m=depth,robot_mask=mask)
    # Sensor RGB uses the same canonical scene/materials as all presentation output.
    # VTK is only the metric-depth rasterizer; do not export its alternate RGB palette.
    with (a.output/"blender.log").open("w") as render_log:
        subprocess.run([a.blender,"-b","-t","6","--python-exit-code","1","-P",
        str(Path(__file__).with_name("render_episode_still.py")),"--","--scene",str(a.scene),
        "--sensor-camera",str(a.camera),"--samples",str(a.samples),"--output",str(a.output/"rgb.png")],
            check=True,stdout=render_log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))
    rgb=np.asarray(Image.open(a.output/"rgb.png").convert("RGB"))
    (a.output/"camera.json").write_text(json.dumps(cam.to_dict(),indent=2))
    (a.output/"sensor.json").write_text(json.dumps(sensor or {"model":"ideal_pinhole"},indent=2))
    np.savez_compressed(a.output/"cloud.npz",points_base=cam.points(depth)[depth>0])
    (a.output/"capture.json").write_text(json.dumps(dict(
        scene=str(a.scene),pixels=int((depth>0).sum()),robot_pixels=int(mask.sum()),
        depth_kind="optical-axis metres; invalid=0",coordinates="robot_base",
        appearance="Canonical Blender studio renderer, same materials/board/lights; sensor perspective",
        sensor_fidelity="Ideal depth with visibility/range clipping; no structured-light reconstruction",
        hardware_version_confirmed=False),indent=2))
    if a.capture_only:
        return
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
