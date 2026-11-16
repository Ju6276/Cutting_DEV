#!/usr/bin/env python3
"""Pinhole RGB-D view of recorded physical scene, using shared robot/knife visuals.

Synthetic ideal depth, not a real sensor or a new physics/rendering scene.
Robot self-mask comes from known robot/knife geometry; no food instance mask is
exported. Ground-truth surface is used ONLY in the separate audit after planning.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import vtk
from PIL import Image,ImageDraw
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
import render_episode as shared
from cutting_world_model.visual_cut import DepthCamera,estimate_plan

def capture(recording, appearance, camera, roots, world_from_base):
    d=dict(np.load(recording/"episode.npz",allow_pickle=False))
    task=json.loads((recording/"task.json").read_text())
    app=dict(np.load(appearance,allow_pickle=False)) if appearance else None
    win=vtk.vtkRenderWindow();win.SetOffScreenRendering(True);win.SetMultiSamples(0)
    win.SetSize(camera.width,camera.height)
    ren=vtk.vtkRenderer();ren.SetBackground(.91,.93,.95);win.AddRenderer(ren)
    world_from_base=np.asarray(world_from_base,float)
    # Reuse the rigid-calibration validator; never infer a frame from one point.
    DepthCamera(2,2,1,1,.5,.5,world_from_base)
    rotation,translation=world_from_base[:3,:3],world_from_base[:3,3]
    ground_base=(shared.presentation_ground(d,task)-translation)@rotation
    if not np.allclose(task["board_pose_geometry"]["board_origin_base"],ground_base.mean(0),atol=1e-6):
        raise ValueError("Recorded board and world/base calibration disagree")
    verts=shared.material_vertices(d,0)
    if app:
        pts=app["points"][0] if app["points"].ndim==3 else app["points"]
        faces=app["faces"]
        colors=(app["colors"]*255).clip(0,255).astype(np.uint8)
    else:
        pts=verts;faces=np.concatenate((d["tri"],d["cut_tri"]))
        colors=shared.material_face_colors(d,"generic")
    poly=shared.polydata(pts,faces)
    color=numpy_to_vtk(colors,deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)
    color.SetName("Recorded display colors");poly.GetCellData().SetScalars(color)
    food=shared.actor(poly,(1,1,1),smooth=True);ren.AddActor(food)
    ground=shared.presentation_ground(d,task)
    board=shared.box(ground.mean(0)-[0,0,.006],
                    [np.ptp(ground[:,0]),np.ptp(ground[:,1]),.012],(.58,.42,.25))
    ren.AddActor(board)
    robots,sources,_=shared.physical_robot_actors(d,task,roots)
    known=[]
    for k,actor,local in robots:
        t=np.eye(4);t[:3,:3]=d["body_mat"][0,k].reshape(3,3);t[:3,3]=d["body_pos"][0,k]
        t=t@local;actor.SetUserMatrix(shared.matrix(t[:3,3],t[:3,:3]))
        ren.AddActor(actor);known.append(actor)
    blade=shared.actor(shared.polydata(d["knife"][0],d["knife_tri"]),(.61,.63,.65),specular=.5)
    ren.AddActor(blade);known.append(blade)
    cam=ren.GetActiveCamera();t=world_from_base@camera.base_from_camera
    cam.SetPosition(*t[:3,3]);cam.SetFocalPoint(*(t[:3,3]+t[:3,2]))
    cam.SetViewUp(*(-t[:3,1]));cam.ParallelProjectionOff()
    if not np.isclose(camera.fx,camera.fy) or not np.allclose([camera.cx,camera.cy],[(camera.width-1)/2,(camera.height-1)/2]):
        raise ValueError("Synthetic renderer expects centred square pixels")
    cam.SetViewAngle(np.degrees(2*np.arctan(camera.height/(2*camera.fy))))
    near,far=.01,2.;cam.SetClippingRange(near,far)
    def read(kind):
        win.Render()
        f=vtk.vtkWindowToImageFilter();f.SetInput(win);f.ReadFrontBufferOff()
        if kind=="depth":f.SetInputBufferTypeToZBuffer()
        else:f.SetInputBufferTypeToRGB()
        f.Update();raw=vtk_to_numpy(f.GetOutput().GetPointData().GetScalars())
        if kind=="rgb":return raw.reshape(camera.height,camera.width,3)[::-1].copy()
        raw=raw.reshape(camera.height,camera.width)[::-1]
        return np.where(raw<1-1e-7,near*far/(far-raw*(far-near)),0.).astype(np.float64)
    try:
        depth=read("depth");rgb=read("rgb")
        food.SetVisibility(False);board.SetVisibility(False)
        robot_depth=read("depth")
    finally:win.Finalize()
    mask=(depth>0)&(robot_depth>0)&(np.abs(depth-robot_depth)<.0002)
    # Independent projection/deprojection audit: visible board pixels vs its physical plane.
    return depth,mask,rgb,(verts-translation)@rotation,dict(robot_mesh_sources=sorted(set(sources)),
        self_mask="known FK robot/knife projection, not a food segmentation label",
        geometry="shared physical blade and existing gripper; no added connector",
        depth="ideal pinhole optical-axis metres; no sensor noise or calibration error")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording",type=Path,required=True)
    p.add_argument("--appearance-scene",type=Path)
    p.add_argument("--source-root",type=Path,action="append",default=[])
    p.add_argument("--camera",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.config.read_text());camera=DepthCamera(**json.loads(a.camera.read_text()))
    depth,robot,rgb,truth,provenance=capture(a.recording,a.appearance_scene,camera,a.source_root,config["render_world_from_base"])
    np.savez_compressed(a.output/"depth.npz",depth_m=depth,robot_mask=robot)
    (a.output/"camera.json").write_text(json.dumps(camera.to_dict(),indent=2))
    (a.output/"config.json").write_text(json.dumps(config,indent=2))
    Image.fromarray(rgb).save(a.output/"rgb.png")
    # Explicitly keep truth out of estimate_plan.
    plan=estimate_plan(depth,camera,config,robot)
    mask=plan.pop("object_mask")
    actual_top=float(((truth-config["board_origin_base"])@np.asarray(config["board_normal_base"])).max())
    audit=dict(provenance,source=str(a.recording),
        source_sha256=hashlib.sha256((a.recording/"episode.npz").read_bytes()).hexdigest(),
        measured_surface_top_m=plan["surface_top_m"],physical_surface_top_m=actual_top,
        surface_error_m=plan["surface_top_m"]-actual_top,
        estimated_start_margin_over_true_top_m=plan["conservative_top_m"]-actual_top,
        scope="perception and goal conversion on recorded scenes, not closed-loop control success")
    (a.output/"plan.json").write_text(json.dumps(plan,indent=2,allow_nan=False))
    (a.output/"audit.json").write_text(json.dumps(audit,indent=2))
    np.savez_compressed(a.output/"points.npz",object_points_base=camera.points(depth)[mask])
    overlay=rgb.copy();overlay[mask]=(overlay[mask]*.6+np.array([30,185,155])*.4).astype(np.uint8)
    image=Image.fromarray(overlay);draw=ImageDraw.Draw(image)
    centre=np.asarray(plan["cut_point_base"]);n=np.asarray(plan["cut_normal_base"])
    along=np.cross(n,np.asarray(config["board_normal_base"]))
    line=np.array([centre-.075*along,centre+.075*along])
    uv=camera.project(line)
    draw.line([tuple(uv[0]),tuple(uv[1])],fill=(245,185,45),width=3)
    image.save(a.output/"perception.png")
    print(json.dumps(dict(plan=plan,audit=audit),indent=2))
if __name__=="__main__":main()
