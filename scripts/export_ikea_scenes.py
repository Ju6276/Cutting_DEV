"""Export unheld objects with the actual IKEA mesh and curved-edge sweep."""
import argparse,json,types,sys,hashlib
import collections, collections.abc
# Upstream DiSECt targets Python 3.8; explicit import compatibility for its settings.
collections.MutableMapping=collections.abc.MutableMapping
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backends/mujoco/scripts"))
from disect_headless import import_disect_cutting

def verify_asset_hash(path, expected, label):
 if not expected:
  raise ValueError(label+" lacks checksum")
 digest=hashlib.sha256()
 with Path(path).open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
 if digest.hexdigest()!=expected:
  raise ValueError(label+" checksum mismatch")

def export(source,platform,output,cutting,source_root):
 meta=json.loads((platform/"cutting_geometry.json").read_text())
 verify_asset_hash(meta["geometry"],meta.get("geometry_sha256"),"CAD knife")
 verify_asset_hash(meta["sdf"],meta.get("sdf_validation",{}).get("sha256"),"Knife SDF")
 g=np.load(meta["geometry"],allow_pickle=False)
 old=torch.load(source,map_location="cpu",weights_only=False)
 settings=dict(old["settings"])
 v=g["vertices_disect"];edge=g["edge_disect"]
 top=float(old["state"].particle_q[:,1].max())
 start=top+.04-float(v[:,1].min())
 for kind in ("ansys","meshio"):
  name=settings["generators"][kind]["filename"]
  settings["generators"][kind]["filename"]=str((source_root/name).resolve())
 settings["initial_y"]=start
 settings["knife_motion"]["position"]=[0,start,0]
 settings["knife_motion"]["rotation"]=[0.,0.,1.,0.]
 settings["static_vertices"]={"include":[],"exclude":[]}
 cfg=output.with_suffix(".json");cfg.write_text(json.dumps(settings))
 sim=cutting.CuttingSim(cutting.load_settings(str(cfg)),adapter="cuda",requires_grad=False,allow_nans=False,log_folder=str(output.parent/"logs"))
 sim.knife.create_mesh=lambda:(v.astype(np.float32),g["faces"].astype(np.int32).reshape(-1))
 sim.setup_free_floating_knife()
 sim.blade_bottom_center_point_offset=torch.tensor(edge[edge[:,1].argmin()],device="cuda",dtype=torch.float32)
 def sweep(self,*args,**kwargs):
  # Pre-processing surface only: below-board extension creates potential springs,
  # not permission for the execution trajectory to penetrate the board.
  high=edge+np.array([0,start,0])
  low=edge+np.array([0,-.02-float(edge[:,1].max()),0])
  vv=np.concatenate([high,low]);n=len(edge)
  ff=np.array([[i,i+1,n+i] for i in range(n-1)]+[[n+i,i+1,n+i+1] for i in range(n-1)])
  self.cut_surface_vertices=vv;self.cut_surface_indices=ff;self.cut_surface_triangles=vv[ff]
  return vv,ff,vv[ff]
 sim.get_trace_cut_surface=types.MethodType(sweep,sim)
 sim.cut();sim.state=sim.model.state();sim.disable_gradients()
 sim.model.knife_edge_vertices=torch.tensor(edge,device="cuda",dtype=torch.float32)
 sim.model.knife_board_vertices=torch.tensor(g["board_vertices_disect"],device="cuda",dtype=torch.float32)
 sim.model.knife_sdf_grid=torch.from_numpy(np.load(meta["sdf"]))
 scene=dict(model=sim.model,state=sim.state,initial_pos=sim.motion.current_pos,initial_rot=sim.motion.current_rot,
            linear_velocity=sim.motion.lin_vel,angular_velocity=sim.motion.ang_vel,spine_height=sim.knife.spine_height,
            settings=dict(sim.settings),tool_geometry=meta)
 torch.save(scene,output)
 print(json.dumps(dict(output=str(output),springs=sim.model.cut_spring_count,particles=sim.model.particle_count,
                       source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),start_center_y=start)),flush=True)

def select_objects(contract,names=None):
 objects=contract["objects"]
 available=[obj["name"] for obj in objects]
 if len(set(available))!=len(available):raise ValueError("Duplicate object names")
 if names:
  unknown=set(names)-set(available)
  if unknown:raise ValueError("Unknown objects: "+", ".join(sorted(unknown)))
  objects=[obj for obj in objects if obj["name"] in names]
 if not objects:raise ValueError("No objects selected")
 return objects

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--platform",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--source-root",type=Path,default=Path("/dataHDD/dong/Projects/SliceIt/disect"))
 p.add_argument("--objects",nargs="+")
 p.add_argument("--contract",type=Path,default=Path(__file__).resolve().parents[1]/"configs/shared/scene_v1.json")
 a=p.parse_args()
 root=Path(__file__).resolve().parents[1]
 contract=json.loads(a.contract.read_text())
 objects=select_objects(contract,a.objects)
 a.output.mkdir(parents=True,exist_ok=True)
 cutting=import_disect_cutting(a.source_root)
 for obj in objects:
  dest=a.output/(obj["name"]+".pt")
  if dest.exists():raise FileExistsError(dest)
  export(root/"backends/mujoco"/obj["disect_source_scene"],a.platform,dest,cutting,a.source_root)
if __name__=="__main__":main()
