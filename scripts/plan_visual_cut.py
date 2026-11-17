#!/usr/bin/env python3
"""Route 1 from calibrated RGB-D export. No simulator or neural network required.

NPZ: depth_m (optical-axis metres), optional robot_mask (known robot projection).
JSON calibration: DepthCamera fields; task JSON: prescribed geometry/allowed ranges.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from cutting_world_model.visual_cut import DepthCamera,estimate_plan,validate_policy_plan,visual_midpoint_plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",choices=("prescribed","visible-midpoint","fixed-midpoint","learned-midpoint"),default="prescribed")
    p.add_argument("--depth",type=Path,required=True)
    p.add_argument("--camera",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--center-checkpoint",type=Path,help="Frozen perception model for learned-midpoint")
    p.add_argument("--checkpoint",type=Path,help="Optional frozen policy interface check; no inference or robot execution")
    a=p.parse_args()
    d=np.load(a.depth,allow_pickle=False)
    camera=DepthCamera(**json.loads(a.camera.read_text()))
    cfg=json.loads(a.config.read_text());mask=d["robot_mask"] if "robot_mask" in d else None
    if (a.selection=="learned-midpoint") != (a.center_checkpoint is not None):
        p.error("--center-checkpoint is required only for learned-midpoint")
    if a.selection in ("fixed-midpoint","learned-midpoint"):
        from cutting_world_model.visual_center import center_plan,load_center
        model=load_center(a.center_checkpoint)[0] if a.center_checkpoint else None
        result=center_plan(d["depth_m"],camera,cfg,mask,model=model)
    else:
        planner=estimate_plan if a.selection=="prescribed" else visual_midpoint_plan
        result=planner(d["depth_m"],camera,cfg,mask)
    mask=result.pop("object_mask")
    if a.checkpoint:
        import torch
        saved=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
        validate_policy_plan(result,saved["metadata"],json.loads(a.config.read_text()))
        result["checkpoint_interface"]={"path":str(a.checkpoint),
            "sha256":hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            "validated":True,"scope":"goal schema and mounting only; not closed-loop execution"}

    if a.center_checkpoint:
        result["center_checkpoint_sha256"]=hashlib.sha256(a.center_checkpoint.read_bytes()).hexdigest()
    result["inputs_sha256"]={str(x):hashlib.sha256(x.read_bytes()).hexdigest()
                             for x in [a.depth,a.camera,a.config]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(a.output.with_suffix(".points.npz"),
                       object_points_base=camera.points(d["depth_m"])[mask])
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
