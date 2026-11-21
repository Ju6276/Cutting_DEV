"""Prepare a provisional fixed Nano camera from the recorded robot/board calibration."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from cutting_world_model.camera_sim import nano_camera

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording",type=Path,required=True)
    p.add_argument("--profile",type=Path,default=Path("configs/vision/mecheye_nano_provisional_v1.json"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    profile=json.loads(a.profile.read_text())
    task=json.loads((a.recording/"task.json").read_text())
    with np.load(a.recording/"episode.npz",allow_pickle=False) as d:
        if "base_position_world" in d and "base_rotation_world" in d:
            world_from_base=np.eye(4)
            world_from_base[:3,3]=d["base_position_world"]
            world_from_base[:3,:3]=d["base_rotation_world"].reshape(3,3)
        else:
            idx=list(d["geom_body"]).index(task["robot_base"])
            world_from_base=np.eye(4)
            world_from_base[:3,3]=d["body_pos"][0,idx]
            world_from_base[:3,:3]=d["body_mat"][0,idx].reshape(3,3)
    board=task["board_pose_geometry"]["board_origin_base"]
    camera=nano_camera(profile,board)
    housing=Path(profile["housing_asset"])
    if not housing.exists():raise FileNotFoundError(housing)
    (a.output/"camera.json").write_text(json.dumps(camera.to_dict(),indent=2))
    (a.output/"rig.json").write_text(json.dumps(dict(
        revision=profile["revision"],housing_asset=str(housing),
        housing_appearance=profile.get("housing_appearance"),
        housing_sha256=hashlib.sha256(housing.read_bytes()).hexdigest(),
        world_from_camera=(world_from_base@camera.base_from_camera).tolist(),
        world_from_base=world_from_base.tolist(),
        mounting="fixed eye-to-hand; front of robot looking down; provisional",
        optical_registration="approximate housing-centre fit; not factory extrinsics",
        support="support structure not modelled; placement proposal, not lab replica",
        source_recording=str(a.recording)),indent=2))
    (a.output/"board.json").write_text(json.dumps(dict(board_origin_base=board,world_from_base=world_from_base.tolist()),indent=2))
    print(a.output)
if __name__=="__main__":main()
