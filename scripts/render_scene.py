"""Canonical presentation entry: recorded geometry, pinned appearance and explicit robot."""
import argparse,json,subprocess,sys,tempfile,os
from pathlib import Path
import numpy as np
from cutting_world_model.render_contract import load_profile,appearance_reference,validate_platform

def main():
    p=argparse.ArgumentParser(description=__doc__)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument("--evaluation",type=Path)
    src.add_argument("--recording",type=Path,help="Single episode directory; initial still only")
    p.add_argument("--object",help="Required for a single episode; exact appearance registry identity")
    p.add_argument("--group",type=int,default=0)
    p.add_argument("--snapshot",action="store_true")
    p.add_argument("--robot-platform",choices=("ur5e","ur10e"),required=True)
    p.add_argument("--backend",choices=("DiSECt","BladeMaster"),required=True)
    p.add_argument("--source-root",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sensor-rig",type=Path,help="Optional fixed external camera assembly JSON")
    p.add_argument("--view",choices=("close","wide"),default="close")
    p.add_argument("--video",action="store_true")
    p.add_argument("--samples",type=int,default=96)
    p.add_argument("--blender",default="/usr/bin/blender")
    a=p.parse_args()
    if a.video and (a.snapshot or a.recording):p.error("Video requires a completed evaluation recording")
    scripts=Path(__file__).resolve().parent
    a.output.parent.mkdir(parents=True,exist_ok=True)
    scene=a.output.with_suffix(".scene.npz")
    # Keep temporary wrappers beside the output on the server, never in /tmp or a local checkout.
    with tempfile.TemporaryDirectory(prefix="render_input_",dir=a.output.parent) as tmp:
        evaluation=a.evaluation
        if a.recording:
            if not a.object:p.error("--recording requires --object")
            if a.backend!="DiSECt":p.error("Single-episode wrapper supports FEM recordings; use evaluation for MPM")
            task=json.loads((a.recording/"task.json").read_text())
            validate_platform(task,a.robot_platform)
            appearance_reference(load_profile()[0],a.object)
            evaluation=Path(tmp);root=evaluation/"geometry/0000";group=root/"object"
            group.mkdir(parents=True)
            for name in ("episode.npz","task.json"):
                (group/name).symlink_to((a.recording/name).resolve())
            with np.load(a.recording/"episode.npz") as d:
                np.savez_compressed(group/"world_states.npz",**{k:d[k][:1] for k in ("particles","body_pos","body_mat")})
            (root/"report.json").write_text(json.dumps(dict(groups=[dict(name=a.object,path="object",first_world=0)],training={})))
            a.group=0;a.snapshot=True
        command=[sys.executable,str(scripts/"export_evaluation_scene.py"),"--evaluation",str(evaluation),
                 "--group",str(a.group),"--backend",a.backend,"--robot-platform",a.robot_platform,"--output",str(scene)]
        if a.snapshot:command+=["--snapshot"]
        for root in a.source_root:command+=["--source-root",str(root)]
        subprocess.run(command,check=True)
    command=[a.blender,"-b","-t","12","--python-exit-code","1","-P",str(scripts/"render_episode_still.py"),"--",
             "--scene",str(scene),"--output",str(a.output),"--view",a.view,"--samples",str(a.samples)]
    if a.sensor_rig:command+=["--sensor-rig",str(a.sensor_rig)]
    if a.video:command+=["--video"]
    subprocess.run(command,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))

if __name__=="__main__":main()

