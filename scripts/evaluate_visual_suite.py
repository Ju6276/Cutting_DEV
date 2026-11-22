"""Run one frozen pre-cut visual handoff per configured food, with a shared wall budget."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--blademaster-config",type=Path,required=True)
    p.add_argument("--center-checkpoint",type=Path,required=True)
    p.add_argument("--render-python",type=Path,required=True)
    p.add_argument("--camera-profile",type=Path,default=ROOT/"configs/vision/mecheye_nano_v4_550_lab_v1.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,default=7200)
    p.add_argument("--per-food-seconds",type=float,default=1200)
    p.add_argument("--seed",type=int,default=20261122)
    a=p.parse_args()
    if min(a.wall_seconds,a.per_food_seconds)<=0:p.error("Positive wall budgets required")
    config=json.loads(a.blademaster_config.read_text())
    names=[g["name"] for g in config["groups"]]
    if len(set(names))!=len(names):p.error("Food names must be unique")
    a.output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    results=[]
    def write(state,**values):
        data=dict(state=state,elapsed_seconds=time.monotonic()-began,
                  expected_foods=names,results=results,**values)
        path=a.output/"progress.json"
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
        tmp.replace(path)
        return data
    for name in names:
        remaining=a.wall_seconds-(time.monotonic()-began)
        if remaining<5:
            write("budget_exhausted")
            break
        budget=min(a.per_food_seconds,remaining)
        write("running",current_food=name)
        command=[sys.executable,"-u",str(ROOT/"scripts/evaluate_visual_cut.py"),
                 "--policy",str(a.policy),"--blademaster-config",str(a.blademaster_config),
                 "--center-checkpoint",str(a.center_checkpoint),
                 "--selection","learned-midpoint","--group",name,
                 "--render-python",str(a.render_python),"--camera-profile",str(a.camera_profile),
                 "--seed",str(a.seed),"--wall-seconds",str(max(1,budget-5)),
                 "--output",str(a.output/name)]
        (a.output/(name+"_command.json")).write_text(json.dumps(command,indent=2))
        with (a.output/(name+".log")).open("w") as log:
            try:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=budget)
                code=result.returncode
            except subprocess.TimeoutExpired:
                code=None
        summary=a.output/name/"summary.json"
        status=a.output/name/"progress.json"
        details=json.loads(summary.read_text()) if summary.exists() else {}
        current=details.get("status",json.loads(status.read_text()) if status.exists() else {})
        episodes=details.get("episodes",[])
        results.append(dict(food=name,returncode=code,state=current.get("state","no_report"),
                            phase=current.get("phase"),error=current.get("error"),
                            completed=code==0 and current.get("state")=="completed" and len(episodes)==1,
                            episodes=episodes))
        write("running",current_food=name)
    complete=len(results)==len(names) and all(r["completed"] for r in results)
    data=write("completed" if complete else "incomplete")
    (a.output/"summary.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    return 0 if complete else 1

if __name__=="__main__":
    raise SystemExit(main())
