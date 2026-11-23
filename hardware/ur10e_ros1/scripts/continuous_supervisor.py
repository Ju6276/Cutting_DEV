"""Independent motion watchdog and restoration for authorized 10 mm air MPC test."""
import subprocess,json,time,os,argparse,sys
from settings import MODULE, record_directory
ap=argparse.ArgumentParser();ap.add_argument("--execute",action="store_true");ap.add_argument("--goal-record");ap.add_argument("--bounded-stiffness",action="store_true");args=ap.parse_args()
from pathlib import Path
import rospy
from controller_manager_msgs.srv import ListControllers,SwitchController,SwitchControllerRequest
from dynamic_reconfigure.client import Client
r=record_directory("mpc_air")
if not args.execute:
 cmd=[sys.executable,str(MODULE/"scripts/continuous_driver.py")]
 if args.goal_record:cmd += ["--goal-record",args.goal_record]
 if args.bounded_stiffness:cmd += ["--bounded-stiffness"]
 sys.exit(subprocess.call(cmd))
rospy.init_node("continuous_mpc_supervisor",anonymous=True)
params=rospy.get_param("/fdcc_shared");out=dict(started=time.time(),distance_m=.010,duration_limit_s=20)
def stop():
 req=SwitchControllerRequest();req.stop_controllers=["fdcc_shared"];req.strictness=1;req.timeout=1.
 return rospy.ServiceProxy("/controller_manager/switch_controller",SwitchController)(req).ok
assert not any(c.state=="running" and any(r.resources for r in c.claimed_resources) for c in rospy.ServiceProxy("/controller_manager/list_controllers",ListControllers)().controller), "Motion controller already active"
proc=None
try:
 with (r/"continuous_execute.log").open("w") as f:
  begin=time.time();start=time.monotonic()
  cmd=[sys.executable,str(MODULE/"scripts/continuous_driver.py"),"--execute","--seconds","20"]
  if args.goal_record:cmd+=["--goal-record",args.goal_record]
  if args.bounded_stiffness:cmd+=["--bounded-stiffness"]
  out["command"]=cmd
  proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   assert time.monotonic()-start<100,"Overall supervisor timeout"
   latest=r/"continuous_latest.txt"
   if latest.exists() and latest.stat().st_mtime>=begin:
    ipc=Path(latest.read_text().strip());hb=ipc/"heartbeat.json"
    if hb.exists():
     h=json.loads(hb.read_text())
     if h["active"]:assert time.monotonic()-h["monotonic"]<.5,"Independent heartbeat watchdog"
   time.sleep(.03)
  out["exit_code"]=proc.returncode
except Exception as e:
 out["error"]=str(e);out["emergency_stop_ok"]=stop()
 if proc and proc.poll() is None:proc.terminate();proc.wait(timeout=4)
finally:
 out["stop_ok"]=stop()
 for key in ["solver","stiffness"]+["pd_gains/"+a for a in ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]]:
  original=params
  for part in key.split("/"):original=original[part]
  try:
   c=Client("/fdcc_shared/"+key,timeout=3);available=c.get_configuration()
   v={k:v for k,v in original.items() if k in available and not isinstance(v,dict)}
   got=c.update_configuration(v);assert all(got[k]==x for k,x in v.items())
  except Exception as e:out.setdefault("restore_errors",{})[key]=str(e)
 out["controllers"]=[[c.name,c.state] for c in rospy.ServiceProxy("/controller_manager/list_controllers",ListControllers)().controller]
 out["finished"]=time.time();(r/("continuous_supervisor_"+time.strftime("%Y%m%d_%H%M%S")+".json")).write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))

if out.get("error") or out.get("exit_code") != 0 or out.get("restore_errors"): sys.exit(1)
