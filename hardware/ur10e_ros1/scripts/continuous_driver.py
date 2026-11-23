"""Continuous FDCC with asynchronous GPU MPC; air commissioning only.
Default is read-only. FDCC stays active between planning updates.
"""
import argparse,copy,json,math,time,threading,subprocess,sys
from pathlib import Path
from settings import MODULE, PLANNER_PYTHON, record_directory
import rospy,tf2_ros,PyKDL
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped,PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import ListControllers,SwitchController,SwitchControllerRequest
from dynamic_reconfigure.client import Client
ap=argparse.ArgumentParser();ap.add_argument("--execute",action="store_true");ap.add_argument("--seconds",type=float,default=20);ap.add_argument("--goal-record");ap.add_argument("--bounded-stiffness",action="store_true");args=ap.parse_args()
assert 2<=args.seconds<=20
root=record_directory("mpc_air")
ipc=root/(("continuous_execute_" if args.execute else "continuous_dry_")+time.strftime("%Y%m%d_%H%M%S"));ipc.mkdir(); (root/"continuous_latest.txt").write_text(str(ipc))
rospy.init_node("continuous_native_mpc",anonymous=True)
buf=tf2_ros.Buffer();listener=tf2_ros.TransformListener(buf)
data={};lock=threading.Lock();active=False;heartbeat=time.monotonic();finish=threading.Event();abort=threading.Event()
def callback(k,m):
 with lock:data[k]=(m,time.monotonic())
subs=[rospy.Subscriber("/joint_states",JointState,lambda m:callback("joint",m),queue_size=1),rospy.Subscriber("/wrench",WrenchStamped,lambda m:callback("wrench",m),queue_size=1),rospy.Subscriber("/ur_hardware_interface/robot_program_running",Bool,lambda m:callback("program",m),queue_size=1)]
def stop():
 req=SwitchControllerRequest();req.stop_controllers=["fdcc_shared"];req.strictness=1;req.timeout=1.
 return rospy.ServiceProxy("/controller_manager/switch_controller",SwitchController)(req).ok
def norm(x):return math.sqrt(sum(a*a for a in x))
def distance(a,b):return norm([x-y for x,y in zip(a,b)])
def transform_msg(m,t):
 t=t.transform;f=PyKDL.Frame(PyKDL.Rotation.Quaternion(t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w),PyKDL.Vector(t.translation.x,t.translation.y,t.translation.z));w=m.wrench
 v=f*PyKDL.Wrench(PyKDL.Vector(w.force.x,w.force.y,w.force.z),PyKDL.Vector(w.torque.x,w.torque.y,w.torque.z))
 out=WrenchStamped();out.header.stamp=m.header.stamp;out.header.frame_id="ati_sensing_frame"
 out.wrench.force.x,out.wrench.force.y,out.wrench.force.z=[v.force[k] for k in range(3)];out.wrench.torque.x,out.wrench.torque.y,out.wrench.torque.z=[v.torque[k] for k in range(3)]
 return out,[v.force[k] for k in range(3)]+[v.torque[k] for k in range(3)]
params=rospy.get_param("/fdcc_shared");names=params["joints"]
def snapshot():
 with lock:d=dict(data)
 now=time.monotonic()
 for k in ("joint","wrench"):
  assert k in d and now-d[k][1]<.1,"Missing/stale "+k
 j=d["joint"][0];w=d["wrench"][0];assert abs((rospy.Time.now()-w.header.stamp).to_sec())<.1
 t=buf.lookup_transform("base_link","knife_control",rospy.Time(0),rospy.Duration(.03))
 assert abs((rospy.Time.now()-t.header.stamp).to_sec())<.1
 tr=t.transform;pose=[tr.translation.x,tr.translation.y,tr.translation.z,tr.rotation.x,tr.rotation.y,tr.rotation.z,tr.rotation.w]
 out,ft=transform_msg(w,buf.lookup_transform("ati_sensing_frame",w.header.frame_id,rospy.Time(0),rospy.Duration(.03)))
 q=[j.position[j.name.index(n)] for n in names];qd=[j.velocity[j.name.index(n)] for n in names]
 assert all(math.isfinite(v) for v in pose+q+qd+ft),"Nonfinite feedback"
 raw=w.wrench;assert norm([raw.force.x,raw.force.y,raw.force.z])<8 and norm([raw.torque.x,raw.torque.y,raw.torque.z])<.5,"Wrench guard"
 assert max(map(abs,qd))<.03,"Joint speed guard"
 if args.execute:runtime.check()
 return dict(q=q,qd=qd,pose=pose,wrench=ft,stamp=w.header.stamp.to_sec()),out
def write(p,x):
 tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def watch():
 while not finish.wait(.01):
  if active and time.monotonic()-heartbeat>.15:
   abort.set()
   try:stop()
   except Exception:pass
   return
from native_runtime import NativeRuntime
runtime=NativeRuntime()
time.sleep(2)
controllers=rospy.ServiceProxy("/controller_manager/list_controllers",ListControllers)().controller
assert not any(c.state=="running" and any(r.resources for r in c.claimed_resources) for c in controllers),"Motion controller already active"
samples=[]
for _ in range(10):
 s,_=snapshot();assert max(map(abs,s["qd"]))<.001;samples.append(s);time.sleep(.05)
origin=samples[-1];quat0=origin["pose"][3:]
(ipc/"robot.urdf").write_text(rospy.get_param("/robot_description"))
t=buf.lookup_transform("knife_control","knife_mount",rospy.Time(0),rospy.Duration(.03)).transform
write(ipc/"feedback.json",dict(rows=samples,controller_parameters=params,control_from_mount=[t.translation.x,t.translation.y,t.translation.z,t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w]))
if args.goal_record:
 prior=json.loads(Path(args.goal_record).read_text())
 requested_goal=prior["origin"]["pose"]
 assert distance(requested_goal[:3],origin["pose"][:3])<.015,"Return target outside authorized range"
 f=json.loads((ipc/"feedback.json").read_text());f["commissioning_goal"]=requested_goal;write(ipc/"feedback.json",f)
if args.bounded_stiffness:
 f=json.loads((ipc/"feedback.json").read_text());f["commissioning_stiffness_bounds"]=[3000,5000];write(ipc/"feedback.json",f)
def bounds(s):
 assert distance(s["pose"][:3],origin["pose"][:3])<.02,"20 mm displacement guard"
 assert distance(s["pose"][:2],origin["pose"][:2])<.003,"3 mm lateral guard"
 assert max(abs(a-b) for a,b in zip(s["q"],origin["q"]))<.1,"Joint displacement guard"
 dot=abs(sum(a*b for a,b in zip(s["pose"][3:],quat0)))
 assert 2*math.acos(min(1.,dot))<math.radians(2),"Orientation guard"
def valid_response(z):
 assert "error" not in z,z.get("error")
 m=z["mapped"];assert len(m["target_pose"])==7
 assert all(math.isfinite(x) for v in m.values() for x in (v if isinstance(v,list) else [v]))
 assert abs(m["error_scale"]-.2)<1e-9
 assert 1000<=m["stiffness"][0]<=10000 and 40<=m["stiffness"][4]<=200
 assert m["stiffness"][1:4]==[2000.,2000.,80.] and m["stiffness"][5]==80.
 assert m["p_gains"][:2]==[.2,.2] and .0199<=m["p_gains"][2]<=.1001 and .199<=m["p_gains"][4]<=3.001
 assert m["d_gains"]==[0.]*6 and m["selection"]==[1.,1.,.5,1.,.5,1.]
 assert distance(m["target_pose"][:3],origin["pose"][:3])<.015
 assert distance(m["target_pose"][:2],origin["pose"][:2])<.002,"Mapped lateral target exceeds 2 mm"
 assert abs(sum(a*b for a,b in zip(m["target_pose"][3:],quat0)))>math.cos(math.radians(.5))

clients={};worker=None;servo_thread=None;servo_error=[];frames=[];records=[];command_lock=threading.Lock()
command=None;plan_stamp=None;last_good=0;heartbeat=time.monotonic();servo_started=None;goal_pose=None
out=dict(mode="continuous_fdcc_asynchronous_mpc_air_only",execute=args.execute,origin=origin,
 fixed_error_scale=.2,stiffness_bounds=([3000,5000] if args.bounded_stiffness else [1000,10000]),samples=64,horizon=4,planned_axes="vertical only; sawing/pitch fixed zero in prediction and execution",
 prediction_limit="coarse ideal servo, held wrench; planning slower than 50 ms; not a real-time cutting validation",
 limits=dict(force_N=8,torque_Nm=.5,displacement_m=.02,lateral_m=.003,maximum_plan_age_s=7),
 plans=[],status="starting",motion_commands_sent=0)
wp=rospy.Publisher("/fdcc_shared/ft_sensor_wrench",WrenchStamped,queue_size=1) if args.execute else None
tp=rospy.Publisher("/fdcc_shared/target_frame",PoseStamped,queue_size=1) if args.execute else None
axes=["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]
def set_mapped(m):
 updates={"solver":{"error_scale":m["error_scale"]},"stiffness":dict(zip(axes,m["stiffness"]))}
 updates.update({"pd_gains/"+a:{"p":m["p_gains"][i],"d":m["d_gains"][i]} for i,a in enumerate(axes)})
 for key,values in updates.items():
  got=clients[key].update_configuration(values)
  assert all(abs(float(got[k])-float(v))<1e-8 for k,v in values.items()),"Configuration readback mismatch"
def servo():
 global heartbeat,active,servo_started,last_good
 step_samples=[];next_frame=time.monotonic()+.05;next_target=0;target=None;count=0
 try:
  servo_started=time.monotonic()
  while not finish.is_set():
   now=time.monotonic();s,w=snapshot();bounds(s)
   with command_lock:cmd=copy.deepcopy(command);stamp=plan_stamp
   assert cmd is not None
   assert now-stamp<7,"Planner feedback older than 7 s"
   assert not abort.is_set(),"External watchdog stop"
   a=cmd["action"];assert a[4:]==[0.,0.]
   if now>=next_target:
    v=[s["pose"][k]+.8*(goal_pose[k]-s["pose"][k]) for k in range(3)]+goal_pose[3:]
    target=PoseStamped();target.header.frame_id="base_link"
    target.pose.position.x,target.pose.position.y,target.pose.position.z=v[:3]
    target.pose.orientation.x,target.pose.orientation.y,target.pose.orientation.z,target.pose.orientation.w=v[3:]
    next_target=now+.05
   if args.execute:
    target.header.stamp=rospy.Time.now();wp.publish(w);tp.publish(target);out["motion_commands_sent"]+=1
   step_samples.append(s);heartbeat=now
   # Supervisor checks this independently of planning and ROS callback thread.
   if count%10==0:write(ipc/"heartbeat.json",dict(monotonic=now,active=active))
   count+=1
   if now>=next_frame:
    wh=[z["wrench"] for z in step_samples[-3:]]
    while len(wh)<3:wh.insert(0,wh[0])
    frame=dict(sequence=len(frames),joint_q=s["q"],joint_qd=s["qd"],wrench_history=wh,
     peak_wrench=[max(abs(z["wrench"][k]) for z in step_samples) for k in range(6)],
     last_wrench=s["wrench"],action=a,monotonic=now,pose=s["pose"],plan_sequence=cmd["sequence"])
    frames.append(frame);records.extend(step_samples);step_samples=[];next_frame=now+.05
    error=distance(s["pose"][:3],goal_pose[:3])
    angle=2*math.acos(min(1.,abs(sum(a*b for a,b in zip(s["pose"][3:],goal_pose[3:])))))
    last_good=last_good+1 if error<.001 and angle<math.radians(1) else 0
    if last_good>=11:out["status"]="air_goal_reached";finish.set();break
    if len(frames)%20==0:
     print(json.dumps(dict(elapsed=round(now-servo_started,2),distance_to_goal_mm=error*1000,delta_mm=[1000*(a-b) for a,b in zip(s["pose"][:3],origin["pose"][:3])],gain=.2,plan=cmd["sequence"])),flush=True)
   time.sleep(.004)
 except Exception as e:
  servo_error.append(str(e));out["status"]="aborted";out["error"]=str(e);abort.set();finish.set()
 finally:
  if args.execute:out["servo_stop_ok"]=stop()
  active=False
  write(ipc/"heartbeat.json",dict(monotonic=time.monotonic(),active=False))
try:
 if args.execute:
  for key in ["solver","stiffness"]+["pd_gains/"+a for a in axes]:clients[key]=Client("/fdcc_shared/"+key,timeout=3)
 workerlog=(ipc/"planner.log").open("w")
 worker=subprocess.Popen([PLANNER_PYTHON,str(MODULE/"scripts/continuous_planner.py"),str(ipc)],stdout=workerlog,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+60
 while not (ipc/"ready.json").exists():
  assert worker.poll() is None,"Planner startup failed"
  assert time.monotonic()<deadline,"Planner warmup timeout"
  snapshot();time.sleep(.05)
 ready=json.loads((ipc/"ready.json").read_text());goal_pose=ready["goal"];out["planner_ready"]=ready
 assert distance(goal_pose[:3],origin["pose"][:3])<.015
 requested_frame=-1;seq=0;pending=False;started=None
 while True:
  assert worker.poll() is None,"Planner process exited"
  if not pending:
   newframes=copy.deepcopy(frames[requested_frame+1:]);stamp=time.monotonic() if not newframes else newframes[-1]["monotonic"]
   write(ipc/("request_%03d.json"%seq),dict(sequence=seq,frames=newframes,snapshot_monotonic=stamp))
   requested_frame+=len(newframes);pending=True;response=ipc/("response_%03d.json"%seq);deadline=time.monotonic()+8
  if response.exists():
   z=json.loads(response.read_text());valid_response(z);assert z["action"][4:]==[0.,0.]
   assert time.monotonic()-z["input_monotonic"]<7,"Stale initial/result plan"
   if args.execute:set_mapped(z["mapped"])
   with command_lock:command=z;plan_stamp=z["input_monotonic"]
   out["plans"].append(z)
   if started is None:
    s,w=snapshot();bounds(s)
    assert distance(s["pose"][:3],origin["pose"][:3])<.0005,"Robot moved before activation"
    if args.execute:
     wp.publish(w);req=SwitchControllerRequest();req.start_controllers=["fdcc_shared"];req.strictness=2;req.timeout=1.
     heartbeat=time.monotonic();active=True
     assert rospy.ServiceProxy("/controller_manager/switch_controller",SwitchController)(req).ok
    started=time.monotonic();out["activation_monotonic"]=started
    servo_thread=threading.Thread(target=servo,daemon=True);servo_thread.start()
    threading.Thread(target=watch,daemon=True).start()
   seq+=1;pending=False
  if started is not None and (finish.is_set() or time.monotonic()-started>=args.seconds):
   if out["status"]=="starting":out["status"]="duration_complete_without_goal"
   break
  assert time.monotonic()<deadline,"Planner response timeout"
  if started is None:snapshot()
  time.sleep(.01)
 if servo_error:raise RuntimeError(servo_error[0])
except Exception as e:
 out["status"]="aborted";out["error"]=str(e);print("ERROR",str(e),flush=True)
finally:
 finish.set()
 if args.execute:out["final_stop_ok"]=stop();active=False
 if servo_thread:servo_thread.join(timeout=2)
 for key,c in clients.items():
  original=params
  for part in key.split("/"):original=original[part]
  try:
   available=c.get_configuration();values={k:v for k,v in original.items() if k in available and not isinstance(v,dict)}
   got=c.update_configuration(values)
   assert all(got[k]==v for k,v in values.items()),"Restore mismatch"
  except Exception as e:out.setdefault("restore_errors",{})[key]=str(e)
 (ipc/"stop").touch()
 if worker:
  try:worker.wait(timeout=3)
  except subprocess.TimeoutExpired:worker.terminate()
 out["controllers"]=[[c.name,c.state] for c in rospy.ServiceProxy("/controller_manager/list_controllers",ListControllers)().controller]
 try:out["final_feedback"]=snapshot()[0]
 except Exception as e:out["final_feedback_error"]=str(e)
 out["frames"]=frames;out["samples"]=records
 if records:
  out["max_displacement_mm"]=1000*max(distance(z["pose"][:3],origin["pose"][:3]) for z in records)
  out["final_delta_mm"]=[1000*(a-b) for a,b in zip(records[-1]["pose"][:3],origin["pose"][:3])]
  out["final_goal_error_mm"]=1000*distance(records[-1]["pose"][:3],goal_pose[:3])
  out["peak_force_N"]=max(norm(z["wrench"][:3]) for z in records)
  out["peak_torque_Nm"]=max(norm(z["wrench"][3:]) for z in records)
 runtime.close()
 write(ipc/"result.json",out);print(json.dumps({k:v for k,v in out.items() if k not in ("frames","samples","plans")},indent=2),flush=True)
 print(str(ipc),flush=True)

if out["status"] == "aborted" or out.get("restore_errors"): sys.exit(1)
