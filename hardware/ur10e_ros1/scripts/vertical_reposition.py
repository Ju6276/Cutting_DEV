"""Plan an explicit base-Z reposition; send only with --execute."""
import argparse,json,time,math,threading,xml.etree.ElementTree as ET,sys
from pathlib import Path
import rospy,tf2_ros,PyKDL,actionlib
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction,FollowJointTrajectoryGoal
from controller_manager_msgs.srv import ListControllers,SwitchController,SwitchControllerRequest
from settings import MODULE, record_directory
from native_runtime import NativeRuntime
ap=argparse.ArgumentParser();ap.add_argument("--execute",action="store_true");ap.add_argument("--delta-z-m",type=float,required=True);a=ap.parse_args()
assert 0 < abs(a.delta_z_m) <= .2, "Reposition limited to 200 mm"
root=record_directory("reposition")
record=root/(("execute_" if a.execute else "plan_")+time.strftime("%Y%m%d_%H%M%S")+".json")
rospy.init_node("lower_table_vertical_reposition",anonymous=True)
buf=tf2_ros.Buffer();listener=tf2_ros.TransformListener(buf)
data={};lock=threading.Lock()
def cb(k,m):
 with lock:data[k]=(m,time.monotonic())
subs=[rospy.Subscriber("/joint_states",JointState,lambda m:cb("joint",m),queue_size=1),
 rospy.Subscriber("/wrench",WrenchStamped,lambda m:cb("wrench",m),queue_size=1)]
runtime=NativeRuntime();time.sleep(1.5)
names=rospy.get_param("/fdcc_shared/joints")
def norm(x):return math.sqrt(sum(v*v for v in x))
def read():
 with lock:d=dict(data)
 now=time.monotonic()
 for k in ("joint","wrench"):assert k in d and now-d[k][1]<.1,"Stale "+k
 j=d["joint"][0];w=d["wrench"][0];t=buf.lookup_transform("base_link","knife_control",j.header.stamp,rospy.Duration(.03))
 assert abs((rospy.Time.now()-t.header.stamp).to_sec())<.1
 t=t.transform;ww=w.wrench
 s=dict(q=[j.position[j.name.index(n)] for n in names],qd=[j.velocity[j.name.index(n)] for n in names],
 pose=[t.translation.x,t.translation.y,t.translation.z,t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w],
 wrench=[ww.force.x,ww.force.y,ww.force.z,ww.torque.x,ww.torque.y,ww.torque.z],monotonic=now)
 assert all(math.isfinite(v) for k in ("q","qd","pose","wrench") for v in s[k]),"Nonfinite feedback"
 return s
origin=read();assert max(map(abs,origin["qd"]))<.001,"Start not stationary"
xml=ET.fromstring(rospy.get_param("/robot_description"))
by_child={j.find("child").get("link"):j for j in xml.findall("joint")}
segments=[];cursor="knife_control"
while cursor!="base_link":
 j=by_child[cursor];segments.append(j);cursor=j.find("parent").get("link")
chain=PyKDL.Chain();low=[];high=[];chain_names=[]
for j in reversed(segments):
 o=j.find("origin")
 xyz=[float(v) for v in (o.get("xyz","0 0 0") if o is not None else "0 0 0").split()]
 rpy=[float(v) for v in (o.get("rpy","0 0 0") if o is not None else "0 0 0").split()]
 f=PyKDL.Frame(PyKDL.Rotation.RPY(*rpy),PyKDL.Vector(*xyz))
 kind=j.get("type");name=j.get("name")
 if kind=="fixed":joint=PyKDL.Joint(name,getattr(PyKDL.Joint,"None"))
 else:
  assert kind in ("revolute","continuous")
  axis=j.find("axis");axis=[float(v) for v in (axis.get("xyz","1 0 0") if axis is not None else "1 0 0").split()]
  joint=PyKDL.Joint(name,f.p,f.M*PyKDL.Vector(*axis),PyKDL.Joint.RotAxis)
  lim=j.find("limit");low.append(float(lim.get("lower","-6.283185307")));high.append(float(lim.get("upper","6.283185307")));chain_names.append(name)
 chain.addSegment(PyKDL.Segment(j.find("child").get("link"),joint,f))
assert chain_names==names
def ja(x):
 out=PyKDL.JntArray(len(x))
 for i,v in enumerate(x):out[i]=v
 return out
fk=PyKDL.ChainFkSolverPos_recursive(chain);f0=PyKDL.Frame()
assert fk.JntToCart(ja(origin["q"]),f0)>=0
tf=PyKDL.Frame(PyKDL.Rotation.Quaternion(*origin["pose"][3:]),PyKDL.Vector(*origin["pose"][:3]))
print("FK vs TF discrepancy",PyKDL.diff(f0,tf).vel.Norm(),PyKDL.diff(f0,tf).rot.Norm(),flush=True)
# ROS TF is interpolated from the state publisher; do not require 10 um
# equality to a distinct encoder sample. Internal IK residual stays 10 um.
assert PyKDL.diff(f0,tf).vel.Norm()<2e-4 and PyKDL.diff(f0,tf).rot.Norm()<1e-3,"URDF/FK does not match actual TF"
qmin,qmax=ja(low),ja(high)
velsolver=PyKDL.ChainIkSolverVel_pinv(chain)
ik=PyKDL.ChainIkSolverPos_NR_JL(chain,qmin,qmax,fk,velsolver,200,1e-8)
duration=max(4.,abs(a.delta_z_m)/.01);dt=.05;count=round(duration/dt);qs=[];poses=[];seed=ja(origin["q"]);max_error=0.
for i in range(count+1):
 u=i/count;s=10*u**3-15*u**4+6*u**5
 target=PyKDL.Frame(f0.M,PyKDL.Vector(f0.p[0],f0.p[1],f0.p[2]+a.delta_z_m*s));outq=PyKDL.JntArray(6)
 code=ik.CartToJnt(seed,target,outq);assert code>=0,("IK failed",i,code)
 q=[outq[k] for k in range(6)]
 assert all(lo+1e-4<v<hi-1e-4 for lo,v,hi in zip(low,q,high)),"Joint limit"
 if qs:assert max(abs(x-y) for x,y in zip(q,qs[-1]))<.01,"IK branch discontinuity"
 actual=PyKDL.Frame();assert fk.JntToCart(outq,actual)>=0
 err=PyKDL.diff(actual,target);max_error=max(max_error,err.vel.Norm())
 assert err.vel.Norm()<1e-5 and err.rot.Norm()<1e-5
 qs.append(q);poses.append([actual.p[k] for k in range(3)]);seed=outq
velocities=[[0.]*6]+[[(qs[i+1][k]-qs[i-1][k])/(2*dt) for k in range(6)] for i in range(1,count)]+[[0.]*6]
max_qd=max(abs(v) for row in velocities for v in row)
if max_qd>.06:
 scale=max_qd/.06;duration*=scale;dt*=scale;velocities=[[v/scale for v in row] for row in velocities]
assert duration<=40,"Required trajectory too slow for this commissioning budget"
goal=FollowJointTrajectoryGoal();goal.trajectory.joint_names=names
for i,(q,v) in enumerate(zip(qs,velocities)):
 p=JointTrajectoryPoint();p.positions=q;p.velocities=v;p.time_from_start=rospy.Duration(i*dt);goal.trajectory.points.append(p)
result=dict(mode="free_space_vertical_position_reposition_not_MPC",execute=a.execute,origin=origin,
 target_position_m=poses[-1],displacement_m=[0,0,a.delta_z_m],duration_s=duration,points=len(qs),
 fk_max_error_m=max_error,planned_max_joint_speed=max(abs(v) for row in velocities for v in row),
 planned_positions=qs,status="plan_verified",samples=[])
record.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ("planned_positions","samples")},indent=2),flush=True)
if not a.execute:runtime.close();sys.exit(0)
runtime.check();now=read()
assert max(abs(x-y) for x,y in zip(now["q"],origin["q"]))<.001,"Robot moved while planning"
assert norm(now["wrench"][:3])<3 and norm(now["wrench"][3:])<.25,"Unexpected initial wrench"
states=rospy.ServiceProxy("/controller_manager/list_controllers",ListControllers)().controller
assert not any(c.state=="running" and any(r.resources for r in c.claimed_resources) for c in states),"Motion controller already running"
def switch(start=(),stop=()):
 req=SwitchControllerRequest();req.start_controllers=list(start);req.stop_controllers=list(stop);req.strictness=2;req.timeout=1.
 return rospy.ServiceProxy("/controller_manager/switch_controller",SwitchController)(req).ok
client=actionlib.SimpleActionClient("/scaled_pos_joint_traj_controller/follow_joint_trajectory",FollowJointTrajectoryAction)
heartbeat=time.monotonic();finished=threading.Event();active=False;watch_error=[]
def watchdog():
 while not finished.wait(.02):
  if active and time.monotonic()-heartbeat>.25:
   watch_error.append("Execution monitor heartbeat missed")
   try:client.cancel_goal();switch(stop=["scaled_pos_joint_traj_controller"])
   except Exception:pass
   return
threading.Thread(target=watchdog,daemon=True).start()
try:
 assert switch(start=["scaled_pos_joint_traj_controller"])
 assert client.wait_for_server(rospy.Duration(3)),"Trajectory action unavailable"
 runtime.check();read()
 active=True;heartbeat=time.monotonic()
 goal.trajectory.header.stamp=rospy.Time.now()+rospy.Duration(.2)
 client.send_goal(goal);start=time.monotonic();next_log=0
 while time.monotonic()-start<duration+5:
  runtime.check();s=read();heartbeat=time.monotonic();assert not watch_error,watch_error
  d=[x-y for x,y in zip(s["pose"][:3],origin["pose"][:3])]
  assert norm(d[:2])<.005,"Lateral deviation"
  assert min(0,a.delta_z_m)-.005<d[2]<max(0,a.delta_z_m)+.005,"Vertical bound"
  assert max(map(abs,s["qd"]))<.1,"Joint speed"
  assert 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(s["pose"][3:],origin["pose"][3:])))))<math.radians(2),"Orientation"
  assert norm(s["wrench"][:3])<8 and norm(s["wrench"][3:])<.5,"Force/torque guard"
  result["samples"].append(s)
  if time.monotonic()>=next_log:
   print(json.dumps(dict(elapsed=round(time.monotonic()-start,1),delta_mm=[round(x*1000,2) for x in d],force_N=round(norm(s["wrench"][:3]),2))),flush=True);next_log=time.monotonic()+2
  if client.get_state() in (2,3,4,5,8,9):break
  time.sleep(.01)
 action_result=client.get_result();result["action_state"]=client.get_state()
 assert action_result is not None and client.get_state()==3 and action_result.error_code==0,("Trajectory unsuccessful",client.get_state(),str(action_result))
 final=read();result["final"]=final
 result["actual_displacement_mm"]=[1000*(x-y) for x,y in zip(final["pose"][:3],origin["pose"][:3])]
 result["target_error_mm"]=1000*norm([x-y for x,y in zip(final["pose"][:3],poses[-1])])
 assert result["target_error_mm"]<2,"Final position error"
 result["status"]="completed"
except Exception as e:
 result["status"]="aborted";result["error"]=str(e);print("ABORT",str(e),flush=True)
finally:
 client.cancel_goal();result["stop_ok"]=switch(stop=["scaled_pos_joint_traj_controller"]);active=False;finished.set();runtime.close()
 result["controllers"]=[(c.name,c.state) for c in rospy.ServiceProxy("/controller_manager/list_controllers",ListControllers)().controller]
 record.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ("samples","planned_positions","origin","final")},indent=2),flush=True)
 print(str(record),flush=True)

if result.get("status") == "aborted": sys.exit(1)
