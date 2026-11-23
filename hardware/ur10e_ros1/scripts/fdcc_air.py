#!/usr/bin/env python3
"""Bounded airborne FDCC commissioning. Default is read-only; never a cutting runner."""
import argparse,copy,json,math,threading,time
from pathlib import Path
from settings import record_directory
from native_runtime import NativeRuntime
import rospy,tf2_ros,PyKDL
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped,PoseStamped
from std_msgs.msg import Bool
from controller_manager_msgs.srv import ListControllers,SwitchController,SwitchControllerRequest

ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true')
ap.add_argument('--distance-m',type=float,default=.02);ap.add_argument('--speed-m-s',type=float,default=.005)
ap.add_argument('--position-only',action='store_true')
ap.add_argument('--cycle-origin-record')
args=ap.parse_args()
assert 0<args.distance_m<=.02 and 0<args.speed_m_s<=.01
ramp_s=args.distance_m/args.speed_m_s;hold_s=2.;total_s=2*ramp_s+2*hold_s
root=record_directory("fdcc_air")
rospy.init_node('native_fdcc_air_check',anonymous=True)
assert rospy.get_param('/fdcc_shared/tool/mass') == 0, 'Native compensated wrench requires zero extra payload compensation'
buf=tf2_ros.Buffer();listener=tf2_ros.TransformListener(buf)
force_reference=None
lock=threading.Lock();data={};abort=threading.Event();finished=threading.Event()
def callback(key,msg):
 with lock:data[key]=(msg,time.monotonic())
subs=[rospy.Subscriber('/joint_states',JointState,lambda m:callback('joint',m),queue_size=1),
      rospy.Subscriber('/wrench',WrenchStamped,lambda m:callback('wrench',m),queue_size=1),
      rospy.Subscriber('/ur_hardware_interface/robot_program_running',Bool,lambda m:callback('program',m),queue_size=1)]
def distance(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def xyz(t):return [t.translation.x,t.translation.y,t.translation.z]
def transform_wrench(m,t):
 p=t.transform.translation;q=t.transform.rotation;w=m.wrench
 f=PyKDL.Frame(PyKDL.Rotation.Quaternion(q.x,q.y,q.z,q.w),PyKDL.Vector(p.x,p.y,p.z))
 z=f*PyKDL.Wrench(PyKDL.Vector(w.force.x,w.force.y,w.force.z),PyKDL.Vector(w.torque.x,w.torque.y,w.torque.z))
 out=WrenchStamped();out.header.stamp=m.header.stamp;out.header.frame_id='ati_sensing_frame'
 out.wrench.force.x,out.wrench.force.y,out.wrench.force.z=[z.force[i] for i in range(3)]
 out.wrench.torque.x,out.wrench.torque.y,out.wrench.torque.z=[z.torque[i] for i in range(3)]
 return out
def snapshot():
 with lock:state=dict(data)
 now=time.monotonic()
 for key in ['joint','wrench']:
  if key not in state or now-state[key][1]>.1:raise RuntimeError('Missing/stale '+key)
 j=state['joint'][0];w=state['wrench'][0]
 if abs((rospy.Time.now()-w.header.stamp).to_sec())>.1:raise RuntimeError('Stale sensor timestamp')
 vals=list(j.position)+list(j.velocity)+[w.wrench.force.x,w.wrench.force.y,w.wrench.force.z,w.wrench.torque.x,w.wrench.torque.y,w.wrench.torque.z]
 if not all(math.isfinite(v) for v in vals):raise RuntimeError('Nonfinite feedback')
 fn=math.sqrt(sum(v*v for v in vals[-6:-3]));tn=math.sqrt(sum(v*v for v in vals[-3:]))
 absolute_force_limit=20. if args.position_only else 8.
 if fn>absolute_force_limit or tn>.5:raise RuntimeError('Absolute air-test wrench bound exceeded: %s %s'%(fn,tn))
 if force_reference is not None:
  delta=math.sqrt(sum((a-b)**2 for a,b in zip(vals[-6:-3],force_reference)))
  if delta>8.:raise RuntimeError('Force change from stationary baseline exceeds 8 N: '+str(delta))
 tf=buf.lookup_transform('base_link','knife_control',rospy.Time(0),rospy.Duration(.03))
 if abs((rospy.Time.now()-tf.header.stamp).to_sec())>.1:raise RuntimeError('Stale tool transform')
 return state,j,w,tf
def stop_controller():
 req=SwitchControllerRequest();req.stop_controllers=['fdcc_shared'];req.strictness=1;req.timeout=1.0
 return rospy.ServiceProxy('/controller_manager/switch_controller',SwitchController)(req).ok
runtime=NativeRuntime()
time.sleep(2)
controllers=rospy.ServiceProxy('/controller_manager/list_controllers',ListControllers)().controller
for c in controllers:
 if c.state=='running' and any(r.resources for r in c.claimed_resources):raise RuntimeError('Another motion controller is running: '+c.name)
state,j,w,pose=snapshot()
if max(abs(x) for x in j.velocity)>.001:raise RuntimeError('Robot not stationary')
if args.position_only:
 stiffness=rospy.get_param('/fdcc_shared/stiffness')
 assert all(stiffness[k]==1. for k in ['sel_x','sel_y','sel_z','sel_ax','sel_ay','sel_az']), 'Position-only configuration not active'
 force_samples=[]
 for _ in range(50):
  _,_,sample,_=snapshot()
  force_samples.append([sample.wrench.force.x,sample.wrench.force.y,sample.wrench.force.z])
  time.sleep(.01)
 force_reference=[sum(v[i] for v in force_samples)/len(force_samples) for i in range(3)]
 state,j,w,pose=snapshot()
start_q=dict(zip(j.name,j.position));start_pos=xyz(pose.transform)
start_rotation=pose.transform.rotation
cycle_origin=list(start_pos);approach_s=0.
if args.cycle_origin_record:
 previous=json.loads(Path(args.cycle_origin_record).read_text())
 cycle_origin=previous['start_position_base_link_m']
 assert len(cycle_origin)==3 and all(math.isfinite(x) for x in cycle_origin)
 assert distance(cycle_origin,start_pos)<.025,'Recovery exceeds original envelope'
 assert distance(cycle_origin[:2],start_pos[:2])<.002,'Recovery lateral distance exceeds 2 mm'
 approach_s=distance(cycle_origin,start_pos)/args.speed_m_s+hold_s
 total_s+=approach_s
target=PoseStamped();target.header.frame_id='base_link'
target.pose.position.x,target.pose.position.y,target.pose.position.z=start_pos
target.pose.orientation=copy.deepcopy(pose.transform.rotation)
if args.execute:runtime.check()
wp=rospy.Publisher('/fdcc_shared/ft_sensor_wrench',WrenchStamped,queue_size=1) if args.execute else None
tp=rospy.Publisher('/fdcc_shared/target_frame',PoseStamped,queue_size=1) if args.execute else None
rows=[];report={'execute':args.execute,'start_position_base_link_m':start_pos,'motion':{'base_z_m':args.distance_m,'speed_m_s':args.speed_m_s,'ramp_s':ramp_s,'hold_s':hold_s,'orientation':'fixed'},'guards':'feedback age 100 ms, force 8 N, torque 0.5 Nm, tool displacement 25 mm, lateral displacement 3 mm, orientation 2 deg, each joint 0.1 rad, joint speed 0.03 rad/s','note':'native UR compensated wrench transformed to ATI reference; FDCC extra gravity compensation disabled'}
report['solver_config']=rospy.get_param('/fdcc_shared/solver')
report['cycle_origin_base_link_m']=cycle_origin
report['approach_s']=approach_s
report['position_only']=args.position_only
report['force_reference_sensor_frame_N']=force_reference
report['guards']={'absolute_force_N':20. if args.position_only else 8.,'force_change_N':8. if args.position_only else None,'absolute_torque_Nm':.5,'max_displacement_m':args.distance_m+.005,'lateral_m':.003,'orientation_deg':2,'joint_displacement_rad':.1,'joint_speed_rad_s':.03,'feedback_timeout_s':.1}
report['stiffness_config']=rospy.get_param('/fdcc_shared/stiffness')
active=False;heartbeat=time.monotonic();failure=None
def watchdog():
 while not finished.wait(.025):
  if active and time.monotonic()-heartbeat>.2:
   abort.set()
   try:stop_controller()
   except Exception as e:report['watchdog_stop_error']=str(e)
   return
thread=threading.Thread(target=watchdog,daemon=True);thread.start()
try:
 if args.execute:
  time.sleep(.2)
  state,j,w,pose=snapshot()
  tf=buf.lookup_transform('ati_sensing_frame',w.header.frame_id,rospy.Time(0),rospy.Duration(.03))
  wp.publish(transform_wrench(w,tf));target.header.stamp=rospy.Time.now();tp.publish(target);time.sleep(.05)
  req=SwitchControllerRequest();req.start_controllers=['fdcc_shared'];req.strictness=2;req.timeout=1.0
  heartbeat=time.monotonic();active=True
  assert rospy.ServiceProxy('/controller_manager/switch_controller',SwitchController)(req).ok,'FDCC activation failed'
 start=time.monotonic()
 while time.monotonic()-start<total_s and not rospy.is_shutdown():
  if abort.is_set():raise RuntimeError('Watchdog stopped test')
  elapsed=time.monotonic()-start
  state,j,w,pose=snapshot()
  if args.execute:runtime.check()
  pos=xyz(pose.transform)
  if distance(pos,start_pos)>args.distance_m+.005:raise RuntimeError('Tool displacement guard')
  if max(abs(v-start_q[n]) for n,v in zip(j.name,j.position))>.1:raise RuntimeError('Joint displacement guard')
  if max(abs(v) for v in j.velocity)>.03:raise RuntimeError('Joint velocity guard')
  if distance(pos[:2],start_pos[:2])>.003:raise RuntimeError('Lateral displacement guard')
  rot=pose.transform.rotation
  dot=abs(sum(getattr(rot,k)*getattr(start_rotation,k) for k in ['x','y','z','w']))
  angle=2*math.acos(min(1.0,dot))
  if angle>math.radians(2):raise RuntimeError('Orientation guard')
  cycle_t=elapsed-approach_s
  if elapsed<approach_s:
   fraction=min(1.,elapsed/max(1e-6,approach_s-hold_s))
   desired=[a+fraction*(b-a) for a,b in zip(start_pos,cycle_origin)]
  else:
   offset=args.distance_m*min(cycle_t/ramp_s,1) if cycle_t<ramp_s+hold_s else args.distance_m*max(0,1-(cycle_t-ramp_s-hold_s)/ramp_s)
   desired=[cycle_origin[0],cycle_origin[1],cycle_origin[2]+offset]
  tf=buf.lookup_transform('ati_sensing_frame',w.header.frame_id,rospy.Time(0),rospy.Duration(.03))
  out=transform_wrench(w,tf)
  target.pose.position.x,target.pose.position.y,target.pose.position.z=desired;target.header.stamp=rospy.Time.now()
  if args.execute:wp.publish(out);tp.publish(target)
  rows.append({'elapsed_s':elapsed,'position':pos,'target_z':target.pose.position.z,'target_xyz':desired,'cycle_elapsed_s':cycle_t,'joint_q':list(j.position),'wrench':[out.wrench.force.x,out.wrench.force.y,out.wrench.force.z,out.wrench.torque.x,out.wrench.torque.y,out.wrench.torque.z]})
  heartbeat=time.monotonic();time.sleep(.01)
 report['status']='completed'
except Exception as e:
 failure=e;report['status']='aborted';report['error']=str(e)
finally:
 if active:
  try:report['controller_stopped']=stop_controller()
  except Exception as e:report['stop_error']=str(e)
 finished.set();report['rows']=rows
 if rows:
  report['max_z_m']=max(x['position'][2]-cycle_origin[2] for x in rows if x.get('cycle_elapsed_s',0)>=0) if any(x.get('cycle_elapsed_s',0)>=0 for x in rows) else 0.
  report['return_error_m']=distance(rows[-1]['position'],cycle_origin)
  report['tracking_passed']=report.get('status')=='completed' and report['max_z_m']>=.8*args.distance_m and report['return_error_m']<=.002
 name='air_execute' if args.execute else 'air_readonly'
 (root/(name+'_'+time.strftime('%Y%m%d_%H%M%S')+'.json')).write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
runtime.close()
if failure:raise failure
