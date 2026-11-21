#!/usr/bin/env python3
"""Read-only ROS snapshot. No publishers or controller switching."""
import argparse
import json
import time
from pathlib import Path
import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import ListControllers

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output)
if out.exists():
    p.error('--output must be a new directory to preserve previous evidence')
out.mkdir(parents=True)
rospy.init_node('capture_mpc_feedback', anonymous=True, disable_signals=True)
buf = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(buf)
rows = []
def receive(msg):
    rows.append(dict(received=time.time(), stamp=msg.header.stamp.to_sec(),
                     names=list(msg.name), q=list(msg.position), qd=list(msg.velocity)))
sub = rospy.Subscriber('/joint_states', JointState, receive, queue_size=1000)
time.sleep(4)
sub.unregister()
if len(rows) < 2:
    raise RuntimeError('No joint feedback received')
if time.time()-rows[-1]['received'] > 1:
    raise RuntimeError('Joint feedback stopped during capture')
rospy.wait_for_service('/controller_manager/list_controllers', timeout=5)
controllers = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)().controller
transforms = {}
for base, tip in [('base_link','tool0'), ('tool0','tool0_controller')]:
    try:
        t = buf.lookup_transform(base, tip, rospy.Time(0), rospy.Duration(1))
        v, q = t.transform.translation, t.transform.rotation
        transforms[base+' <- '+tip] = dict(xyz=[v.x,v.y,v.z], xyzw=[q.x,q.y,q.z,q.w])
    except Exception as exc:
        transforms[base+' <- '+tip] = dict(error=str(exc))
result = dict(captured_unix=time.time(), count=len(rows),
    rate_hz=(len(rows)-1)/(rows[-1]['received']-rows[0]['received']),
    max_stamp_age_s=max(r['received']-r['stamp'] for r in rows),
    last=rows[-1], transforms=transforms,
    controllers=[dict(name=c.name,state=c.state,type=c.type) for c in controllers],
    fdcc_parameters=rospy.get_param('/fdcc'))
(out/'feedback.json').write_text(json.dumps(result, indent=2))
(out/'joints.json').write_text(json.dumps(rows))
(out/'robot_description.urdf').write_text(rospy.get_param('/robot_description'))
print(json.dumps(result, indent=2))
