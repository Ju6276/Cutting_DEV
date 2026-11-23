#!/usr/bin/env python3
"""Read-only snapshot: ordered joints and installed tool frames; no motion."""
import argparse
import datetime
import json
from pathlib import Path
import time
import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import ListControllers
from settings import record_directory

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--output", type=Path)
a = ap.parse_args()
rospy.init_node("cutting_capture_pose", anonymous=True)
buf = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(buf)
names = rospy.get_param("/fdcc_shared/joints")
time.sleep(.5)
j = rospy.wait_for_message("/joint_states", JointState, timeout=3)
if abs((rospy.Time.now() - j.header.stamp).to_sec()) > .1:
    raise RuntimeError("Stale joints")
if max(abs(j.velocity[j.name.index(n)]) for n in names) > .001:
    raise RuntimeError("Capture requires stationary joints")
poses = {}
for frame in ("knife_control", "knife_mount", "knife_tip", "ati_sensing_frame", "tool0_controller"):
    msg = buf.lookup_transform("base_link", frame, j.header.stamp, rospy.Duration(1))
    p, q = msg.transform.translation, msg.transform.rotation
    poses[frame] = dict(position_m=[p.x,p.y,p.z], quaternion_xyzw=[q.x,q.y,q.z,q.w])
out = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           reference_frame="base_link", control_frame="knife_control", joint_names=names,
           joint_positions_rad=[j.position[j.name.index(n)] for n in names], poses=poses,
           meaning="Reference only; no automatic motion or cutting endpoint calibration",
           controllers=[dict(name=c.name,state=c.state) for c in
               rospy.ServiceProxy("/controller_manager/list_controllers",ListControllers)().controller])
path = a.output or record_directory("poses") / (time.strftime("%Y%m%d_%H%M%S") + ".json")
with path.open("x") as f:
    json.dump(out,f,indent=2,allow_nan=False)
print(path)
