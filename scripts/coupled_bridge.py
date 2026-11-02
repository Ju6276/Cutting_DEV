#!/usr/bin/env python3
"""Expose synchronized DiSECt snapshots in a separate SliceIt ROS instance."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rospy
from std_srvs.srv import Trigger, TriggerResponse

from collect_disect import snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sliceit-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(rospy.myargv()[1:])
    sys.path.insert(0, str(Path(args.sliceit_root) / "scripts"))
    from public_ros_bridge import PublicBridge

    class RecordingBridge(PublicBridge):
        def __init__(self):
            super().__init__(args.config, args.output)
            self.snapshot_count = 0
            self.snapshot_service = rospy.Service("/world_model/snapshot", Trigger, self.capture)

        def capture(self, request):
            with self.lock:
                if self.sim is None:
                    return TriggerResponse(False, "DiSECt has not been loaded")
                directory = self.output / "snapshots"
                directory.mkdir(exist_ok=True)
                path = directory / ("%06d.npz" % self.snapshot_count)
                self.snapshot_count += 1
                frame = snapshot(self.sim)
                frame["spring_reference_ke"] = self.initial_ke.copy()
                frame["ros_time"] = np.array(rospy.get_time())
                np.savez_compressed(path, **frame)
                return TriggerResponse(True, json.dumps({"path": str(path), "episode": self.episode}))

    rospy.init_node("world_model_disect_bridge")
    bridge = RecordingBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
