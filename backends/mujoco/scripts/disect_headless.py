"""Import the original DiSECt cutting package on a headless server.

disect/cutting/__init__.py imports ros_visualizer, which calls matplotlib.use('Qt5agg') at import
time and fails without a display. None of the scripts here use the ROS visualizer.
"""
import sys
import types


def import_disect_cutting(disect_root):
    sys.path.insert(0, str(disect_root))
    if 'disect.cutting.ros_visualizer' not in sys.modules:
        stub = types.ModuleType('disect.cutting.ros_visualizer')
        stub.ROSVisualizer = None
        sys.modules['disect.cutting.ros_visualizer'] = stub
    import disect.cutting
    return disect.cutting
