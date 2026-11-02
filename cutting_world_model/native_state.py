"""Decode source controller/filter snapshots at their completed callback boundaries."""
import json

import numpy as np


NATIVE_STATE_LAYOUT = 'native_ros_recurrent_v1'


def decode_controller_snapshot(response):
    if not response.success:
        raise RuntimeError(response.message)
    data = np.asarray(json.loads(response.message), dtype=np.float64)
    if data.ndim != 1 or len(data) < 4 or data[0] != 1 or not np.isfinite(data).all():
        raise ValueError('Incomplete or non-finite recurrent-state packet')
    joints = int(data[3])
    if joints < 1 or joints != data[3] or len(data) != 44 + 5 * joints:
        raise ValueError('Invalid recurrent-state joint count')
    fields = dict(schema_version=data[:1].copy(), sequence=data[1:2].copy(), stamp=data[2:3].copy())
    layout = [(name, joints) for name in (
        'controller_measured_q', 'controller_measured_qd', 'controller_q',
        'controller_velocity', 'controller_previous_position_command')]
    layout += [('controller_previous_error', 6), ('controller_cached_tip_pose', 7),
               ('controller_target_pose', 7), ('controller_sensor_wrench', 6),
               ('controller_target_wrench', 6), ('controller_tare_compensation', 6),
               ('controller_command_valid', 1), ('hardware_stamp', 1)]
    offset = 4
    for name, width in layout:
        fields[name] = data[offset:offset + width].copy()
        offset += width
    return fields


def decode_filter_snapshot(response):
    if not response.success:
        raise RuntimeError(response.message)
    fields = {name: np.asarray(value, dtype=np.float64)
              for name, value in json.loads(response.message).items()}
    shapes = dict(schema_version=(1,), sequence=(1,), stamp=(1,), raw_stamp=(1,), topic2_stamp=(1,),
                  filter_history_count=(1,), filter_initialized=(1,), filter_offset=(6,),
                  filter_added_wrench=(6,), filter_flags=(2,))
    for name, shape in shapes.items():
        if name not in fields or fields[name].shape != shape:
            raise ValueError('Invalid filter snapshot field: ' + name)
    for name in ('filter_history', 'filter_zi'):
        if name not in fields or fields[name].ndim != 2 or fields[name].shape[1] != 6:
            raise ValueError('Invalid filter snapshot field: ' + name)
    if fields['schema_version'].tolist() != [1.] or any(not np.isfinite(v).all() for v in fields.values()):
        raise ValueError('Unsupported or non-finite filter snapshot')
    return fields


class NativeStateRecorder:
    def __init__(self, arm, namespace):
        import rospy
        from std_srvs.srv import Trigger
        self.clock = rospy.get_time
        namespace = '/' + namespace.strip('/')
        self.topics = {
            'controller': namespace + '/cartesian_compliance_controller/recurrent_state',
            'filter': namespace + '/' + arm.ft_topic.strip('/') + '/filtered/recurrent_state',
        }
        self.controller_service = rospy.ServiceProxy(self.topics['controller'], Trigger)
        self.controller_service.wait_for_service(timeout=60)
        self.filter_service = rospy.ServiceProxy(self.topics['filter'], Trigger)
        self.filter_service.wait_for_service(timeout=60)

    def read(self, boundary):
        controller = decode_controller_snapshot(self.controller_service())
        sensor = decode_filter_snapshot(self.filter_service())
        if controller['controller_command_valid'][0] != 1:
            raise RuntimeError('Native controller has not completed its first hardware write after starting')
        frame = {name: value for name, value in controller.items() if name.startswith('controller_')}
        frame.update({name: value for name, value in sensor.items() if name.startswith('filter_')})
        for name in ('filter_history', 'filter_zi'):
            frame[name] = frame[name].reshape(-1, 6)
        frame['filter_added_age'] = np.array([boundary - sensor['topic2_stamp'][0]])
        frame['filter_raw_age'] = np.array([boundary - sensor['raw_stamp'][0]])
        frame['controller_sample_age'] = np.array([boundary - controller['stamp'][0]])
        frame['filter_sample_age'] = np.array([boundary - sensor['stamp'][0]])
        frame['controller_hardware_ros_time'] = controller['hardware_stamp'].copy()
        frame['observation_ros_time'] = np.array(boundary)
        for source, packet in (('controller', controller), ('filter', sensor)):
            frame[source + '_ros_time'] = packet['stamp'].copy()
            frame[source + '_sequence'] = packet['sequence'].copy()
        return frame
