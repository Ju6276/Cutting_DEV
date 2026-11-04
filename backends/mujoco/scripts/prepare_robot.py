"""Import the current SliceIt arm and attached links in its base-link frame."""
import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from rigid_inertia import combine_fixed_inertias


def adjacent_collision_pairs(robot, base):
    """ODE excludes joint-connected rigid bodies, including their fixed links."""
    groups = {base: base}
    pending = list(robot.findall('joint'))
    adjacent = set()
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            if parent not in groups:
                continue
            if joint.get('type') == 'fixed':
                groups[child] = groups[parent]
            else:
                groups[child] = child
                adjacent.add(frozenset((groups[parent], child)))
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('URDF joint tree is disconnected')
    collision_links = [link.get('name') for link in robot.findall('link')
                       if link.find('collision') is not None]
    return sorted((first, second)
                  for index, first in enumerate(collision_links)
                  for second in collision_links[index + 1:]
                  if frozenset((groups[first], groups[second])) in adjacent)


def prepare(task_path, source_roots, output):
    task = json.loads(Path(task_path).read_text())
    source = ET.fromstring(task['robot_xml'])
    base = task['robot_base']
    included = {base}
    joints = source.findall('joint')
    while True:
        children = {j.find('child').get('link') for j in joints
                    if j.find('parent').get('link') in included}
        if children <= included:
            break
        included |= children
    robot = ET.Element('robot', name='sliceit_b_bot')
    extension = ET.SubElement(robot, 'mujoco')
    ET.SubElement(extension, 'compiler', fusestatic='false', discardvisual='true', strippath='false', inertiafromgeom='false')
    packages = {}
    for root in source_roots:
        for package in Path(root).rglob('package.xml'):
            packages[ET.parse(package).getroot().findtext('name')] = package.parent
    mesh_sources = []
    for link in source.findall('link'):
        if link.get('name') not in included:
            continue
        link = copy.deepcopy(link)
        for visual in link.findall('visual'):
            link.remove(visual)
        for mesh in link.findall('.//mesh'):
            uri = mesh.get('filename')
            if uri.startswith('package://'):
                package, relative = uri[len('package://'):].split('/', 1)
                path = packages[package] / relative
                if not path.exists():
                    raise FileNotFoundError(path)
                mesh.set('filename', str(path))
                mesh_sources.append(uri)
        robot.append(link)
    for joint in joints:
        if joint.find('parent').get('link') in included:
            robot.append(copy.deepcopy(joint))
    inertia_groups = combine_fixed_inertias(robot, base)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    urdf = output / 'robot.urdf'
    ET.ElementTree(robot).write(urdf)
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.option.timestep = task['environment']['robot_control_dt']
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    collision_excludes = adjacent_collision_pairs(robot, base)
    for first, second in collision_excludes:
        spec.add_exclude(bodyname1=first, bodyname2=second)
    by_name = {j.get('name'): j for j in joints}
    for name in task['joint_names']:
        effort = float(by_name[name].find('limit').get('effort'))
        spec.add_actuator(name=name + '_torque', target=name,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gainprm=[1] + [0] * 9, gear=[1, 0, 0, 0, 0, 0],
                          ctrllimited=True, ctrlrange=[-effort, effort])
    model = spec.compile()
    binary = output / 'robot.mjb'
    mujoco.mj_saveModel(model, str(binary))
    metadata = {k: task[k] for k in ('robot_base', 'robot_tip', 'joint_names')}
    metadata.update({'inertia_groups': inertia_groups, 'mesh_sources': mesh_sources, 'links': sorted(included),
                     'adjacent_collision_excludes': collision_excludes,
                     'source_task': str(Path(task_path).resolve())})
    (output / 'robot_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'nq': model.nq, 'nv': model.nv, 'nu': model.nu, 'ngeom': model.ngeom,
                      'nbody': model.nbody, 'model': str(binary)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--source-root', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.task, args.source_root, args.output)
