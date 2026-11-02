"""Combine URDF fixed-link inertias without changing rigid-body dynamics."""
import xml.etree.ElementTree as ET

import numpy as np


def origin(element):
    transform = np.eye(4)
    if element is None:
        return transform
    transform[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    r, p, y = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    transform[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                          [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                          [-sp, cp*sr, cp*cr]]
    return transform


def parallel_axis(position):
    return np.dot(position, position) * np.eye(3) - np.outer(position, position)


def combine_fixed_inertias(robot, base):
    links = {link.get('name'): link for link in robot.findall('link')}
    joints = robot.findall('joint')
    transforms = {base: (base, np.eye(4))}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in pending[:]:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in transforms:
                continue
            group, transform = transforms[parent]
            transforms[child] = ((group, transform @ origin(joint.find('origin')))
                                 if joint.get('type') == 'fixed' else (child, np.eye(4)))
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('URDF joint tree is disconnected')
    groups = {}
    for name, link in links.items():
        inertial = link.find('inertial')
        if inertial is None:
            continue
        group, transform = transforms[name]
        transform = transform @ origin(inertial.find('origin'))
        mass = float(inertial.find('mass').get('value'))
        values = inertial.find('inertia').attrib
        inertia = np.array([[float(values['i'+a+b] if 'i'+a+b in values else values['i'+b+a])
                             for b in 'xyz'] for a in 'xyz'])
        rotation, position = transform[:3, :3], transform[:3, 3]
        groups.setdefault(group, []).append((mass, position, rotation @ inertia @ rotation.T))
        link.remove(inertial)
    report = []
    for group, items in groups.items():
        mass = sum(m for m, _, _ in items)
        if mass == 0:
            continue
        com = sum(m * p for m, p, _ in items) / mass
        inertia = sum(i + m * parallel_axis(p - com) for m, p, i in items)
        inertial = ET.SubElement(links[group], 'inertial')
        ET.SubElement(inertial, 'mass', value=str(mass))
        ET.SubElement(inertial, 'origin', xyz=' '.join(map(str, com)))
        ET.SubElement(inertial, 'inertia', **{'i'+a+b: str(inertia[i,j])
                      for i,a in enumerate('xyz') for j,b in enumerate('xyz') if i <= j})
        report.append({'group': group, 'source_bodies': len(items), 'mass': mass,
                       'com': com.tolist(), 'inertia': inertia.tolist()})
    return report
