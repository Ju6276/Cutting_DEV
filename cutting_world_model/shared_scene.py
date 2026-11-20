"""Shared deterministic case specification; backend material mechanisms remain explicit."""
import copy
import hashlib
import json
import random
from pathlib import Path

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def sample_clearances(seed, name, worlds, bounds):
    if type(seed) is not int or seed < 0 or type(worlds) is not int or worlds < 1:
        raise ValueError('Nonnegative seed and positive world count required')
    generator = random.Random(f'{seed}:{name}:start-clearance-v1')
    return [generator.uniform(*bounds) for _ in range(worlds)]

def configs(contract, prepared):
    if contract['revision']!='paired_scene_v1':
        raise ValueError('Unsupported paired scene revision')
    if prepared['contract_sha256']!=fingerprint(contract):
        raise ValueError('Prepared scenes belong to another contract')
    common=contract['common']
    if common['ground_friction_mu']!=common['knife_friction_mu']:
        raise ValueError('Current BM preset shares blade/board coefficients; explicit equality required')
    from .episode_conditions import validate as validate_conditions
    conditions=validate_conditions(common['episode_conditions'])
    task=copy.deepcopy(common['task'])
    from .scene_alignment import validate_alignment
    low, high = validate_alignment(task['scene_alignment'])['clearance_range_m']
    worlds=common['worlds_per_object']
    if type(worlds) is not int or worlds<1:
        raise ValueError('Positive integer worlds_per_object required')
    seed=contract['case_seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer case_seed required')
    names=[o['name'] for o in contract['objects']]
    if not names or len(set(names)) != len(names):
        raise ValueError('Nonempty unique object names required')
    d=dict(description=contract['scope'],seed=seed,parallel=False,
           force_limits=common['force_limits'],force_observation=common['force_observation'],
           task=copy.deepcopy(task),episode_conditions=copy.deepcopy(conditions),groups=[])
    b=dict(description=contract['scope'],source_root=contract['blademaster_source_root'],
           worlds=worlds*len(contract['objects']),clearance_m=high,
           start_clearance_m=[],case_seed=seed,
           settle_seconds=common['settle_seconds'],
           force_limits=common['force_limits'],force_observation=common['force_observation'],
           task=copy.deepcopy(task),episode_conditions=copy.deepcopy(conditions),execution='batched',groups=[])
    if contract.get('robot_platform'):
        d['robot_platform']=contract['robot_platform']
        b['robot_platform']=contract['robot_platform']
    for obj in contract['objects']:
        item=prepared['objects'][obj['name']]
        if item['blademaster_geometry_asset'] != obj['blademaster_geometry_asset']:
            raise ValueError('Prepared BladeMaster asset path differs from shared source')
        if item['elastic']!=obj['elastic']:
            raise ValueError('Prepared elastic parameters differ from shared source')
        clearances=sample_clearances(seed,obj["name"],worlds,(low,high))
        b['start_clearance_m'].extend(clearances)
        d['groups'].append(dict(name=obj['name'],scene=item['scene'],worlds=worlds,
            clearance=[low,high],start_clearance_m=clearances,
            material={'youngs_modulus':conditions['youngs_modulus'],'density':conditions['density'],
                      'cut_spring_stiffness':conditions['cut_strength']},
            target_speed=conditions['target_speed'],
            sim_dt=obj['disect_sim_dt'],force_model='net',friction_velocity=.01,
            settle_seconds=common['settle_seconds']))
        b['groups'].append(dict(name=obj['name'],worlds=worlds,material=obj['blademaster_material'],
            geometry_asset=obj['blademaster_geometry_asset'],material_overrides=obj['elastic'],
            contact_friction_mu=common['knife_friction_mu']))
    return d,b

def load_configs(contract_path, prepared_path):
    contract=json.loads(Path(contract_path).read_text())
    prepared=json.loads(Path(prepared_path).read_text())
    for item in prepared['objects'].values():
        if hashlib.sha256(Path(item['scene']).read_bytes()).hexdigest()!=item['scene_sha256']:
            raise ValueError('Prepared scene file changed since verification')
        if hashlib.sha256(Path(item['blademaster_geometry_asset']).read_bytes()).hexdigest()!=item['blademaster_geometry_sha256']:
            raise ValueError('BladeMaster geometry changed since verification')
    return configs(contract,prepared)
