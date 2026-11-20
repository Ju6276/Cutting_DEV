import copy
from pathlib import Path
import pytest
from cutting_world_model.shared_baselines import validate_current_interface
from cutting_world_model.robot_platform import RobotPlatform


def metadata():
    return {'free_cut_task':{'version':'pose_depth_v3'},
            'robot_platform':{'revision':'ur10e_test','asset_root':'/old/path',
                'controller_sha256':'abc','runtime_sha256':{'controller.json':'abc','robot.mjb':'xyz'}}}


def test_robot_identity_allows_asset_root_relocation():
    saved=metadata();actual=copy.deepcopy(saved)
    actual['robot_platform']['asset_root']='/new/path'
    validate_current_interface(saved,actual)


@pytest.mark.parametrize('key',['controller_sha256','runtime_sha256','revision'])
def test_robot_identity_rejects_changed_controller_or_compiled_robot(key):
    saved=metadata();actual=copy.deepcopy(saved)
    actual['robot_platform'][key]=({'robot.mjb':'changed'} if key=='runtime_sha256' else 'changed')
    with pytest.raises(ValueError,match='robot_platform'):
        validate_current_interface(saved,actual)


def test_ur10e_checkpoint_without_execution_hashes_is_rejected():
    saved=metadata();actual=copy.deepcopy(saved)
    del saved['robot_platform']['controller_sha256']
    with pytest.raises(ValueError,match='lacks verified identity'):
        validate_current_interface(saved,actual)


def test_legacy_ur5e_identity_has_no_new_hash_requirement():
    saved={'free_cut_task':{'version':'pose_depth_v3'}}
    validate_current_interface(saved,copy.deepcopy(saved))


def test_actual_robot_manifest_hashes_are_recorded():
    root=Path('/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_estimated_v1')
    if not (root/'platform.json').exists():pytest.skip('UR10e assets absent')
    import hashlib
    p=RobotPlatform.load(root).metadata['robot_platform']
    assert p['controller_sha256']==hashlib.sha256((root/'controller.json').read_bytes()).hexdigest()
    assert p['runtime_sha256']['robot.mjb']==hashlib.sha256((root/'robot.mjb').read_bytes()).hexdigest()
