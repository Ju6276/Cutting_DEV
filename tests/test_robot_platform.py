import json
import os
from pathlib import Path
import pytest
from cutting_world_model.robot_platform import RobotPlatform


def test_uncompiled_platform_is_not_silently_legacy(tmp_path):
    (tmp_path/"platform.json").write_text("{}")
    with pytest.raises(ValueError,match="not compiled"):
        RobotPlatform.load(tmp_path)


def test_modified_robot_assets_are_rejected(tmp_path):
    (tmp_path/"robot.mjb").write_bytes(b"modified")
    (tmp_path/"platform.json").write_text(json.dumps({"runtime":{"sha256":{"robot.mjb":"wrong"}}}))
    with pytest.raises(ValueError,match="hash mismatch"):
        RobotPlatform.load(tmp_path)


def test_ur10e_real_robot_chain_sensor_and_fdcc(tmp_path):
    root=Path(os.environ.get("CUTTING_UR10E_PLATFORM",
        "/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_estimated_v1"))
    if not (root/"platform.json").exists():
        pytest.skip("prepared user UR10e CAD assets not installed")
    from scripts.check_robot_platform import check
    check(root,tmp_path/"report.json","cpu")
    bundle=RobotPlatform.load(root)
    with pytest.raises(ValueError,match="not yet a validated cutting scene"):
        bundle.require_cutting_ready()
