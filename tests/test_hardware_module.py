"""Offline contracts for the real module. Never imports ROS or commands hardware."""
import importlib.util
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import pytest

MODULE=Path(__file__).resolve().parents[1]/"hardware/ur10e_ros1"

def load(name):
    sys.path.insert(0,str(MODULE/"scripts"))
    try:
        spec=importlib.util.spec_from_file_location(name,MODULE/"scripts"/(name+".py"))
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)

def test_tool_append_preserves_calibrated_arm(tmp_path):
    combine=load("prepare_description").combine
    arm='<robot name="real"><link name="base_link"/><link name="tool0"/><joint name="factory" type="fixed"><origin xyz="0.123 0 0"/><parent link="base_link"/><child link="tool0"/></joint></robot>'
    tool='<robot name="tool"><link name="tool0"/>'
    for name in ("knife_control","knife_compliance","ati_sensing_frame"):
        tool+=f'<link name="{name}"/><joint name="fixed_{name}" type="fixed"><parent link="tool0"/><child link="{name}"/></joint>'
    tool+='</robot>'
    out=ET.fromstring(combine(arm,tool,tmp_path))
    old=ET.fromstring(arm)
    assert [ET.tostring(e) for e in out][:len(old)]==[ET.tostring(e) for e in old]
    with pytest.raises(ValueError,match="already present"):
        combine(ET.tostring(out,encoding="unicode"),tool,tmp_path)
    with pytest.raises(ValueError,match="must be fixed"):
        combine(arm,tool.replace('type="fixed"','type="revolute"',1),tmp_path)

def test_runtime_rejects_paused_and_stale_without_opening_socket():
    cls=load("native_runtime").NativeRuntime
    runtime=cls.__new__(cls)
    runtime.error=None;runtime.state=2;runtime.stamp=time.monotonic()
    runtime.check()
    runtime.state=4
    with pytest.raises(AssertionError,match="not PLAYING"):runtime.check()
    runtime.state=2;runtime.stamp=time.monotonic()-1
    with pytest.raises(AssertionError,match="stale"):runtime.check()
    runtime.stamp=time.monotonic();runtime.error="disconnected"
    with pytest.raises(AssertionError,match="disconnected"):runtime.check()
