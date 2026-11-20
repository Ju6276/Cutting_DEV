import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from cutting_gpu_backend.disect import validate_sdf_payload
from scripts.export_ikea_scenes import verify_asset_hash


def test_packed_sdf_reconstructs_original_npy_checksum_and_rejects_tampering(tmp_path):
    values=np.arange(41,dtype=np.float32)
    path=tmp_path/"knife_sdf.npy"
    np.save(path,values)
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    validate_sdf_payload(torch.from_numpy(values.copy()),expected)
    changed=values.copy();changed[13]+=.125
    with pytest.raises(ValueError,match="payload checksum mismatch"):
        validate_sdf_payload(torch.from_numpy(changed),expected)
    with pytest.raises(ValueError,match="float32"):
        validate_sdf_payload(torch.from_numpy(values.astype(np.float64)),expected)


@pytest.mark.parametrize("kind",["geometry","sdf"])
def test_export_rejects_asset_changed_after_manifest(kind,tmp_path):
    p=tmp_path/(kind+".npy")
    np.save(p,np.arange(12,dtype=np.float32))
    expected=hashlib.sha256(p.read_bytes()).hexdigest()
    verify_asset_hash(p,expected,kind)
    np.save(p,np.arange(12,dtype=np.float32)+1)
    with pytest.raises(ValueError,match="checksum mismatch"):
        verify_asset_hash(p,expected,kind)
    with pytest.raises(ValueError,match="lacks checksum"):
        verify_asset_hash(p,None,kind)


def test_actual_ikea_sdf_payload_checksum_without_gpu():
    root=Path("/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_estimated_v1")
    if not (root/"cutting_geometry.json").exists():pytest.skip("user CAD asset absent")
    meta=json.loads((root/"cutting_geometry.json").read_text())
    verify_asset_hash(meta["geometry"],meta["geometry_sha256"],"geometry")
    verify_asset_hash(meta["sdf"],meta["sdf_validation"]["sha256"],"SDF")
    validate_sdf_payload(torch.from_numpy(np.load(meta["sdf"],allow_pickle=False)),
                         meta["sdf_validation"]["sha256"])
