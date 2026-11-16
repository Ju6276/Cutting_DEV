"""Create versioned DiSECt coefficient variants without replacing source scenes."""
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from cutting_world_model.shared_scene import fingerprint, configs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,default=Path('configs/shared/scene_v1.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    root=Path('backends/mujoco').resolve()
    sys.path.insert(0,str(root/'third_party/disect'))
    contract=json.loads(a.contract.read_text());common=contract['common']
    result=dict(contract_sha256=fingerprint(contract),objects={})
    for obj in contract['objects']:
        source=root/obj['disect_source_scene']
        scene=torch.load(source,map_location='cuda:0',weights_only=False)
        settings=scene['settings']
        elastic=dict(young_modulus=settings['young'],poisson_ratio=settings['poisson'],density=settings['density'])
        if elastic!=obj['elastic']:raise ValueError('Source material differs: '+obj['name'])
        scene['model'].sdf_mu.fill_(common['knife_friction_mu'])
        scene['model'].contact_mu=common['ground_friction_mu']
        for key in ('sdf_mu','surface_sdf_mu'):settings[key]=common['knife_friction_mu']
        settings['ground_mu']=common['ground_friction_mu']
        dest=(a.output/(obj['name']+'.pt')).resolve();torch.save(scene,dest)
        result['objects'][obj['name']]=dict(scene=str(dest),elastic=elastic,
            source_scene=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            blademaster_geometry_asset=obj['blademaster_geometry_asset'],
            blademaster_geometry_sha256=hashlib.sha256(Path(obj['blademaster_geometry_asset']).read_bytes()).hexdigest(),
            scene_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    d,b=configs(contract,result)
    for name,cfg in [('disect',d),('blademaster',b)]:
        (a.output/(name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
    print(a.output/'manifest.json')
if __name__=='__main__':main()
