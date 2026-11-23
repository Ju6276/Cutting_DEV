from pathlib import Path
import sys, json, math, argparse
import torch
ap=argparse.ArgumentParser(description="Offline controller fixture; no ROS or robot connection.")
ap.add_argument("--record-dir",type=Path,required=True,help="Directory with platform/controller.json, platform/platform.json and robot_description.candidate.urdf")
a=ap.parse_args()
root=a.record_dir
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'backends/mujoco/cutting_gpu_backend'))
from controller import FDCCController,ControllerConfig,SerialChain
parameters=json.loads((root/'platform/controller.json').read_text())
chain=SerialChain((root/'robot_description.candidate.urdf').read_text(),parameters['robot_base_link'],parameters['end_effector_link'],parameters['joints'])
config=ControllerConfig.from_parameters(parameters)
q0=torch.tensor([json.loads((root/'platform/platform.json').read_text())['q_initial_rad']],dtype=torch.float64)
c=FDCCController(chain,config);c.reset(q0)
rows=[];expect=[]
for i in range(30):
 q=q0+.01*torch.sin(torch.arange(6,dtype=torch.float64)+i*.2)
 frames,_,_=chain(q,mass_matrix=False)
 target=frames[chain.tip].clone()
 target[:,:3]+=torch.tensor([[.002,-.003,-.004]],dtype=torch.float64)
 wrench=torch.tensor([[.2*i,-.1*i,.3*i,.01*i,-.02*i,.005*i]],dtype=torch.float64)
 command=c.update(q,target,wrench)
 assert torch.isfinite(command).all()
 rows.append(torch.cat((q,target,wrench),dim=-1)[0].tolist())
 expect.append(command[0].tolist())
text=' '.join(map(str,q0[0].tolist()))+'\n'+'\n'.join(' '.join(map(str,r)) for r in rows)+'\n'
(root/'controller_fixture_input.txt').write_text(text)
(root/'controller_fixture_expected.json').write_text(json.dumps(expect,indent=2)+'\n')
print('Prepared 30 updates; nonzero tool mass, transformed wrench, persistent controller velocity; no hardware access.')
