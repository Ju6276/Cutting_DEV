"""CUDA graph execution of the unchanged analytic FDCC integration.
Commissioning-only fixed error_scale profile. No robot communication here.
"""
import inspect,textwrap,torch
from cutting_world_model.shared_baselines import AnalyticFDCCModel
from cutting_gpu_backend.controller.fdcc import FDCCController,SlicingActionMapper
class NativeGainModel(AnalyticFDCCModel):
    pass
_source=textwrap.dedent(inspect.getsource(AnalyticFDCCModel.predict))
_source=_source.replace("arguments = mapper(action.double(), goal[..., :7].double(), current_pose)","arguments = mapper(action.double(), goal[..., :7].double(), current_pose)\n    arguments['error_scale'] = q.new_full((len(q),), self.native_gain)")
_scope=dict(AnalyticFDCCModel.predict.__wrapped__.__globals__)
exec(_source,_scope)
NativeGainModel.predict=_scope["predict"]
class GraphIntegrator:
 def __init__(self,model):
  self.model=model;self.cache={};self.constants={}
 def __call__(self,q,qd,wrench,motion,action,goal,tare):
  inputs=(q,qd,wrench,motion,action,goal,tare)
  key=tuple((tuple(x.shape),x.dtype,x.device.index) for x in inputs)
  if key not in self.cache:
   static=[x.clone() for x in inputs];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):self.compute(*static)
   torch.cuda.current_stream().wait_stream(stream)
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):output=self.compute(*static)
   self.cache[key]=(static,graph,output)
  static,graph,output=self.cache[key]
  for dst,src in zip(static,inputs):dst.copy_(src)
  graph.replay()
  qout,infos=output
  if bool((infos!=0).any()) or not bool(torch.isfinite(qout).all()):
   raise FloatingPointError("Invalid FDCC solve during CUDA graph")
  return qout.clone()
 def compute(self,*args):
  original=torch.Tensor.new_tensor
  def constant(t,data,*a,**kw):
   if isinstance(data,torch.Tensor):raise TypeError("Dynamic new_tensor is not a capture constant")
   key=(str(t.device),str(t.dtype),repr(data),repr(a),repr(sorted(kw.items())))
   if key not in self.constants:
    if torch.cuda.is_current_stream_capturing():raise RuntimeError("Unwarmed CUDA constant")
    self.constants[key]=original(t,data,*a,**kw)
   return self.constants[key]
  torch.Tensor.new_tensor=constant
  try:return self._compute(*args)
  finally:torch.Tensor.new_tensor=original
 def _compute(self,q,qd,wrench,motion,action,goal,tare):
  infos=[]
  def solve_capture(a,b):
   solution,info=torch.linalg.solve_ex(a,b,check_errors=False);infos.append(info);return solution
  update_source=textwrap.dedent(inspect.getsource(FDCCController.update)).replace("torch.linalg.solve(","solve_capture(")
  scope=dict(FDCCController.update.__globals__);scope["solve_capture"]=solve_capture
  exec(update_source,scope)
  class CapturedController(FDCCController):pass
  CapturedController.update=scope["update"]
  controller=CapturedController(self.model.chain,self.model.controller_config)
  controller.reset(q,qd);controller.tare_compensation=tare
  spec=self.model.spec
  mapper=SlicingActionMapper(max_twist=spec.metadata["controller"]["max_twist"],robot_control_dt=spec.metadata["environment"]["robot_control_dt"],pitch_control=spec.metadata["free_cut_task"].get("pitch_control"),agent_control_dt=spec.dt)
  mapper.reset(len(q),q.device,q.dtype);mapper.added_motion_command=motion.clone()
  current_pose=self.model.chain(q,mass_matrix=False)[0][self.model.chain.tip]
  arguments=mapper(action,goal[:,:7],current_pose);arguments["error_scale"]=q.new_full((len(q),),self.model.native_gain)
  for _ in range(self.model.substeps):
   command=controller.update(q,sensor_wrench=wrench,**arguments)
   q=controller.enforce_position_limits(command,spec.dt/self.model.substeps)
  return q,torch.stack(infos)
class FastNativeModel(NativeGainModel):
 def __init__(self,*args,native_gain=.2,**kwargs):
  super().__init__(*args,**kwargs);self.native_gain=native_gain;self.integrator=GraphIntegrator(self)
 def fast_q(self,state,action,goal):
  spec=self.spec;q=spec.field(state,"joint_q").double()
  tare=q.new_zeros((len(q),6)) if self.tare_compensation is None else self.tare_compensation.to(q).expand(len(q),-1)
  return self.integrator(q,spec.field(state,"joint_qd").double(),spec.field(state,"wrench_history")[:,-1].double(),spec.field(state,"motion_command").double(),action.double(),goal.double(),tare)
_source=textwrap.dedent(inspect.getsource(AnalyticFDCCModel.predict))
start=_source.index("    q0 = spec.field")
end=_source.index("    fields = {name:")
_source=_source[:start]+'    q0 = spec.field(state, "joint_q").double()\n    q = self.fast_q(state, action, goal)\n    wrench = spec.field(state, "wrench_history")[:, -1].double()\n'+_source[end:]
_scope=dict(AnalyticFDCCModel.predict.__wrapped__.__globals__);exec(_source,_scope);FastNativeModel.predict=_scope["predict"]
