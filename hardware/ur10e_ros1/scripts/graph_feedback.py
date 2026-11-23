"""Capture unchanged actual-feedback state derivation; no model-predicted q."""
import torch
from cutting_world_model.pose_depth import depth_progress
class GraphFeedback:
 def __init__(self,spec,goal):self.spec=spec;self.goal=goal;self.constants={};self.cache=None
 def raw(self,old,q,qd,wh,peak,last,action):
  spec=self.spec
  fields={n:spec.field(old,n).clone() for n in spec.physical_fields+spec.history_fields}
  fields.update(joint_q=q,joint_qd=qd,wrench_history=wh,peak_wrench=peak,last_wrench=last)
  fields["cut_progress"]=depth_progress(spec.fk(q),spec.board_geometry,fields["depth_span"],spec.field(old,"cut_progress"))
  fields["board_contact"]=torch.zeros_like(fields["board_contact"])
  return spec.derive(old,fields,action,self.goal)
 def compute(self,*inputs):
  original=torch.Tensor.new_tensor;original_get=torch.Tensor.__getitem__
  def constant(t,data,*a,**kw):
   key=(str(t.device),str(t.dtype),repr(data),repr(a),repr(kw))
   if key not in self.constants:
    assert not torch.cuda.is_current_stream_capturing()
    self.constants[key]=original(t,data,*a,**kw)
   return self.constants[key]
  def indexed(t,index):
   def convert(i):
    if not isinstance(i,list):return i
    key=("index",str(t.device),tuple(i))
    if key not in self.constants:
     assert not torch.cuda.is_current_stream_capturing()
     self.constants[key]=torch.tensor(i,dtype=torch.long,device=t.device)
    return self.constants[key]
   return original_get(t,tuple(convert(i) for i in index) if isinstance(index,tuple) else convert(index))
  torch.Tensor.new_tensor=constant;torch.Tensor.__getitem__=indexed
  try:return self.raw(*inputs)
  finally:torch.Tensor.new_tensor=original;torch.Tensor.__getitem__=original_get
 def __call__(self,*inputs):
  if self.cache is None:
   static=[t.clone() for t in inputs];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):self.compute(*static)
   torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):out=self.compute(*static)
   self.cache=static,g,out
  static,g,out=self.cache
  for dst,src in zip(static,inputs):dst.copy_(src)
  g.replay()
  return tuple(t.clone() for t in out)
