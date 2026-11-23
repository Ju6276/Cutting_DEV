"""Read-only RTDE program-state monitor; does not send control inputs."""
import socket,struct,time,threading
from settings import ROBOT_IP
class NativeRuntime:
 def __init__(self):
  self.state=None;self.stamp=0;self.error=None;self.closed=False
  self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def loop(self):
  try:
   s=socket.create_connection((ROBOT_IP,30004),timeout=1);self.socket=s
   def send(c,p=b""):s.sendall(struct.pack("!HB",len(p)+3,c)+p)
   def receive():
    def exact(n):
     b=b""
     while len(b)<n:
      v=s.recv(n-len(b))
      if not v:raise ConnectionError("RTDE disconnected")
      b+=v
     return b
    n,c=struct.unpack("!HB",exact(3));return c,exact(n-3)
   send(86,struct.pack("!H",2));assert receive()==(86,b"\x01")
   send(79,struct.pack("!d",10)+b"runtime_state")
   c,p=receive();assert c==79 and p[1:]==b"UINT32"
   send(83);assert receive()==(83,b"\x01")
   while not self.closed:
    c,p=receive()
    if c==85:_,self.state=struct.unpack("!BI",p);self.stamp=time.monotonic()
  except Exception as e:self.error=str(e)
 def check(self):
  assert self.error is None,self.error
  assert time.monotonic()-self.stamp<.35,"RTDE runtime feedback stale"
  assert self.state==2,"Robot actual runtime is not PLAYING: "+str(self.state)
 def close(self):
  self.closed=True
  try:self.socket.close()
  except Exception:pass
