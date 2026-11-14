from types import SimpleNamespace
import json
import pytest
import cutting_world_model.training_snapshot as snapshot

def test_snapshot_records_real_global_ids_without_stepping(tmp_path,monkeypatch):
 calls=[]
 def capture(env,backend,path):
  calls.append(env);path.mkdir();(path/'report.json').write_text('{}')
 monkeypatch.setattr(snapshot,'capture_group',capture)
 members=[SimpleNamespace(env='a'),SimpleNamespace(env='b')]
 env=SimpleNamespace(members=members,groups=[SimpleNamespace(name='cucumber',worlds=2),SimpleNamespace(name='potato',worlds=3)],worlds=5)
 status=dict(iteration=20,wm_updates=4,policy_updates=2,training_stage='joint')
 report=snapshot.capture_training_environment(env,'backend',tmp_path/'capture',status)
 assert calls==['a','b']
 assert [g['first_world'] for g in report['groups']]==[0,2]
 assert report['training']==status and report['worlds']==5
 assert json.loads((tmp_path/'capture/report.json').read_text())==report
 with pytest.raises(FileExistsError):snapshot.capture_training_environment(env,'backend',tmp_path/'capture',status)

def test_worker_capture_does_not_replace_parent_current_state(tmp_path):
 class Member:
  current=object()
  def send(self,command,path):assert command=='capture';self.path=path
  def _receive_message(self):return 'ok',None
 member=Member();old=member.current
 env=SimpleNamespace(members=[member],groups=[SimpleNamespace(name='prism',worlds=2)],worlds=2)
 snapshot.capture_training_environment(env,'backend',tmp_path/'capture',{})
 assert member.current is old

def test_worker_error_cannot_produce_complete_report(tmp_path):
 class Member:
  def send(self,*args):pass
  def _receive_message(self):return 'error','failed'
 env=SimpleNamespace(members=[Member()],groups=[SimpleNamespace(name='prism',worlds=2)],worlds=2)
 with pytest.raises(RuntimeError,match='failed'):
  snapshot.capture_training_environment(env,'backend',tmp_path/'capture',{})
 assert not (tmp_path/'capture/report.json').exists()
