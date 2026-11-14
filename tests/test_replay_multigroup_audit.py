"""Short/long episodes must keep per-world replay windows correct after ring wrap."""
import pytest
import torch
from cutting_world_model.online_replay import BatchedOnlineReplay


@pytest.mark.parametrize('worlds',[1,3,7])
def test_mixed_terminal_periods_ring_wrap_and_oversized_sampling(worlds):
    torch.manual_seed(11)
    replay=BatchedOnlineReplay(worlds,3,'cpu',capacity=32,goal_dim=7)
    ids=torch.arange(worlds)
    episodes=torch.zeros(worlds,dtype=torch.long)
    steps=torch.zeros(worlds,dtype=torch.long)
    periods=1+2*(ids%3)
    for _ in range(50):
        state=torch.stack((ids,episodes,steps),-1).float()
        nxt=state.clone();nxt[:,2]+=1
        ended=steps+1>=periods
        action=torch.zeros(worlds,6);action[:,0]=ids
        goal=torch.zeros(worlds,7);goal[:,0]=episodes
        replay.add(state,action,action,goal,torch.ones(worlds),nxt,ended,torch.zeros_like(ended))
        episodes+=ended.long();steps=torch.where(ended,0,steps+1)
    assert replay.buffer.size==32*worlds
    # Larger than stored rows deliberately exercises replacement sampling.
    batch=replay.buffer.sample_windows(4096,8)
    assert batch.states.shape==(9,4096,3)
    torch.testing.assert_close(batch.states[:,:,0],batch.states[:1,:,0].expand(9,-1))
    torch.testing.assert_close(batch.states[:,:,1],batch.states[:1,:,1].expand(9,-1))
    increments=batch.states[1:,:,2]-batch.states[:-1,:,2]
    torch.testing.assert_close(increments,batch.valid_steps.float())
    assert (batch.done.sum(0)==1).all()  # each sampled short episode ends once
    torch.testing.assert_close(batch.xi[:,0],batch.states[0,:,1])
    starts=replay.buffer.sample_starts(4096,10)
    assert (starts.hist_states[:,:,0]==starts.state[None,:,0]).all()
    assert (starts.hist_states[:,:,1]==starts.state[None,:,1]).all()
