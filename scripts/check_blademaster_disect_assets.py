"""Load migrated DiSECt assets through UR5e and run bounded reference episodes.

The pose-task reward comes directly from GPUEnvironment.step. Reports retain
material separation separately from pose success. No learning or reward tuning.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"backends/mujoco")]
from cutting_world_model.multi_env import MultiConditionEnvironment
from cutting_world_model.free_cut import validate_topology
from cutting_gpu_backend.reference import KnifeTrajectory
from scripts.compare_potato_preparation import metrics,cpu_row,stable_windows
from cutting_world_model.provenance import git_revision


def ground_stats(material):
    """Report physical contact geometry separately from extrapolated ghost nodes.

    DiSECt duplicates cut-tet nodes on the empty side of each subdomain. Those
    contact_mask=0 nodes are FEM degrees of freedom, not occupied material points.
    Ground contact uses mask-enabled nodes and interpolated cut-edge points.
    """
    worlds=material.worlds
    q=material.state_field("particle_q")
    flat=q.reshape(-1,3)
    mask=material.model.contact_mask.reshape(worlds,-1)>0
    edges=material.model.cut_edge_indices.reshape(-1,2).long()
    t=material.model.cut_edge_coords.reshape(-1,1)
    virtual=(flat[edges[:,0]]*(1-t)+flat[edges[:,1]]*t).reshape(worlds,-1,3)
    def stats(values):
        return dict(count=int(values.numel()),
                    minimum_y_mm=float(values.min()*1000) if values.numel() else None,
                    below_board=int((values<0).sum()),
                    below_minus_1mm=int((values<-.001).sum()),
                    below_minus_5mm=int((values<-.005).sum()))
    groups=[]
    for w in range(worlds):
        real=q[w,:,1][mask[w]]
        cut=virtual[w,:,1]
        groups.append(dict(all_fem_nodes=stats(q[w,:,1]),
                           ghost_nodes=stats(q[w,:,1][~mask[w]]),
                           contact_nodes=stats(real),cut_surface_points=stats(cut),
                           physical_contact_geometry=stats(torch.cat((real,cut)))))
    return dict(representation="physical_contact_geometry_v1",worlds=groups,
                contact_ke=float(material.model.contact_ke),
                contact_kd=float(material.model.contact_kd),
                contact_distance_m=float(material.model.contact_distance))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objects",nargs="+",default=["banana","cake","tofu","luncheon"])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-steps",type=int,default=180)
    p.add_argument("--wall-seconds",type=int,default=900)
    p.add_argument("--settle-trace-seconds",type=float,default=0.)
    a=p.parse_args()
    if a.max_steps<1 or a.wall_seconds<1 or a.settle_trace_seconds<0:
        raise ValueError("Budgets must be positive")
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    start=time.monotonic()
    def timeout(*_):
        raise TimeoutError("Asset validation wall budget")
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(a.wall_seconds)
    report=dict(revision=git_revision(ROOT),scope="UR5e fixed references; no optimizer",
                max_steps=a.max_steps,wall_seconds=a.wall_seconds,objects={})
    try:
        for name in a.objects:
            cfg=json.loads((ROOT/"configs/blademaster_disect"/("pose_"+name+".json")).read_text())
            cfg["parallel"]=False
            cfg["groups"][0]["worlds"]=1
            if a.settle_trace_seconds:
                cfg["groups"][0]["settle_seconds"]=0.
            multi=None
            record=dict(config=cfg,status="starting")
            report["objects"][name]=record
            try:
                multi=MultiConditionEnvironment.from_config(str(ROOT/"backends/mujoco"),cfg)
                e=multi.members[0].env
                frame=e.current
                validate_topology(e.physics.material)
                record.update(topology_validated=True,ground_initial=ground_stats(e.physics.material),settled_initial=cpu_row(metrics(e.physics.material)),
                              fixed_particles=int((e.physics.material.model.particle_inv_mass==0).sum()),
                              particles=frame["particle_q"].shape[1],springs=frame["cut_spring_ke"].shape[1])
                if a.settle_trace_seconds:
                    material=e.physics.material
                    trace=[dict(time_s=0.,**cpu_row(metrics(material)))]
                    peak=0.
                    for j in range(round(a.settle_trace_seconds/.05)):
                        peak=max(peak,float(material.advance_stationary(.05).max()))
                        trace.append(dict(time_s=(j+1)*.05,**cpu_row(metrics(material))))
                    accepted=stable_windows(trace)
                    record.update(status="settling_trace",trace=trace,stable_windows=accepted,
                                  peak_knife_force_during_settling=peak)
                    if peak>0 or any(any(row["released"]) for row in trace):
                        raise ValueError("Settling contacted the knife or damaged material")
                    print(name,"settling trace complete",flush=True)
                    continue
                gap=float(frame["knife_board_gap"][0,0])
                distance=max(0.,gap-cfg["task"]["board"]["stop_gap_m"])
                speed=float(e.target_speed[0])
                e.physics.set_reference_trajectory(KnifeTrajectory.constant([0.,-speed,0.],max(distance/speed,1e-6),kind="board_ending_press"))
                rows=[]
                record["rows"]=rows
                for step in range(min(a.max_steps,cfg["task"]["horizon_steps"])):
                    frame,reward,done,trunc,info=e.step(torch.zeros(1,6,device=e.device),reference=True)
                    rows.append(dict(step=step+1,reward=float(reward[0]),success=bool(info["success"][0]),
                                     collision=bool(info["collision"][0]),outside=bool(info["outside"][0]),
                                     timeout=bool(info["timeout"][0]),separated=bool(info["separated"][0]),
                                     released=int(info["released_springs"][0]),
                                     peak_force_n=float(info["peak_force"][0]),limit_utilization=float(info["safety_margin"][0]),
                                     ground=ground_stats(e.physics.material),
                                     gap_m=float(frame["knife_min_board_gap"].min()),
                                     goal_error=(frame["observation"][0,:3]*frame["observation"].new_tensor(e.heads.max_distance)).tolist()))
                    if (step+1) % 20 == 0:
                        print(name,"step",step+1,"released",rows[-1]["released"],"gap",rows[-1]["gap_m"],flush=True)
                    if bool(done[0] or trunc[0]):
                        break
                record.update(status="completed" if bool(done[0] or trunc[0]) else "step_budget",
                              rows=rows,return_sum=sum(row["reward"] for row in rows),
                              final=cpu_row(metrics(e.physics.material)),ground_final=ground_stats(e.physics.material))
            except TimeoutError:
                record.update(status="wall_budget")
                raise
            except Exception as exc:
                record.update(status="error",error=repr(exc))
            finally:
                if multi is not None:
                    multi.close()
                report["elapsed_seconds"]=time.monotonic()-start
                (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(name,record["status"],record.get("error",""),flush=True)
    finally:
        signal.alarm(0)
    if any(r["status"]=="error" for r in report["objects"].values()):
        raise RuntimeError("One or more assets failed; inspect report.json")


if __name__=="__main__":
    main()
