"""Compose shared-studio frames with measured tool-distance/task-rule overlays."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--recording',type=Path,required=True)
 p.add_argument('--render',type=Path,required=True,help='Shared studio video path with its .json sidecar')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 report=json.loads((a.recording/'comparison.json').read_text())
 meta=json.loads(a.render.with_suffix('.json').read_text())
 if report['status']!='complete' or meta['preview_only']:raise ValueError('Complete physical recording and rendering required')
 rows=report['rows'];fps=meta['fps'];speed=meta['speed']
 task=report['config']['task'];motion=task['tool_motion']
 free_mm=motion['free_m']*1000;failure_mm=motion['failure_m']*1000
 final_mm=task['completion_position']['radius_m']*1000
 failure_cost=task['weights']['failure']
 plot_max=max(failure_mm*1.4,max(x['distance_m']*1000 for x in rows)*1.1)
 initial=json.loads((a.recording/'geometry/0000/report.json').read_text())
 initial_xy=np.array(initial['training']['ee_pose_base'][0][:2])
 initial_distance=float(np.linalg.norm(initial_xy-np.array(rows[0]['target_pose'][:2])))
 times=np.arange(len(rows)+1)*.05
 indices=np.searchsorted(times,np.arange(0,times[-1],speed/fps)).clip(0,len(times)-1)
 indices=np.r_[indices,np.full(fps,len(times)-1)]
 folder=Path(meta['frame_directory'])
 font_path=subprocess.check_output(['fc-match','-f','%{file}',':lang=zh'],text=True)
 def font(n):return ImageFont.truetype(font_path,n)
 fonts={n:font(n) for n in (18,22,24,28,32)}
 ink=(28,40,51);muted=(92,106,118);blue=(23,115,159);orange=(188,98,30)
 stop=int(report['modes']['path_and_final']['step'])
 W,H=1800,1110
 encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
 cache={}
 def frame(i):
  if i not in cache:
   path=folder/f'{i:04d}.png'
   cache[i]=Image.open(path).convert('RGB').resize((880,624),Image.Resampling.LANCZOS)
  return cache[i]
 names={'running':'进行中','success':'到位成功','tool_motion_failure':'横向越界，回合终止',
        'timeout':'超时','outside':'工作空间越界','collision':'超力限'}
 def state(i,mode):
  if i==0:return dict(distance_m=initial_distance,time_s=0.),dict(return_total=0.,position_cost=0.,ending='running')
  row=rows[i-1];return row,row['modes'][mode]
 def position_total(i):
  active=[row['modes']['path_and_final'] for row in rows[:i] if row['modes']['path_and_final']['active']]
  return sum(x['position_cost'] for x in active)+(failure_cost if i>=stop else 0)
 preview_steps={0,int(indices[np.searchsorted(indices,stop)]),int(indices[-1])}
 try:
  for j,index in enumerate(indices):
   i=int(index);right=min(i,stop)
   im=Image.new('RGB',(W,H),(245,247,249));d=ImageDraw.Draw(im)
   d.text((28,12),'刀具横向约束：同一段实际仿真刀路，比较两种规则',font=fonts[32],fill=ink)
   d.text((28,58),f'脚本刀路（非训练效果）· 黄瓜 · {speed:g} 倍速 · 距离相对目标末端位置；使用共用场景的实际录制几何。',font=fonts[22],fill=muted)
   for x,mode,idx,color,title in [(10,'final_only',i,blue,'仅限制最终到位'),(910,'path_and_final',right,orange,'过程 + 最终到位')]:
    im.paste(frame(idx),(x,134))
    d.text((x+18,96),title,font=fonts[28],fill=color)
    row,m=state(idx,mode)
    d.text((x+18,766),f"实际水平偏差：{row['distance_m']*1000:.1f} mm     时刻：{idx*.05:.2f} s",font=fonts[24],fill=ink)
    cost=m.get('position_cost',0.)
    d.text((x+18,802),f"本步位置扣分：{cost:.4f}     任务累计回报：{m['return_total']:.2f}",font=fonts[22],fill=ink)
    label=names.get(m['ending'],m['ending'])
    if mode=='path_and_final' and i>stop:label+='（画面保持）'
    d.text((x+18,837),label,font=fonts[24],fill=color)
   # Exact measured distance through time, with the alternative episode held at its terminal event.
   x0,y0,cw,ch=85,900,1560,130
   def xy(t,mm):return (x0+cw*t/max(times[-1],.05),y0+ch*(1-mm/plot_max))
   for mm,color,label in [(free_mm,(112,152,126),'免费范围'),(final_mm,(126,137,148),'最终范围'),(failure_mm,orange,'过程失败边界')]:
    y=xy(0,mm)[1];d.line((x0,y,x0+cw,y),fill=color,width=1)
    d.text((x0+cw+8,y-12),f'{mm:g}',font=fonts[18],fill=color)
   data=[(0.,initial_distance*1000)]+[(row['time_s'],row['distance_m']*1000) for row in rows[:i]]
   if len(data)>1:d.line([xy(*v) for v in data],fill=blue,width=3)
   if i>=stop:
    point=rows[stop-1];a1=xy(point['time_s'],point['distance_m']*1000);a2=xy(i*.05,point['distance_m']*1000)
    d.line([a1,a2],fill=orange,width=4);d.ellipse((a1[0]-5,a1[1]-5,a1[0]+5,a1[1]+5),fill=orange)
   d.text((28,901),'mm',font=fonts[18],fill=muted)
   d.text((85,1038),f'{free_mm:g} mm 内无额外扣分；{free_mm:g}–{failure_mm:g} mm 二次增长；超过 {failure_mm:g} mm：失败 −{failure_cost:g}，结束回合。',font=fonts[22],fill=ink)
   d.text((85,1074),'右侧终止后保持最后画面；没有继续累计奖励，也没有模拟真实硬件的刹停过程。',font=fonts[18],fill=muted)
   if i in preview_steps and (j==0 or i!=int(indices[j-1])):
    im.save(a.output.with_name(a.output.stem+f'_step_{i:03d}.png'))
   encoder.stdin.write(im.tobytes())
 finally:
  encoder.stdin.close()
  if encoder.wait()!=0:raise RuntimeError('Video encoding failed')
 a.output.with_suffix('.json').write_text(json.dumps(dict(recording=str(a.recording),renderer=str(a.render),
   scope=report['scope'],frames=len(indices),fps=fps,stop_step=stop,final=report['modes'],
   active_position_cost_total=position_total(stop),geometry='Shared studio renderer; actual recorded geometry; no extra crack/proxy overlay'),indent=2))
if __name__=='__main__':main()
