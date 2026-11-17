import subprocess,numpy as np,csv
from pathlib import Path
p=Path('outputs/frontcourt_handoff_20260927'); root=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw');rows=[]
for name,t in [('match1_original.mp4',2828),('match1.mp4',115360*1001/60000)]:
 cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-ss',str(t),'-i',str(root/name),'-frames:v','30','-vf','scale=960:540','-pix_fmt','gray','-f','rawvideo','pipe:1']
 r=subprocess.run(cmd,capture_output=True,check=True);a=np.frombuffer(r.stdout,np.uint8).reshape(-1,540,960)[:,180:480,100:850]
 d=np.abs(np.diff(a.astype(np.int16),axis=0)).mean(axis=(1,2));even=float(np.median(d[::2]));odd=float(np.median(d[1::2]));print(name,'frames',len(a),'alternating medians',even,odd,'ratio',max(even,odd)/min(even,odd))
 for i,x in enumerate(d):rows.append(dict(file=name,clip_start_sec=t,adjacent_pair_index=i,mean_pixel_difference=float(x),scope='Different active-play intervals; not aligned same scene'))
with (p/'original_vs_edited_temporal_probe.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
