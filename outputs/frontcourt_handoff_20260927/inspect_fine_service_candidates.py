import csv,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;rs=list(csv.DictReader((p/'effective_service_search_intervals.csv').open(encoding='utf-8-sig')));cap=cv2.VideoCapture(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw\match1.mp4');boxes={'002':(200,500,1000,1700),'003':(1800,500,2600,1700),'004':(950,500,1750,1700),'005':(2650,500,3450,1700),'006':(650,500,1450,1700),'007':(2400,500,3200,1700),'048':(1500,400,2300,1600)};manifest=[]
for r in rs:
 n=r['candidate_id'][-3:];a=float(r['service_leading_to_play_search_start_sec']);b=float(r['service_leading_to_play_search_end_sec']);cells=[]
 for t in np.linspace(a,b,12):
  f=round(t*60000/1001);cap.set(1,f);ok,im=cap.read();assert ok;x1,y1,x2,y2=boxes[n];im=cv2.resize(im[y1:y2,x1:x2],(300,450));cv2.rectangle(im,(0,0),(300,30),(0,0,0),-1);cv2.putText(im,f'{t:.3f} F{f}',(5,21),0,.5,(255,255,255),1);cells.append(im);manifest.append(dict(candidate_id=n,time_sec=t,source_frame=f,evidence=f'{n}_effective_service_fine_grid.jpg'))
 cv2.imwrite(str(p/f'{n}_effective_service_fine_grid.jpg'),np.vstack([np.hstack(cells[i:i+4]) for i in range(0,12,4)]))
cap.release()
with (p/'effective_service_fine_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=manifest[0]);w.writeheader();w.writerows(manifest)
