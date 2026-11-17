import csv,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;r=list(csv.DictReader((p/'rally_label_review.csv').open(encoding='utf-8-sig')));cap=cv2.VideoCapture(r'C:\Users\ta239\Research\softtennis_match_study\study\data\raw\match1.mp4');manifest=[]
for x in r:
 n=x['candidate_id'][-3:]
 if n not in ['002','003','004','005','006','007','048']:continue
 start=float(x['candidate_start_sec']);tiles=[]
 for offset in [-2,-1,0,1,2,3,4,5,6]:
  t=start+offset;f=round(t*60000/1001);cap.set(1,f);ok,im=cap.read();assert ok;im=cv2.resize(im,(640,360));cv2.putText(im,f'{n} t{t:.1f} F{f}',(8,345),0,.6,(0,255,255),2);tiles.append(im);manifest.append(dict(candidate_id=n,time_sec=t,source_frame=f,evidence=f'{n}_service_context_grid.jpg'))
 cv2.imwrite(str(p/f'{n}_service_context_grid.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,9,3)]))
cap.release()
with (p/'service_context_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=manifest[0]);w.writeheader();w.writerows(manifest)
print('7 context grids, 63 timestamped frames saved; no contacts confirmed')
