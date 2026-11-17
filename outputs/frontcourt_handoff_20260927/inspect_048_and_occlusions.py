import csv,os,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;os.environ['YOLO_CONFIG_DIR']=str(p)
from ultralytics import YOLO
import torch
root=Path(r'C:\Users\ta239\Research\softtennis_match_study\study');cap=cv2.VideoCapture(str(root/'data/raw/match1.mp4'))
for n,center in [('005',8648),('006',10536)]:
 cells=[]
 for offset in [-24,-18,-12,-6,0,6,12,18,24]:
  f=center+offset;cap.set(1,f);ok,im=cap.read();assert ok;crop=cv2.resize(im[900:1450,1400:2500],(600,300));cv2.rectangle(crop,(0,0),(600,30),(0,0,0),-1);cv2.putText(crop,f'{n} F{f} delta{offset:+d}',(8,22),0,.6,(255,255,255),1);cells.append(crop)
 cv2.imwrite(str(p/f'{n}_occlusion_temporal_context.jpg'),np.vstack([np.hstack(cells[i:i+3]) for i in range(0,9,3)]))
f=125911;cap.set(1,f);ok,im=cap.read();assert ok;preview=cv2.resize(im,(1920,1080));model=YOLO(root/'models/yolo11m.pt');pred=model.predict(im,classes=[0],conf=.25,imgsz=1280,device=0 if torch.cuda.is_available() else 'cpu',verbose=False)[0];rows=[]
for i,b in enumerate(pred.boxes):
 x1,y1,x2,y2=b.xyxy[0].cpu().tolist();x=(x1+x2)/2
 if not (850<=y2<=1850 and 350<=x<=3500):continue
 rows.append(dict(candidate_id='048',source_frame=f,detection_index=i,x1=x1,y1=y1,x2=x2,y2=y2,confidence=float(b.conf.item())))
 cv2.rectangle(preview,(int(x1/2),int(y1/2)),(int(x2/2),int(y2/2)),(0,255,255),2);cv2.putText(preview,f'D{i}',(int(x1/2),int(y1/2)-7),0,.7,(0,255,255),2)
cv2.putText(preview,'048 F125911 provisional service bracket +30F',(20,1040),0,.7,(255,255,255),2);cv2.imwrite(str(p/'048_refined_anchor_person_candidates.jpg'),preview)
with (p/'048_snapshot_person_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
cap.release();print('048 detections and two occlusion sequences saved')
