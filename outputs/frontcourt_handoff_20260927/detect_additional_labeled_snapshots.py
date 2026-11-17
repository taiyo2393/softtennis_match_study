import os,csv,json
from pathlib import Path
p=Path(__file__).resolve().parent;os.environ['YOLO_CONFIG_DIR']=str(p/'ultralytics_local')
import cv2,torch
from ultralytics import YOLO
root=Path(r'C:\Users\ta239\Research\softtennis_match_study\study');model=YOLO(root/'models/yolo11m.pt');cap=cv2.VideoCapture(str(root/'data/raw/match1.mp4'));rows=[]
for n,t in [('001',1.450950),('025',1000.601100)]:
 f=round(t*60000/1001)+30;cap.set(1,f);ok,im=cap.read();assert ok;preview=cv2.resize(im,(1920,1080));result=model.predict(im,classes=[0],conf=.25,imgsz=1280,device=0 if torch.cuda.is_available() else 'cpu',verbose=False)[0]
 for i,b in enumerate(result.boxes):
  x1,y1,x2,y2=b.xyxy[0].cpu().tolist();x,y=(x1+x2)/2,y2
  # Wide court vicinity for review, never equated with four match players.
  if not (850<=y<=1850 and 350<=x<=3500):continue
  rows.append(dict(candidate_id=n,sample_frame=f,detection_index=i,x1=x1,y1=y1,x2=x2,y2=y2,confidence=float(b.conf.item()),player_selection_status='unreviewed_person_detection_only',anchor_status='legacy_contact_validity_unreviewed'))
  cv2.rectangle(preview,(round(x1/2),round(y1/2)),(round(x2/2),round(y2/2)),(0,255,255),2);cv2.putText(preview,f'D{i}',(round(x1/2),round(y1/2)-8),0,.7,(0,255,255),2)
 cv2.putText(preview,f'{n} F{f}: legacy serve +30F; validity pending',(20,1040),0,.7,(255,255,255),2);cv2.imwrite(str(p/f'{n}_legacy_anchor_snapshot_candidates.jpg'),preview)
cap.release()
with (p/'additional_snapshot_person_detections.csv').open('w',encoding='utf-8-sig',newline='') as s:w=csv.DictWriter(s,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print('Snapshot detections saved; no tracker identities, validity or ground truth inferred')
