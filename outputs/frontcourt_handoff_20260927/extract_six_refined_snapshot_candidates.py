import csv,os,json
from pathlib import Path
p=Path(__file__).resolve().parent;os.environ['YOLO_CONFIG_DIR']=str(p);import cv2,torch
from ultralytics import YOLO
root=Path(r'C:\Users\ta239\Research\softtennis_match_study\study');spec=[('002',2223,2229),('003',4201,4223),('004',6757,6773),('005',8607,8618),('006',10495,10506),('007',15552,15563)];contacts=[];detections=[];cap=cv2.VideoCapture(str(root/'data/raw/match1.mp4'));model=YOLO(root/'models/yolo11m.pt')
for n,a,b in spec:
 f=b+30;contacts.append(dict(candidate_id='match1_rally_window_'+n,contact_candidate_first_frame=a,contact_candidate_last_frame=b,sample_candidate_frame=f,actual_post_contact_min_sec=30*1001/60000,actual_post_contact_max_sec=(b+30-a)*1001/60000,review_status='ai_visual_bracket_human_pending',effective_serve_status='sequence_supported_not_human_confirmed',evidence=f'{n}_effective_service_fine_grid.jpg'))
 cap.set(1,f);ok,im=cap.read();assert ok;preview=cv2.resize(im,(1920,1080));r=model.predict(im,classes=[0],conf=.25,imgsz=1280,device=0 if torch.cuda.is_available() else 'cpu',verbose=False)[0]
 for i,box in enumerate(r.boxes):
  x1,y1,x2,y2=box.xyxy[0].cpu().tolist();x=(x1+x2)/2
  if not (850<=y2<=1850 and 350<=x<=3500):continue
  detections.append(dict(candidate_id=n,source_frame=f,detection_index=i,x1=x1,y1=y1,x2=x2,y2=y2,confidence=float(box.conf.item()),selection_status='unreviewed_person_detection_not_four_player_identity'))
  cv2.rectangle(preview,(int(x1/2),int(y1/2)),(int(x2/2),int(y2/2)),(0,255,255),2);cv2.putText(preview,f'D{i}',(int(x1/2),int(y1/2)-6),0,.6,(0,255,255),2)
 cv2.putText(preview,f'{n} F{f}: provisional contact bracket +30F',(20,1040),0,.7,(255,255,255),2);cv2.imwrite(str(p/f'{n}_refined_anchor_person_candidates.jpg'),preview)
cap.release()
for name,rows in [('six_service_contact_brackets.csv',contacts),('six_service_snapshot_person_candidates.csv',detections)]:
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print('6 provisional contact brackets and snapshot person detections saved; no human labels changed; 048 remains unresolved')
