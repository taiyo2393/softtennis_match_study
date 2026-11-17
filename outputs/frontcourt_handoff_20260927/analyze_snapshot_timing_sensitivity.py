import csv,json,math,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;root=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study');H=np.array(json.loads((root/'config/match1_court_plane_calibration.json').read_text())['image_to_court_homography'],np.float32)
allpos=[];features=[];summ=[]
specs=[('044',115395,115403,p/'full_rally_044/bytetrack.csv',{1:'near_rear',2:'near_front',20:'far_front',422:'far_rear'}),('056',154111,154116,p/'serve_phase_056_deferred/bytetrack_four_player_reconnection.csv',{1:'near_front',2:'near_rear',4:'far_front',42:'far_rear'})]
for n,lo,hi,path,slots in specs:
 with path.open(encoding='utf-8-sig',newline='') as f:rs=[r for r in csv.DictReader(f) if lo<=int(r['source_frame'])<=hi and int(r['track_id']) in slots]
 for frame in range(lo,hi+1):
  rr=[r for r in rs if int(r['source_frame'])==frame];assert len(rr)==4,(n,frame,len(rr));coords={}
  for r in rr:
   x,y=cv2.perspectiveTransform(np.float32([[[float(r['bottom_center_x']),float(r['bottom_center_y'])]]]),H)[0,0];slot=slots[int(r['track_id'])];coords[slot]=(float(x),float(y));allpos.append(dict(candidate_id=n,source_frame=frame,slot=slot,tracker_id=r['track_id'],x=float(x),y=float(y),status='candidate_bbox_bottom_not_ground_truth'))
  row=dict(candidate_id=n,source_frame=frame)
  for side in ['near','far']:
   a,b=coords[side+'_front'],coords[side+'_rear'];row[side+'_lateral_gap']=abs(a[0]-b[0]);row[side+'_longitudinal_gap']=abs(a[1]-b[1])
  features.append(row)
 for key in ['near_lateral_gap','near_longitudinal_gap','far_lateral_gap','far_longitudinal_gap']:
  vals=[r[key] for r in features if r['candidate_id']==n];summ.append(dict(candidate_id=n,feature=key,minimum=min(vals),maximum=max(vals),range=max(vals)-min(vals),normalization='x divided by court width; y by court length; no mixed-axis distance',status='exploratory_timing_sensitivity_only'))
for name,rows in [('timing_sensitivity_positions.csv',allpos),('timing_sensitivity_features.csv',features),('timing_sensitivity_summary.csv',summ)]:
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
cap=cv2.VideoCapture(str(root/'data/raw/match1.mp4'))
for n,f,path,slots in [('044',115403,specs[0][3],specs[0][4]),('056',154116,specs[1][3],specs[1][4])]:
 cap.set(1,f);ok,im=cap.read();assert ok;cv2.imwrite(str(p/f'{n}_position_annotation_raw.jpg'),im)
 with path.open(encoding='utf-8-sig') as s:rr=[r for r in csv.DictReader(s) if int(r['source_frame'])==f and int(r['track_id']) in slots]
 cells=[]
 for r in rr:
  x,y=round(float(r['bottom_center_x'])),round(float(r['bottom_center_y']));crop=im[y-200:y+200,x-300:x+300].copy();assert crop.shape==(400,600,3)
  cv2.circle(crop,(300,200),5,(0,255,255),2);cv2.putText(crop,f'{slots[int(r["track_id"])]} bbox bottom',(8,30),0,.65,(0,255,255),2);cells.append(crop)
 cv2.imwrite(str(p/f'{n}_footpoint_candidate_review.jpg'),np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])]))
cap.release()
print(json.dumps(summ,indent=2))
