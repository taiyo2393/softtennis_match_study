import cv2,numpy as np,csv
from pathlib import Path
p=Path('outputs/frontcourt_handoff_20260927');cap=cv2.VideoCapture(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw\match1.mp4');rows=[]
for label,first,last,box in [('044',115360,115380,(1250,750,1750,1250)),('056',154075,154091,(750,700,1550,1750))]:
 prev=None
 cap.set(1,first)
 for f in range(first,last+1):
  ok,im=cap.read();assert ok
  x1,y1,x2,y2=box;roi=im[y1:y2,x1:x2]
  if prev is not None:
   d=np.abs(roi.astype(np.int16)-prev.astype(np.int16));rows.append(dict(candidate=label,previous_frame=f-1,source_frame=f,roi_mean_absolute_difference=round(float(d.mean()),5),fraction_channel_values_changed=round(float(np.mean(d!=0)),5)))
  prev=roi.copy()
cap.release()
with (p/'contact_frame_adjacent_image_difference.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for r in rows:print(r)
