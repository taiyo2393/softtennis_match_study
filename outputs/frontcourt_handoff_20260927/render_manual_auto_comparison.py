import csv,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent
with (p/'manual_auto_position_comparison.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
cells=[]
for r in rows:
 im=cv2.imread(str(p/f'benchmark_{r["candidate_id"]}_raw.jpg'));x,y=round(float(r['auto_x_px'])),round(float(r['auto_y_px']));left,top=x-200,y-120;crop=im[top:top+240,left:left+400].copy()
 cv2.drawMarker(crop,(200,120),(0,255,255),cv2.MARKER_CROSS,18,2);mx,my=round(float(r['manual_x_px']))-left,round(float(r['manual_y_px']))-top;cv2.circle(crop,(mx,my),7,(255,255,0),2);cv2.line(crop,(200,120),(mx,my),(255,255,255),1);cv2.rectangle(crop,(0,0),(400,30),(0,0,0),-1);cv2.putText(crop,f'{r["candidate_id"]} {r["image_slot"]} {float(r["image_distance_px"]):.1f}px',(5,22),0,.48,(255,255,255),1);cells.append(crop)
cv2.imwrite(str(p/'manual_auto_footpoint_comparison.jpg'),np.vstack([np.hstack(cells[i:i+4]) for i in range(0,12,4)]))
