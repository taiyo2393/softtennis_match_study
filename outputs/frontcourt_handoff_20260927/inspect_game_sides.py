import csv,cv2,numpy as np
from pathlib import Path
p=Path('outputs/frontcourt_handoff_20260927'); rows=list(csv.DictReader((p/'rally_label_review.csv').open(encoding='utf-8-sig')))
cap=cv2.VideoCapture(r'C:\Users\ta239\Research\softtennis_match_study\study\data\raw\match1.mp4'); tiles=[]
for n in [1,5,15,21,26,33,39,44,53,55,56]:
 f=round((float(rows[n-1]['candidate_start_sec'])+2)*60000/1001); cap.set(1,f); ok,im=cap.read(); assert ok
 im=cv2.resize(im,(960,540)); cv2.putText(im,f'{n:03} F{f}',(20,520),0,1,(0,255,255),2); cv2.imwrite(str(p/f'game_side_probe_{n:03}.jpg'),im); tiles.append(im)
while len(tiles)%3: tiles.append(np.zeros((540,960,3),np.uint8))
cv2.imwrite(str(p/'game_side_probe_overview.jpg'),np.vstack([np.hstack(tiles[j:j+3]) for j in range(0,len(tiles),3)]));cap.release()
