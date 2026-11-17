from pathlib import Path
import cv2,numpy as np
p=Path(__file__).resolve().parent
cap=cv2.VideoCapture(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw\match1.mp4')
for label,first,last,box in [('044',115365,115373,(1250,750,1750,1250)),('056',154079,154087,(750,700,1550,1750))]:
 cells=[]
 for f in range(first,last+1):
  cap.set(1,f);ok,im=cap.read();assert ok
  x1,y1,x2,y2=box; crop=im[y1:y2,x1:x2]
  if label=='056':crop=cv2.resize(crop,(400,525))
  cv2.rectangle(crop,(0,0),(220,36),(0,0,0),-1);cv2.putText(crop,f'F{f}',(10,27),0,.8,(255,255,255),2);cells.append(crop)
 cv2.imwrite(str(p/f'{label}_contact_every_frame_review.jpg'),np.vstack([np.hstack(cells[i:i+3]) for i in range(0,9,3)]))
cap.release()
