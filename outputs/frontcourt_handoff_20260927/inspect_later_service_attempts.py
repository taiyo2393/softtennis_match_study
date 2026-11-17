import cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;cap=cv2.VideoCapture(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw\match1.mp4')
for n,ts in [('002',range(34,43)),('004',np.arange(108.5,117,1)),('005',np.arange(138.5,147,1)),('007',range(253,262))]:
 cells=[]
 for t in ts:
  f=round(float(t)*60000/1001);cap.set(1,f);ok,im=cap.read();assert ok;im=cv2.resize(im,(640,360));cv2.putText(im,f'{n} t{float(t):.1f} F{f}',(8,345),0,.6,(0,255,255),2);cells.append(im)
 cv2.imwrite(str(p/f'{n}_later_service_context_grid.jpg'),np.vstack([np.hstack(cells[i:i+3]) for i in range(0,9,3)]))
cap.release()
