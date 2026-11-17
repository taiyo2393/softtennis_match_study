import cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;cap=cv2.VideoCapture(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw\match1.mp4');fps=60000/1001
for n,times in [('025',[990,994,998,1000,1002,1004,1006,1008,1010,1012,1014,1016]),('048',[2095,2098,2102,2106,2110,2114,2118,2120,2122,2124,2126,2130]),('050',[2254,2258,2262,2266,2270,2274,2278,2282,2286,2290,2294,2298])]:
 cells=[]
 for t in times:
  f=round(t*fps);cap.set(1,f);ok,im=cap.read();assert ok;im=cv2.resize(im,(640,360));cv2.putText(im,f'{n} t{t} F{f}',(8,345),0,.65,(0,255,255),2);cells.append(im)
 cv2.imwrite(str(p/f'{n}_expanded_boundary_context.jpg'),np.vstack([np.hstack(cells[i:i+3]) for i in range(0,12,3)]))
cap.release()
