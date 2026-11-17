import cv2,json
from pathlib import Path
p=Path(__file__).resolve().parent;cap=cv2.VideoCapture(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw\match1.mp4')
for n,f in [('001',117),('007',15593),('056',154116)]:
 cap.set(1,f);ok,im=cap.read();assert ok;cv2.imwrite(str(p/f'benchmark_{n}_raw.jpg'),im)
cap.release()
