"""Aspect-preserving coarse timeline; does not assign contact ground truth."""
from pathlib import Path
import cv2
import numpy as np
OUT=Path(__file__).resolve().parent
SOURCE=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw\match1.mp4')
cap=cv2.VideoCapture(str(SOURCE))
tiles=[]
for index in range(114635,115806,30):
    cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,frame=cap.read()
    assert ok
    tile=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile,(0,330),(240,360),(0,0,0),-1)
    cv2.putText(tile,f'F{index}',(8,352),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    tiles.append(tile)
for page in range((len(tiles)+11)//12):
    group=tiles[page*12:(page+1)*12]
    while len(group)<12:
        group.append(np.zeros_like(tiles[0]))
    sheet=np.vstack([np.hstack(group[i:i+3]) for i in range(0,12,3)])
    cv2.imwrite(str(OUT/f'044_serve_sequence_{page}.jpg'),sheet)
for name,first,last in [('first',114720,114770),('second',115345,115395)]:
    group=[]
    for index in range(first,last+1,2 if name=='second' else 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=cap.read()
        assert ok
        tile=cv2.resize(frame[700:1400,1200:1900],(350,350),interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile,(0,320),(240,350),(0,0,0),-1)
        cv2.putText(tile,f'F{index}',(8,344),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        group.append(tile)
    while len(group)%4:
        group.append(np.zeros_like(group[0]))
    sheet=np.vstack([np.hstack(group[i:i+4]) for i in range(0,len(group),4)])
    cv2.imwrite(str(OUT/f'044_{name}_attempt_detail.jpg'),sheet)
cap.release()
