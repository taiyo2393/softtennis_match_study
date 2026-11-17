"""Render contact sheets for the two tracker pilots' full candidate windows."""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'data/raw/match1.mp4'
cap=cv2.VideoCapture(str(SOURCE))
fps=cap.get(cv2.CAP_PROP_FPS)
for name, a, b in [('044',114635,115804),('056',153926,154915)]:
    frames=list(range(a,b+1,round(fps)))
    frames.extend(range(a,a+round(fps*5),round(fps/4)))
    frames.extend(range(b-round(fps*5),b+1,round(fps/4)))
    frames=sorted(set(f for f in frames if a<=f<=b))
    tiles=[]
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,f)
        ok,frame=cap.read()
        if not ok: raise RuntimeError(f)
        tile=cv2.resize(frame,(640,360))
        cv2.rectangle(tile,(0,0),(200,32),(0,0,0),-1)
        cv2.putText(tile,f'F{f} {f/fps:.2f}s',(7,24),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2)
        tiles.append(tile)
    for page in range((len(tiles)+15)//16):
        group=tiles[page*16:(page+1)*16]
        while len(group)<16:group.append(np.zeros_like(group[0]))
        sheet=np.vstack([np.hstack(group[i:i+4]) for i in range(0,16,4)])
        cv2.imwrite(str(OUT/f'rally_{name}_{page:02}.jpg'),sheet)
    print(name,len(frames),len(tiles))
cap.release()
