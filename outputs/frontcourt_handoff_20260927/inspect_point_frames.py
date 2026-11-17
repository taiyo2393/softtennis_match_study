"""Render fine-grained original-frame sheets around serve and point endings."""
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
OUT=Path(__file__).resolve().parent
windows={
    '044_start':(114635,114900,10),
    '044_end':(115735,115815,4),
    '056_start':(153926,154126,8),
    '056_end':(154840,154970,5),
}
cap=cv2.VideoCapture(str(ROOT/'data/raw/match1.mp4'))
for name,(start,end,step) in windows.items():
    images=[]
    for frame in range(start,end+1,step):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,img=cap.read()
        if not ok: raise RuntimeError(frame)
        # Show the full playing court with frame id and original broadcast score.
        thumb=cv2.resize(img,(960,540))
        cv2.rectangle(thumb,(0,0),(180,27),(0,0,0),-1)
        cv2.putText(thumb,str(frame),(5,21),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        images.append(thumb)
    for page in range((len(images)+7)//8):
        group=images[page*8:(page+1)*8]
        while len(group)<8:group.append(np.zeros_like(group[0]))
        sheet=np.vstack([np.hstack(group[i:i+2]) for i in range(0,8,2)])
        cv2.imwrite(str(OUT/f'{name}_{page}.jpg'),sheet)
    print(name,len(images))
cap.release()
