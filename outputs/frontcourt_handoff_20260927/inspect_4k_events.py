"""Crop the original 4K frames around visible serves and endings."""
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT=Path(__file__).resolve().parent
# Crop coordinates are in the 3840x2160 source image.
events={
    '044_serve':(114660,114745,4,(1200,350,2050,1450)),
    '056_serve':(154065,154103,2,(550,850,1550,2100)),
    '044_finish':(115768,115791,2,(2450,1050,3840,2160)),
    '056_finish':(154865,154910,3,(1150,350,2450,1450)),
}
cap=cv2.VideoCapture(str(ROOT/'data/raw/match1.mp4'))
for name,(first,last,step,(x1,y1,x2,y2)) in events.items():
    images=[]
    for idx in range(first,last+1,step):
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,frame=cap.read()
        if not ok:raise RuntimeError(idx)
        crop=frame[y1:y2,x1:x2]
        tile=cv2.resize(crop,(540,600))
        cv2.rectangle(tile,(0,0),(160,30),(0,0,0),-1)
        cv2.putText(tile,str(idx),(5,23),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        images.append(tile)
    for page in range((len(images)+11)//12):
        group=images[page*12:(page+1)*12]
        while len(group)<12:group.append(np.zeros_like(group[0]))
        sheet=np.vstack([np.hstack(group[i:i+4]) for i in range(0,12,4)])
        cv2.imwrite(str(OUT/f'{name}_4k_{page}.jpg'),sheet)
    print(name,len(images))
cap.release()
