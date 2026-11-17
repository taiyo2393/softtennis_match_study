import cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;cap=cv2.VideoCapture(r'C:\Users\ta239\Research\softtennis_match_study\study\data\raw\match1.mp4');tiles=[]
for t in np.linspace(2099.2,2100.2,12):
 f=round(t*60000/1001);cap.set(1,f);ok,im=cap.read();assert ok;crop=cv2.resize(im[400:1600,1100:1900],(300,450));cv2.rectangle(crop,(0,0),(300,30),(0,0,0),-1);cv2.putText(crop,f'{t:.3f} F{f}',(5,21),0,.5,(255,255,255),1);tiles.append(crop)
cv2.imwrite(str(p/'048_revised_contact_detail.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,12,4)]));cap.release()
