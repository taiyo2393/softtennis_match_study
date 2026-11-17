import csv,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;rows=list(csv.DictReader((p/'uncertain_score_motion_queue.csv').open(encoding='utf-8-sig')));panels=[]
for r in rows:
 n=int(r['candidate_id'][-3:])-1;im=cv2.imread(str(p/f'scoreboard_scan_{n//3:02d}.jpg'));y=(n%3)*540
 a=im[y+36:y+288,:900];b=im[y+288:y+540,:900];title=np.zeros((28,1200,3),np.uint8);cv2.putText(title,f'{n+1:03} START / END+1s',(8,21),0,.6,(255,255,255),1);panels.append(np.vstack([title,np.hstack([cv2.resize(a,(600,168)),cv2.resize(b,(600,168))])]))
for i in range(3):cv2.imwrite(str(p/f'exception_scores_{i}.jpg'),np.vstack(panels[i*4:i*4+4]))
